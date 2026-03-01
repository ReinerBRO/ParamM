import os
import threading
from typing import List, Optional

from openai import OpenAI


_dotenv_loaded = False
_dotenv_lock = threading.Lock()
_relay_key_idx = 0
_relay_key_lock = threading.Lock()
_openai_client = None
_openai_client_sig = None
_openai_client_lock = threading.Lock()


def _load_dotenv_if_present(path: str = ".env") -> None:
    """Load .env values once without overriding already exported variables."""
    global _dotenv_loaded
    with _dotenv_lock:
        if _dotenv_loaded:
            return
        if not os.path.exists(path):
            _dotenv_loaded = True
            return
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        _dotenv_loaded = True


def _disable_unsupported_socks_proxy() -> None:
    proxy_vars = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ]
    has_socks_proxy = any(
        (os.getenv(v) or "").lower().startswith("socks") for v in proxy_vars
    )
    if not has_socks_proxy:
        return
    try:
        import socksio  # type: ignore  # noqa: F401
    except ImportError:
        for v in proxy_vars:
            if (os.getenv(v) or "").lower().startswith("socks"):
                os.environ.pop(v, None)


def get_relay_api_keys() -> List[str]:
    """Collect OpenAI/relay keys from env in priority order."""
    _load_dotenv_if_present()
    keys: List[str] = []
    primary = os.getenv("OPENAI_API_KEY")
    if primary:
        keys.append(primary)
    for i in range(2, 26):
        k = os.getenv(f"ZZZ_API_KEY_{i}")
        if k:
            keys.append(k)
    # Deduplicate while keeping order
    seen = set()
    uniq = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def get_next_api_key(round_robin: bool = True) -> Optional[str]:
    """Pick one API key, optionally rotating across relay keys."""
    global _relay_key_idx
    keys = get_relay_api_keys()
    if not keys:
        return None
    if not round_robin or len(keys) == 1:
        return keys[0]
    with _relay_key_lock:
        key = keys[_relay_key_idx % len(keys)]
        _relay_key_idx += 1
    return key


def get_openai_base_url() -> Optional[str]:
    """Resolve base URL. If relay keys are present, default to zhizengzeng."""
    _load_dotenv_if_present()
    base = os.getenv("OPENAI_BASE_URL")
    if base:
        return base
    if any(os.getenv(f"ZZZ_API_KEY_{i}") for i in range(2, 26)):
        return "https://api.zhizengzeng.com/v1"
    return None


def get_openai_client(round_robin: bool = False) -> OpenAI:
    """
    Build OpenAI-compatible client.
    - round_robin=True: return a fresh client with rotated API key.
    - round_robin=False: reuse a singleton client.
    """
    global _openai_client, _openai_client_sig
    key = get_next_api_key(round_robin=round_robin)
    base_url = get_openai_base_url()
    _disable_unsupported_socks_proxy()
    kwargs = {}
    if key:
        kwargs["api_key"] = key
    if base_url:
        kwargs["base_url"] = base_url

    if round_robin:
        return OpenAI(**kwargs)

    sig = (key, base_url)
    with _openai_client_lock:
        if _openai_client is None or _openai_client_sig != sig:
            _openai_client = OpenAI(**kwargs)
            _openai_client_sig = sig
    return _openai_client
