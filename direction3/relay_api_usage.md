# Zhizengzeng Relay Usage (Memory-style)

This follows the same pattern used in Memory (`memorychain/utils.py` and `hebbian/utils.py`):
- create OpenAI client with `api_key` + `base_url`
- call `client.chat.completions.create(...)`
- rotate API keys to reduce single-key rate limits

## Environment

Load `.env` first:

```bash
cd /data/user/jzhu997/projects/ParamAgent
set -a
source ./.env
set +a
```

## Python Example

```python
import os
from openai import OpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

keys = [
    os.getenv(f"ZZZ_API_KEY_{i}")
    for i in range(2, 26)
    if os.getenv(f"ZZZ_API_KEY_{i}")
]
if not keys:
    raise RuntimeError("No ZZZ_API_KEY_2..25 found in environment")

# Simple round-robin key selection
api_key = keys[0]
client = OpenAI(api_key=api_key, base_url=BASE_URL)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "hello"}],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

## Yunwu Relay Usage

Yunwu is also OpenAI-compatible:

- `https://yunwu.ai`
- `https://yunwu.ai/v1`
- `https://yunwu.ai/v1/chat/completions`

### Quick curl test

```bash
curl https://yunwu.ai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${YUNWU_API_KEY}" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7
  }'
```

### Using 24 keys from `.yunwu`

Assume `.yunwu` contains one API key per line (24 lines total):

```bash
cd /data/user/user06/ParamAgent
mapfile -t YUNWU_KEYS < .yunwu

export OPENAI_BASE_URL="https://yunwu.ai/v1"

# Map into existing ParamAgent key slots used by run scripts
for i in $(seq 2 25); do
  idx=$((i - 2))
  export "ZZZ_API_KEY_${i}=${YUNWU_KEYS[$idx]}"
done
```

Python call is unchanged (OpenAI-compatible):

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("ZZZ_API_KEY_2"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://yunwu.ai/v1"),
)

resp = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "hello"}],
    temperature=0.7,
    max_tokens=128,
)
print(resp.choices[0].message.content)
```
