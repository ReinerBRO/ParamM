import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'npu.sh'


def make_fake_ssh(tmp_path: Path) -> Path:
    stub = tmp_path / 'ssh'
    stub.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'target="$1"\n'
        'echo "SSH_TARGET=${target}"\n'
    )
    stub.chmod(0o755)
    return stub


def run_npu(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    make_fake_ssh(tmp_path)
    env = os.environ.copy()
    env['PATH'] = f"{tmp_path}:{env['PATH']}"
    return subprocess.run(
        ['bash', str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_supports_mutsumi_target(tmp_path: Path) -> None:
    result = run_npu(tmp_path, 'mutsumi')
    assert result.returncode == 0, result.stderr
    assert '===== mutsumi =====' in result.stdout
    assert 'SSH_TARGET=mutsumi' in result.stdout


def test_supports_eren_target(tmp_path: Path) -> None:
    result = run_npu(tmp_path, 'eren')
    assert result.returncode == 0, result.stderr
    assert '===== eren =====' in result.stdout
    assert 'SSH_TARGET=eren' in result.stdout


def test_accepts_host_arg_with_sh_suffix(tmp_path: Path) -> None:
    result = run_npu(tmp_path, 'mutsumi.sh')
    assert result.returncode == 0, result.stderr
    assert '===== mutsumi =====' in result.stdout
    assert 'SSH_TARGET=mutsumi' in result.stdout


def test_all_includes_all_four_nodes(tmp_path: Path) -> None:
    result = run_npu(tmp_path, 'all')
    assert result.returncode == 0, result.stderr
    for target in ('miku', 'yui', 'mutsumi', 'eren'):
        assert f'===== {target} =====' in result.stdout
        assert f'SSH_TARGET={target}' in result.stdout
