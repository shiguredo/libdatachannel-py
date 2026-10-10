"""callback リークの回帰テスト

callback が wrapper 自身を捕捉していても、 close() を呼べば解放されることを確認する。
nanobind のリーク警告はインタプリタ終了時に stderr へ出るため、 子プロセスで実行する。
"""

import subprocess
import sys
from pathlib import Path

# 子プロセスの正当な実行時間より十分大きい値
_LEAK_REPRODUCTION_TIMEOUT = 60


def _run_leak_reproduction() -> subprocess.CompletedProcess[str]:
    """callback リークの再現スクリプトを子プロセスで実行する"""
    script = Path(__file__).with_name("leak_reproduction_callbacks.py")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=_LEAK_REPRODUCTION_TIMEOUT,
        check=False,
    )


def test_callback_closure_is_released_by_close() -> None:
    """callback が wrapper を捕捉していても close() で解放されること

    nanobind の std::function caster は callable を強参照するため、 wrapper を捕捉する
    callback を登録したまま close() を呼ばないと、 Python の GC から見えない循環ができて
    インスタンスが解放されず、 終了時に "nanobind: leaked" が stderr へ出る。
    """
    result = _run_leak_reproduction()

    assert result.returncode == 0, (
        f"リークの再現スクリプトが失敗した: returncode={result.returncode} "
        f"stderr={result.stderr[-2000:]}"
    )
    assert "nanobind: leaked" not in result.stderr, (
        f"callback を登録したインスタンスが解放されていない: stderr={result.stderr[-2000:]}"
    )
