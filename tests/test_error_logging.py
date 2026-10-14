"""エラーログの共通処理のテスト

handle_error が structlog の API で動くこと (logging の API を使わないこと) と、
Debug ログが有効な場合だけスタックトレースを出力することを検証する。
"""

import importlib.util
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import structlog

# examples/ は配布物に含まれないため、 importlib で直接読み込む。
# examples/whip.py は portaudio などを import するため、 共有部分だけを読む
_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "error_logging.py"


def _load_error_logging() -> ModuleType:
    """examples/error_logging.py を読み込む"""
    spec = importlib.util.spec_from_file_location("error_logging", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["error_logging"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """structlog の設定をテストごとに戻す

    テストでログレベルを変えるため、 後続のテストに影響させない。
    """
    yield
    structlog.reset_defaults()


def test_handle_error_logs_error_message(capsys: pytest.CaptureFixture[str]) -> None:
    """handle_error が例外を投げずにエラーメッセージを出力すること

    structlog の logger は logging の isEnabledFor を持たないため、 レベル判定に
    使うと AttributeError になっていた。 ここでは出力内容だけを確認する。
    """
    error_logging = _load_error_logging()

    error_logging.handle_error("test context", ValueError("boom"))

    captured = capsys.readouterr()
    assert "Error test context: boom" in captured.out


def test_handle_error_logs_traceback_at_debug_level(capsys: pytest.CaptureFixture[str]) -> None:
    """Debug ログが有効な場合はスタックトレースが出力されること"""
    error_logging = _load_error_logging()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))

    try:
        raise ValueError("boom")
    except ValueError as error:
        error_logging.handle_error("test context", error)

    captured = capsys.readouterr()
    assert "Traceback (most recent call last)" in captured.out
    assert "ValueError: boom" in captured.out


def test_handle_error_suppresses_traceback_at_info_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Info レベルではスタックトレースを出力しないこと

    structlog は出力しないレベルでは例外を整形しないため、 レベル判定を自前で
    行う必要がない。
    """
    error_logging = _load_error_logging()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

    try:
        raise ValueError("boom")
    except ValueError as error:
        error_logging.handle_error("test context", error)

    captured = capsys.readouterr()
    assert "Error test context: boom" in captured.out
    assert "Traceback" not in captured.out
