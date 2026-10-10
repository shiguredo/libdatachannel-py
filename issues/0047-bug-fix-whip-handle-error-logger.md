# examples の handle_error が structlog の logger で AttributeError になる

- Priority: High
- Created: 2026-10-10
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-whip-handle-error-logger
- Polished: 2026-10-10

## 目的

`examples/whip.py` と `examples/whep.py` が共有する `handle_error` が、 例外を処理しようとした時点で `AttributeError` になる。 エラー時に呼ばれる関数がエラーになるため、 参照実装としての異常時の振る舞いが壊れている。 例外時に確実にログを残せるようにする。

## 優先度根拠

- 実測 (`uv run --no-sync` + timeout) で `handle_error` を直接呼ぶと `AttributeError: 'BoundLoggerFilteringAtInfo' object has no attribute 'isEnabledFor'` になる
- `_send_encoded_audio` は `audio_send_loop` スレッドから `handle_error` を呼ぶため、 例外が送出されると音声送信スレッドが停止する (`examples/whip.py` の on_output 経路でも同じ)
- `handle_error` は whep.py からも import されているため、 影響は 2 つの example に及ぶ
- examples は本ライブラリの参照実装であり、 利用者が同じパターンをコピーする

## 現状

- `handle_error` は `logger.error(f"Error {context}: {error}")` のあとに `if logger.isEnabledFor(logging.DEBUG): traceback.print_exc()` を実行する
- structlog は `structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))` で設定されており、 この wrapper に `isEnabledFor` は無い
- 実測: `handle_error("test", ValueError("x"))` → `logger.error` の出力後に `AttributeError`
- そのため例外時は、 ログは出るが `AttributeError` が送出されて呼び出し元のループが止まる

## 設計方針

- `handle_error` を structlog の API に合わせて書き直す。 `logger.error(...)` で出力し、 スタックトレースはレベル判定を自前でせず `logger.debug(..., exc_info=error)` に任せる (structlog は出力しないレベルでは例外を整形しないため、 `logging.DEBUG` の比較そのものが不要になる)
- `whip.py` の `handle_error` は `whep.py` から `from whip import handle_error` で共有されている。 共有したまま検証できるよう、 この関数を `examples/error_logging.py` に切り出し、 `whip.py` はそこから import する (whep.py の import はそのまま動く)
- 検証は `examples/whip.py` を import せずにできるようにする (`whip.py` は portaudio / uvc / webcodecs を import するため、 テストから直接読めない)。 `examples/rtp_timestamp.py` と同じく importlib で `examples/error_logging.py` を読み込み、 `handle_error` を呼ぶ
- `structlog.testing` などのテスト用の差し替えではなく、 `capsys` で実際の出力を確認する。 Debug 有効時のトレースバックは `structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(...))` で再現し、 `structlog.reset_defaults()` で元に戻す

## 完了条件

- `tests/test_error_logging.py` で `handle_error` を呼んでも `AttributeError` が出ないこと
- 例外時に `logger.error` の出力 (`Error <context>: <error>`) が残ること
- Debug ログが有効な場合はスタックトレースが出力され、 Info レベルでは出力されないこと
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `handle_error` (examples/whip.py、 examples/whep.py から import)
- 関連: 0010 (RTP timestamp の wrap 漏れ。 例外処理の前提が `handle_error` に依存している)
