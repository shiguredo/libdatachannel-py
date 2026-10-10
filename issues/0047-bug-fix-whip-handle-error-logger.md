# examples の handle_error が structlog の logger で AttributeError になる

- Priority: High
- Created: 2026-10-10
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-whip-handle-error-logger
- Polished: {YYYY-MM-DD}

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

- `handle_error` を structlog の API に合わせて書き直す。 Debug 時のスタックトレースは `logger.debug(..., exc_info=True)` のように structlog で扱える形にするか、 トレースバック出力をやめて `logger.error` だけにする
- `logging.DEBUG` の比較に structlog の logger を使わない (`structlog.is_configured()` や `logger.is_enabled_for` ではなく、 標準の `logging.getLogger()` を参照する形は避け、 structlog 側で完結させる)
- whep.py と重複しないよう、 共有する関数として 1 箇所だけ直す
- 例外時の振る舞いを検証できる形にする (examples は import が重いため、 検証方法を明確にする)

## 完了条件

- `handle_error` が `AttributeError` を出さないこと (`uv run --no-sync python -c` で `handle_error` を直接呼び、 例外が出ないことを確認する)
- 例外時に `logger.error` の出力が残ること
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `handle_error` (examples/whip.py、 examples/whep.py から import)
- 関連: 0010 (RTP timestamp の wrap 漏れ。 例外処理の前提が `handle_error` に依存している)
