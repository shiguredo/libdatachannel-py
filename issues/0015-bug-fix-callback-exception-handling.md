# Python callback 内の例外の扱いが経路ごとに不統一である

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-callback-exception-handling
- Polished: {YYYY-MM-DD}

## 目的

Python callback 内で例外が発生した場合の振る舞いが経路ごとに異なる。threadpool 上の callback は libdatachannel 側で PLOG_WARNING のログ 1 行のみで future に格納され、誰も get しないため例外が消える。例外を利用者に観測可能な形に統一する。

## 優先度根拠

- 利用者のコールバック内のバグ (例外) がログ 1 行で消えるため、デバッグが困難
- 経路ごとの不統一は、将来の修正 (IceUdpMuxListener 経路の対応等) の判断を難しくする

## 現状

- threadpool 上で実行される callback (state change 等の内部 task 経由) の例外は、libdatachannel の threadpool (`src/impl/threadpool.hpp`) が PLOG_WARNING でログして rethrow し、packaged_task の future に格納されるが、get する呼び出し元が存在しない
- Channel / PeerConnection / WebSocketServer の callback は libdatachannel 側の try/catch 内で呼ばれる
- binding 側はどの経路でも、例外を利用者に提示する仕組みを持たない
- IceUdpMuxListener 経路 (try/catch 自体が無い) は [[0012-bug-fix-ice-udp-mux-listener-callback-exception]] で対応する

## 設計方針

- 方針を先に決める:
  - 案 A: 例外を握らず、logging (英語) で明示的に記録する
  - 案 B: 例外を再送出する (伝播先の threadpool future で消えるため、意味が薄い)
  - 案 C: 例外を専用の on_error callback にルーティングする
- リポジトリの規約 (エラーメッセージは英語、握り潰しの回避) を踏まえ、案 A をベースにログレベルと文言を統一する
- callback 内例外の扱いを README または docstring で明文化する

## 完了条件

- callback 内例外の扱いがドキュメント化され、経路ごとの差異が明記されていること
- threadpool 経路で例外が観測可能になること (テストまたは検証手順で確認)
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `PyMediaHandlerImpl`、`bind_plihandler`、`bind_rembhandler` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/impl/threadpool.hpp`
- 関連 issue: [[0012-bug-fix-ice-udp-mux-listener-callback-exception]]、[[0005-bug-fix-destructor-callback-deadlock]]
