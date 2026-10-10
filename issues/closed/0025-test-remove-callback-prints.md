# テスト系が callback 内 print を使い続けている

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-remove-callback-prints
- Polished: {YYYY-MM-DD}

## 目的

test_track や WebSocket 系テストが callback 内 print を使い続けている。issues/pending/0005 で実証済みの「pytest capture + callback 内 print → hang」経路を自前で踏んでおり、test_destruct_without_explicit_close (print を撤去済み) と不整合。

## 優先度根拠

- hang は pytest capture 設定次第で再現し得る (0005 の実証テーブルが記録している)
- 英語 print は「テストメッセージは全て日本語にすること」にも反する
- [[0024-refactor-deduplicate-peerconnection-tests]] (重複解消) と同時に着手すると効率的

## 現状

- test_peerconnection.py の test_track は 5 箇所のコールバックで print を使用する
- test_websocket.py / test_websocketserver.py も callback 内 print を使用する
- test_destruct_without_explicit_close は callback 内 print を一切行わない方針 (issues/pending/0005 由来) を実装済み
- callback 外の print ("Success" 等) も pytest capture 下では意味がない

## 設計方針

- callback 内 print を削除する。デバッグ出力が必要な箇所は pytest の capture と整合する手段に置き換えるか、テストとして意味のない出力は削除する
- callback 外の print も削除する
- [[0024-refactor-deduplicate-peerconnection-tests]] と同時着手する場合は、重複解消後の 1 箇所に集約する

## 完了条件

- callback 内 print が 0 件になること
- 全テストが pytest の標準 capture で PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `test_track` (tests/test_peerconnection.py)、`test_websocket` (tests/test_websocket.py)、`test_websocketserver` (tests/test_websocketserver.py)
- 関連 issue: [[0005-bug-fix-destructor-callback-deadlock]]、[[0024-refactor-deduplicate-peerconnection-tests]]、[[0028-fmt-fix-language-convention-violations]]
