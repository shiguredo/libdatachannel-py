# WebSocketServer のテストが稀に native crash する

- Priority: Medium
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-websocketserver-native-crash
- Polished: {YYYY-MM-DD}

## 目的

tests/test_websocketserver.py::test_websocketserver を実行したとき、 稀に native crash でテストプロセスごと落ちる。 落ちるとテストスイート全体が中断し、 CI の leg が失敗してリリースの公開も止まる。 原因を特定して解消する。

## 優先度根拠

- 発生するとテストプロセスごと落ちるため、 pytest では検出できない (pytest-timeout も rerun プラグインも効かない)。 CI ではテストスイート全体を 1 回だけやり直す暫定対応で回避しており、 根本対応が無い
- リリース判定 (タグ push) では wheel の leg が 1 つでも失敗すると publish がブロックされるため、 無視できない
- [[0003-bug-fix-websocketserver-destructor-gil-release]] は destructor の GIL hang、 [[0002-bug-fix-websocket-destructor-gil-release]] は WebSocket の destructor を扱っており、 本 issue は native crash という別の症状

## 現状

- 実測 1: macOS 26 arm64 / Python 3.14 / ビルドした wheel を install した環境で、 テストスイート全体の実行中に `tests/test_websocketserver.py::test_websocketserver` のテスト名が出力された約 18 ms 後に `Trace/BPT trap: 5` (SIGTRAP)。 終了コード 133。 同じコマンドの再実行 21 回は成功 (頻度 1/22)
- 実測 2: CI (wheel.yml の build_macos / macos-26_arm64 / 3.14) で同じ症状により leg が失敗。 他の leg は成功
- クラッシュ時のスタックは未取得 (core dump 未採取)
- 対象テストは 127.0.0.1 の固定ポート 48080 を使い、 15 秒のポーリングと末尾での明示破棄を行う。 callback 内に print が残っている ([[0025-test-remove-callback-prints]] の対象)
- WebSocketServer の binding は `src/bind_libdatachannel.cpp` の `bind_websocketserver` にある

## 設計方針

- まず再現条件を特定する (OS / Python 版 / free-threading か否か / 実行順序 / ポート再利用の有無)。 再現用のスクリプトを用意し、 core dump またはサンプリングでクラッシュ時のスタックを取る
- 原因に応じて修正する (native 側の破棄順序、 コールバックからの Python 呼び出し、 サーバーの停止手順など)
- 修正後は、 クラッシュしていた条件で繰り返し実行して安定を確認する
- CI の再試行 (テストスイート全体を 1 回だけやり直す暫定対応) は、 本 issue の修正が入るまで残す

## 完了条件

- 再現条件が特定され、 修正によって 100 回連続でクラッシュしないこと
- CI の再試行を外しても安定して PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: tests/test_websocketserver.py、 src/bind_libdatachannel.cpp の `bind_websocketserver`
- 関連 issue: [[0002-bug-fix-websocket-destructor-gil-release]]、 [[0003-bug-fix-websocketserver-destructor-gil-release]]、 [[0025-test-remove-callback-prints]]
