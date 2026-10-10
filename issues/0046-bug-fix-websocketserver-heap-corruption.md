# WebSocketServer のテストが Linux で間欠的に SIGABRT する (ヒープ破壊)

- Priority: High
- Created: 2026-10-10
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-websocketserver-heap-corruption
- Polished: {YYYY-MM-DD}

## 目的

`tests/test_websocketserver.py::test_websocketserver` が Linux (ubuntu-24.04 x86_64 / Python 3.14) で間欠的に SIGABRT して leg が落ちる。 glibc の malloc がヒープ破壊を検出した形で、 プロセスごと落ちる。 破壊箇所を特定して修正する。

## 優先度根拠

- 2026-10-09 の wheel ワークフロー (27 ジョブ中 24 success / 2 skipped) で、 ubuntu-24.04 x86_64 / Python 3.14 の leg だけが `Process completed with exit code 134` で失敗した。 同じ leg を再実行すると成功しており、 間欠的に起きる
- 落ちているのは `tests/test_websocketserver.py` の `test_websocketserver` が `server.stop()` を呼んでいる箇所で、 faulthandler のネイティブスタックは `libc.so.6` の `abort` から `__libc_message` 系に伸びていた。 glibc の malloc がヒープの破壊を検出して `abort()` した形である
- 同じ leg で他のテスト (packetizer の追加テストを含む 142 件) はすべて PASSED しており、 落ちるのは WebSocketServer のテストだけである
- libdatachannel の WebSocketServer は受け入れ用の thread から Python callback を呼び、 `stop()` は受け入れ thread を join するだけで確立済みの接続には触れない ([[0038-bug-fix-websocketserver-native-crash]] の調査で確認済み)。 0038 では mbedTLS をスレッドセーフにビルドして macOS の SIGTRAP を解消したが、 そのときの検証は安定性 10 回 + TLS probe 98 ラウンドでクラッシュ 0 (95% 上側限界 約 3%) であり、 残存率を 0 とは確認していない

## 現状

- 失敗した leg: wheel ワークフローの `build_ubuntu (ubuntu-24.04_x86_64, ubuntu-24.04, 3.14)`
- 症状: `Fatal Python error: Aborted` で exit code 134
- faulthandler が出力したスタックの要点
  - `tests/test_websocketserver.py` の `test_websocketserver` (`server.stop()` の呼び出し)
  - `Binary file "/lib/x86_64-linux-gnu/libc.so.6", at abort+0xdf`
  - `Binary file "/lib/x86_64-linux-gnu/libc.so.6", at +0x297b6` と `+0xa90d5` (glibc の `__libc_message` 系)
- `gh run rerun --failed` による再実行では成功した
- 手元 (macOS 26 arm64 / Python 3.12) では 0038 の修正以降に再現していない
- `tests/test_websocketserver.py` の `test_websocketserver` は `port = 0` で動的確保し、 callback の完了をイベントで待つ形になっている (0038 で変更済み)

### 原因 (未特定)

ヒープ破壊が起きていることはスタックから裏付けられるが、 破壊している箇所は特定できていない。 候補は次のとおり。

- libdatachannel の WebSocketServer の経路 (受け入れ thread から呼ぶ Python callback、 サーバー側 WebSocket の寿命)
- 0038 で mbedTLS をスレッドセーフにした後も残る並行初期化の経路
- Python と C++ をまたぐ参照の取り扱い (callback のクロージャと C++ の WebSocket の循環)

## 設計方針

- まず再現条件を絞る。 手元で再現しないため、 glibc の malloc 検査を強めた状態 (`MALLOC_CHECK_=3` と `MALLOC_PERTURB_`) で `tests/test_websocketserver.py` を繰り返し実行し、 Linux で再現するかどうかを確かめる
- 再現したら、 破壊を検出した時点の全 thread のスタック (`faulthandler.enable()` と `faulthandler.dump_traceback_later()`) と malloc のエラーメッセージ (`free(): invalid pointer` / `corrupted size vs. prev_size` など) を取得する
- 0038 と同じ観点 (mbedTLS のスレッド対応、 WebSocketServer の停止と破棄の順序、 callback の実行 thread) を一次資料で確認し、 破壊箇所を絞り込む
- 原因が binding 側にある場合は binding で、 libdatachannel 側にある場合は upstream への報告を前提にした回避を入れる
- 再現率が低いため、 修正後は再現条件で繰り返し実行して安定を確認する (試行回数と 0/N の 95% 上側限界を報告する)

## 完了条件

- 破壊箇所が特定されていること (特定できない場合は、 残る候補と次の調査手順が issue に記録されていること)
- 修正によって、 再現条件で繰り返し実行してもクラッシュしないこと (試行回数と 0/N の 95% 上側限界を報告する)
- CI (wheel.yml の leg) が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 失敗した leg: wheel ワークフローの `build_ubuntu (ubuntu-24.04_x86_64, ubuntu-24.04, 3.14)`
- 対象テスト: `tests/test_websocketserver.py` の `test_websocketserver`
- libdatachannel v0.24.0: `source/src/impl/websocketserver.cpp` (受け入れ thread と `stop()`)、 `source/src/impl/channel.cpp` (callback の実行)
- 関連 issue: [[0038-bug-fix-websocketserver-native-crash]] (mbedTLS のスレッド対応)、 [[0039-bug-fix-nanobind-del-not-called]] (callback の寿命)
