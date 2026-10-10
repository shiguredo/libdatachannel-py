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

### 原因 (特定済み)

`CMakeLists.txt` の「古い `_deps` を捨てる」ガードが、 無効化されたままの行に誤マッチして発動しないため、 スレッド非対応の mbedTLS が CI で使われ続けていた。

- ガードは `if(NOT MBEDTLS_CONFIG_CONTENT MATCHES "#define MBEDTLS_THREADING_C")` で、 部分一致のため無効化されたままの行 `//#define MBEDTLS_THREADING_C` にも一致する。 `cmake -P` で誤マッチを再現した
- `wheel.yml` は `restore-keys` の前方一致で `_deps` を復元するため、 `hashFiles('CMakeLists.txt')` が変わっていても古いキャッシュが使われる
- 決定的な証拠: 同じコミット (`c5ea89b` = 0037 の mbedTLS スレッド対応をマージ済み) の wheel run で、 leg によって挙動が分かれた
  - `build_ubuntu (ubuntu-24.04_x86_64, ubuntu-24.04, 3.14)`: ログが `-- MbedTLS already built` (再ビルドなし) → `tests/test_websocketserver.py` の `server.stop()` で exit 134
  - `build_ubuntu (ubuntu-24.04_armv8, ubuntu-24.04-arm, 3.13)`: ログが `-- Building MbedTLS...` (新規ビルド) → `test_websocketserver` を含む 4 件が PASSED
- 経路: libdatachannel の `impl/tlstransport.cpp` が複数 thread から `psa_crypto_init()` を呼び、 スレッド対応の無い mbedTLS では内部状態 (PSA / entropy) が壊れてヒープ破壊になる。 0038 の crash report の `mbedtls_entropy_func` / `mbedtls_ctr_drbg_seed` 経路と一致する
- 手元 (macOS 26 arm64) の `_deps` には `#define MBEDTLS_THREADING_C` があるため再現しない

## 設計方針

- `CMakeLists.txt` のガードを、 コメント行に誤マッチしない形にする (行頭から `#define MBEDTLS_THREADING_C` を探す)
- `wheel.yml` の `_deps` キャッシュキーと `restore-keys` の接頭辞に世代を付け (v2)、 修正前に作られたキャッシュを復元させない
- 修正後は、 影響していた leg のログで `-- Building MbedTLS...` (再ビルド) が出ることと、 `_deps` の mbedTLS ライブラリにスレッド用の mutex シンボルがあることを確認する
- Linux で `tests/test_websocketserver.py` をヒープ検査付きで繰り返す検証は、 環境が用意できる場合に補助的に行う

## 完了条件

- `CMakeLists.txt` のガードが、 無効化されたままの行 (`//#define MBEDTLS_THREADING_C`) では再ビルドが走り、 有効な define では走らないこと (`cmake -P` の確認で示す)
- CI (wheel.yml) の全 leg が PASS すること
- 影響していた leg のログで `-- Building MbedTLS...` (再ビルド) が出ること
- `CHANGES.md` の `## develop` の `### misc` に記録すること (利用者に見える挙動は変わらないため)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること
- 破壊箇所が特定できない場合は、 残る候補と次の調査手順が issue に記録されていること (今回特定できたため対象外)

## 参考

- 失敗した leg: wheel ワークフローの `build_ubuntu (ubuntu-24.04_x86_64, ubuntu-24.04, 3.14)`
- 対象テスト: `tests/test_websocketserver.py` の `test_websocketserver`
- libdatachannel v0.24.0: `source/src/impl/websocketserver.cpp` (受け入れ thread と `stop()`)、 `source/src/impl/channel.cpp` (callback の実行)
- 関連 issue: [[0038-bug-fix-websocketserver-native-crash]] (mbedTLS のスレッド対応)、 [[0039-bug-fix-nanobind-del-not-called]] (callback の寿命)
