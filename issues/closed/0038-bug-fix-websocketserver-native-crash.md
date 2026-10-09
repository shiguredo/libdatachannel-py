# WebSocketServer のテストが稀に native crash する

- Priority: Medium
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-websocketserver-native-crash
- Polished: 2026-10-09

## 目的

tests/test_websocketserver.py::test_websocketserver を実行したとき、 稀に native crash でテストプロセスごと落ちる。 落ちるとテストスイート全体が中断し、 CI の leg が失敗してリリースの公開も止まる。 原因を特定して解消する。

## 優先度根拠

- 発生するとテストプロセスごと落ちるため、 pytest では検出できない (pytest-timeout も rerun プラグインも効かない)。 CI ではテストスイート全体を 1 回だけやり直す暫定対応で回避しており、 根本対応が無い
- リリース判定 (タグ push) では wheel の leg が 1 つでも失敗すると publish がブロックされるため、 無視できない
- [[0003-bug-fix-websocketserver-destructor-gil-release]] は destructor の GIL hang、 [[0002-bug-fix-websocket-destructor-gil-release]] は WebSocket の destructor を扱っており、 本 issue は native crash という別の症状

## 現状

- 実測 1: macOS 26 arm64 / Python 3.14 / ビルドした wheel を install した環境で、 テストスイート全体の実行中に `tests/test_websocketserver.py::test_websocketserver` のテスト名が出力された約 18 ms 後に `Trace/BPT trap: 5` (SIGTRAP)。 終了コード 133。 22 回中 1 回の観測 (率の推定は不能。 95% 信頼区間は約 0.12%〜23%)
- 実測 2: CI (wheel.yml の build_macos / macos-26_arm64 / 3.14) で同じ症状により leg が失敗。 失敗ログでは `tests/test_websocket.py::test_close_is_idempotent PASSED` の直後に `Trace/BPT trap: 5` が出ており、 `test_websocketserver` の実行中に落ちている。 同じログに `nanobind: leaked 87 functions!` も出ている (別途対応)
- 実測 3: ユーザー環境 (macOS 26.6.2 arm64 / Python 3.12.12) で `Fatal Python error: Segmentation fault`。 クラッシュ位置はテストの `time.sleep(1)` (ポーリング中)
- **クラッシュ時のスタックは crash report から取得済み** (`~/Library/Logs/DiagnosticReports/`)
  - `python3.12-2026-10-08-155041.ips`: EXC_BREAKPOINT / SIGTRAP。 faulting thread は libdatachannel の **RTC poll** スレッドで、 `_xzm_xzone_malloc_freelist_outlined` (malloc の freelist 検査) ← `mbedtls_md_setup` / `mbedtls_entropy_func` / `mbedtls_ctr_drbg_seed` / `psa_crypto_init` ← `rtc::impl::TlsTransport::TlsTransport` ← `rtc::impl::WebSocket::initTlsTransport()` ← `rtc::impl::TcpTransport::processConnect` ← `rtc::impl::PollService::runLoop()`。 **ヒープ破壊**である
  - `python3.12-2026-10-09-115914.ips` (実測 3): SIGSEGV。 アクセス先は main thread の Stack Guard 領域の直下で、 main thread のスタック枯渇またはフレーム破壊
  - `.ips` の "faulting thread" は main thread ではない。 main thread が `time.sleep` 中と表示されるのは、 「Python から見える最後の実行位置」が出ているだけである
- 原因: **mbedTLS をスレッドセーフでない設定でビルドしていた** (`MBEDTLS_THREADING_C` / `MBEDTLS_THREADING_PTHREAD` が無効)。 libdatachannel は `src/impl/tlstransport.cpp` で `psa_crypto_init()` を呼び、 これを client の poll thread と server の RTC poll thread から同時に実行するため、 mbedTLS 内部の PSA / entropy の状態が壊れてヒープ破壊を起こす。 mbedTLS 自身の `mbedtls_config.h` にも「複数 thread から PSA 関数を呼ぶ場合は `MBEDTLS_THREADING_C` が必要」と書かれている
- 対象テストは 127.0.0.1 の固定ポート 48080 を使い、 15 秒のポーリングと末尾での明示破棄を行う。 callback 内に print が残っている ([[0025-test-remove-callback-prints]] の対象)
- WebSocketServer の binding は `src/bind_libdatachannel.cpp` の `bind_websocketserver` にある。 `on_client` は `std::function<void(shared_ptr<WebSocket>)>` で、 accept thread から libdatachannel の recursive_mutex を保持したまま Python callback を呼ぶ

## 設計方針

- crash report の faulting thread のスタックを一次証拠として原因を特定する (取得済み)
- mbedTLS をスレッドセーフにビルドする (`MBEDTLS_THREADING_C` / `MBEDTLS_THREADING_PTHREAD` を mbedTLS のビルドに追加する)。 config の変更は mbedTLS の構造体レイアウトを変えるため、 libdatachannel も同じ設定で再ビルドする
- テストの `time.sleep` によるポーリングをイベント待ちに置き換える。 破棄経路は `reset_callbacks()` を使わない (libdatachannel の mutex と GIL が inversions して恒停することを実測済み)
- 修正後は、 クラッシュしていた条件で繰り返し実行して安定を確認する (試行回数と 0/N の 95% 上側限界を報告する)
- CI の再試行 (テストスイート全体を 1 回だけやり直す暫定対応) は本 issue で外す。 外した状態で不安定なテストが他にあれば別途対応する

## 完了条件

- 原因 (mbedTLS のスレッド対応欠落によるヒープ破壊) が crash report のスタックで裏付けられていること
- 修正後、 TLS を使うテストを繰り返し実行してクラッシュしないこと (試行回数 N と 0/N の 95% 上側限界 (rule of three で 3/N) を報告する。 N=22 では 13.6% が限界で根拠として弱いため、 N は 100 以上とする)
- CI の再試行を外しても安定して PASS すること (prek.yml / wheel.yml のリトライを削除する)
- テストから `time.sleep` によるポーリングが無くなっていること。 例外は (a) `time.sleep(0)` の測定用 yield、 (b) 恒停する既知テスト (`test_destruct_without_explicit_close`) 内の待ち (実行検証ができないため差分を作らない)、 (c) RTP の再送間隔 (受信通知 callback が無くイベント待ちにできない) の 3 つで、 いずれも理由コメントを付ける
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `CMakeLists.txt`
  - mbedTLS のビルドに `MBEDTLS_THREADING_C` と `MBEDTLS_THREADING_PTHREAD` を追加した。 libdatachannel は複数の thread から `psa_crypto_init()` を呼ぶため、 これが無いと mbedTLS 内部の PSA / entropy の状態が壊れて TLS の初期化中にヒープ破壊を起こす (Windows には pthread が無いため `if(NOT WIN32)` で対象外にした)
  - `MBEDTLS_THREADING_C` は mbedTLS の構造体レイアウトを変えるため、 libdatachannel も同じ設定でビルドし直す必要がある
  - 設定が古いままのビルド済み `_deps` を検出したら mbedTLS と libdatachannel を作り直すようにした (CI のキャッシュに古い `_deps` が復元され、 修正が無言で効かなくなっていたため)
- `tests/test_websocketserver.py` / `tests/test_websocket.py` / `tests/test_peerconnection.py`
  - `time.sleep` によるポーリングを `threading.Event` の待ちに置き換えた。 例外は (a) 測定用の `time.sleep(0)`、 (b) 恒停する既知テスト (`test_destruct_without_explicit_close`) 内の待ち、 (c) RTP の再送間隔の 3 つで、 いずれも理由コメントを付けた
  - WebSocketServer のテストは `port = 0` で動的確保し、 実際のポートを `server.port()` から取るようにした (固定ポートによる bind 衝突を排除)
  - 待ち時間は従来のポーリングと同等以上 (15〜20 秒) にした
- `tests/test_websocketserver.py` の破棄経路で `reset_callbacks()` は使わない。 accept / processor thread が libdatachannel の mutex を保持したまま Python callback の GIL を待つ状況では相互待ちになり恒停するため
- `.github/workflows/prek.yml` / `wheel.yml`
  - pytest のリトライ (continue-on-error + Warn/Retry) を削除し、 失敗をそのまま job の失敗として扱うようにした
- `CHANGES.md` の `## develop` に `[FIX]` エントリを追加した
- 検証
  - クラッシュのスタック (crash report) が `psa_crypto_init` → `mbedtls_ctr_drbg_seed` → `mbedtls_entropy_func` の経路で malloc の freelist 検査に落ちていることを確認した
  - スイート (121 passed / 12 skipped / 1 deselected) を直列で 10 回連続実行し、 すべて成功 (15〜19 秒/回)
  - TLS を使うサーバー + クライアントを 98 ラウンド (逐次 60 + 8 並列 × 40) 作る probe でクラッシュ 0 件
  - `libmbedcrypto.a` に `mbedtls_threading_psa_globaldata_mutex` / `psa_rngdata_mutex` / `key_slot_mutex` が組み込まれたことを `nm` で確認
  - CI (retry なし) で全 leg が成功した
- クラッシュ率は修正前で「22 回中 1 回の観測」であり、 率の推定はできない。 修正後の 0/N (N = 安定性 10 回 + probe 98 ラウンド) から言える 95% 上側限界は rule of three で約 3% 以下である (クラッシュ率 0 を主張するものではない)
- 残件: Python と C++ をまたぐ参照サイクル (callback のクロージャ ↔ C++ の WebSocket) による nanobind のリーク警告は残る。 nanobind は `fprintf(stderr, ...)` で報告するだけで exit code を変えないため CI は落ちない。 別 issue として扱う

## 参考

- 対象: tests/test_websocketserver.py、 src/bind_libdatachannel.cpp の `bind_websocketserver`
- 関連 issue: [[0002-bug-fix-websocket-destructor-gil-release]]、 [[0003-bug-fix-websocketserver-destructor-gil-release]]、 [[0025-test-remove-callback-prints]]
