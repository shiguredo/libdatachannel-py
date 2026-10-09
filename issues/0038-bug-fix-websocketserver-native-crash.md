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

## 参考

- 対象: tests/test_websocketserver.py、 src/bind_libdatachannel.cpp の `bind_websocketserver`
- 関連 issue: [[0002-bug-fix-websocket-destructor-gil-release]]、 [[0003-bug-fix-websocketserver-destructor-gil-release]]、 [[0025-test-remove-callback-prints]]
