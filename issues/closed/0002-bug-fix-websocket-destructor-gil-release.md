# WebSocket の destructor 内で発生する GIL 保持 hang を修正する

- Priority: High
- Created: 2026-05-18
- Completed: 2026-10-09
- Polished: 2026-10-09
- Branch: feature/fix-websocket-destructor-gil-release

## 目的

`WebSocket.close()` / `WebSocket.force_close()` は GIL を保持したまま binding から呼ばれており、 受信 callback を実行中の内部 thread との間でロック順逆転が起きて Python プロセス全体が hang する。 本 issue では `WebSocket` を対象に、 これらの API と `__del__` の close 経路を GIL 解放下で実行する仕組みを C++ binding に追加する。 ただし TLS/TCP の close handshake は別 thread が完了させる必要があるため、 `close()` は `state == Closing` の場合に polling せず即 return する例外規則を入れる。 C++ destructor 自身 (`~WebSocket()` の `remoteClose()` / `resetCallbacks()`) は GIL を保持したまま走るため、 callback 実行中の窓では依然として恒停し得る (実測済み)。 そこは [[0005-bug-fix-destructor-callback-deadlock]] の範囲とする。

## 優先度根拠

- `WebSocket` は signaling・転送等で頻繁に使用されるため、 destruct 時 hang の影響を受けやすい
- 影響度は `PeerConnection` ([[0001-bug-fix-peer-connection-destructor-gil-release]]) と同等であり、 一緒に解消する方がリリースノートの整合も取りやすい
- 後続の `WebSocketServer` ([[0003-bug-fix-websocketserver-destructor-gil-release]]) は `on_client` callback の引数で `WebSocket` インスタンスを渡すため、 本 issue で destruct 時の自動 `close()` の挙動を binding 側に確定させる必要がある

## 現状

- `src/bind_libdatachannel.cpp` の `WebSocket` bindings は `.def("close", &WebSocket::close)` と直接バインドしており、 GIL を保持したまま `close()` が実行される。 `force_close` も同様に `.def("force_close", &WebSocket::forceClose)` と直接バインドされている
- `WebSocket` には state API があり (`Connecting / Open / Closing / Closed`)、 `ready_state()` で close の完了を polling で待てる構造である。 ただし libdatachannel の `WebSocket::close()` は `Connecting/Open` のときしか動作せず、 `Closing` 以降は内部で何もしない。 `Closed` へ進むには TLS/TCP close handshake (別 thread) の完了を待つ必要があるため、 polling を盲目的に行うと対向応答遅延で timeout までブロックする
- **恒停の機構 (実測)**: 受信 callback が rtc の poll thread で実行されている窓で destruct すると、 poll thread は callback mutex を保持したまま GIL を待ち、 destructor (GIL 保持) は同じ mutex を待つためロック順逆転で恒停する。 恒停時のスタックは main thread が `~WebSocket()` → `closeTransports()` → `Transport::onRecv()` → `std::recursive_mutex::lock()`、 poll thread が `Channel::flushPendingMessages()` → `PyGILState_Ensure` → `take_gil` である
- **再現条件 (実測)**: サーバーが 1 ms 間隔で push し続け、 内部 thread が受信 callback を実行している最中に close 経路を呼ぶ (または破棄する)。 callback 内の I/O の有無は恒停に影響しない (callback が何もしない場合も恒停する)。 echo サーバーに接続して destruct するだけでは恒停しないため、 修正の判別には push し続けるサーバーが要る
- 恒停時は main thread が native の mutex 待ちになるため pytest-timeout の SIGALRM handler が走らない (実測: `@pytest.mark.timeout(15)` を付けても発火せず 25 秒以上停止し、 外部から kill する必要があった)。 CI ([[0022-test-enable-ci-tests]] で追加) では job の `timeout-minutes` まで停止する
- `force_close()` も同じ恒停点 (`closeTransports()`) に入る (実測: 2 回とも恒停)
- `src/libdatachannel/__init__.py` には Python wrapper class が無い ([[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法で wrapper 方式は撤回され、 binding 側に `__del__` を生やす方式に統一された)

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

[[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法 (binding 側の `close_peer_connection` + `.def("__del__", ...)` + `nb::is_weak_referenceable()`) と同型の実装を `WebSocket` に適用する。

- `close_websocket` を `namespace {}` に追加する。 ロジック:
  1. `self.readyState() == WebSocket::State::Closed` なら no-op で早期 return する
  2. `self.readyState() == WebSocket::State::Closing` なら **polling せず即 return** する。 残りの状態遷移は `~WebSocket()` 側の destructor 処理に委ねる。 この特別扱いは「`WebSocket::close()` は `Connecting/Open` のときしか動かず、 `Closing` 以降は内部で何もしない」 「対向応答遅延で polling が 30 秒待たされるのを避ける」 という理由をコードコメントに明記する
  3. それ以外は `self.close()` を呼び、 `WebSocket::State::Closed` に達するまで polling する。 polling 定数 (`kPollInterval=10ms` / `kCloseTimeout=30s`) は関数内 `constexpr` として持ち、 timeout 時は GIL を再取得 (`nb::gil_scoped_acquire`) してから `RuntimeWarning` を出して return する (残処理は destructor に委ねる)。 `PyErr_WarnEx` が負を返した場合 (filterwarnings=error 等で警告が例外に昇格した場合) は `nb::python_error` を投げる。 0001 の `close_peer_connection` と同じ扱いにする
- `force_close_websocket` も追加する。 `forceClose()` は同じ恒停点に入るため、 GIL 解放下で呼ぶ。 `forceClose()` は `closeTransports()` まで同期で進むため polling はしない
- `.def("close", &close_websocket, nb::call_guard<nb::gil_scoped_release>())` / `.def("force_close", &force_close_websocket, nb::call_guard<nb::gil_scoped_release>())` に差し替える
- `.def("__del__", ...)` を追加し、 GIL release 下で `self.resetCallbacks()` で callback を解除してから `close_websocket` を呼ぶ (解除を先に行い、 close が timeout で例外を投げても解除が残るようにする)。 `resetCallbacks()` も GIL 解放下で実行することで、 受信 callback を実行中の内部 thread が GIL を取得して処理を終えられる。 また `__del__` から投げた例外は呼び出し側で捕捉できないため `RuntimeWarning` として記録するだけで握り潰す (0001 の `PeerConnection` binding と同型)
- `nb::class_<WebSocket, Channel>` に `nb::is_weak_referenceable()` を指定する (test で weakref により `__del__` 発火を検証するため)

### 2. テスト (tests/test_websocket.py)

- **恒停し得る検証は pytest プロセス内で実行しない**。 検証スクリプトは `tests/hang_reproduction_websocket.py` (pytest が collect しない名前) に置き、 `subprocess.run([sys.executable, <スクリプト>, mode, iterations], timeout=180, capture_output=True)` で子プロセスとして実行して `returncode` を検証する。 子プロセスが恒停した場合は親側で `subprocess.TimeoutExpired` になりテストが失敗するため、 CI の job timeout まで停止しない。 pytest-timeout は恒停時に発火しないため使わない (理由をテストのコメントに残す)
- 子プロセス側では、 恒停の窓を作るために 1 ms 間隔で push し続けるサーバーを立て、 受信 callback の実行中に close 経路を呼ぶ。 callback が実行中であることは `threading.Event` で同期する (`Event.wait()` は GIL を解放するため callback 側の進行を妨げない)。 恒停に必要なのは「内部 thread が callback mutex を保持したまま GIL を待ち続ける状態」であり、 callback 内の I/O の有無ではない
  - [[0025-test-remove-callback-prints]] が callback 内の `print` の除去を進めているため `print` は使わない
  - 検証するのは `close()` と `force_close()` の 2 経路。 どちらも呼び出し後に `ready_state()` が `Closed` であることを検証する。 `close()` は対向との close handshake を待つため 1 回あたり 10 秒程度かかるので 1 回、 `force_close()` は 5 回反復する
  - 子プロセスは最後に `os._exit(0)` で終了する。 C++ destructor は GIL を保持したまま走り、 callback 実行中の窓では依然として恒停し得るため (スコープ外を参照)、 検証は close 経路に絞る
- 併せて weakref で破棄が完了することを検証し (`__del__` の呼び出しそのものは nanobind の dealloc 経由で観測できないため、 破棄の完了に留める)、 接続を開いた状態から `close()` を 2 回呼んでも 2 回目が即時完了すること (`test_close_is_idempotent`) を検証する (未接続の WebSocket は初期状態が `Closed` で早期 return しか通らないため、 実接続してから検証する)
- `test_del_releases_native` を 0001 のテスト方式に合わせて追加する
- 既存テストが PASS することを確認する

### 3. CHANGES.md

- `## develop` に 0001 のエントリとは別の `[FIX]` エントリを追加する (担当者行 `- @voluntas` を付ける)
- 文言には「`WebSocket` の destruct 時の GIL 保持 hang を修正する」 「`__del__` で `close()` が自動的に呼ばれる」 「`close()` 自身も GIL 解放下で `Closed` まで待機し、 30 秒で完了しなかった場合は `RuntimeWarning` を出す」 「`state==Closing` の場合は polling せず即 return する」 「`force_close()` も GIL 解放下で実行する」 を明記する

## 完了条件

- `prek run --all-files pytest` (prek.toml の pytest フック = `uv run pytest -v --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close`) が PASS する。 拡張モジュールを install 済みであること (`make develop` 相当)
- CI (wheel.yml の 24 leg / prek.yml の `ty` ジョブ) の pytest が PASS する。 既知の恒停テスト (`tests/test_peerconnection.py::test_destruct_without_explicit_close`) は CI でも `--deselect` で除外されている。 本 issue で追加する恒停再現テストは子プロセスで実行するため CI でも実行される
- `close()` と `force_close()` の恒停再現テスト (子プロセス + timeout) が、 **修正前は恒停して timeout で失敗し、 修正後は完走する** こと (実測済み: 未修正のビルドでは 2 テストとも timeout、 修正後は 6 テストすべて PASS)
- 既知の恒停テスト ([[0005-bug-fix-destructor-callback-deadlock]]) は対象外とする。 `make test` は `make develop` (フルビルド) を実行し恒停テストを除外しないため、 完了条件には使わない
- C++ 側の public `~WebSocket()` の恒停 (callback 実行中に破棄が走る場合) は本 issue の対象外とする (スコープ外を参照)。 `close()` が `Closed` に到達するのは Connecting / Open から呼んだ場合で、 `Closing` の場合は polling せず即 return する
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `close_websocket` を匿名 namespace に追加した。 `Closed` は no-op、 `Closing` は polling せず即 return し、 それ以外は `close()` して `Closed` まで polling する (10 ms 間隔 / 30 秒)。 30 秒で到達しなかった場合は GIL を再取得して `RuntimeWarning` を出し、 警告が例外に昇格した場合は `nb::python_error` を投げる。 残処理は C++ デストラクタに委ねる
  - `force_close_websocket` を匿名 namespace に追加した。 `Closed` は no-op、 それ以外は `forceClose()` を呼ぶ (同期で `Closed` に到達するため polling しない)
  - `WebSocket` の `close` / `force_close` の binding を GIL 解放下で実行するように差し替え、 `__del__` を追加した。 `__del__` は GIL 解放下で `resetCallbacks()` を呼んでから `close_websocket` を呼び、 例外は `RuntimeWarning` として記録するだけで握り潰す
  - `nb::class_<WebSocket, Channel>` に `nb::is_weak_referenceable()` を指定した
- `tests/test_websocket.py` / `tests/hang_reproduction_websocket.py`
  - 恒停を再現する検証スクリプトを追加した。 1 ms 間隔で push し続けるサーバーを立て、 受信 callback の実行中に `close()` / `force_close()` を呼ぶ。 恒停時は pytest-timeout が発火しないため、 検証は子プロセスで実行し `subprocess.run` の timeout (180 秒) で打ち切る
  - `test_close_does_not_hang_while_receiving` / `test_force_close_does_not_hang` / `test_del_releases_native` / `test_close_is_idempotent` を追加した
  - 実測: 未修正のビルドでは `close()` / `force_close()` の検証が恒停して timeout で失敗し、 修正後は WebSocket の 6 テストすべて PASS。 全体は 84 passed / 12 skipped / 1 deselected
- `CHANGES.md` の `## develop` に `[FIX]` エントリを追加した

## スコープ外 (関連する未解決問題)

- 本 issue は binding 側の close 経路 (`close()` / `force_close()` / `__del__` の close) から GIL 保持を取り除くアプローチである。 明示 `close()` が 30 秒で `Closed` に達しなかった場合も、 `Closing` で早期 return した場合も、 続く public `~WebSocket()` の `remoteClose()` / `resetCallbacks()` は GIL を保持したまま走る。 受信 callback が実行中の窓では引き続き hang し得る (実測済み)。 根本対応は [[0005-bug-fix-destructor-callback-deadlock]] に集約する
- 送信系 API の GIL 保持は [[0032-bug-fix-send-gil-deadlock]] で対応済み (`WebSocket.send()` も GIL 解放下で実行される)。 本 issue は `WebSocket` の close 経路 (`close()` / `force_close()` / `__del__`) のみを対象とする
- `DataChannel.close()` / `Track.close()` は GIL を保持したまま呼ばれるが、 close 経路に `closeTransports()` を伴わないため恒停の実測根拠が無く、 本 issue の対象外とする

## 参考

- 既存ブランチ (試行錯誤の履歴): `feature/fix-destructor-gil-release`
  - `85b144a` (`close()` を GIL release で実行) / `6736371` (Python wrapper 追加) / `ab2f2b8` (`Closing` 状態の polling 早期 return) は wrapper 方式と `wait_for_closed` template に基づく試行錯誤であり、 0001 の解決方法で撤回された。 cherry-pick せず、 develop に取り込まれた 0001 の実装 (`close_peer_connection` / `.def("__del__")` / `nb::is_weak_referenceable()`) を踏襲すること。
- 関連 issue: [[0038-bug-fix-websocketserver-native-crash]] (WebSocketServer の native crash。 本 issue の destructor hang とは別の症状) / [[0001-bug-fix-peer-connection-destructor-gil-release]] (`close_peer_connection` / `__del__` / `is_weak_referenceable` の実装元) / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] / [[0005-bug-fix-destructor-callback-deadlock]]
- libdatachannel 関連コード位置 (シンボル名で特定する):
  - `_deps/libdatachannel/v0.24.0/source/src/websocket.cpp` の public `~WebSocket()` (impl `remoteClose()` と `resetCallbacks()` を呼ぶ) と public `WebSocket::close()`
  - `_deps/libdatachannel/v0.24.0/source/src/impl/websocket.cpp` の impl `WebSocket::close()` (`Connecting/Open` のときのみ動作) と `closeTransports()` (`State::Closed` への遷移と `triggerClosed()` 呼び出し)
  - `_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp` の `synchronized_callback::operator()` (mutex 保持実行)
