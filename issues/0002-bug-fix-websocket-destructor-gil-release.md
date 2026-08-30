# WebSocket の destructor 内で発生する GIL 保持 hang を修正する

- Priority: High
- Created: 2026-05-18
- Polished: 2026-08-30
- Model: Opus 4.7
- Branch: feature/fix-websocket-destructor-gil-release

## 目的

`WebSocket` を明示的に `close()` せずに destruct すると、 libdatachannel 本体の close 経路で内部 thread の停止と TLS/TCP の close handshake が GIL を保持したまま走る。 内部 cleanup が Python callback の GIL 待ちに入ると Python プロセス全体が hang する。

本 issue では `WebSocket` を対象に、 destruct 前に GIL release で同期 `close()` する仕組みを C++ binding に追加する。 ただし TLS/TCP の close handshake は別 thread が完了させる必要があるため、 `state==Closing` の場合は polling せず即 return する例外規則を入れる。

## 優先度根拠

- `WebSocket` は signaling・転送等で頻繁に使用されるため、 destruct 時 hang の影響を受けやすい。
- 影響度は `PeerConnection` ([[0001-bug-fix-peer-connection-destructor-gil-release]]) と同等であり、 一緒に解消する方がリリースノートの整合も取りやすい。
- 後続の `WebSocketServer` ([[0003-bug-fix-websocketserver-destructor-gil-release]]) は `on_client` callback の引数で `WebSocket` インスタンスを渡すため、 本 issue で destruct 時の自動 `close()` の挙動を binding 側に確定させる必要がある。

## 現状

- `src/bind_libdatachannel.cpp` の `WebSocket` bindings は `close` を直接バインドしており、 GIL を保持したまま `close()` が実行される。
- `WebSocket` には state API があり (`Connecting / Open / Closing / Closed`)、 close の同期完了を polling で待てる構造である。 ただし libdatachannel の `WebSocket::close()` は `Connecting/Open` のときしか動作せず、 `Closing` 以降は内部で何もしない。 `Closed` へ進むには TLS/TCP close handshake (別 thread) の完了を待つ必要があるため、 polling を盲目的に行うと対向応答遅延で timeout までブロックする。
- `WebSocket` を明示 `close()` せずに destruct すると、 public `~WebSocket()` が `remoteClose()` → `closeTransports()` を GIL 保持下で実行し、 途中で `on_closed` 等の Python callback を呼んで GIL 待ちに入ると Python プロセス全体が hang する。
- `src/libdatachannel/__init__.py` には Python wrapper class が無い (0001 の解決方法で wrapper 方式は撤回され、 binding 側に `__del__` を生やす方式に統一された)。

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

[[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法 (binding 側の `close_peer_connection` + `.def("__del__", ...)` + `nb::is_weak_referenceable()`) と同型の実装を `WebSocket` に適用する。

- `close_websocket` を `namespace {}` に追加する。 ロジック:
  1. `self.readyState() == WebSocket::State::Closed` なら no-op で早期 return する。
  2. `self.readyState() == WebSocket::State::Closing` なら **polling せず即 return** する。 残りの状態遷移は `~WebSocket()` 側の destructor 処理に委ねる。 この特別扱いは「`WebSocket::close()` は `Connecting/Open` のときしか動かず、 `Closing` 以降は内部で何もしない」 「対向応答遅延で polling が 30 秒待たされるのを避ける」 という理由をコードコメントに明記する。
  3. それ以外は `self.close()` を呼び、 `WebSocket::State::Closed` に達するまで polling する。 polling 定数 (`kPollInterval=10ms` / `kCloseTimeout=30s`) は関数内 `constexpr` として持ち、 timeout 時は `RuntimeWarning` を出して return する (残処理は destructor に委ねる)。
- `.def("close", &close_websocket, nb::call_guard<nb::gil_scoped_release>())` に差し替える。
- `.def("__del__", ...)` を追加し、 GIL release 下で `close_websocket` を呼ぶ。 `__del__` から投げた例外は呼び出し側で捕捉できないため `RuntimeWarning` として記録するだけで握り潰す (0001 の `PeerConnection` binding と同型)。
- `nb::class_<WebSocket, Channel>` に `nb::is_weak_referenceable()` を指定する (test で weakref により `__del__` 発火を検証するため)。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- 変更不要。 0001 の解決方法で wrapper class 方式は撤回されたため、 本 issue でも Python wrapper は追加せず binding 側の `__del__` 方式に統一する。

### 3. テスト (tests/test_websocket.py)

- `test_destruct_without_explicit_close` を新規追加する。 既存 `tests/conftest.py` の `echo_websocket_server` fixture に接続して Open 状態にし、 明示 `close()` を呼ばずに destruct しても hang せず終了することを検証する。
- 検証方法は 0001 の `tests/test_peerconnection.py` に合わせ、 `weakref` で `__del__` 発火を実検証し、 `recwarn` で `RuntimeWarning` 0 件を検証する。 hang 時の上限として `@pytest.mark.timeout` を指定する。
- 既存テストが PASS することを確認する。

### 4. CHANGES.md

- 0001 の実装手順により後続 issue は別 PR で着手するため、 `## develop` に 0001 のエントリとは別の `[FIX]` エントリを追加する。
- 文言には「`WebSocket` の destruct 時の GIL 保持 hang を修正する」 「`__del__` で `close()` が自動的に呼ばれる」 「`state==Closing` の場合は polling せず即 return する」 を明記する。

## 完了条件

- `uv sync && make test` で全テストが PASS する。
- `tests/test_websocket.py::test_destruct_without_explicit_close` が、 明示 `close()` を呼ばずに `ws` を destruct しても hang せず終了する。
- `tests/test_websocket.py::test_del_releases_native` と `test_close_is_idempotent` が PASS する (0001 のテスト方式に合わせて追加)。
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `close_websocket` を匿名 namespace に追加。 `Closed` / `Closing` の早期 return 規則と理由コメントを併記する。
  - `WebSocket` bindings の `.def("close", ...)` を `&close_websocket` + `nb::call_guard<nb::gil_scoped_release>()` に差し替え、 `.def("__del__", ...)` を追加し、 `nb::class_<WebSocket, Channel>` に `nb::is_weak_referenceable()` を指定する。
- `src/libdatachannel/__init__.py`
  - 変更なし (Python wrapper は追加しない)。
- `tests/test_websocket.py`
  - `test_destruct_without_explicit_close` / `test_del_releases_native` / `test_close_is_idempotent` を追加。
- `CHANGES.md`
  - `## develop` セクションに `[FIX]` エントリを追加。

## スコープ外 (関連する未解決問題)

本 issue は destruct 到達前に明示 `close()` で `Closed` まで進めて destructor 経路の負担を減らすアプローチであり、 完全には hang を防げない。 30 秒 timeout 後に `close_websocket` が return しても、 続く public `~WebSocket()` の `remoteClose()` 経路は依然 GIL 保持下で走る。 callback が I/O block する条件下では引き続き hang し得るため、 根本対応は [[0005-bug-fix-destructor-callback-deadlock]] に集約する。

## 参考

- 既存ブランチ (試行錯誤の履歴): `feature/fix-destructor-gil-release`
  - `85b144a` (`close()` を GIL release で実行) / `6736371` (Python wrapper 追加) / `ab2f2b8` (`Closing` 状態の polling 早期 return) は wrapper 方式と `wait_for_closed` template に基づく試行錯誤であり、 0001 の解決方法で撤回された。 cherry-pick せず、 develop に取り込まれた 0001 の実装 (`close_peer_connection` / `.def("__del__")` / `nb::is_weak_referenceable()`) を踏襲すること。
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] (`close_peer_connection` / `__del__` / `is_weak_referenceable` の実装元) / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] / [[0005-bug-fix-destructor-callback-deadlock]]
- libdatachannel 関連コード位置 (シンボル名で特定する):
  - `_deps/libdatachannel/v0.24.0/source/src/websocket.cpp` の public `~WebSocket()` (impl `remoteClose()` と `resetCallbacks()` を呼ぶ) と public `WebSocket::close()`
  - `_deps/libdatachannel/v0.24.0/source/src/impl/websocket.cpp` の impl `WebSocket::close()` (`Connecting/Open` のときのみ動作) と `closeTransports()` (`State::Closed` への遷移と `triggerClosed()` 呼び出し)
  - `_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp` の `synchronized_callback::operator()` (mutex 保持実行)
