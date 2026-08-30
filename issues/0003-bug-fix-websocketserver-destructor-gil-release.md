# WebSocketServer の destructor 内で発生する GIL 保持 hang を修正する

- Priority: Medium
- Created: 2026-05-18
- Polished: 2026-08-30
- Model: Opus 4.7
- Branch: feature/fix-websocketserver-destructor-gil-release

## 目的

`WebSocketServer` を明示的に `stop()` せずに destruct すると、 libdatachannel 本体の `~WebSocketServer()` 配下で `stop()` が呼ばれ、 内部の `tcpServer->close()` と `mThread.join()` が GIL を保持したまま走る。 受け入れ thread が Python callback (`on_client` 等) の GIL 待ちに入ると Python プロセス全体が hang する。

本 issue では `WebSocketServer` を対象に、 destruct 前に GIL release で同期 `stop()` する仕組みを C++ binding に追加する ([[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法と同じ binding 側 `__del__` 方式)。 `WebSocketServer` は state API を持たないため polling は行わず、 GIL release だけを担保する。 `on_client` callback に渡される `WebSocket` インスタンスの自動 `close()` は [[0002-bug-fix-websocket-destructor-gil-release]] の binding 側 `__del__` により担保されるため、 本 issue では追加対応しない。

## 優先度根拠

- `WebSocketServer` はサーバー用途の利用者のみが触る API のため、 `PeerConnection` ([[0001-bug-fix-peer-connection-destructor-gil-release]]) や `WebSocket` ([[0002-bug-fix-websocket-destructor-gil-release]]) ほど影響範囲は広くない。
- しかし destruct 時の hang は debug 困難な事象であり、 一連の修正 ([[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]) と整合した形で閉じる必要がある。
- `on_client` callback で受け取る `WebSocket` の自動 `close()` は 0002 の binding 側 `__del__` で担保されるため、 本 issue では追加対応しない。

## 現状

- `src/bind_libdatachannel.cpp` の `WebSocketServer` bindings は `stop` を `&WebSocketServer::stop` で直接バインドしており、 GIL を保持したまま `stop()` が実行される。 `__del__` binding も無いため、 明示 `stop()` を呼ばずに destruct した場合も GIL 保持下で `stop()` が走る。
- `WebSocketServer` には state API がないため、 stop の完了を polling で確認することはできない。 `stop()` の戻り時点で `tcpServer->close()` と `mThread.join()` が完了している前提に乗る。
- 利用者が `del server` あるいは function スコープ抜けで destruct した場合に、 `~WebSocketServer()` 内 `stop()` が GIL 保持下で走り、 callback の GIL 待ちと噛み合って Python プロセス全体が hang する。

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

[[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法 (binding 側の `close_peer_connection` + `.def("__del__", ...)` + `nb::is_weak_referenceable()`) と同型の実装を `WebSocketServer` に適用する。

- `stop_websocket_server` を `namespace {}` に追加する。 `self.stop()` を呼ぶだけで、 polling は行わない (state API が無い)。 `stop()` の戻り時点で `tcpServer->close()` と `mThread.join()` が完了している。
- `.def("stop", &WebSocketServer::stop)` を `.def("stop", &stop_websocket_server, nb::call_guard<nb::gil_scoped_release>())` に差し替える。
- `.def("__del__", ...)` を追加し、 GIL release 下で `stop_websocket_server` を呼ぶ。 `__del__` から投げた例外は呼び出し側で捕捉できないため `RuntimeWarning` として記録するだけで握り潰す (0001 の `PeerConnection` binding と同型)。
- `nb::class_<WebSocketServer>` に `nb::is_weak_referenceable()` を指定する (test で weakref により `__del__` 発火を検証するため)。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- 変更不要。 0001 の解決方法で wrapper class 方式は撤回されたため、 本 issue でも Python wrapper は追加せず binding 側の `__del__` 方式に統一する。

### 3. テスト (tests/test_websocketserver.py)

- `test_destruct_without_explicit_close` を新規追加する。 内容は「`WebSocketServer` を明示 `stop()` を呼ばずに destruct しても hang せず終了する」 ことを検証する。
- 検証方法は 0001 / 0002 のテストに合わせ、 `weakref` で `__del__` 発火を実検証し、 `@pytest.mark.timeout` で hang 時の上限を指定する。 polling が無いため `RuntimeWarning` 経路は存在せず、 `recwarn` は使わない。
- 既存テストが PASS することを確認する。

### 4. CHANGES.md

- 0001 / 0002 の実装手順により後続 issue は別 PR で着手するため、 `## develop` に 0001 / 0002 のエントリとは別の `[FIX]` エントリを追加する。
  - 「`WebSocketServer` を明示的に `stop()` せずに destruct した場合の GIL 保持 hang を修正する」
  - 「`WebSocketServer.__del__` で `stop()` が自動的に呼ばれる」

## 完了条件

- `uv sync && make test` で全テストが PASS する。
- `tests/test_websocketserver.py::test_destruct_without_explicit_close` が、 明示 `stop()` を呼ばずに `server` を destruct しても hang せず終了し、 weakref により `__del__` 発火が検証できる。
- `CHANGES.md` の `## develop` に 0001 / 0002 とは別の `[FIX]` エントリが追加されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `stop_websocket_server` を匿名 namespace に追加する (polling なし)。
  - `WebSocketServer` bindings の `.def("stop", ...)` を `&stop_websocket_server` + `nb::call_guard<nb::gil_scoped_release>()` に差し替え、 `.def("__del__", ...)` を追加し、 `nb::class_<WebSocketServer>` に `nb::is_weak_referenceable()` を指定する。
- `src/libdatachannel/__init__.py`
  - 変更なし (Python wrapper は追加しない)。
- `tests/test_websocketserver.py`
  - `test_destruct_without_explicit_close` を追加する。
- `CHANGES.md`
  - `## develop` セクションに 0001 / 0002 とは別の `[FIX]` エントリを追加する。

## 参考

- 既存ブランチ (試行錯誤の履歴): `feature/fix-destructor-gil-release`
  - `85b144a` (`stop()` を GIL release で実行) / `6736371` (Python wrapper 追加) / `f4a1703` (test 追加) / `5869135` (`on_client` 仕様の明記) / `7f8112d` (test コメントを wrapper 実装と整合) は wrapper 方式に基づく試行錯誤であり、 0001 の解決方法で撤回された。 cherry-pick せず、 develop に取り込まれた 0001 の実装 (`close_peer_connection` / `.def("__del__")` / `nb::is_weak_referenceable()`) を踏襲すること。
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] (`close_peer_connection` / `__del__` / `is_weak_referenceable` の実装元) / [[0002-bug-fix-websocket-destructor-gil-release]] (`on_client` 経由の `WebSocket` の自動 `close()` の実装元) / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] / [[0005-bug-fix-destructor-callback-deadlock]]
- libdatachannel 関連コード位置 (シンボル名で特定する):
  - `_deps/libdatachannel/v0.24.0/source/src/websocketserver.cpp` の public `~WebSocketServer()` (impl の `stop()` を呼ぶ) と public `WebSocketServer::stop()`
  - `_deps/libdatachannel/v0.24.0/source/src/impl/websocketserver.cpp` の `WebSocketServer::stop()` (`tcpServer->close()` + `mThread.join()`) と `WebSocketServer::~WebSocketServer()` (公開 destructor からも `stop()` を呼ぶ)

## スコープ外 (関連する未解決問題)

- `stop()` に timeout は導入しない。 `stop()` が完了しない異常状態では destruct も完了しないが、 これは `tcpServer->close()` や `mThread.join()` の挙動に依存するため、 timeout の有無は別途設計判断が必要。 本 issue ではスコープ外とし、 必要に応じて別 issue で扱う。
- 本 issue は destruct 到達前に GIL release で `stop()` を完了させて destructor 経路の負担を減らすアプローチであり、 完全には hang を防げない。 callback が I/O block する条件下では引き続き hang し得るため、 根本対応は [[0005-bug-fix-destructor-callback-deadlock]] に集約する。
