# WebSocketServer の destructor 内で発生する GIL 保持 hang を修正する

- Priority: Medium
- Created: 2026-05-18
- Model: Opus 4.7
- Branch: feature/fix-websocketserver-destructor-gil-release

## 目的

`WebSocketServer` を明示的に `stop()` せずに destruct すると、 libdatachannel 本体の `~WebSocketServer()` 配下で `stop()` が呼ばれ、 内部の `tcpServer->close()` と `mThread.join()` が GIL を保持したまま走る。 受け入れ thread が Python callback (`on_client` 等) の GIL 待ちに入ると Python プロセス全体が hang する。

本 issue では `WebSocketServer` を対象に、 destruct 前に GIL release で同期 `stop()` する仕組みを C++ binding と Python wrapper に追加する。 `WebSocketServer` は state API を持たないため polling は行わず、 GIL release だけを担保する。 加えて `on_client` callback に渡される `WebSocket` インスタンスは Python wrapper にラップされない仕様を明確化する。

## 優先度根拠

- `WebSocketServer` はサーバー用途の利用者のみが触る API のため、 `PeerConnection` ([[0001-bug-fix-peer-connection-destructor-gil-release]]) や `WebSocket` ([[0002-bug-fix-websocket-destructor-gil-release]]) ほど影響範囲は広くない。
- しかし destruct 時の hang は debug 困難な事象であり、 一連の修正 ([[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]) と整合した形で閉じる必要がある。
- `on_client` callback で受け取る `WebSocket` が wrapper されない (= `__del__` 経由の自動 `close()` が走らない) という周辺仕様も合わせて明文化する。

## 現状

- `src/bind_libdatachannel.cpp` の `WebSocketServer` bindings は `stop` を `&WebSocketServer::stop` で直接バインドしており、 GIL を保持したまま `stop()` が実行される。
- `WebSocketServer` には state API がないため、 stop の完了を polling で確認することはできない。 `stop()` の戻り時点で `tcpServer->close()` と `mThread.join()` が完了している前提に乗る。
- `src/libdatachannel/__init__.py` に Python wrapper class が無い。
- 利用者が `del server` あるいは function スコープ抜けで destruct した場合に、 `~WebSocketServer()` 内 `stop()` が GIL 保持下で走り、 callback の GIL 待ちと噛み合って Python プロセス全体が hang する。
- `on_client` callback の引数で渡される `WebSocket` インスタンスは nanobind の native class であり、 Python wrapper に置き換わらない。 これは現状ドキュメント化されておらず、 利用者が wrapper の挙動を期待すると混乱する。

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

- `WebSocketServer` bindings の `.def("stop", &WebSocketServer::stop)` を `.def("stop", &WebSocketServer::stop, nb::call_guard<nb::gil_scoped_release>())` に差し替える。
- polling は不要 (state API が無い)。 `wait_for_closed` ヘルパーには依存しない。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- `from .libdatachannel_ext import WebSocketServer as _WebSocketServer` を追加する。
- `class WebSocketServer(_WebSocketServer)` を新規追加し、 `__del__` 内で `try/except Exception: pass` で囲んで `self.stop()` を呼ぶ。
- docstring に以下を明記する。
  - 「明示的に `stop()` を呼ぶことを推奨。 `__del__` はセーフティネット」
  - 「`on_client` callback の引数で渡される `WebSocket` インスタンスは Python wrapper ではなく nanobind の native class のため、 `__del__` 経由の自動 close は走らない。 callback 内で `WebSocket` を保持して使う場合は明示的に `close()` を呼ぶこと」

### 3. テスト (tests/test_websocketserver.py)

- `test_destruct_without_explicit_close` を新規追加する。 内容は「`WebSocketServer` を明示 `stop()` を呼ばずに destruct しても hang せず終了する」 ことを検証する。
- 既存テストが PASS することを確認する。

### 4. CHANGES.md

- 既存の `[FIX]` エントリ ([[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] と同 PR にまとめる場合は同一エントリに加筆) に以下を追記する。
  - 「`WebSocketServer` を明示的に `stop()` せずに destruct した場合の GIL 保持 hang を修正する」
  - 「公開クラス `WebSocketServer` が Python wrapper に置き換わる」
  - 「`on_client` callback に渡される `WebSocket` は wrapper されない仕様を docstring に明記した」

## 完了条件

- `uv sync && make test` で全テストが PASS する。
- `tests/test_websocketserver.py::test_destruct_without_explicit_close` が、 明示 `stop()` を呼ばずに `server` を destruct しても hang せず終了する。
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加 (or 既存エントリに加筆) されている。
- `on_client` 経由の `WebSocket` が wrapper されない仕様が docstring に明記されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `WebSocketServer` bindings の `.def("stop", ...)` に `nb::call_guard<nb::gil_scoped_release>()` を追加。
- `src/libdatachannel/__init__.py`
  - `from .libdatachannel_ext import WebSocketServer as _WebSocketServer` を追加。
  - `class WebSocketServer(_WebSocketServer)` を追加し `__del__` で `stop()` を呼ぶ。
  - docstring に `on_client` callback の `WebSocket` が wrapper されない仕様を明記。
- `tests/test_websocketserver.py`
  - `test_destruct_without_explicit_close` を追加。
- `CHANGES.md`
  - `## develop` セクションに `[FIX]` エントリを追加 (or 0001 / 0002 のエントリに加筆)。

## 参考

- 仕切り直しサマリ: `/tmp/destructor-gil-release-summary.md`
- 既存ブランチ (参考実装): `feature/fix-destructor-gil-release`
  - 該当コミット: `85b144a` (`stop()` を GIL release で実行), `6736371` (Python wrapper 追加), `f4a1703` (test 追加), `5869135` (`on_client` 仕様の明記), `7f8112d` (test コメントを wrapper 実装と整合)
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]
- libdatachannel 関連コード位置:
  - `_deps/libdatachannel/v0.24.0/source/src/impl/websocketserver.cpp:61-68` (`stop()` 内 `tcpServer->close()` + `mThread.join()`)

## スコープ外 (関連する未解決問題)

- `stop()` に timeout は導入しない。 `stop()` が完了しない異常状態では destruct も完了しないが、 これは `tcpServer->close()` や `mThread.join()` の挙動に依存するため、 timeout の有無は別途設計判断が必要 (レビュー指摘 I-6)。 本 issue ではスコープ外とし、 必要に応じて別 issue で扱う。
