# IceUdpMuxListener の destructor 内で発生する GIL 保持 hang を修正する

- Priority: Medium
- Created: 2026-05-18
- Polished: 2026-08-30
- Model: Opus 4.7
- Branch: feature/fix-ice-udp-mux-listener-destructor-gil-release

## 目的

`IceUdpMuxListener` を明示的に `stop()` せずに destruct すると、 libdatachannel (および libjuice) の cleanup 経路で内部 thread が `thread_join` 等によって同期 join される。 これらの処理は Python オブジェクト破棄経路 (= GIL を保持したまま) で走るため、 thread 側が Python callback の GIL 待ちに入った瞬間に Python プロセス全体が hang する。

本 issue では `IceUdpMuxListener` を対象に、 destruct 前に GIL release で同期 `stop()` する仕組みを C++ binding に追加する ([[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法と同じ binding 側 `__del__` 方式)。 state API を持たないため polling は行わず、 GIL release だけを担保する。

## 優先度根拠

- `IceUdpMuxListener` は UDP MUX 機能を使う利用者のみが触る API のため、 `PeerConnection` ([[0001-bug-fix-peer-connection-destructor-gil-release]]) や `WebSocket` ([[0002-bug-fix-websocket-destructor-gil-release]]) ほど影響範囲は広くない。
- ただし destruct 時の hang は debug 困難な事象であり、 同じ仕組みで一連の修正 ([[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]]) と整合した形で閉じる必要がある。

## 現状

- `src/bind_libdatachannel.cpp` の `IceUdpMuxListener` bindings は `stop` を `&IceUdpMuxListener::stop` で直接バインドしており、 GIL を保持したまま `stop()` が実行される。
- `IceUdpMuxListener` には state API がないため、 stop の完了を polling で確認することはできない。 `stop()` の戻り時点で内部 thread の `thread_join` が完了している前提に乗る。
- `src/libdatachannel/__init__.py` に Python wrapper class が無い。
- 利用者が `del listener` あるいは function スコープ抜けで destruct した場合、 `~IceUdpMuxListener()` 配下の cleanup が GIL 保持下で走り、 callback の GIL 待ちと噛み合って Python プロセス全体が hang する。
- `tests/` ディレクトリには `IceUdpMuxListener` を直接対象とするテストファイルが現状存在しない。

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

[[0001-bug-fix-peer-connection-destructor-gil-release]] の解決方法 (binding 側の `close_peer_connection` + `.def("__del__", ...)` + `nb::is_weak_referenceable()`) と同型の実装を `IceUdpMuxListener` に適用する。

- `stop_ice_udp_mux_listener` を `namespace {}` に追加する。 `self.stop()` を呼ぶだけで、 polling は行わない (state API が無い)。 `stop()` の戻り時点で内部 thread の `thread_join` が完了している。
- `.def("stop", &IceUdpMuxListener::stop)` を `.def("stop", &stop_ice_udp_mux_listener, nb::call_guard<nb::gil_scoped_release>())` に差し替える。
- `.def("__del__", ...)` を追加し、 GIL release 下で `stop_ice_udp_mux_listener` を呼ぶ。 `__del__` から投げた例外は呼び出し側で捕捉できないため `RuntimeWarning` として記録するだけで握り潰す (0001 の `PeerConnection` binding と同型)。
- `nb::class_<IceUdpMuxListener>` に `nb::is_weak_referenceable()` を指定する (test で weakref により `__del__` 発火を検証するため)。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- 変更不要。 0001 の解決方法で wrapper class 方式は撤回されたため、 本 issue でも Python wrapper は追加せず binding 側の `__del__` 方式に統一する。

### 3. テスト (tests/test_ice_udp_mux_listener.py)

- 新規ファイル `tests/test_ice_udp_mux_listener.py` を作成し、 `test_destruct_without_explicit_close` を追加する。 内容は「`IceUdpMuxListener` を明示 `stop()` を呼ばずに destruct しても hang せず終了する」 ことを検証する。
- 検証方法は 0001 / 0002 / 0003 のテストに合わせ、 `weakref` で `__del__` 発火を実検証し、 `@pytest.mark.timeout` で hang 時の上限を指定する。 polling が無いため `RuntimeWarning` 経路は存在せず、 `recwarn` は使わない。
- ファイル名はテストディレクトリ内の既存命名 (`test_<lower_case>.py`) に従う。
- localhost で利用可能な UDP ポートで listener を立ち上げ、 immediately destruct する最小ケースを書く。

### 4. CHANGES.md

- 0001 / 0002 / 0003 の実装手順により後続 issue は別 PR で着手するため、 `## develop` に 0001 / 0002 / 0003 のエントリとは別の `[FIX]` エントリを追加する。
  - 「`IceUdpMuxListener` を明示的に `stop()` せずに destruct した場合の GIL 保持 hang を修正する」
  - 「`IceUdpMuxListener.__del__` で `stop()` が自動的に呼ばれる」

## 完了条件

- `uv sync && make test` で全テストが PASS する。
- `tests/test_ice_udp_mux_listener.py::test_destruct_without_explicit_close` が、 明示 `stop()` を呼ばずに `listener` を destruct しても hang せず終了し、 weakref により `__del__` 発火が検証できる。
- `CHANGES.md` の `## develop` に 0001 / 0002 / 0003 とは別の `[FIX]` エントリが追加されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `stop_ice_udp_mux_listener` を匿名 namespace に追加する (polling なし)。
  - `IceUdpMuxListener` bindings の `.def("stop", ...)` を `&stop_ice_udp_mux_listener` + `nb::call_guard<nb::gil_scoped_release>()` に差し替え、 `.def("__del__", ...)` を追加し、 `nb::class_<IceUdpMuxListener>` に `nb::is_weak_referenceable()` を指定する。
- `src/libdatachannel/__init__.py`
  - 変更なし (Python wrapper は追加しない)。
- `tests/test_ice_udp_mux_listener.py`
  - 新規ファイルを作成し、 `test_destruct_without_explicit_close` を追加。
- `CHANGES.md`
  - `## develop` セクションに 0001 / 0002 / 0003 とは別の `[FIX]` エントリを追加する。

## 参考

- 既存ブランチ (試行錯誤の履歴): `feature/fix-destructor-gil-release`
  - `f67b9ee` (`IceUdpMuxListener` を hang 対策の対象に追加) は wrapper 方式に基づく試行錯誤であり、 0001 の解決方法で撤回された。 cherry-pick せず、 develop に取り込まれた 0001 の実装 (`close_peer_connection` / `.def("__del__")` / `nb::is_weak_referenceable()`) を踏襲すること。
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] (`close_peer_connection` / `__del__` / `is_weak_referenceable` の実装元) / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]]
- libdatachannel / libjuice 関連コード位置 (シンボル名で特定する):
  - `_deps/libdatachannel/v0.24.0/source/deps/libjuice/src/conn_mux.c` の `conn_mux_registry_cleanup` (内部の `thread_join`)
  - `_deps/libdatachannel/v0.24.0/source/src/iceudpmuxlistener.cpp` の public `stop()`

## スコープ外 (関連する未解決問題)

- `stop()` に timeout は導入しない。 `stop()` が完了しない異常状態では destruct も完了しないが、 これは内部 thread の `thread_join` の挙動に依存するため、 timeout の有無は別途設計判断が必要。 本 issue ではスコープ外とし、 必要に応じて別 issue で扱う。
