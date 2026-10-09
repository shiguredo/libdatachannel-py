# IceUdpMuxListener の destructor 内で発生する GIL 保持 hang を修正する

- Priority: Medium
- Created: 2026-05-18
- Polished: 2026-10-09
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
- `nb::class_<IceUdpMuxListener>` に `nb::is_weak_referenceable()` を指定する (test で weakref を使うため)。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- 変更不要。 0001 の解決方法で wrapper class 方式は撤回されたため、 本 issue でも Python wrapper は追加せず binding 側の `__del__` 方式に統一する。

### 3. テスト (tests/test_ice_udp_mux_listener.py)

- 新規ファイル `tests/test_ice_udp_mux_listener.py` を作成し、 以下を追加する。
  - `test_stop_releases_gil`: `stop()` の呼び出し中に GIL を待つ thread が進行することで、 GIL 解放を実測する ([[0003-bug-fix-websocketserver-destructor-gil-release]] と同じ方式。 free-threading ビルドでは skip)
  - `test_destruct_without_explicit_close`: 明示 `stop()` を呼ばずに破棄しても恒停せず終了すること (weakref が死ぬこと) を検証する。 **`weakref` で `__del__` の発火は検証できない**: nanobind 3 の `tp_dealloc` は C++ destructor を直接呼び、 CPython の finalizer (`tp_finalize`) を呼ばないため、 基底クラスのインスタンス破棄時に `__del__` は実行されない ([[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] の実装時に実測して判明済み)
  - `test_del_calls_stop_on_python_subclass`: Python サブクラスでは破棄時に `__del__` が実行され、 その中から `super().__del__()` (= binding の stop) を呼べること、 停止後に同じ UDP ポートを bind できること (= stop が実際に走ったこと) を検証する
- ファイル名はテストディレクトリ内の既存命名 (`test_<lower_case>.py`) に従う。

### 4. CHANGES.md

- 0001 / 0002 / 0003 の実装手順により後続 issue は別 PR で着手するため、 `## develop` に 0001 / 0002 / 0003 のエントリとは別の `[FIX]` エントリを追加する。
  - 「`IceUdpMuxListener` を明示的に `stop()` せずに destruct した場合の GIL 保持 hang を修正する」
  - 「`IceUdpMuxListener.__del__` から GIL 解放下で `stop()` が呼ばれる (Python サブクラスでは破棄時に実行される)」

## 完了条件

- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する。
- `tests/test_ice_udp_mux_listener.py` の新規 3 テスト (`test_stop_releases_gil` / `test_destruct_without_explicit_close` / `test_del_calls_stop_on_python_subclass`) が PASS する。
- `IceUdpMuxListener.stop()` が GIL 解放下で実行されること (GIL を待つ thread が停止中に進行することで実測)。
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
- 検証 (レビュー): `stop()` を GIL 解放下で実行しない場合は `test_stop_releases_gil` が失敗する (GIL を解放しない呼び出しを同形で回すと進行 0 になる)。 `__del__` を削除した場合は `super().__del__()` の `AttributeError` を、 `stop()` を呼ばない場合は停止後のポート bind 成功を検出して失敗する
- 既知の制約: nanobind 3 の `tp_dealloc` は CPython の finalizer を呼ばないため、 基底クラスのインスタンスを破棄する経路では `__del__` は実行されない。 破棄時に GIL を保持したまま走る C++ 側の公開デストラクタ (`stop()` 呼び出し) の恒停は binding 側では解消できず、 根本対応は [[0005-bug-fix-destructor-callback-deadlock]] / [[0039-bug-fix-nanobind-del-not-called]] に集約する ([[0003-bug-fix-websocketserver-destructor-gil-release]] と同じ整理)
- libdatachannel 関連コード位置 (シンボル名で特定する):
  - `_deps/libdatachannel/v0.24.0/source/deps/libjuice/src/conn_mux.c` の `conn_mux_registry_cleanup` (内部の `thread_join`)
  - `_deps/libdatachannel/v0.24.0/source/src/iceudpmuxlistener.cpp` の public `stop()`

## スコープ外 (関連する未解決問題)

- `stop()` に timeout は導入しない。 `stop()` が完了しない異常状態では destruct も完了しないが、 これは内部 thread の `thread_join` の挙動に依存するため、 timeout の有無は別途設計判断が必要。 本 issue ではスコープ外とし、 必要に応じて別 issue で扱う。
