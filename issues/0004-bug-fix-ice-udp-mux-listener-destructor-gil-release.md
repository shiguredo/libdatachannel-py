# IceUdpMuxListener の destructor 内で発生する GIL 保持 hang を修正する

- Priority: Medium
- Created: 2026-05-18
- Model: Opus 4.7
- Branch: feature/fix-ice-udp-mux-listener-destructor-gil-release

## 目的

`IceUdpMuxListener` を明示的に `stop()` せずに destruct すると、 libdatachannel (および libjuice) の cleanup 経路で内部 thread が `thread_join` 等によって同期 join される。 これらの処理は Python オブジェクト破棄経路 (= GIL を保持したまま) で走るため、 thread 側が Python callback の GIL 待ちに入った瞬間に Python プロセス全体が hang する。

本 issue では `IceUdpMuxListener` を対象に、 destruct 前に GIL release で同期 `stop()` する仕組みを C++ binding と Python wrapper に追加する。 state API を持たないため polling は行わず、 GIL release だけを担保する。

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

- `IceUdpMuxListener` bindings の `.def("stop", &IceUdpMuxListener::stop)` を `.def("stop", &IceUdpMuxListener::stop, nb::call_guard<nb::gil_scoped_release>())` に差し替える。
- polling は不要 (state API が無い)。 `wait_for_closed` ヘルパーには依存しない。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- `from .libdatachannel_ext import IceUdpMuxListener as _IceUdpMuxListener` を追加する。
- `class IceUdpMuxListener(_IceUdpMuxListener)` を新規追加し、 `__del__` 内で `try/except Exception: pass` で囲んで `self.stop()` を呼ぶ。
- docstring に「明示的に `stop()` を呼ぶことを推奨。 `__del__` はセーフティネット」 を明記する。

### 3. テスト (tests/test_ice_udp_mux_listener.py)

- 新規ファイル `tests/test_ice_udp_mux_listener.py` を作成し、 `test_destruct_without_explicit_close` を追加する。 内容は「`IceUdpMuxListener` を明示 `stop()` を呼ばずに destruct しても hang せず終了する」 ことを検証する。
- ファイル名はテストディレクトリ内の既存命名 (`test_<lower_case>.py`) に従う。
- localhost で利用可能な UDP ポートで listener を立ち上げ、 immediately destruct する最小ケースを書く。

### 4. CHANGES.md

- 既存の `[FIX]` エントリ ([[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] と同 PR にまとめる場合は同一エントリに加筆) に以下を追記する。
  - 「`IceUdpMuxListener` を明示的に `stop()` せずに destruct した場合の GIL 保持 hang を修正する」
  - 「公開クラス `IceUdpMuxListener` が Python wrapper に置き換わる」

## 完了条件

- `uv sync && make test` で全テストが PASS する。
- `tests/test_ice_udp_mux_listener.py::test_destruct_without_explicit_close` が、 明示 `stop()` を呼ばずに `listener` を destruct しても hang せず終了する。
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加 (or 既存エントリに加筆) されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `IceUdpMuxListener` bindings の `.def("stop", ...)` に `nb::call_guard<nb::gil_scoped_release>()` を追加。
- `src/libdatachannel/__init__.py`
  - `from .libdatachannel_ext import IceUdpMuxListener as _IceUdpMuxListener` を追加。
  - `class IceUdpMuxListener(_IceUdpMuxListener)` を追加し `__del__` で `stop()` を呼ぶ。
- `tests/test_ice_udp_mux_listener.py`
  - 新規ファイルを作成し、 `test_destruct_without_explicit_close` を追加。
- `CHANGES.md`
  - `## develop` セクションに `[FIX]` エントリを追加 (or 0001 / 0002 / 0003 のエントリに加筆)。

## 参考

- 仕切り直しサマリ: `/tmp/destructor-gil-release-summary.md`
- 既存ブランチ (参考実装): `feature/fix-destructor-gil-release`
  - 該当コミット: `f67b9ee` (`IceUdpMuxListener` を hang 対策の対象に追加)
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]]
- libdatachannel / libjuice 関連コード位置:
  - `_deps/libdatachannel/v0.24.0/source/deps/libjuice/src/conn_mux.c:288` (`conn_mux_registry_cleanup` 内の `thread_join`)
  - `_deps/libdatachannel/v0.24.0/source/src/iceudpmuxlistener.cpp` (`stop()` の public 実装)

## スコープ外 (関連する未解決問題)

- `stop()` に timeout は導入しない。 `stop()` が完了しない異常状態では destruct も完了しないが、 これは内部 thread の `thread_join` の挙動に依存するため、 timeout の有無は別途設計判断が必要。 本 issue ではスコープ外とし、 必要に応じて別 issue で扱う。
