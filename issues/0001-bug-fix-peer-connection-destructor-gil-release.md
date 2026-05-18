# PeerConnection の destructor 内で発生する GIL 保持 hang を修正する

- Priority: High
- Created: 2026-05-18
- Model: Opus 4.7
- Branch: feature/fix-peer-connection-destructor-gil-release

## 目的

`PeerConnection` を明示的に `close()` せずに destruct すると、 libdatachannel 本体の `~PeerConnection()` 内で `remoteClose()` を経由して `mProcessor.join()` が走る。 この cleanup は Python オブジェクト破棄経路 (= GIL を保持したまま) で実行されるため、 内部 cleanup task が Python callback の GIL 待ちに入った瞬間に Python プロセス全体が長時間 hang する。

本 issue では `PeerConnection` を対象に、「destruct 前に GIL release で同期 `close()` する」 仕組みを C++ binding と Python wrapper に追加し、 利用者が明示 `close()` を呼び忘れた場合の hang を回避する。

## 優先度根拠

- `PeerConnection` は本ライブラリの中心 API であり、 ほぼ全ての利用シナリオで触る。
- destruct 時の hang は Python プロセス全体の停止に直結し、 テスト・本番運用の両方で容易に踏み得る。
- `tests/test_peerconnection.py::test_leak` は本問題のため skip されており、 リグレッション検知が機能していない。
- 後続の `WebSocket` / `WebSocketServer` / `IceUdpMuxListener` 対応 ([[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]) で共通利用する `wait_for_closed` ヘルパーは本 issue で先行導入する。

## 現状

- `src/bind_libdatachannel.cpp` の `PeerConnection` bindings では `close` を `&PeerConnection::close` 等で直接バインドしており、 GIL を保持したまま `close()` が実行される。
- `src/libdatachannel/__init__.py` には Python wrapper class が無く、 `close()` を呼び忘れた場合のセーフティネットが無い。
- 利用者が `del pc` あるいはスコープ抜けによる暗黙の destruct を行うと、 `~PeerConnection()` 内 `mProcessor.join()` が GIL 保持下で動き、 callback の GIL 待ちと噛み合って Python プロセス全体が hang する。
- `tests/test_peerconnection.py::test_leak` は本問題が再現するため skip されており、 回帰検知が止まっている。

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

- `PeerConnection::close` のバインドを関数化し、 以下のロジックを持つ `close_peer_connection` を `namespace {}` 内に追加する。
  - `self.state() == PeerConnection::State::Closed` なら no-op で早期 return する (冪等性)。
  - そうでなければ `self.close()` を呼び、 後述の共通ヘルパー `wait_for_closed` で `state==Closed` まで polling する。
- `.def("close", &close_peer_connection, nb::call_guard<nb::gil_scoped_release>())` のように、 `nb::call_guard<nb::gil_scoped_release>()` を必ず指定して GIL を release した状態で実行する。
- 共通 polling ヘルパー `wait_for_closed` を `namespace {}` の冒頭に追加する。 仕様:
  - template 化し、 state ゲッタ (`std::function` 不要、 lambda で十分) と Closed 値、 timeout 警告メッセージを受け取る。
  - polling 定数は `constexpr auto kPollInterval = std::chrono::milliseconds(10);` と `constexpr auto kCloseTimeout = std::chrono::seconds(30);` を採用する。 根拠コメント (10 ms: busy loop でない最小値 / 30 s: 通常 close の数秒に対し余裕を持った上限) をコード内に必ず残す。
  - timeout 到達時は `nb::gil_scoped_acquire` で GIL を再取得し、 `PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1)` を呼ぶ。 戻り値が `< 0` の場合 (例: `filterwarnings=error` で warning が例外に昇格された場合) は pending exception を放置せず `throw nb::python_error();` で伝播させる。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- `from .libdatachannel_ext import PeerConnection as _PeerConnection` で内部参照を退避する。
- `class PeerConnection(_PeerConnection)` を新規追加し、 `__del__` 内で `self.close()` を `try/except Exception: pass` で囲んで呼ぶ。
- C++ binding 側で `close()` は GIL release されているため、 destruct 経路から呼んでも他 thread が動ける。
- wrapper の docstring には以下を明記する。
  - 「リソースの確実な解放のため、 明示的に `close()` を呼ぶことを推奨する」
  - 「`close()` を呼び忘れた場合のセーフティネットとして `__del__` で `close()` を呼ぶ」
  - 「`__del__` は GC タイミングに依存するため、 close 完了時刻が予測しにくい」

### 3. テスト (tests/test_peerconnection.py)

- 既存の skip された `test_leak` の skip を解除し、 `test_destruct_without_explicit_close` にリネームする。 内容は「`PeerConnection` を明示 `close()` せず関数スコープ抜け / `del` で destruct しても hang せず終了する」 ことを検証する。
- `test_close_is_idempotent` を新規追加し、 `close()` を 2 回呼んでも 2 回目が no-op で即時 return することを検証する (経過時間が `kCloseTimeout` を超えないこと等)。
- 未使用 import (`import pytest` 等) が残る場合は整理する。

### 4. CHANGES.md

- `## develop` セクションに `[FIX]` エントリを追加する。 文言例:
  - `- [FIX] PeerConnection を明示的に close() せずに destruct した場合に発生する GIL 保持 hang を修正する`
  - 補足として「公開クラス `PeerConnection` が Python wrapper に置き換わる」 「`__del__` 経由で `close()` が呼ばれる」 「close() polling timeout 30 秒で `RuntimeWarning` を出す」 を箇条書きで明記する。
- 担当者行はエントリの次行に 2 文字分インデントで `- @voluntas` 等を書く。

## 完了条件

- `uv sync && make test` で全テストが PASS する (skip しないこと)。
- `tests/test_peerconnection.py::test_destruct_without_explicit_close` が、 明示 `close()` を呼ばずに `pc` を destruct しても hang せず終了する。
- `tests/test_peerconnection.py::test_close_is_idempotent` が PASS する (2 回目 `close()` が即時 return する)。
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `<chrono>` / `<thread>` の include を追加。
  - 匿名 namespace の冒頭に `wait_for_closed` (template) と `close_peer_connection` を追加。
  - `PeerConnection` bindings の `.def("close", ...)` を `&close_peer_connection` + `nb::call_guard<nb::gil_scoped_release>()` に差し替え。
- `src/libdatachannel/__init__.py`
  - `from .libdatachannel_ext import PeerConnection as _PeerConnection` を追加。
  - `class PeerConnection(_PeerConnection)` を追加し `__del__` で `close()` を呼ぶ。
- `tests/test_peerconnection.py`
  - 既存 `test_leak` の skip を解除して `test_destruct_without_explicit_close` にリネーム。
  - `test_close_is_idempotent` を追加。
- `CHANGES.md`
  - `## develop` セクションに `[FIX]` エントリを追加。

## 参考

- 仕切り直しサマリ: `/tmp/destructor-gil-release-summary.md`
- 既存ブランチ (参考実装): `feature/fix-destructor-gil-release`
  - 該当コミット: `0d376df` (`PeerConnection.close()` を GIL release で同期実行), `02d9628` (Python wrapper 追加), `6afe6b3` (timeout 追加), `2e7dd5a` (RuntimeWarning), `a51b82d` (polling 定数の根拠コメント), `ab2f2b8` (関数化と早期 return), `44dedf5` (`PyErr_WarnEx` の戻り値検査), `23e31fa` (`test_close_is_idempotent`)
- 関連 issue: [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]
- libdatachannel 関連コード位置:
  - `_deps/libdatachannel/v0.24.0/source/src/peerconnection.cpp:40-46` (`~PeerConnection()` public)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:82` (`~PeerConnection()` impl の `mProcessor.join()`)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:87-95` (`close()` impl)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:97-106` (`remoteClose()` impl)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:1321-1323` (`changeState(Closed)` の同期 callback 実行)
  - `_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp:79` (`synchronized_callback::operator()` の mutex 保持実行)

## スコープ外 (関連する未解決問題)

- pytest stdout capture + callback での `print` + `Track.on_closed` 等を組み合わせた条件下で、 `gc.collect()` を強制呼びすると依然として hang する事象が観測されている。 これは libdatachannel 内の `synchronized_callback` が mutex を保持したまま同期実行する構造に由来し、 binding 単体では完全解決できない。 本 issue では `gc.collect()` を含むテストは導入しない。 別 issue として切り出して検討する (案 I: nanobind の `tp_dealloc` フック、 案 G/J: callback wrapper の非同期化 等)。
