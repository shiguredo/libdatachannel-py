# WebSocket の destructor 内で発生する GIL 保持 hang を修正する

- Priority: High
- Created: 2026-05-18
- Model: Opus 4.7
- Branch: feature/fix-websocket-destructor-gil-release

## 目的

`WebSocket` を明示的に `close()` せずに destruct すると、 libdatachannel 本体の close 経路で内部 thread の停止と TLS/TCP の close handshake が GIL を保持したまま走る。 内部 cleanup が Python callback の GIL 待ちに入ると Python プロセス全体が hang する。

本 issue では `WebSocket` を対象に、 destruct 前に GIL release で同期 `close()` する仕組みを C++ binding と Python wrapper に追加する。 ただし TLS/TCP の close handshake は別 thread が完了させる必要があるため、 `state==Closing` の場合は polling せず即 return する例外規則を入れる。

## 優先度根拠

- `WebSocket` は signaling・転送等で頻繁に使用されるため、 destruct 時 hang の影響を受けやすい。
- 影響度は `PeerConnection` ([[0001-bug-fix-peer-connection-destructor-gil-release]]) と同等であり、 一緒に解消する方がリリースノートの整合も取りやすい。
- 後続の `WebSocketServer` ([[0003-bug-fix-websocketserver-destructor-gil-release]]) では `WebSocket` の wrapper の存在が前提になる箇所がある (`on_client` callback の限定事項) ため、 本 issue で wrapper を確定させる必要がある。

## 現状

- `src/bind_libdatachannel.cpp` の `WebSocket` bindings は `close` を直接バインドしており、 GIL を保持したまま `close()` が実行される。
- `WebSocket` には state API があり (`Connecting / Open / Closing / Closed`)、 close の同期完了を polling で待てる構造である。 しかし `Closing` 以降は libdatachannel 内部の `close()` が no-op になり、 `Closed` へ進めるのは TLS/TCP close handshake (別 thread) に委ねられる。 すなわち polling を盲目的に行うと対向応答遅延で timeout までブロックする。
- `src/libdatachannel/__init__.py` に Python wrapper class が無い。
- 利用者の暗黙 destruct 時、 libdatachannel 内部の close 経路で GIL 保持下の hang が起こりうる。

## 設計方針

### 1. C++ binding 側 (src/bind_libdatachannel.cpp)

- [[0001-bug-fix-peer-connection-destructor-gil-release]] で導入された共通ヘルパー `wait_for_closed` (template 関数, polling 定数 `kPollInterval=10ms` / `kCloseTimeout=30s`) を再利用する。
- `close_websocket` を `namespace {}` に追加する。 ロジック:
  1. `self.readyState() == WebSocket::State::Closed` なら no-op で早期 return する。
  2. `self.readyState() == WebSocket::State::Closing` なら **polling せず即 return** する。 残りの状態遷移は `~WebSocket()` 側の destructor 処理に委ねる。 この特別扱いは「`WebSocket::close()` は `Connecting/Open` のときしか動かず、 `Closing` 以降は内部で何もしない」 「対向応答遅延で polling が 30 秒待たされるのを避ける」 という理由をコードコメントに明記する。
  3. それ以外は `self.close()` を呼び、 `wait_for_closed(self, [](WebSocket& s) { return s.readyState(); }, WebSocket::State::Closed, "<warning message>")` で `Closed` まで待つ。
- `.def("close", &close_websocket, nb::call_guard<nb::gil_scoped_release>())` のように差し替える。

### 2. Python wrapper 側 (src/libdatachannel/__init__.py)

- `from .libdatachannel_ext import WebSocket as _WebSocket` を追加する。
- `class WebSocket(_WebSocket)` を新規追加し、 `__del__` 内で `try/except Exception: pass` で囲んで `self.close()` を呼ぶ。
- docstring に明示 `close()` 推奨と `__del__` セーフティネットの位置付けを明記する。

### 3. テスト (tests/test_websocket.py)

- `test_destruct_without_explicit_close` を新規追加する。 内容は「`WebSocket` を明示 `close()` せず destruct しても hang せず終了する」。
- `tests/test_websocketserver.py` の既存 server fixture を流用するか、 同等の最小サーバーを別途立てる。
- 既存テストが PASS することを確認する。

### 4. CHANGES.md

- [[0001-bug-fix-peer-connection-destructor-gil-release]] と同じ `## develop` の `[FIX]` エントリに `WebSocket` 分を追記するか、 別エントリとして追加する (1 PR にまとめる場合は前者、 分割する場合は後者)。
- 文言には「公開クラス `WebSocket` が Python wrapper に置き換わる」 「`__del__` で `close()` が自動的に呼ばれる」 「`state==Closing` の場合は polling せず即 return する」 を明記する。

## 完了条件

- `uv sync && make test` で全テストが PASS する。
- `tests/test_websocket.py::test_destruct_without_explicit_close` が、 明示 `close()` を呼ばずに `ws` を destruct しても hang せず終了する。
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加 (or 既存エントリに加筆) されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること。

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `close_websocket` を匿名 namespace に追加。 `Closed` / `Closing` の早期 return 規則と理由コメントを併記する。
  - `WebSocket` bindings の `.def("close", ...)` を `&close_websocket` + `nb::call_guard<nb::gil_scoped_release>()` に差し替え。
- `src/libdatachannel/__init__.py`
  - `from .libdatachannel_ext import WebSocket as _WebSocket` を追加。
  - `class WebSocket(_WebSocket)` を追加し `__del__` で `close()` を呼ぶ。
- `tests/test_websocket.py`
  - `test_destruct_without_explicit_close` を追加。
- `CHANGES.md`
  - `## develop` セクションに `[FIX]` エントリを追加 (or 0001 のエントリに加筆)。

## 参考

- 仕切り直しサマリ: `/tmp/destructor-gil-release-summary.md`
- 既存ブランチ (参考実装): `feature/fix-destructor-gil-release`
  - 該当コミット: `85b144a` (`close()` を GIL release で実行), `6736371` (Python wrapper 追加), `ab2f2b8` (`Closing` 状態の polling 早期 return)
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] (`wait_for_closed` ヘルパー導入元) / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]
- libdatachannel 関連コード位置:
  - `_deps/libdatachannel/v0.24.0/source/src/impl/websocket.cpp:69` (`~WebSocket()` impl は `PLOG` のみだが close 経路は別)
  - `_deps/libdatachannel/v0.24.0/source/src/websocket.cpp` (`close()` の public 実装)
  - `_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp:79` (`synchronized_callback::operator()` の mutex 保持実行)
