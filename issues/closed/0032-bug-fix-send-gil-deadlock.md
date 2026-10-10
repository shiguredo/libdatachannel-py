# DataChannel.send などの送信系 binding が GIL を保持して受信経路の callback とデッドロックする

- Priority: High
- Created: 2026-10-08
- Completed: 2026-10-08
- Branch: feature/fix-send-gil-deadlock
- Polished: 2026-10-08
- Reporter: @Geomglot

## 目的

映像トラック (H.264) とデータチャネルを同時に使う構成で、 asyncio の main thread から `DataChannel.send()` を頻繁に呼んでいると、 受信側のパケットロスが多い条件下で Python プロセスが恒久的にデッドロックする (CPU 使用率はほぼ 0%)。

送信系 binding が GIL を保持したまま libdatachannel の送信経路 (SCTP → DTLS) に入るため、 受信経路の worker thread は Python callback (`PliHandler` の `on_pli` など) の実行に必要な GIL を取得できない。 worker thread は内部ロックを保持したまま GIL を待ち、 main thread は GIL を保持したまま送信経路の内部ロックを待つという循環待ちになり、 復帰不能になる。

`PeerConnection.close()` / `PeerConnection.__del__` で導入済みの GIL 解放を送信系 binding にも適用し、 この循環待ちを解消する。 `WebSocket.close()` / `WebSocketServer.stop()` / `IceUdpMuxListener.stop()` への同種の適用は [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] が担当する (いずれも open)。

## 優先度根拠

- 復帰不能の恒久デッドロックであり、 プロセス再起動以外の回避手段がない
- 実利用 (commaai/teleoprtc / openpilot の webrtcd) で発生している
- 現状の回避策は「受信経路で Python を呼ばない」 (= `PliHandler` の Python callback を使わない) であり、 `examples/whip.py` が示す標準的な使い方を諦める必要がある
- 修正は `PeerConnection` の `close` / `__del__` binding と同じ `nb::call_guard<nb::gil_scoped_release>()` の適用で完結し、 影響範囲と実装コストが小さい

## 現状

報告された環境と操作条件:

- libdatachannel-py 2026.1.0.dev2 (aarch64 / Linux、 Python 3.12)
- 利用元は commaai/teleoprtc (openpilot の webrtcd)
- 映像トラック (H.264) とデータチャネルを同時に使用し、 asyncio の main thread から `DataChannel.send()` を頻繁に呼ぶ
- 受信側でパケットロスが多い (Wi-Fi が弱い) ときに発生しやすい
- 回避策として `PliHandler` の Python callback を使わないようにしている

報告されたバックトレース (gdb / py-spy):

```
Main thread (GIL 保持):
  ___pthread_mutex_lock
  rtc::impl::DtlsTransport::send
  rtc::impl::SctpTransport::handleWrite / WriteCallback
  usrsctp_sendv
  rtc::impl::SctpTransport::send
  rtc::impl::DataChannel::outgoing
  rtc::DataChannel::send
    (Python) channel.send(...)

RTC worker:
  PyEval_RestoreThread
  PyGILState_Ensure
  rtc::PliHandler::incoming
  rtc::MediaHandler::incomingChain
  rtc::impl::Track::incoming
  rtc::impl::DtlsTransport::ReadCallback
  rtc::impl::DtlsTransport::doRecv
```

このバックトレースは報告者が採取したもので、 中間フレームは省略されている。 `Track::incoming` に至る実際の経路は下記 2 のとおり `DtlsSrtpTransport` の demux と `PeerConnection::forwardMedia` / `dispatchMedia` を経由する。

このリポジトリと libdatachannel v0.24.0 で確認した内容:

1. 送信系 binding は GIL を保持したまま呼ばれる (`src/bind_libdatachannel.cpp`)。 いずれも `nb::call_guard<nb::gil_scoped_release>()` が付与されていない。
   - `bind_channel` の `Channel.send` (message_variant 版 / (data, size) 版)
   - `bind_datachannel` の `DataChannel.send` (message_variant 版 / (data, size) 版)
   - `bind_track` の `Track.send` と `Track.send_frame` (いずれも両版)
   - `bind_websocket` の `WebSocket.send` (両版)
2. 受信経路は内部ロックを保持したまま Python callback に到達する。 本リポジトリは MbedTLS でビルドするため、 `DtlsTransport::doRecv` は `mRecvMutex` を保持し、 その内側で `mSslMutex` を保持したまま `mbedtls_ssl_read` を呼ぶ。 `mbedtls_ssl_read` は入力を `DtlsTransport::ReadCallback` から得るため、 `ReadCallback` → `DtlsSrtpTransport::demuxMessage` → `DtlsSrtpTransport::recvMedia` → `mSrtpRecvCallback` (= `PeerConnection::forwardMedia`) → `PeerConnection::dispatchMedia` → `Track::incoming` → `MediaHandler::incomingChain` → `PliHandler::incoming` と進む。 `PliHandler` の callback は nanobind の `std::function` wrapper 経由で Python を呼ぶため GIL が必要になる。 worker thread はこの 2 つのロック (`mRecvMutex` / `mSslMutex`) を保持したまま GIL を待つ。
3. 送信経路は `SctpTransport::send` の `mSendMutex`、 `SctpTransport::handleWrite` の `mWriteMutex`、 `DtlsTransport::send` の `mSslMutex` を取得する。 GIL はプロセス全体で 1 つしかないため、 main thread が GIL を保持したまま送信経路で内部ロックを待つ限り、 worker thread は callback を完了できず、 保持しているロックを解放できない。 なお `DataChannel.send` は `SctpTransport::handleWrite` → `DtlsTransport::send` で worker thread が保持する `mSslMutex` を取得するため、 この経路が循環待ちになる。

つまり「main thread が GIL を握ったまま送信経路のロックを待つ」 「worker thread が受信経路のロックを握ったまま GIL を待つ」 というロック順序逆転であり、 送信系 binding が GIL を解放しないことが循環待ちの片側を作っている。 送信系 binding が GIL を解放すれば、 main thread がロック待ちに入っても worker thread は callback を完了してロックを解放できる。

## 設計方針

- 送信系 binding に `nb::call_guard<nb::gil_scoped_release>()` を付与する。 `PeerConnection` の `close` / `__del__` binding と同じ方式と揃える。 `WebSocket.close()` / `WebSocketServer.stop()` / `IceUdpMuxListener.stop()` には本 issue では付与しない ([[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] の担当)。
  - `bind_channel` の `Channel.send`、 `bind_datachannel` の `DataChannel.send`、 `bind_track` の `Track.send` と `Track.send_frame`、 `bind_websocket` の `WebSocket.send`
  - `(data, size)` 版が残る場合はそちらにも付与する ([[0006-bug-fix-send-size-out-of-bounds-read]] の判断に依存)
- GIL 解放後も、 送信経路から同期的に呼ばれる Python callback は安全である。 nanobind の `include/nanobind/stl/function.h` の `pyfunc_wrapper_t::operator()` が `gil_scoped_acquire` してから Python callable を呼ぶため、 呼び出し元が GIL を解放していても問題ない。 実際に `DataChannel.send` は `SctpTransport::updateBufferedAmount` → `PeerConnection::forwardBufferedAmount` → `impl::Channel::triggerBufferedAmount` 経由で `on_buffered_amount_low` を同期的に呼び得る。
- 引数の変換 (bytes / str → `std::vector<std::byte>` / `message_variant`) は `call_guard` の適用前に完了するため、 GIL 解放中に Python オブジェクトへ触れる箇所はない。
- Free Threading ビルドでは GIL が存在しないため `nb::gil_scoped_release` が解放する対象は無く、 送信経路の呼び出し結果には影響しない。 なお nanobind の実装 (`include/nanobind/nb_misc.h`) は Free Threading ビルドでも no-op にはならず、 `PyEval_SaveThread()` / `PyEval_RestoreThread()` で thread state を detach / attach する (同じ call_guard は `PeerConnection.close()` / `PeerConnection.__del__` で既に使われている)。
- 本 issue のスコープは報告された送信系 binding に限定する。 同じ構造の問題は他のブロッキング API (`set_local_description` 等) にもあり得るが、 [[0005-bug-fix-destructor-callback-deadlock]] と同様に別途判断する。

## 完了条件

- `Channel.send` / `DataChannel.send` / `Track.send` / `Track.send_frame` / `WebSocket.send` が GIL を解放して実行されること
- 映像を送る側の Track の media handler chain に `PliHandler` の `on_pli` を登録し、 受信側から PLI を発生させる 2 つの `PeerConnection` のループバック構成で、 main thread が映像トラックと同じ `PeerConnection` の `DataChannel.send()` を連続実行している間も worker thread の callback が実行されることを `threading.Event` で検証するテストを `tests/test_peerconnection.py` に追加し、 PASS すること。 恒久デッドロックはタイミング依存で発生するため、 このテストは callback 経路の regression 検証であり、 修正前のコードでも必ず失敗するとは限らない
  - 既存のループバックテストと同じく `@pytest.mark.timeout(...)` をテストに個別指定する。 ただしデッドロックが再発した場合は main thread が GIL を保持したまま native lock で停止するため pytest-timeout は発火せず、 テストは CI の job timeout まで停止する (GIL に依存しない watchdog などの対策は [[0023-test-set-pytest-default-timeout]] と併せて判断する)
- 既存テスト全件が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `bind_channel` / `bind_datachannel` / `bind_track` / `bind_websocket` の `send` (2 オーバーロード) と `send_frame` の計 10 箇所に `nb::call_guard<nb::gil_scoped_release>()` を付与し、 送信経路のロック待ちの間 GIL を解放するようにした
- `bind_channel` の直前に GIL を解放する理由をコメントで明記し、 他の 3 セクションからも参照できるようにした (送信経路が各トランスポートの内部ロックを取得すること、 GIL 解放中も同期 callback は安全であること、 同一 Track への並行 send と送信中の `close()` は呼び出し側で直列化すること)
- 映像トラックと DataChannel を同時に使うループバック構成で、 `DataChannel.send()` を連続実行している間に `PliHandler` の callback が実行されることを検証する regression テストを `tests/test_peerconnection.py` に追加した。 callback が送信の外側で実行されても検知できないため、 送信中フラグで判定する方式にした
- `CHANGES.md` の `## develop` に `[FIX]` エントリを追加した
- なお `tests/test_peerconnection.py::test_destruct_without_explicit_close` は [[0005-bug-fix-destructor-callback-deadlock]] が扱う既知の恒停を持つため、 完了条件の「既存テスト全件が PASS」からは除外して判定した

## 参考

- 報告元: commaai/teleoprtc (openpilot の webrtcd) の利用者
- 関連 issue: [[0005-bug-fix-destructor-callback-deadlock]] (destructor 経路の根本対応) / [[0001-bug-fix-peer-connection-destructor-gil-release]] (`Channel.close()` にも同種の問題があり得ると明記) / [[0006-bug-fix-send-size-out-of-bounds-read]] (send の `(data, size)` オーバーロード) / [[0024-refactor-deduplicate-peerconnection-tests]] (追加テストのループバックセットアップの共通化)
- 対象シンボル: `bind_channel` / `bind_datachannel` / `bind_track` / `bind_websocket` / `close_peer_connection` (`src/bind_libdatachannel.cpp`)、 `examples/whip.py` の `PliHandler`
- libdatachannel v0.24.0: `DtlsTransport::doRecv` / `DtlsTransport::send` / `DtlsTransport::ReadCallback` (`src/impl/dtlstransport.cpp`)、 `DtlsSrtpTransport::demuxMessage` / `DtlsSrtpTransport::recvMedia` (`src/impl/dtlssrtptransport.cpp`)、 `SctpTransport::doRecv` / `SctpTransport::send` / `SctpTransport::handleWrite` (`src/impl/sctptransport.cpp`)、 `Track::incoming` (`src/impl/track.cpp`)、 `Channel::triggerBufferedAmount` (`src/impl/channel.cpp`)、 `PeerConnection::forwardMedia` / `PeerConnection::dispatchMedia` / `PeerConnection::forwardBufferedAmount` (`src/impl/peerconnection.cpp`)
- nanobind (2.13.0 / 3.0.1 / 3.1.0 で確認): `include/nanobind/stl/function.h` (`pyfunc_wrapper_t::operator()` が `gil_scoped_acquire` してから Python callable を呼ぶ)、 `include/nanobind/nb_misc.h` (`gil_scoped_release` は Free Threading ビルドでも `PyEval_SaveThread()` / `PyEval_RestoreThread()` を呼ぶ)
