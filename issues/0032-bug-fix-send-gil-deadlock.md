# DataChannel.send などの送信系 binding が GIL を保持して受信経路の callback とデッドロックする

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-send-gil-deadlock
- Polished: {YYYY-MM-DD}
- Reporter: @Geomglot

## 目的

映像トラック (H.264) とデータチャネルを同時に使う構成で、 asyncio の main thread から `DataChannel.send()` を頻繁に呼んでいると、 受信側のパケットロスが多い条件下で Python プロセスが恒久的にデッドロックする (CPU 使用率はほぼ 0%)。

送信系 binding が GIL を保持したまま libdatachannel の送信経路 (SCTP → DTLS) に入るため、 受信経路の worker thread は Python callback (`PliHandler` の `on_pli` など) の実行に必要な GIL を取得できない。 worker thread は内部ロックを保持したまま GIL を待ち、 main thread は GIL を保持したまま送信経路の内部ロックを待つという循環待ちになり、 復帰不能になる。

`PeerConnection.close()` / `WebSocket.close()` / `WebSocketServer.stop()` / `IceUdpMuxListener.stop()` で導入済みの GIL 解放を送信系 binding にも適用し、 この循環待ちを解消する。

## 優先度根拠

- 復帰不能の恒久デッドロックであり、 プロセス再起動以外の回避手段がない
- 実利用 (commaai/teleoprtc / openpilot の webrtcd) で発生している
- 現状の回避策は「受信経路で Python を呼ばない」 (= `PliHandler` の Python callback を使わない) であり、 `examples/whip.py` が示す標準的な使い方を諦める必要がある
- 修正は既存の `close` / `__del__` binding と同じ `nb::call_guard<nb::gil_scoped_release>()` の適用で完結し、 影響範囲と実装コストが小さい

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

このリポジトリと libdatachannel v0.24.0 で確認した内容:

1. 送信系 binding は GIL を保持したまま呼ばれる (`src/bind_libdatachannel.cpp`)。 いずれも `nb::call_guard<nb::gil_scoped_release>()` が付与されていない。
   - `bind_channel` の `Channel.send` (message_variant 版 / (data, size) 版)
   - `bind_datachannel` の `DataChannel.send` (message_variant 版 / (data, size) 版)
   - `bind_track` の `Track.send` と `Track.send_frame` (いずれも両版)
   - `bind_websocket` の `WebSocket.send` (両版)
2. 受信経路は内部ロックを保持したまま Python callback に到達する。 `DtlsTransport::doRecv` は `mRecvMutex` を保持したまま `recv()` で上位へ渡し、 `SctpTransport::doRecv` は `mRecvMutex` を保持したまま `processData()` → `Track::incoming` → `MediaHandler::incomingChain` → `PliHandler::incoming` と進む。 `PliHandler` の callback は nanobind の `std::function` wrapper 経由で Python を呼ぶため GIL が必要になる。
3. 送信経路は `SctpTransport::send` の `mSendMutex`、 `SctpTransport::handleWrite` の `mWriteMutex`、 `DtlsTransport::send` の `mSslMutex` を取得する。 GIL はプロセス全体で 1 つしかないため、 main thread が GIL を保持したまま送信経路で内部ロックを待つ限り、 worker thread は callback を完了できず、 保持しているロックを解放できない。

つまり「main thread が GIL を握ったまま送信経路のロックを待つ」 「worker thread が受信経路のロックを握ったまま GIL を待つ」 というロック順序逆転であり、 送信系 binding が GIL を解放しないことが循環待ちの片側を作っている。 送信系 binding が GIL を解放すれば、 main thread がロック待ちに入っても worker thread は callback を完了してロックを解放できる。

## 設計方針

- 送信系 binding に `nb::call_guard<nb::gil_scoped_release>()` を付与する。 既存の `close` / `stop` / `__del__` binding と同じ方式と揃える。
  - `bind_channel` の `Channel.send`、 `bind_datachannel` の `DataChannel.send`、 `bind_track` の `Track.send` と `Track.send_frame`、 `bind_websocket` の `WebSocket.send`
  - `(data, size)` 版が残る場合はそちらにも付与する ([[0006-bug-fix-send-size-out-of-bounds-read]] の判断に依存)
- GIL 解放後も、 送信経路から同期的に呼ばれる Python callback は安全である。 nanobind の `include/nanobind/stl/function.h` の `pyfunc_wrapper_t::operator()` が `gil_scoped_acquire` してから Python callable を呼ぶため、 呼び出し元が GIL を解放していても問題ない。 実際に `DataChannel.send` は `SctpTransport::updateBufferedAmount` → `PeerConnection::forwardBufferedAmount` → `impl::Channel::triggerBufferedAmount` 経由で `on_buffered_amount_low` を同期的に呼び得る。
- 引数の変換 (bytes / str → `std::vector<std::byte>` / `message_variant`) は `call_guard` の適用前に完了するため、 GIL 解放中に Python オブジェクトへ触れる箇所はない。
- Free Threading ビルドでは `nb::gil_scoped_release` は no-op になるため、 `FREE_THREADED` 指定時の挙動に影響しない。
- 本 issue のスコープは報告された送信系 binding に限定する。 同じ構造の問題は他のブロッキング API (`set_local_description` 等) にもあり得るが、 [[0005-bug-fix-destructor-callback-deadlock]] と同様に別途判断する。

## 完了条件

- `Channel.send` / `DataChannel.send` / `Track.send` / `Track.send_frame` / `WebSocket.send` が GIL を解放して実行されること
- 受信経路の Python callback (`PliHandler` の `on_pli`) を登録した 2 つの `PeerConnection` のループバック構成で、 main thread が `DataChannel.send()` を連続実行している間も worker thread の callback が実行されることを `threading.Event` と timeout で検証するテストを追加し、 PASS すること
- 既存テスト全件が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 報告元: commaai/teleoprtc (openpilot の webrtcd) の利用者
- 関連 issue: [[0005-bug-fix-destructor-callback-deadlock]] (destructor 経路の根本対応) / [[0001-bug-fix-peer-connection-destructor-gil-release]] (`Channel.close()` にも同種の問題があり得ると明記) / [[0006-bug-fix-send-size-out-of-bounds-read]] (send の `(data, size)` オーバーロード)
- 対象シンボル: `bind_channel` / `bind_datachannel` / `bind_track` / `bind_websocket` / `close_peer_connection` (`src/bind_libdatachannel.cpp`)、 `examples/whip.py` の `PliHandler`
- libdatachannel v0.24.0: `DtlsTransport::doRecv` / `DtlsTransport::send` (`src/impl/dtlstransport.cpp`)、 `SctpTransport::doRecv` / `SctpTransport::send` / `SctpTransport::handleWrite` (`src/impl/sctptransport.cpp`)、 `Track::incoming` (`src/impl/track.cpp`)、 `Channel::triggerBufferedAmount` (`src/impl/channel.cpp`)、 `PeerConnection::forwardBufferedAmount` (`src/impl/peerconnection.cpp`)
- nanobind (3.0.1 / 3.1.0 で確認): `include/nanobind/stl/function.h` (`pyfunc_wrapper_t::operator()` が `gil_scoped_acquire` してから Python callable を呼ぶ)
