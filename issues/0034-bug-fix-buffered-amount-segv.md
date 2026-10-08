# Channel.buffered_amount() を Python から呼ぶと SIGSEGV する

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-buffered-amount-segv
- Polished: 2026-10-08

## 目的

`DataChannel` / `Track` / `WebSocket` の `buffered_amount()` が SIGSEGV し、 送信バッファ量を Python から取得できない。 派生クラスのインスタンスから呼ぶこの経路を解消する。

## 優先度根拠

- 公開 API の呼び出しだけでプロセスが復帰不能に落ちる
- `on_buffered_amount_low` / `set_buffered_amount_low_threshold` と組み合わせた送信フロー制御で必要になる API であり、 [[0026-test-add-missing-binding-tests]] が未テストの binding として挙げている

## 現状

- `DataChannel` / `Track` / `WebSocket` のいずれでも `buffered_amount()` を呼ぶと SIGSEGV する (exit code 139)。 同じオブジェクトの `available_amount()` / `max_message_size()` / `is_open()` / `label()` は正常に動作する
- binding は `bind_channel` の `.def("buffered_amount", &Channel::bufferedAmount)`
- `Channel::bufferedAmount()` (`_deps/libdatachannel/v0.24.0/source/include/rtc/channel.hpp`) は virtual だが、 `DataChannel` / `Track` / `WebSocket` のいずれも override していない。 正常に動作する `isOpen` / `isClosed` / `maxMessageSize` はいずれも派生クラスで override されている
- `Channel` は `private CheshireCat<impl::Channel>` を private 継承しており、 `Channel::bufferedAmount()` は `impl()->bufferedAmount` を返す (`source/src/channel.cpp`)
- 追加の実測:
  - `Channel.buffered_amount(dc)` (未バインド呼び出し) も SIGSEGV する
  - `Channel.max_message_size(dc)` は SIGBUS で落ちる。 `Channel::maxMessageSize()` は `return 0;` のみの実装で `impl()` を参照しないため、 原因は `Channel` の binding 経由の virtual 呼び出しそのものにある
  - `Channel.available_amount(dc)` は正常に動作する (非 virtual)
  - `bind_channel` の binding をラムダ (`[](Channel& self) { return self.bufferedAmount(); }`) に変えても SIGSEGV する (つまりラムダ化では直らない)
  - `bind_datachannel` / `bind_track` に `.def("buffered_amount", &Channel::bufferedAmount)` を追加すると、 `dc.buffered_amount()` / `track.buffered_amount()` が正常に値を返す (この修正で直ることを実測で確認済み)

再現手順:

```python
from libdatachannel import PeerConnection

pc = PeerConnection()
dc = pc.create_data_channel("x")
dc.available_amount()  # 0 (正常)
dc.buffered_amount()  # SIGSEGV
```

## 設計方針

- `bind_datachannel` / `bind_track` / `bind_websocket` に `.def("buffered_amount", &Channel::bufferedAmount)` を追加し、 派生クラスの binding から呼ぶようにする (実測で修正を確認済み)。 原因は `Channel` が 2 番目の基底であるため `Channel` 経由の binding 呼び出しで基底オフセットが加算されず、 virtual 呼び出しが誤った vtable スロットを読むことにある
- `bind_channel` の `buffered_amount` は本 issue では変更しない。 削除すると型スタブの `Channel` クラスからもメソッドが消え、 `Channel` 型でアノテートした変数からの呼び出しが型検査で落ちるため、 削除の是非は [[0036-bug-fix-channel-binding-virtual-dispatch]] で他の virtual メソッドと合わせて判断する (本 issue の判断は 0036 が上書きする)。 実施順は本 issue → [[0036-bug-fix-channel-binding-virtual-dispatch]] とする
- `Channel` の binding 経由で virtual メソッドを呼ぶと同種の現象が起きる (`Channel.is_closed(dc)` は SIGSEGV、 `Channel.max_message_size(dc)` は SIGBUS、 `Channel.is_open(dc)` は `DataChannel::close()` を実行する静かな誤動作)。 これも本 issue のスコープ外とし、 同じく [[0036-bug-fix-channel-binding-virtual-dispatch]] で扱う
- 修正後は 3 クラスすべてで `buffered_amount()` が 0 以上の値を返すことをテストで確認する

## 完了条件

- `DataChannel` / `Track` / `WebSocket` で `buffered_amount()` が SIGSEGV せず、 0 以上の値を返すこと
- 3 クラスを対象にしたテストが `tests/` に追加されていること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること
- スコープ外: `bind_channel` の virtual メソッド binding を未バインドで呼ぶ経路 (`Channel.is_closed(dc)` など) は [[0036-bug-fix-channel-binding-virtual-dispatch]] で扱う

## 参考

- 対象シンボル: `bind_channel` の `buffered_amount` (src/bind_libdatachannel.cpp)、 `Channel::bufferedAmount` (libdatachannel)
- 関連 issue: [[0026-test-add-missing-binding-tests]]、 [[0036-bug-fix-channel-binding-virtual-dispatch]]
