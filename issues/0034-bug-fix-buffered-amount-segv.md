# Channel.buffered_amount() を Python から呼ぶと SIGSEGV する

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-buffered-amount-segv
- Polished: {YYYY-MM-DD}

## 目的

`DataChannel` / `Track` / `WebSocket` の `buffered_amount()` が SIGSEGV し、 送信バッファ量を Python から取得できない。 プロセスが異常終了する経路を解消する。

## 優先度根拠

- 公開 API の呼び出しだけでプロセスが復帰不能に落ちる
- `on_buffered_amount_low` / `set_buffered_amount_low_threshold` と組み合わせた送信フロー制御で必要になる API であり、 [[0026-test-add-missing-binding-tests]] が未テストの binding として挙げている

## 現状

- `DataChannel` / `Track` / `WebSocket` のいずれでも `buffered_amount()` を呼ぶと SIGSEGV する (exit code 139)。 同じオブジェクトの `available_amount()` / `max_message_size()` / `is_open()` / `label()` は正常に動作する
- binding は `bind_channel` の `.def("buffered_amount", &Channel::bufferedAmount)`
- `Channel::bufferedAmount()` (`_deps/libdatachannel/v0.24.0/source/include/rtc/channel.hpp`) は virtual だが、 `DataChannel` / `Track` / `WebSocket` のいずれも override していない。 正常に動作する `isOpen` / `isClosed` / `maxMessageSize` はいずれも派生クラスで override されている
- `Channel` は `private CheshireCat<impl::Channel>` を private 継承しており、 `Channel::bufferedAmount()` は `impl()->bufferedAmount` を返す (`source/src/channel.cpp`)

再現手順:

```python
from libdatachannel import PeerConnection

pc = PeerConnection()
dc = pc.create_data_channel("x")
dc.available_amount()  # 0 (正常)
dc.buffered_amount()  # SIGSEGV
```

## 設計方針

- 原因を特定する (nanobind の virtual メンバ関数ポインタの扱いと、 private 継承経由の `impl()` 呼び出しのどちらが原因か)
- 修正方法の候補: binding を virtual メンバ関数ポインタではなくラムダ (`[](Channel& self) { return self.bufferedAmount(); }`) にする、 派生クラスごとに binding する、 など
- 修正後は 3 クラスすべてで `buffered_amount()` が 0 以上の値を返すことをテストで確認する

## 完了条件

- `DataChannel` / `Track` / `WebSocket` で `buffered_amount()` が SIGSEGV せず、 0 以上の値を返すこと
- 3 クラスを対象にしたテストが `tests/` に追加されていること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_channel` の `buffered_amount` (src/bind_libdatachannel.cpp)、 `Channel::bufferedAmount` (libdatachannel)
- 関連 issue: [[0026-test-add-missing-binding-tests]]
