# Channel の binding 経由で virtual メソッドを呼ぶと落ちる、 または誤った関数が実行される

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-channel-binding-virtual-dispatch
- Polished: {YYYY-MM-DD}

## 目的

`DataChannel` / `Track` / `WebSocket` は `Channel` を 2 番目の基底として継承しており、 `bind_channel` で binding した virtual メソッドを `Channel.is_closed(dc)` のように未バインドで呼ぶと、 基底オフセットが加算されないため誤った vtable スロットが読まれる。 SIGSEGV する経路と、 静かに誤った関数を実行する経路を解消する。

## 優先度根拠

- `Channel.is_closed(dc)` は SIGSEGV (exit code 139)、 `Channel.max_message_size(dc)` は SIGBUS (exit code 138) でプロセスが落ちる
- `Channel.is_open(dc)` は落ちないが `rtc::DataChannel::close()` を実行する (開いているチャネルを閉じてしまう静かな誤動作)
- いずれも Python から 1 行で到達できる公開 API

## 現状

再現手順:

```python
from libdatachannel import Channel, PeerConnection

pc = PeerConnection()
dc = pc.create_data_channel("x")
Channel.is_closed(dc)  # SIGSEGV
Channel.buffered_amount(dc)  # SIGSEGV
Channel.max_message_size(dc)  # SIGBUS
Channel.is_open(dc)  # 落ちないが DataChannel::close() を実行する
```

- 原因 (lldb とオブジェクトレイアウトの実測): `class DataChannel final : private CheshireCat<impl::DataChannel>, public Channel` のように `Channel` は 2 番目の基底で、 オブジェクト先頭から 24 バイトの位置にある。 `Channel` 経由の binding 呼び出しでは基底オフセットが加算されないため、 virtual 呼び出しがオブジェクト先頭の vtable を誤って参照する。 非 virtual の `Channel.available_amount(dc)` は impl ポインタがたまたま互換なため正常に動作する
- lldb での実測: `Channel.max_message_size(dc)` が実行したのは `rtc::DataChannel::send(std::byte const*, unsigned long)`、 `Channel.is_closed(dc)` は `rtc::DataChannel::send(std::variant<...>)`、 `Channel.buffered_amount(dc)` はどの rtc 関数にも入らず不正アドレスへジャンプする
- [[0034-bug-fix-buffered-amount-segv]] は `buffered_amount` を派生クラス側にも binding して通常の呼び出しを直す。 本 issue はその適用後に `bind_channel` 側の virtual メソッド binding を扱う (実施順は 0034 → 本 issue。 0034 が `bind_channel` 側を残す判断を本 issue が上書きする)

## 設計方針

- virtual メソッドの binding を派生クラス側 (`nb::class_<DataChannel, Channel>` など) に付け直し、 `bind_channel` から削除する。 nanobind は登録先クラスのポインタで第一引数を受け取るため、 派生クラスに登録すれば基底オフセット (24 バイト) が C++ の暗黙変換で加算される
- ラムダ化 (`[](Channel& self) { ... }`) と `nb::cast<Channel&>(self)` は無効である。 nanobind の型変換は継承を判定するが基底オフセットを加算せず同じポインタを返すためで、 最小再現と lldb の実測で確認済み
- `bind_channel` から virtual メソッドを削除すると型スタブの `Channel` クラスからもメソッドが消え、 `Channel` 型でアノテートした変数からの呼び出しが型検査で落ちる。 この影響を評価し、 削除する範囲を決める
- 削除後も派生 3 クラス (DataChannel / Track / WebSocket) の binding が各メソッドを提供し続けることを確認する
- `bind_channel` の virtual メソッド (close / send 2 種 / is_open / is_closed / max_message_size / buffered_amount) を全数確認する
- `Channel` は Python から生成できないため、 テストは派生クラスのインスタンスを `Channel` 経由で呼ぶ形にする

## 完了条件

- `bind_channel` の virtual メソッドを未バインドで呼んでも落ちないこと (`Channel.is_closed(dc)` は SIGSEGV、 `Channel.buffered_amount(dc)` は SIGSEGV、 `Channel.max_message_size(dc)` は SIGBUS になる)
- 落ちない経路でも誤った関数が実行されないこと (実測: `Channel.is_open(dc)` は `DataChannel::close()`、 `Channel.close(dc)` は `DataChannel::isOpen()`、 `Channel.send(dc, b"ab")` は `DataChannel::isClosed()`、 `Channel.send(dc, b"ab", 2)` は `DataChannel::maxMessageSize()` を実行していた)
- `bind_channel` の virtual メソッドを対象にしたテストが `tests/` に追加されていること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_channel` の virtual メソッド binding (src/bind_libdatachannel.cpp)、 `Channel::isClosed` / `Channel::maxMessageSize` / `Channel::bufferedAmount` (libdatachannel)
- 関連 issue: [[0034-bug-fix-buffered-amount-segv]]
