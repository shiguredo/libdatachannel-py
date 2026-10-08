# Channel の binding 経由で virtual メソッドを呼ぶと落ちる、 または誤った関数が実行される

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-channel-binding-virtual-dispatch
- Polished: 2026-10-08

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

- `bind_channel` から virtual メソッドの binding を削除する。 対象は `channel.hpp` の virtual に対応する 7 binding (close / send 2 種 / is_open / is_closed / max_message_size / buffered_amount) で確定であり、 実装時に範囲を判断する余地は残さない
- 派生 3 クラス (DataChannel / Track / WebSocket) には 7 メソッドすべての binding が既にある (`buffered_amount` は [[0034-bug-fix-buffered-amount-segv]] で追加済み)。 付け直しの作業は不要であり、 再登録してはならない (同名の binding が二重になる)
- 非 virtual の binding (on_open / on_closed / on_error / on_message 2 種 / on_buffered_amount_low / set_buffered_amount_low_threshold / reset_callbacks / receive / peek / available_amount / on_available) は削除しない。 これらは派生クラス側に binding が無く、 未調整のポインタでも impl の shared_ptr が同一の impl オブジェクトを指すため正しく動作する
- 削除後の未バインド呼び出し (`Channel.is_closed(dc)` など) は AttributeError になる。 SIGSEGV / SIGBUS と、 落ちずに誤った関数を実行する問題がこれで解消する
- ラムダ化 (`[](Channel& self) { ... }`) と `nb::cast<Channel&>(self)` は無効である。 nanobind の型変換は継承を判定するが基底オフセットを加算せず同じポインタを返すためで、 最小再現と lldb の実測で確認済み
- 型スタブは binding から生成されるため `class Channel` から 7 メソッドが消える (派生 3 クラスの宣言には残る)。 リポジトリ内に `Channel` 型でアノテートして呼ぶ箇所は無いため `make typecheck` は落ちない
- テストは `tests/test_channel.py` に追加する。 派生クラスのインスタンス経由で 7 メソッドが従来どおり動作することと、 `Channel` に virtual メソッドの binding が存在しないこと (`hasattr` が偽) を確認する
- `CHANGES.md` の `## develop` に公開 API の変更を含む `[FIX]` エントリを追加する。 あわせて [[0032-bug-fix-send-gil-deadlock]] の `[FIX]` エントリから `Channel.send()` の記載を除く (削除後に存在しなくなるため)
- `bind_channel` 直前の GIL 解放のコメントは、 対象クラスの列挙から `Channel` を除いて `DataChannel` / `Track` / `WebSocket` にする (派生 3 クラスの send から参照されている)

## 完了条件

- `bind_channel` の virtual メソッドの binding が削除され、 close / send 2 種 / is_open / is_closed / max_message_size / buffered_amount のすべてで `hasattr(Channel, <name>)` が偽になること
- 落ちない経路で誤った関数が実行される問題も、 binding が無くなることで解消していること (実測していた誤動作: `Channel.is_open(dc)` は `DataChannel::close()`、 `Channel.close(dc)` は `DataChannel::isOpen()`、 `Channel.send(dc, b"ab")` は `DataChannel::isClosed()`、 `Channel.send(dc, b"ab", 2)` は `DataChannel::maxMessageSize()` を実行していた)
- 削除後も派生 3 クラス (DataChannel / Track / WebSocket) で 7 メソッドが従来どおり動作すること
- `tests/test_channel.py` に上記を検証するテストが追加されていること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_channel` の virtual メソッド binding (src/bind_libdatachannel.cpp)、 `Channel::close` / `Channel::send` / `Channel::isClosed` / `Channel::maxMessageSize` / `Channel::bufferedAmount` (libdatachannel)
- 関連 issue: [[0034-bug-fix-buffered-amount-segv]] (同じ原因で `buffered_amount` を派生クラス側にも binding 済み)、 [[0032-bug-fix-send-gil-deadlock]] (送信系 binding に GIL 解放を付けた issue。 `Channel.send()` の binding 削除に伴い `CHANGES.md` の記載を修正する)、 [[0026-test-add-missing-binding-tests]] (Channel 系 binding のテストを追加する issue。 本 issue の削除後は Channel 側のメソッドが消えるため、 テストは派生クラス経由になる)
