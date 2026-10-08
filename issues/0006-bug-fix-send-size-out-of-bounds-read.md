# send(data, size) で size が data の長さを超過したときにバッファ外読み込みを行う

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/change-remove-send-size-overload
- Polished: 2026-10-08

## 目的

DataChannel / Track / WebSocket の `send` と Track の `send_frame` に存在する (data, size) オーバーロードが、 size に data の長さを超える値を渡すだけでヒープのバッファ外を読み、 その内容をネットワークへ送信する。 情報漏洩とプロセスのクラッシュにつながるメモリ安全性の欠陥を解消する。

## 優先度根拠

- 公開 API の 1 行の呼び出し (`dc.send(b"a", 10**9)`) で SIGBUS によりプロセスが落ちる (実測: exit 138、 例外は出ない)
- 読み出したメモリ内容がそのまま対向に送信される (実測: loopback の DataChannel で `dc1.send(b"A" * 16, 1024)` を呼ぶと受信側は 1024 バイトを受信し、 16 バイト目以降の 1008 バイトは 'A' 以外のヒープ内容だった)
- 影響する binding が 4 箇所あり、 いずれも同じパターン
- (data, size) オーバーロードは Python から見て存在価値がなく (size は `len(data)` から導出でき、 前方部分送信はスライスで等価)、 廃止しても失う機能がない

## 現状

### 再現手順 1: クラッシュ (未接続のまま観測できる)

```python
from libdatachannel import PeerConnection

pc = PeerConnection()
dc = pc.create_data_channel("chat")
dc.send(b"a", 10**9)  # SIGBUS (exit 138)。 例外は出ない
```

実測 (別プロセス + timeout、 exit code は signal を含む):

| 呼び出し | 結果 |
|---|---|
| `dc.send(b"a", 10**9)` | SIGBUS (exit 138) |
| `dc.send(b"a", 10**7)` | SIGBUS になることがある (実測では 10 回中 6 回。 境界はヒープ配置に依存する)。 `2 * 10**6` 以下では落ちず `RuntimeError: DataChannel not open` になり、 `2 * 10**7` 以上は安定して SIGBUS |
| `track.send(b"a", 10**9)` / `track.send_frame(b"a", 10**9, FrameInfo(0))` | SIGBUS |
| `ws.send(b"a", 10**9)` | SIGBUS |
| 同じ長さの有効なバッファを渡した対照 (`dc.send(b"a" * 10**9, 10**9)`) | `RuntimeError: DataChannel not open` で落ちない |

未接続のオブジェクトでも、 例外 (`DataChannel not open` など) が出る前に範囲外の読み出しが起きている。 size を大きくすると例外に到達する前に落ちる。

### 再現手順 2: 情報漏洩 (接続後)

loopback の PeerConnection で DataChannel を確立し、 ペイロード 16 バイトに対して size 1024 を渡すと、 受信側は 1024 バイトを受信し、 16 バイト目以降は 'A' 以外のヒープ内容 (`e02d05cb07000000d02ce00001000000` など) だった。 size 100000 なら 100000 バイトが送信される。 WebSocket も echo サーバに対して同じで、 16 バイト + size 4096 で 4096 バイトがサーバに届く。

### 原因

- `bind_datachannel` / `bind_track` / `bind_websocket` 内の (data, size) オーバーロードは size を検証せず `self.send(data.data(), size)` に渡す (4 箇所。 Channel 側の binding は [[0036-bug-fix-channel-binding-virtual-dispatch]] で削除済みなので対象外。 再登録しないこと)
- カスタム type_caster (`type_caster<std::vector<std::byte>>`) は bytes を data の長さ分だけ確保した vector にコピーする
- libdatachannel 側は state を見る前に `[data, data + size)` をそのまま読む (`src/datachannel.cpp` の `DataChannel::send(const byte*, size_t)` は `Message(data, data + size, Message::Binary)`、 `src/track.cpp` の `Track::send` / `Track::sendFrame` は `binary(data, data + size)`、 `src/websocket.cpp` の `WebSocket::send` は `make_message(data, data + size, Message::Binary)`)
- size に負数を渡した場合は nanobind のオーバーロード解決が失敗して `TypeError` になる (実測。 `OverflowError` ではない) ため、 問題は `size > len(data)` の経路
- size < `len(data)` は前方部分送信として動作する (実測: `send(b"AAAAAAAA", 4)` は 4 バイト送信) が、 type_caster が `len(data)` 全体をコピーするため `data[:size]` + 1 引数版の方が速く、 Python から見た存在価値がない
- (data, size) オーバーロードは 2025.1.0 / 2025.1.2 / 2026.1.0.dev0〜dev2 の各タグに含まれるリリース済みの公開 API であり、 型スタブにも宣言されている

## 設計方針

- (data, size) オーバーロードを廃止する。 Python の bytes / str は長さを持つため 1 引数版だけで足り、 前方部分送信はスライスで等価になる。 廃止により未定義動作の経路そのものが無くなる
  - 対象は `DataChannel.send` / `Track.send` / `Track.send_frame` / `WebSocket.send` の 2 引数版 4 箇所
  - 削除対象は「2 引数版のみ」とし、 1 引数版 (`message_variant` と `binary` + `FrameInfo`) は変更しない
  - Channel 側の binding は [[0036-bug-fix-channel-binding-virtual-dispatch]] で削除済みのため再登録しない
- リリース済み API の削除なので後方互換性はなくなる。 変更履歴には `[CHANGE]` として記録し、 影響 (2 引数で呼んでいたコードは `TypeError` になること、 代替は 1 引数版とスライスであること) を書く
- `tests/test_channel.py` の `test_send_with_size_on_unconnected_objects` は 2 引数版を呼んで `RuntimeError` を期待しているため、 廃止に合わせて書き換える (1 引数版の未接続時の例外は現在どのテストでもカバーされていないため、 1 引数版で書き直して経路を残す)
- 2 引数版が使えなくなったこと (`TypeError` になること) と、 1 引数版で同じ送信ができることをテストで固定する

## 完了条件

- (data, size) オーバーロードが削除され、 data の長さを超える size を渡しても未定義動作 (範囲外読み出し / SIGBUS) に到達しないこと
- `tests/test_channel.py` の 2 引数版を呼ぶテストが削除または 1 引数版に書き換えられ、 全テストが PASS すること
- 2 引数で呼ぶと `TypeError` になり、 1 引数版 (`bytes` / `str`) で送信できることを検証するテストが追加されていること
- 1 引数版で前方部分送信ができること (スライスで等価) がテストで示されていること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `CHANGES.md` の `## develop` に `[CHANGE]` エントリが追加されていること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_datachannel`、 `bind_track`、 `bind_websocket`、 `type_caster<std::vector<std::byte>>` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/datachannel.cpp` (`DataChannel::send(const byte*, size_t)`)、 `src/track.cpp` (`Track::send` / `Track::sendFrame`)、 `src/websocket.cpp` (`WebSocket::send`)
- 関連 issue: [[0036-bug-fix-channel-binding-virtual-dispatch]] (Channel 側の binding 削除。 Channel の 2 引数版は再登録しない)、 [[0007-bug-fix-nalunit-empty-size-segv]] (binding 側の入力検証の前例。 本 issue は廃止するため検証は追加しない)、 [[0026-test-add-missing-binding-tests]] (`Track.send_frame` のテストを計画しており、 廃止に合わせた更新が必要)
