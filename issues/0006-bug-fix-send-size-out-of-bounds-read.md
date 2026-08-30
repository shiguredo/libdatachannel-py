# send(data, size) で size が data の長さを超過したときにバッファ外読み込みを行う

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-send-size-out-of-bounds-read
- Polished: {YYYY-MM-DD}

## 目的

Channel / DataChannel / Track / WebSocket の send と Track の send_frame に存在する (data, size) オーバーロードが、size に data の長さを超える値を渡すだけでヒープのバッファ外を読み、その内容をネットワークへ送信し得る。情報漏洩とクラッシュにつながるメモリ安全性の欠陥を解消する。

## 優先度根拠

- 公開 API の 1 行の呼び出し (`send(b"abc", 100)`) で未定義動作に到達する
- 読み出したメモリ内容がそのまま RTP / WebSocket フレームとして対向に送信され得る (情報漏洩)
- 影響する binding が 5 箇所あり、いずれも同じパターン

## 現状

再現手順:

```python
from libdatachannel import PeerConnection

pc = PeerConnection()
dc = pc.create_data_channel("chat")
dc.send(b"abc", 100)  # data の長さ 3 を超える size
```

- `bind_channel` / `bind_datachannel` / `bind_track` / `bind_websocket` 内の (data, size) オーバーロードは size を検証せず `self.send(data.data(), size)` に渡す
- カスタム type_caster (`type_caster<std::vector<std::byte>>`) は bytes を data の長さ分だけ確保した vector にコピーする
- libdatachannel 側は `[data, data + size)` をそのまま読む (`src/datachannel.cpp` の `DataChannel::send(const byte*, size_t)`、`src/track.cpp` の `Track::send` / `Track::sendFrame`、`src/websocket.cpp` の `WebSocket::send`)
- Python 側では size は `len(data)` から導出可能であり、(data, size) オーバーロードには Python から見た存在価値がない
- size に負数を渡した場合は nanobind が OverflowError で弾くため、問題は `size > len(data)` の経路

## 設計方針

- (data, size) オーバーロードを廃止する (推奨)。Python の bytes / str は長さを持つため、message_variant 版だけで足りる
- C API 互換の理由で残す場合は `size != data.size()` のとき `nb::value_error` を投げる
- いずれの場合も、超過した size を渡した呼び出しが例外になることを検証するテストを追加する

## 完了条件

- data の長さを超える size を渡しても未定義動作に到達しないこと
- 検証テストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_channel`、`bind_datachannel`、`bind_track`、`bind_websocket`、`type_caster<std::vector<std::byte>>` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/datachannel.cpp` (`DataChannel::send(const byte*, size_t)`)、`src/track.cpp` (`Track::send` / `Track::sendFrame`)、`src/websocket.cpp` (`WebSocket::send`)
