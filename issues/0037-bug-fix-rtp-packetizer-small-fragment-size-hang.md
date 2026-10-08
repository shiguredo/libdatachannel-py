# RtpPacketizer の max_fragment_size に小さい値を渡すとハングし、 メモリを消費し続ける

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-rtp-packetizer-small-fragment-size-hang
- Polished: {YYYY-MM-DD}

## 目的

H264RtpPacketizer / H265RtpPacketizer / AV1RtpPacketizer の `max_fragment_size` に小さい値を渡すと、 `outgoing()` が無限ループに陥ってメモリを消費し続けるか、 libdatachannel 内部の `std::length_error` / `std::out_of_range` が `ValueError: vector` / `IndexError: vector` として現れる。 Python から数値を 1 つ渡すだけでプロセスが固まり、 KeyboardInterrupt も効かない経路を塞ぐ。

## 優先度根拠

- `max_fragment_size` に 1〜4 を渡すと `outgoing()` が無限ループし、 RSS が 6.5 GB から 23 GB へ約 6 秒で増加した (実測)。 OOM でマシンごと落ちうる
- ハング中は GIL を保持したままなのでプロセス全体が固まり、 KeyboardInterrupt も効かない (`outgoing` の binding に `nb::call_guard<nb::gil_scoped_release>()` が付いていない。 実測でハング中は他のスレッドが 1 度も動かないことを確認)
- 例外になる場合も `ValueError: vector` / `IndexError: vector` という libdatachannel 内部のメッセージで、 原因が分からない
- 既定値 (1220) と既存のテスト / サンプル (1200) では発生しないが、 明示的に小さい値を渡すだけで到達する

## 現状

再現手順 (`NalUnit::Separator.Length` のときは先頭 4 バイトが NAL 長として読まれるため、 NAL 4 バイトは長さ 4 バイトを前置した 8 バイトのメッセージで渡す):

```python
from libdatachannel import H264RtpPacketizer, Message, NalUnit, RtpPacketizationConfig

cfg = RtpPacketizationConfig(1234, "cname", 96, H264RtpPacketizer.CLOCK_RATE)
packetizer = H264RtpPacketizer(NalUnit.Separator.Length, cfg, 2)  # 2 でハングする
frame = bytes([0x00, 0x00, 0x00, 0x04, 0x65, 0x00, 0x00, 0x00])  # NAL 長 4 バイト + NAL 4 バイト
message = Message(len(frame))
for i, b in enumerate(frame):
    message[i] = b
packetizer.outgoing([message], lambda out: None)  # 無限ループ (メモリを消費し続ける)
```

実測 (1 件 1 プロセス、 timeout 10 秒。 NAL 4 バイト / H265 NAL 5 バイト / AV1 OBU 6 バイト):

| `max_fragment_size` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 1220 |
|---|---|---|---|---|---|---|---|---|---|---|
| H264 (NAL 4 バイト) | `ValueError` | `ValueError` | ハング | ハング | 正常 | 正常 | 正常 | 正常 | 正常 | 正常 |
| H265 (NAL 5 バイト) | `ValueError` | `ValueError` | `ValueError` | ハング | ハング | 正常 | 正常 | 正常 | 正常 | 正常 |

| `max_fragment_size` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| AV1 (OBU 6 バイト) | `IndexError` | ハング | 正常 | 正常 | 正常 | 正常 | 正常 | 正常 | 正常 |

1000 バイトの NAL でも同じで、 H264 は 0 / 1 で `ValueError`、 2 でハング、 H265 は 0 / 1 / 2 で `ValueError`、 3 でハングする。

### 原因

H264 / H265 の `generateFragments` (`source/src/nalunit.cpp` / `source/src/h265nalunit.cpp`) は、 分割が必要なとき (`size() > max_fragment_size`) に次の計算をする。

- `c = ceil(size() / max_fragment_size)` (フラグメント数。 `max_fragment_size` が 0 なら無限大)
- `m1 = uint16_t(int(ceil(size() / c)))` (`c` が無限大のときは 0)
- `m2 = m1 - ヘッダサイズ` (H264 は 2、 H265 は 3。 size_t のため `m1 < ヘッダサイズ` でアンダーフローする)

`m1 < ヘッダサイズ` のときは範囲外のイテレータ対を作るため、 libc++ の range コンストラクタが `std::length_error` を投げ、 Python では `ValueError: vector` になる。 `m1 == ヘッダサイズ` のときは 0 バイトのフラグメントを追加し続け、 `offset += m2` が進まないため無限ループになる (空のフラグメントを確保し続けるためメモリが増え続ける)。

AV1 の `AV1RtpPacketizer::outgoing` (`source/src/av1rtppacketizer.cpp`) は `payload(std::min(max_fragment_size, size + 1))` を作るため、 `max_fragment_size` が 0 のとき `payload.at(0)` が `std::out_of_range` になり Python では `IndexError: vector` になる。 1 のときは `payloadRemaining` が 0 になり `index` が進まないため無限ループになる。

壊れる条件は `max_fragment_size` だけでなく NAL / OBU のサイズにも依存する (例: H264 は 4 バイトの NAL で 3 を渡すとハングするが、 1000 バイトの NAL では 3 は正常)。 libdatachannel master でも未修正である。

### binding 側

- `bind_rtp_packetizer` 系は `"max_fragment_size"_a = RtpPacketizer::DefaultMaxFragmentSize` をそのまま受け取り、 下限を検証していない
- `outgoing` の binding には `nb::call_guard<nb::gil_scoped_release>()` が付いていないため、 ハングすると GIL を保持したままプロセス全体が固まる
- 既存のテスト (`tests/test_peerconnection.py`) とサンプル (`examples/whip.py`) は 1200 / 1220 しか渡しておらず、 境界値のテストはない ([[0026-test-add-missing-binding-tests]] の対象)

## 設計方針

- 「壊れる条件」は `max_fragment_size` と入力サイズの組み合わせで決まるため、 `max_fragment_size` の下限検証だけでは防げない。 次のどれを取るかを実装時に決める
  - `max_fragment_size` の下限を検証する (`nb::value_error`)。 単独では不十分なため、 防げない組み合わせが残ることを issue に記録する
  - `outgoing` の binding で、 渡されたメッセージのサイズと `max_fragment_size` から壊れる条件 (上記の式) を判定し、 該当する場合は `nb::value_error` を投げる
  - libdatachannel 側の修正を upstream に報告することを前提にし、 binding には当面の回避 (上記のいずれか) を入れる
- どの案でも、 ハングする組み合わせが例外になることをテストで固定する (テスト自体がハングしない形で書く。 例: `max_fragment_size` と入力サイズの組み合わせを引数化し、 例外になることだけを確認する)
- 例外メッセージは libdatachannel 内部のものではなく、 何が問題かを示すものにする

## 完了条件

- `max_fragment_size` に小さい値を渡してもハングせず、 メモリを消費し続けないこと
- 壊れる条件が判定され、 該当する呼び出しが例外になること (または libdatachannel 側の修正を前提とした回避が入っていること)
- H264 / H265 / AV1 の 3 クラスで、 境界の値と正常値をカバーするテストが追加されていること
- 例外になる場合は原因が分かるメッセージになること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_av1rtppacketizer` / `bind_h264rtppacketizer` / `bind_h265rtppacketizer` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `source/src/nalunit.cpp` (`NalUnit::generateFragments`)、 `source/src/h265nalunit.cpp` (`H265NalUnit::generateFragments`)、 `source/src/av1rtppacketizer.cpp` (`AV1RtpPacketizer::outgoing`)
- 関連 issue: [[0007-bug-fix-nalunit-empty-size-segv]] (レビュー中に発見。 binding 側の入力検証という同じテーマ)、 [[0026-test-add-missing-binding-tests]] (Packetizer 系のテスト)
