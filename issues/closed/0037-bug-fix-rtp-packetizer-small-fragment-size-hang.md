# RtpPacketizer の max_fragment_size に小さい値を渡すとハングし、 メモリを消費し続ける

- Priority: High
- Created: 2026-10-08
- Completed: 2026-10-10
- Branch: feature/fix-rtp-packetizer-small-fragment-size-hang
- Polished: 2026-10-10

## 目的

H264RtpPacketizer / H265RtpPacketizer / AV1RtpPacketizer の `max_fragment_size` に小さい値を渡すと、 `outgoing()` が無限ループに陥ってメモリを消費し続けるか、 libdatachannel 内部の `std::length_error` / `std::out_of_range` が `ValueError: vector` / `IndexError: vector` として現れる。 Python から数値を 1 つ渡すだけでプロセスが固まり、 KeyboardInterrupt も効かない経路を塞ぐ。

## 優先度根拠

- `max_fragment_size` にハングする値 (例: H264 の 4 バイト NAL に 2) を渡すと `outgoing()` が無限ループし、 RSS が 6.5 GB から 23 GB へ約 6 秒で増加した (実測)。 追試でも 2.0 MB から 7.2 GB へ 2.8 秒で増加した。 OOM でマシンごと落ちうる
- ハング中は GIL を保持したままなのでプロセス全体が固まり、 KeyboardInterrupt も効かない (`outgoing` の binding に `nb::call_guard<nb::gil_scoped_release>()` が付いていない。 実測でハング中は他のスレッドが 1 度も動かないことを確認)
- AV1 に小さい値を渡すと、 ハングではなくヒープ破壊による即時クラッシュになる (6 バイトの SequenceHeader を渡した後で `max_fragment_size` 2 を渡すと exit 138)。 ハングと違って回避の余地が無くプロセスごと落ちる
- 例外になる場合も `ValueError: vector` / `IndexError: vector` という libdatachannel 内部のメッセージで、 原因が分からない
- 既定値は `RtpPacketizer::DefaultMaxFragmentSize` (1220) で、 明示的に 1200 を渡しているのは `tests/test_peerconnection.py` と `examples/whip.py` だけである。 どちらも正常に動く値で、 小さい値を渡すだけで到達する

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

AV1 は `outgoing` を上書きせず、 private の `AV1RtpPacketizer::fragment` から `AV1RtpPacketizer::fragmentObu` (`source/src/av1rtppacketizer.cpp`) に入る。 `binary payload(std::min(size_t(mMaxFragmentSize), remaining + metadataSize))` を作るため、 `max_fragment_size` が 0 のとき `payload.at(0)` が `std::out_of_range` になり Python では `IndexError: vector` になる。 1 のときは `payloadOffset` が `payload.size()` と等しくなって `payloadRemaining` が 0 になり、 `remaining` と `index` が進まないため無限ループになる。

`fragmentObu` は SequenceHeader OBU を `mSequenceHeader` にキャッシュして呼び出しをまたいで保持し、 次の呼び出しで `payload` の先頭に前置する。 このとき `metadataSize` は `2 + SequenceHeader の長さ`、 `payloadOffset` も同じ値になるため、 `max_fragment_size < 2 + SequenceHeader の長さ` では次のどちらかになる (実測)。

- `max_fragment_size` が 1 のとき: `payload.at(1)` が範囲外になり `IndexError: vector`
- `max_fragment_size` が `2 + SequenceHeader の長さ` 未満のとき: SequenceHeader を前置する `memcpy` が `payload` をはみ出し、 続いて `payload.size() - payloadOffset` が `size_t` でアンダーフローして `memcpy` の長さが巨大になる。 いずれにしてもヒープ破壊で、 その後の症状 (ハング / クラッシュ) は実行環境によって変わる

実際、 6 バイトの SequenceHeader をキャッシュさせた後は `max_fragment_size` 1 で `IndexError`、 2〜7 (`2 + SequenceHeader の長さ` 未満の全域) で `memcpy` の長さが巨大になって即時クラッシュする (実測で exit 138。 素の 6 バイト OBU なら 2 は正常)。 表の AV1 の値は **SequenceHeader を渡していない場合** のものである。

H264 / H265 の壊れる条件は `max_fragment_size` と入力サイズの組み合わせで決まる (例: H264 は 4 バイトの NAL で 3 を渡すとハングするが、 1000 バイトの NAL では 3 は正常)。 `size` 1〜4000 と `max_fragment_size` 0〜40 の全組み合わせを上記の式で再計算したところ、 **H264 は `max_fragment_size` 4〜65535、 H265 は 6〜65535 でハングも範囲外アクセスも 1 件も発生しない** (H264 は 3 以下、 H265 は 5 以下で、 入力サイズによっては発生する)。 65536 以上で壊れるのは、 フラグメント長を `uint16_t` に切り詰める処理 (`maxFragmentSize = uint16_t(int(ceil(size() / fragments_count)))`) があるためで、 実測でも `max_fragment_size` 65536 と 131071 バイトの NAL で `ValueError: vector` になる (65535 と 131072 バイトの NAL は正常)。 つまり「下限を検証しても防げない」のではなく、 「ヘッダサイズの 2 倍 (H264 は 4、 H265 は 6) 未満の下限では防げない」 のが正確である。

さらに AV1 は事前に渡した SequenceHeader の長さにも依存するため、 `max_fragment_size` と入力サイズだけでは壊れる条件を決められない。 libdatachannel v0.24.0 では未修正で、 upstream の master でも同じ算術である。

### binding 側

- `bind_rtp_packetizer` 系は `"max_fragment_size"_a = RtpPacketizer::DefaultMaxFragmentSize` をそのまま受け取り、 下限を検証していない
- `outgoing` の binding には `nb::call_guard<nb::gil_scoped_release>()` が付いていないため、 ハングすると GIL を保持したままプロセス全体が固まる
- Packetizer 系のテストは `tests/test_rtppacketizer.py` に `RtpPacketizer` / `OpusRtpPacketizer` の構築確認しかなく、 `outgoing` を呼ぶテストは無い。 [[0026-test-add-missing-binding-tests]] は未テストの binding を一般的にカバーする issue で、 本 issue は `max_fragment_size` の境界値と壊れる入力を対象にする

## 設計方針

- binding 側で構築時に下限を検証し、 壊れる入力は `nb::value_error` で拒否する。 libdatachannel 本体は Release ビルドで `assert` が消えるため入力の防御が無い (`generateFragments` の `assert(size() > maxFragmentSize)` も同様)
  - `H264RtpPacketizer`: `max_fragment_size` が 4 未満または 65535 超を拒否する。 4〜65535 なら入力サイズによらずハングも範囲外アクセスも起きない (上記の全数確認)
  - `H265RtpPacketizer`: 同じく 6 未満または 65535 超を拒否する
  - `AV1RtpPacketizer`: `max_fragment_size` が 2 未満を拒否する (0 は `payload.at(0)` の範囲外、 1 は `payloadRemaining` が 0 になる)。 上限は無い
- AV1 の SequenceHeader キャッシュ経路 (`max_fragment_size < 2 + SequenceHeader の長さ`) は binding では判定できない。 未知なのはキャッシュ済みの SequenceHeader の長さで、 `AV1RtpPacketizer` は `mSequenceHeader` を公開しておらず、 Python 側から読む手段も無い (クラスが final のため継承もできない)。 判定には libdatachannel と同じ OBU 解析 (TemporalUnit の leb128 走査を含む) を binding に二重実装する必要があり、 保守の負担に見合わないため採らない。 この経路は binding のコメントと issue に記録し、 libdatachannel 側の修正を upstream へ報告することを前提とする (利用者は SequenceHeader より十分大きい `max_fragment_size` を使う)
- `outgoing` の binding に `nb::call_guard<nb::gil_scoped_release>()` を付ける。 検証で防げない経路が残っても、 GIL を保持したまま無限ループに入ってプロセス全体が固まることを避ける
- 例外メッセージは libdatachannel 内部のものではなく、 何が問題かを示す英語のメッセージにし、 期待値と実際の値を含める (例: `max_fragment_size must be at least 4 to fragment an H264 NAL unit, got 2`)
- テストは、 構築時の拒否を 1 プロセスで (`@pytest.mark.timeout(10)` を付けて)、 許可値で恒停しないことを子プロセス + timeout で確認する。 恒停し得るのは構築時に拒否されなかった値だけなので、 子プロセスで確認するのはその範囲になる。 なお `@pytest.mark.timeout(10)` は GIL を保持したままの native ループの中では発火しないため、 恒停の検出は子プロセスの timeout に頼る

## 完了条件

- 範囲外の `max_fragment_size` を構築時に拒否すること。 例外は `ValueError` で、 メッセージはクラス名を前置し、 期待値と実際の値を含むこと (`H264RtpPacketizer: max_fragment_size must be at least 4 to fragment an H264 NAL unit, got 2` / `H264RtpPacketizer: max_fragment_size must be at most 65535 to fragment an H264 NAL unit, got 65536` の形)
- 次の境界値と正常値をテストで固定すること (拒否は構築時の例外なので `@pytest.mark.timeout(10)` を付けて 1 プロセスで確認できる)
  - H264: `max_fragment_size` 1 / 2 / 3 (拒否)、 65536 / 65537 (拒否)、 4 (許可)。 入力は 5 バイト (分割が起きる最小) / 9 バイト (2 * max_fragment_size + 1) / 1000 バイトの NAL、 上限は 65535 と 131070 バイトの NAL
  - H265: 1 / 2 / 3 / 4 / 5 (拒否)、 65536 (拒否)、 6 (許可)。 入力は 7 バイト / 13 バイト / 1000 バイトの NAL、 上限は 65535 と 131070 バイトの NAL
  - AV1: 0 / 1 (拒否) と 2 (許可)。 入力は OBU 6 バイト (SequenceHeader を渡さない前提)
- 許可した値で `outgoing` が恒停せず戻ること。 `outgoing` は引数のメッセージ列を RTP パケットに置き換えるだけで `send` を呼ばず、 引数は Python 側へ書き戻されないため結果を観測できない。 恒停しないことは子プロセスに分離して timeout で確認する
- `outgoing` が GIL を解放すること (他スレッドが動くことを確認する)
- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` が PASS すること
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS すること
- `CHANGES.md` の `## develop` に変更内容が記録されていること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `H264RtpPacketizer` / `H265RtpPacketizer` / `AV1RtpPacketizer` の構築時に `max_fragment_size` の範囲を検証するようにした。 H264 は 4〜65535、 H265 は 6〜65535、 AV1 は 2 以上で、 範囲外は `ValueError` になる (`H264RtpPacketizer: max_fragment_size must be at least 4 to fragment an H264 NAL unit, got 2` の形)
  - 下限はフラグメント計算 (フラグメント数 `ceil(size / max_fragment_size)`、 フラグメント長 `ceil(size / 分割数)` から FU ヘッダを引く) を全数確認して決めた。 下限未満ではフラグメント長が 0 (ハング) やアンダーフロー (範囲外アクセス) になる。 ヘッダ長の 2 倍 (H264 は 4、 H265 は 6) ならば入力サイズによらず起きない
  - 上限は、 フラグメント長を `uint16_t` に切り詰める処理があるための 65535。 65536 以上では切り詰めで長さが 0 や 1 になり、 小さい値と同じ不具合になる (実測で `max_fragment_size` 65536 と 131071 バイトの NAL が `ValueError: vector` になることを確認)
  - `outgoing` (基底 `RtpPacketizer` と映像 3 クラス) を `nb::call_guard<nb::gil_scoped_release>()` で実行するようにした。 検証で防げない経路が残っても、 GIL を保持したまま無限ループに入ってプロセス全体が固まることを避ける
  - AV1 の `max_fragment_size < 2 + SequenceHeader の長さ` の経路は binding からは判定できない (`AV1RtpPacketizer` は `mSequenceHeader` を公開しておらず、 クラスが final のため継承もできない)。 この制約はコードのコメントに残し、 libdatachannel 側の修正を upstream へ報告する前提とする
- `tests/test_rtppacketizer.py`
  - 構築時の範囲外拒否 (H264 1 / 2 / 3 / 65536 / 65537、 H265 1〜5 / 65536、 AV1 0 / 1) を `pytest.raises` で固定した
  - 許可値で `outgoing` が恒停しないことを確認するテストを追加した。 入力は分割の境界 (H264 は 5 / 9 バイト、 H265 は 7 / 13 バイト) と正常系 (1000 バイト)、 上限 (65535 と 131070 バイト) である
  - `outgoing` が GIL を解放して実行されることを、 他 thread の進行で 3 クラスそれぞれ確認するテストを追加した
- `tests/hang_reproduction_packetizer.py` を追加した (恒停し得る `outgoing` を子プロセスで実行して timeout で検出する)
- `CHANGES.md` の `## develop` に `[FIX]` を追記した
- 検証: `tests/test_rtppacketizer.py` 26 passed、 全体 146 passed / 12 skipped / 1 deselected、 `/review-diff-code` 3 周で致命的 0 / 重要 0

## 参考

- 対象シンボル: `bind_av1rtppacketizer` / `bind_h264rtppacketizer` / `bind_h265rtppacketizer` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `source/src/nalunit.cpp` (`NalUnit::generateFragments`)、 `source/src/h265nalunit.cpp` (`H265NalUnit::generateFragments`)、 `source/src/av1rtppacketizer.cpp` (`AV1RtpPacketizer::fragment` / `AV1RtpPacketizer::fragmentObu`)、 `source/include/rtc/av1rtppacketizer.hpp` (`mPacketization` / `mMaxFragmentSize` が private)
- 関連 issue: [[0007-bug-fix-nalunit-empty-size-segv]] (レビュー中に発見。 binding 側の入力検証という同じテーマ)、 [[0026-test-add-missing-binding-tests]] (Packetizer 系のテスト)
