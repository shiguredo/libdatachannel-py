# whip.py の RTP timestamp が 2^32 で wrap せず長時間配信で送信が停止する

- Priority: High
- Created: 2026-08-30
- Completed: 2026-10-10
- Branch: feature/fix-whip-rtp-timestamp-wrap
- Polished: 2026-10-10

## 目的

WHIPClient が RtpPacketizationConfig.timestamp を Python int の加算で更新するため、 累積値が uint32 の範囲を超えた時点で setter が `TypeError` を投げ、 映像・音声の送信が回復不能に停止する。 timestamp は uint32 (`rtppacketizationconfig.hpp` の `uint32_t timestamp;`) であるため、 32 bit の剰余演算で wrap させる。

## 優先度根拠

- start_timestamp が 0 以上 2^32 - 1 の一様乱数のため、video (90 kHz) は平均約 6.6 時間、audio (48 kHz) は平均約 12.4 時間で必ず発火する
- 発火後は毎フレーム例外になり、 送信が再開しない。 利用者から見ると配信が止まったままになる。 なお `handle_error` 自体も structlog の logger に存在しない `logger.isEnabledFor` を呼ぶため `AttributeError` になる (実測。 別 issue で扱う)
- examples は本ライブラリの参照実装であり、同じパターンを利用者がコピーするリスクがある

## 現状

再現手順 (簡略化):

```python
from libdatachannel import RtpPacketizationConfig

config = RtpPacketizationConfig(1, "cname", 111, 48000)
config.timestamp = 0x100000000  # uint32 の範囲外 → TypeError
```

- `WHIPClient._setup_video_encoder` の on_output と `WHIPClient._send_encoded_audio` で `config.timestamp = config.timestamp + elapsed_timestamp` を実行する
- RtpPacketizationConfig.timestamp は uint32 であり、2^32 を超える値の代入は nanobind の範囲チェックで TypeError になる (実測で再現)
- 例外は on_output 内の except Exception で捕まえられ、 `handle_error` を呼ぶ。 ただし `handle_error` は `logger.error` のあとに `logger.isEnabledFor(logging.DEBUG)` を呼び、 structlog の `make_filtering_bound_logger` には `isEnabledFor` が無いため `AttributeError` になる (実測)。 この点は別 issue で扱う。 timestamp は更新されないため、 次フレーム以降も同じ例外を繰り返す
- 差分の int 丸めを毎フレーム独立に行うため、29.97 fps 等の clock rate と整数比でない fps では累積ずれも発生する構造になっている

## 設計方針

- 初期値 `start_timestamp` (乱数) は維持する。 RTP timestamp の初期値は RFC 3550 Section 5.1 で乱数にすることが SHOULD とされており、 libdatachannel の `RtpPacketizationConfig` も同じ趣旨で `startTimestamp` を乱数にしている。 0 から始めると乱数だった現行の挙動からの説明なき逸脱になる
- timestamp の計算に既存の `RtpPacketizationConfig.get_timestamp_from_seconds(seconds, clock_rate)` (静的メソッド。 `uint32_t(int64_t(round(seconds * clock_rate)))` で wrap する) を使い、 その結果に `start_timestamp` を足して 32 bit でマスクする。 自前の変換は作らない
- 初回 dts からの絶対時間方式に変更し、 毎フレームの丸め誤差累積も解消する。 最初のフレームの dts を基準として保持し、 経過秒を渡す (映像は 90000、 音声は 48000)
- timestamp の計算は最初の dts を基準にするため、 0 初期化の `last_video_dts_usec` に依存しなくなる (`last_video_dts_usec` はデバッグログの duration 用に残る)
- timestamp の計算は `examples/rtp_timestamp.py` の純関数 `compute_rtp_timestamp` に切り出し、 フレームごとの基準 dts は `RtpTimestampCalculator` が持つ
- `tests/test_rtp_timestamp.py` から importlib で読み込み、 wrap・丸め・初期値の維持・フレーム列での非累積を検証する (`tests/test_trickle_ice.py` と同じ方式)
- `examples/whip.py` は import 時に uvc / portaudio / webcodecs を要求するため、 example 本体を import するテストは書かない
- whep.py は `frame_info.timestamp` を読むだけで timestamp を加算しないため対象外 (確認済み)

## 完了条件

- `tests/test_rtp_timestamp.py` で、 32 bit の上限を超える入力が例外にならず wrap した値になること、 端数が四捨五入されること、 最初のフレームが初期値 (乱数) になること、 フレーム列を連続で渡しても丸め誤差が累積しないことを検証すること
- 最初のフレームの timestamp が 0 ではなく `start_timestamp` (乱数) になること (映像と音声で `start_timestamp` を足していることをコードで確認する)
- `make develop` のあと `uv run --no-sync python -m pytest tests/ -q --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close` と `prek run --all-files pytest` / `prek run --all-files ty` が PASS すること (deselect は恒停するテストのため必須)
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- wrap しても映像・音声の送信が継続すること (手動確認)。 `examples/whip.py` を `--fake-capture-device` で起動し、 `RtpPacketizationConfig.start_timestamp` を `0xFFFFFF00` 付近に固定して数秒動かしても送信が継続することを確認する

## 解決方法

- `examples/rtp_timestamp.py` (新規)
  - `compute_rtp_timestamp(start_timestamp, first_dts_usec, dts_usec, clock_rate)` で、 最初の dts からの経過時間から RTP timestamp を求める。 `RtpPacketizationConfig.get_timestamp_from_seconds` を使い、 初期値 (乱数) を足して `& 0xFFFFFFFF` で 32 bit に収める
  - フレームごとの基準 dts は `RtpTimestampCalculator` が持つ
- `examples/whip.py`
  - 映像 (`_setup_video_encoder` の `on_output`) と音声 (`_send_encoded_audio`) の timestamp 更新を `RtpTimestampCalculator` に置き換えた。 毎フレームの差分の足し込みをやめ、 丸め誤差の累積をなくした (30 fps では 100 秒あたり数十 ms のずれが累積していた)
  - 使わなくなった `first_video_dts_usec` / `first_audio_dts_usec` / `last_audio_dts_usec` を削除した (`last_video_dts_usec` はデバッグログの duration 用に残る)
- `tests/test_rtp_timestamp.py` (新規)
  - wrap・丸め・初期値の維持 (最初のフレームは初期値のまま)・フレーム列を連続で渡したときの非累積を検証する 12 テスト
- 検証
  - `tests/test_rtp_timestamp.py` 12 passed、 全体 163 passed / 12 skipped / 1 deselected
  - `prek run --all-files pytest` と `prek run --all-files ty` が PASS
  - 手動確認 (WHIP サーバーが要るため未実施): `--fake-capture-device` で `start_timestamp` を `0xFFFFFF00` 付近に固定し、 長時間動かしても送信が継続することを確認する
- 併せて、 `handle_error` が `logger.isEnabledFor` で `AttributeError` になる不具合を別 issue として起票した

## スコープ外 (関連する未解決問題)

- `handle_error` の `AttributeError` (structlog の `make_filtering_bound_logger` に `isEnabledFor` が無い) は別 issue で扱う。 whep.py も同じ関数を共有している

## 参考

- 対象シンボル: `WHIPClient._setup_video_encoder` (映像の on_output)、 `WHIPClient._send_encoded_audio` (音声の送信)、 `WHIPClient._setup_audio_encoder` (音声の config 生成) (examples/whip.py)
- `RtpPacketizationConfig.timestamp` は `uint32_t` (`_deps/libdatachannel/v0.24.0/source/include/rtc/rtppacketizationconfig.hpp`)。 32 bit の剰余演算で wrap させる
