# whip.py の RTP timestamp が 2^32 で wrap せず長時間配信で送信が停止する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-whip-rtp-timestamp-wrap
- Polished: {YYYY-MM-DD}

## 目的

WHIPClient が RtpPacketizationConfig.timestamp を Python int の加算で更新するため、累積値が uint32 の範囲を超えた時点で setter が TypeError を投げ、on_output の except で握り潰されて映像・音声の送信が回復不能に停止する。RTP の timestamp wrap は RFC 3550 上の正規動作であり、wrap するように修正する。

## 優先度根拠

- start_timestamp が 0 以上 2^32 - 1 の一様乱数のため、video (90 kHz) は平均約 6.6 時間、audio (48 kHz) は平均約 12.4 時間で必ず発火する
- 発火後は例外が握り潰され、エラー表示なしに配信が停止する (運用での気づきが遅い)
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
- 例外は on_output 内の except Exception で `handle_error` に握り潰され、timestamp が更新されないまま次フレームも同じ例外を繰り返す
- 差分の int 丸めを毎フレーム独立に行うため、29.97 fps 等の clock rate と整数比でない fps では累積ずれも発生する構造になっている

## 設計方針

- timestamp 更新を `(timestamp + elapsed) & 0xFFFFFFFF` で wrap させる
- 併せて初回 dts からの絶対時間方式 (`(dts_usec - first_dts_usec) * clock_rate // 1000000`) に変更し、毎フレームの丸め誤差累積も解消する
- whep.py に同様のパターンがないか確認し、あれば同時に修正する

## 完了条件

- timestamp が wrap しても送信が継続すること
- wrap を含む timestamp 更新を検証するテストまたは検証手順が明示されていること
- 全テスト PASS すること

## 参考

- 対象シンボル: `WHIPClient._setup_video_encoder`、`WHIPClient._setup_audio_encoder`、`WHIPClient._send_encoded_audio` (examples/whip.py)
- RFC 3550 (RTP timestamp は mod 2^32 で wrap)
