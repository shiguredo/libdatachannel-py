# whip.py と whep.py に WHIP / WHEP 仕様の応答処理の欠落がある

- Priority: Low
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-whip-whep-spec-compliance
- Polished: {YYYY-MM-DD}

## 目的

examples のシグナリング処理に、仕様上 MUST とされる応答処理の欠落がある。201 Created 応答の Content-Type 検証、Location ヘッダー欠落時の警告、WHEP 406 counter-offer の valid-until の取り扱い、Link ヘッダーの引用符対応。examples の仕様遵守を改善する。

## 優先度根拠

- examples は本ライブラリの参照実装であり、仕様遵守の欠落はそのまま利用者の混乱につながる
- ただし機能不全に直結しないため Low。candidate 伝送 ([[0019-bug-fix-whip-whep-ice-candidate-exchange]]) を優先する

## 現状

- 201 Created 応答の Content-Type が application/sdp かを検証していない (RFC 9725 Section 4.2、draft Section 4.2.1 は MUST)
- whip.py / whep.py ともに Location ヘッダー欠落時の警告がなく、session_url が None のまま進み、disconnect 時に DELETE がスキップされる (RFC 9725 Section 4.2 は 201 応答に Location を MUST で要求)
- whep.py は 406 counter-offer の Content-Type の valid-until パラメータ (draft Section 4.2.2、既定 30 秒) をパースせず、PATCH 前に期限切れが起きうる
- `parse_link_header` (examples/whip.py) は一重引用符形式の `rel='ice-server'` に非対応 (RFC 8288 の Link ヘッダー構文では属性値の引用符は一重も許容される)

## 設計方針

- Content-Type 検証の追加、Location 欠落時の logger.warning 追加、valid-until のパースと期限チェック、Link ヘッダーの引用符対応を追加する
- 実装した処理には docs/ 配下の一次資料の節番号を根拠コメントとして明記する

## 完了条件

- 上記 4 項目が実装されていること
- 該当処理に仕様の節番号コメントがあること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `WHIPClient.connect` / `WHIPClient.disconnect` (examples/whip.py)、`WHEPClient.connect` / `WHEPClient.disconnect` (examples/whep.py)、`parse_link_header` (examples/whip.py)
- docs/rfc9725.txt (Section 4.2、6.1)、docs/draft-ietf-wish-whep-03.txt (Section 4.2.1、4.2.2)
