# whip.py と whep.py が ICE candidate を対向に伝送しない

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-whip-whep-ice-candidate-exchange
- Polished: {YYYY-MM-DD}

## 目的

WHIPClient / WHEPClient は disable_auto_gathering=True のまま candidate を含まない offer を POST し、その後 gathering しても得た local candidate を伝送しない (on_local_candidate 未登録、Trickle ICE の PATCH 未実装)。さらに gathering 自体が Link ヘッダーに ICE server がある場合のみ実行されるため、host candidate すら生成されないケースがある。RFC 9725 と draft-ietf-wish-whep-03 の規定に沿って candidate を伝送する。

## 優先度根拠

- RFC 9725 Section 4.3 は、201 Created 応答受信後にバッファした candidate を 1 つの HTTP PATCH で送ることを SHOULD として規定している (docs/rfc9725.txt で確認済み)
- draft-ietf-wish-whep-03 Section 4.4.2 も同様の SHOULD を規定している (docs/draft-ietf-wish-whep-03.txt で確認済み)
- 現状の実装では、Link ヘッダーで ICE server を返さないサーバーとの接続が原理的に成立しにくい

## 現状

- `WHIPClient.connect` (examples/whip.py) と `WHEPClient.connect` (examples/whep.py) は `config.disable_auto_gathering = True` を設定し、set_local_description 後に candidate を含まない offer を POST する
- gathering は `if ice_servers:` のときのみ `gather_local_candidates` を呼ぶ
- `on_local_candidate` を登録しておらず、Trickle ICE 用の PATCH (Content-Type: application/trickle-ice-sdpfrag) も実装していない
- そのため、gathering で得た local candidate が対向に届く経路が存在しない

## 設計方針

- `on_local_candidate` を登録し、candidate をバッファする
- 201 Created 応答後に、application/trickle-ice-sdpfrag の PATCH でバッファした candidate を 1 リクエストで送信する (RFC 9725 Section 4.3、draft Section 4.4.2)
- または gathering 完了を待って candidate を含む offer を POST する方式に変更する (trickle 非対応サーバー向け)。どちらを既定にするかは、対応サーバーの要件を確認して決める
- `if ice_servers:` のガードを外し、ICE server がなくても host candidate を gathering する
- 実装した処理には docs/ 配下の一次資料の節番号を根拠コメントとして明記する

## 完了条件

- local candidate が対向に伝わること (実サーバーまたは同等の検証手順で確認)
- gathering が ICE server の有無に依存しないこと
- WHIP / WHEP ともに該当処理に仕様の節番号コメントがあること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `WHIPClient.connect` (examples/whip.py)、`WHEPClient.connect` (examples/whep.py)
- docs/rfc9725.txt (Section 4.3)、docs/draft-ietf-wish-whep-03.txt (Section 4.4.2)
