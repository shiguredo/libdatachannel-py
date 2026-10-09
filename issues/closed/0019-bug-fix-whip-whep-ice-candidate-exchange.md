# whip.py と whep.py が ICE candidate を対向に伝送しない

- Priority: High
- Created: 2026-08-30
- Completed: 2026-10-09
- Branch: feature/fix-whip-whep-ice-candidate-exchange
- Polished: 2026-10-09

## 目的

WHIPClient / WHEPClient は disable_auto_gathering=True のまま candidate を含まない offer を POST し、その後 gathering しても得た local candidate を伝送しない (on_local_candidate 未登録、Trickle ICE の PATCH 未実装)。さらに gathering 自体が Link ヘッダーに ICE server がある場合のみ実行されるため、host candidate すら生成されないケースがある。RFC 9725 と draft-ietf-wish-whep-03 の規定に沿って candidate を伝送する。

## 優先度根拠

- RFC 9725 Section 4.3.2 は、 201 Created 応答を受信するまで gathering した candidate を保持することを MUST、 受信後にバッファした candidate を 1 つの HTTP PATCH で送ることを SHOULD として規定している (docs/rfc9725.txt 479-484 行)
- draft-ietf-wish-whep-03 Section 4.4.2 も同旨を WHEP player について規定している (docs/draft-ietf-wish-whep-03.txt 1033-1039 行)
- 現状の実装では、Link ヘッダーで ICE server を返さないサーバーとの接続が原理的に成立しにくい

## 現状

- `WHIPClient.connect` (examples/whip.py) と `WHEPClient.connect` (examples/whep.py) は `config.disable_auto_gathering = True` を設定し、set_local_description 後に candidate を含まない offer を POST する
- gathering は `if ice_servers:` のときのみ `gather_local_candidates` を呼ぶ
- `on_local_candidate` を登録しておらず、Trickle ICE 用の PATCH (Content-Type: application/trickle-ice-sdpfrag) も実装していない
- そのため、gathering で得た local candidate が対向に届く経路が存在しない

## 設計方針

- 既定は「早期 offer + Trickle ICE の PATCH」とする (RFC 9725 Section 4.3.2 の SHOULD)
  - `on_local_candidate` を登録して candidate をバッファする
  - 201 Created 応答後に、 `application/trickle-ice-sdpfrag` の PATCH でバッファした candidate を 1 リクエストで送る (RFC 9725 Section 4.3.2 / draft Section 4.4.2)
  - `Content-Type` は `application/trickle-ice-sdpfrag`、 201 応答に ETag があれば `If-Match` に使う。 期待応答は 204 No Content
  - trickle ICE は RFC 9725 Section 4.4.5 で OPTIONAL のため、 PATCH を拒否された場合は警告して接続を継続する
- `if ice_servers:` のガードを外し、 ICE server がなくても host candidate を gathering する
- gathering が完了していない場合は `a=end-of-candidates` を付けない (対向の ICE を早期に完了させないため)
- PATCH body の組み立ては `examples/trickle_ice.py` の純関数に切り出し、 `tests/test_trickle_ice.py` から RFC 9725 Figure 3 の PATCH body と突き合わせる (examples は webcodecs / uvc / blend2d を import するため CI のテスト環境では import できず、 純関数だけを検証する)
- ICE restart (RFC 9725 Section 4.3 の entity-tag 更新と再バッファ) は対象外とする
- 実装した処理には docs/ 配下の一次資料の節番号を根拠コメントとして明記する

## 完了条件

- `tests/test_trickle_ice.py` が RFC 9725 Figure 3 の PATCH body と一致する fragment を返すこと (`uv run pytest tests/test_trickle_ice.py` が PASS)
- gathering が ICE server の有無に依存しないこと (コードと、 ICE server 無しでも PATCH する経路で確認)
- WHIP / WHEP ともに該当処理に仕様の節番号コメントがあること (`rg -n "Section 4\.3\.2" examples/whip.py examples/whep.py` がヒットする)
- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- 実サーバー (MediaMTX 等の WHIP / WHEP サーバー) での動作確認は CI 外の手動確認とする (CI にメディアサーバーを立てる仕組みが無く、 HTTP だけを模したサーバーは規約で禁止されているため)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `examples/trickle_ice.py` (新規) に純関数を切り出した
  - `build_sdp_fragment(local_sdp, candidates, end_of_candidates)`: RFC 9725 Section 4.3.2 / RFC 8840 に従う PATCH body を組み立てる (bundle ポリシーでは offerer-tagged の `m=` 行のみ、 ICE 資格情報は media-level 優先)。 `end_of_candidates` は必須引数にし、 candidate の収集が終わっていない場合は `a=end-of-candidates` を付けない
  - `wait_for_ice_gathering(pc, timeout)`: gathering が完了するまで待つ。 candidate が 1 つ届いた時点で送ると後から届く srflx / relay の candidate が送られないため、 完了 (または上限) まで待つ
- `tests/test_trickle_ice.py` (新規): RFC 9725 Figure 3 と同じ形の PATCH body になること、 端のケース (candidate 無し / `end_of_candidates=False` / ICE 資格情報なし / `m=` なし)、 実 `PeerConnection` での gathering 待機を検証する (8 テスト)
- `examples/whip.py` / `examples/whep.py`
  - `on_local_candidate` を登録して candidate をバッファする
  - ICE server の有無にかかわらず gathering する (host candidate も伝える)
  - 201 Created (whep は 406 counter-offer 後も) のあとに gathering の完了を待ち、 `Content-Type: application/trickle-ice-sdpfrag` + (201 に ETag があれば) `If-Match` で 1 つの PATCH を送る (`send_trickle_ice_patch` を whip.py に置いて whep.py から共有)
  - PATCH が 204 以外なら警告して接続を継続する (RFC 9725 Section 4.4.5 で trickle ICE は OPTIONAL)
- `CHANGES.md` の `## develop` に `[UPDATE]` エントリを追加した
- 実測: `tests/test_trickle_ice.py` は 8 passed、 全体で 115 passed / 12 skipped / 1 deselected、 CI は全 leg PASS。 レビューでは `wait_for_ice_gathering` が srflx candidate を 2/2 取りこぼさずに送ることを実測で確認した
- 対象外: ICE restart (RFC 9725 Section 4.3 の entity-tag 更新と再バッファ)。 実サーバー (MediaMTX 等) での動作確認は CI 外の手動確認とする (CI にメディアサーバーを立てる仕組みが無く、 HTTP だけを模したサーバーは規約で禁止されているため)
- 既知の制約: gathering が既に完了している PeerConnection に対して `gather_local_candidates()` を呼んでも candidate は増えないため、 その場合は PATCH を送らない (警告ログのみ)

## 参考

- 対象シンボル: `WHIPClient.connect` (examples/whip.py)、 `WHEPClient.connect` (examples/whep.py)、 `build_sdp_fragment` (examples/trickle_ice.py)、 `tests/test_trickle_ice.py`
- docs/rfc9725.txt (Section 4.3)、docs/draft-ietf-wish-whep-03.txt (Section 4.4.2)
