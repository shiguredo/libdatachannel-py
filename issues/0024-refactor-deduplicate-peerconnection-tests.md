# test_track と test_destruct_without_explicit_close が約 100 行重複している

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/refactor-deduplicate-peerconnection-tests
- Polished: {YYYY-MM-DD}

## 目的

tests/test_peerconnection.py の 2 つのテストが PeerConnection ペアの構築、相互 SDP / candidate 交換のコールバック 8 関数、track 構築、接続待機ループをほぼ丸ごと複製している。fixture とヘルパーに切り出して重複を解消する。

## 優先度根拠

- 重複により、修正 (例: callback 内 print の削除) が 2 箇所に適用されるため漏れやすい
- 接続待機の assert にメッセージがなく、失敗時に原因が分からない問題も重複先で直る

## 現状

- test_track と test_destruct_without_explicit_close の以下が重複する
  - pc1 / pc2 の構築 (Configuration と port_range 設定)
  - pc1_on_local_description 等 8 つのコールバック
  - pc2_on_track の track 取得
  - Description.Video 構築 (codec / bitrate / ssrc / SDP ラウンドトリップ assert)
  - 接続待機ループ
- 待機失敗時の assert にメッセージがなく、失敗時に「接続が確立しなかったのかトラックが開かなかったのか」が分からない

## 設計方針

- 相互接続済み PeerConnection ペアを返す fixture を作る
- _make_video_media(mid, ssrc) と接続待機ヘルパー (_wait_track_open 等) を切り出す
- 各テストは目的 (renegotiation + close の検証 / recwarn + weakref 検証) に特化した形にする
- 待機失敗時の assert に日本語メッセージを付ける
- [[0026-test-add-missing-binding-tests]] で追加する Track 系テストからもヘルパーを再利用する

## 完了条件

- 重複が解消されていること
- 既存 2 テストと同等の検証内容が保たれていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `test_track`、`test_destruct_without_explicit_close` (tests/test_peerconnection.py)
- 関連 issue: [[0025-test-remove-callback-prints]]、[[0026-test-add-missing-binding-tests]]
