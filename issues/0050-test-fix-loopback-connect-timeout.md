# ループバック接続の確立が 20 秒を超えてテストが断続的に失敗する

- Priority: Medium
- Created: 2026-10-10
- Completed: 2026-10-10
- Branch: develop (直接コミット)
- Polished: 2026-10-10

## 目的

`tests/test_peerconnection.py` のループバックテストが、 負荷の高い CI ランナーで接続確立 (ICE / DTLS のハンドシェイク) に 20 秒以上かかり、 断続的に失敗する。 待ち時間を延ばし、 失敗時に接続状態を出して原因を追えるようにする。

## 優先度根拠

- 実測: develop の wheel run (2026-10-10、 run 38029029277) で `test_request_keyframe_releases_gil_for_incoming_callback` が `AssertionError: 送信側の Track が open しなかった` で失敗した (1 failed, 174 passed)。 同じコードの release ブランチでは同じ leg が成功しており、 環境依存の断続的な失敗
- オファー / アンサーの交換 (gathering 完了・end-of-candidates・アンサー生成) はすべて成功しており、 接続確立だけが 20 秒に収まらなかった

## 現状

- `make_loopback_with_pli` などが `t1_opened.wait(timeout=20)` で接続確立を待つ
- 失敗時のメッセージが「送信側の Track が open しなかった」だけで、 接続状態 (state / ice_state) が分からず原因を追えない

## 設計方針

- 接続確立の待ち時間を 60 秒にする (`_CONNECT_TIMEOUT`)。 ループバックのハンドシェイクが 20 秒を超えるのは異常ではなく、 ランナーの負荷に依存する
- 失敗時のメッセージに `state` と `ice_state` を含め、 次に失敗したときに原因を追えるようにする
- CI のリトライは行わない (issue 0042 の方針を維持する)

## 完了条件

- `tests/test_peerconnection.py` の接続確立の待ちが `_CONNECT_TIMEOUT` (60 秒) になっていること
- 失敗時のメッセージに接続状態が含まれること
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 関連: [[0021-test-fix-flaky-concurrent-datachannel]] (同じく断続的に失敗していたテスト)
- 実測ログ: run 38029029277 の `build_macos (macos-15_arm64, macos-15, 3.13)` leg

## 解決方法

- `tests/test_peerconnection.py`
  - `_CONNECT_TIMEOUT = 60` を追加し、 接続確立 (Track / DataChannel の open) の待ち時間を 20 秒から 60 秒にした
  - `make_loopback_with_pli` の失敗時のメッセージに `state` と `ice_state` を含めた
- 検証
  - 全体 172 passed / 12 skipped / 1 deselected、 `prek run --all-files ty` が PASS
  - `/review-diff-code` の致命的 / 重要指摘が 0 件
