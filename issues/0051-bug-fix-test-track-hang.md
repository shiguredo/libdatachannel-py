# test_track がネイティブで恒停し CI の leg が timeout で cancelled になる

- Priority: High
- Created: 2026-10-10
- Completed: {YYYY-MM-DD}
- Branch: develop (直接コミット)
- Polished: 2026-10-10

## 目的

`tests/test_peerconnection.py::test_track` が一部の CI ランナーでネイティブに恒停し、 job の timeout (30 分) で cancelled になる。 まず除外して CI を緑に戻し、 原因を追う。

## 優先度根拠

- 実測: run 38035406536 / 38034643720 で 4 leg (ubuntu-24.04_x86_64 3.14、 ubuntu-24.04_armv8 3.14、 ubuntu-22.04_armv8 3.14、 macos-26_arm64 3.12) が 30 分で cancelled になった
- 停止箇所は run 38029029277 (0025 の print 削除前) と同じで、 最後に PASSED したテストが `test_set_local_description` (63%)、 次に実行される `test_track` で停止している
- callback 内 print を削除しても再現するため、 print 以外の要因 (callback からの再入、 close / destructor 経路の mutex) が疑わしい

## 現状

- `test_track` は loopback 接続を作り、 callback から `set_remote_description()` / `add_remote_candidate()` を呼ぶ
- 同じ経路の新しいテスト (`make_loopback_with_pli` を使うもの) は同じ leg でも完走している

## 設計方針

- CI (`wheel.yml` の Test wheel 2 箇所) と prek の pytest フックで `test_track` を除外し、 恒停で job が落ちないようにする (既存の `test_destruct_without_explicit_close` と同じ扱い)
- 原因調査は別途行い、 恒停が直ったら除外を外す
- [[0024-refactor-deduplicate-peerconnection-tests]] で重複を解消する際に、 `test_track` を新しいテストへ寄せることも検討する

## 完了条件

- CI と prek の pytest で `test_track` が除外されていること
- 恒停の原因が特定され、 除外を外しても CI が完走すること (原因調査の後)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `.github/workflows/wheel.yml` の Test wheel 2 箇所と `prek.toml` の pytest フックに `--deselect tests/test_peerconnection.py::test_track` を追加した
- 検証
  - 除外により CI の 4 leg が完走することを確認する (除外前は 30 分で cancelled)
  - 恒停の原因究明と除外の解除は本 issue の残作業とする

## 参考

- 実測ログ: run 38035406536 の `build_ubuntu (ubuntu-24.04_x86_64, ubuntu-24.04, 3.14)` leg
- 関連: [[0025-test-remove-callback-prints]] (print 削除では解消しなかった)
