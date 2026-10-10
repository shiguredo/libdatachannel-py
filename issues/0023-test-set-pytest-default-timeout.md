# pytest の timeout 設定が全体で定められていない

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/refactor-set-pytest-default-timeout
- Polished: {YYYY-MM-DD}

## 目的

pytest-timeout は依存に含まれるが pyproject.toml に全体 timeout がなく、@pytest.mark.timeout は 1 テストのみ。接続待ちテストが hang した場合、CI 全体が runner のタイムアウトまで止まる。テスト全体のタイムアウト方針を定める。

## 優先度根拠

- issues/pending/0005 に示された hang 経路 (callback 内 I/O block) は将来的にテストで踏み得る。全体 timeout がないと hang がテスト失敗としてすら観測できない
- CI テスト実行 ([[0022-test-enable-ci-tests]]) と組み合わせたとき、hang は runner タイムアウト (ジョブ全体の停止) に変わってしまう

## 現状

- pyproject.toml の [tool.pytest.ini_options] は testpaths のみ
- @pytest.mark.timeout(120) は test_destruct_without_explicit_close のみ
- test_track は最大 22 秒、test_websocket は 20 秒、test_websocketserver は 15 秒の待機ループを持つ

## 設計方針

- [tool.pytest.ini_options] に全体 timeout を設定する (既存の最長テストに余裕を持たせた値。例: 300 秒)
- 接続待ちテストは個別 marker で必要な値に上書きする方針を明記する
- CI ([[0022-test-enable-ci-tests]]) と組み合わせて hang をテスト失敗として検出できるようにする

## 完了条件

- 全体 timeout が設定されていること
- 個別 marker の運用方針がコメントとして明記されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: pyproject.toml ([tool.pytest.ini_options])
- 関連 issue: [[0022-test-enable-ci-tests]]、[[0005-bug-fix-destructor-callback-deadlock]]
