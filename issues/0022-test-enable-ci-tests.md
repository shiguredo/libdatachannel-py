# CI でテストが一度も実行されていない

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-enable-ci-tests
- Polished: {YYYY-MM-DD}

## 目的

wheel.yml は pytest をコメントアウトしており、テストするはずの build_debug.yml は存在しないファイル (DEPS / run.py / py.typed / libdatachannel_ext.pyi) に依存して手動実行でも確実に失敗する。push・tag・schedule のどれを取っても CI で pytest が動いておらず、リリースタグでは無検証の wheel が PyPI に publish される。

## 優先度根拠

- AGENTS.md は「全てのテストが通らない限りコミットしないこと」を規定するが、CI に検証がなく規律のみに依存している
- リリース物 (PyPI publish) の検証がゼロであることは、配布品質の問題として最優先

## 現状

- wheel.yml の build_ubuntu / build_macos は `# - run: uv run pytest tests/ -v` がコメントアウトされている
- build_debug.yml は `grep '^OPENH264_VERSION=' DEPS`、`uv run python run.py build ...`、`src/libdatachannel/py.typed` と `libdatachannel_ext.pyi` のコピー等、本リポジトリに存在しない構成 (sora-python-sdk 由来) に依存する
- build_debug.yml は workflow_dispatch 専用で自動実行されない

## 設計方針

- build_debug.yml は削除する (sora-python-sdk 由来の残骸であり、本リポジトリの構成に合わせる修正の価値がない)
- wheel.yml に、ビルド済み wheel を fresh な環境に install して pytest を実行する step を追加する。最低でも 3.12 と 3.14t (free-threading) の 2 環境を対象にする
- テスト実行の前提 (aiohttp テストサーバー、ポート競合) を考慮して、Ubuntu runner で実行する形にする
- tests に OpenH264 等の外部依存はないため、追加のセットアップは不要であることを確認済み

## 完了条件

- CI (push または workflow_dispatch) で pytest が実行され PASS すること
- 3.14t 環境で tests/test_free_threading.py が skip されずに実行されること ([[0016-bug-fix-freethreaded-silently-disabled]] のビルド整合とセット)
- build_debug.yml が削除されていること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: .github/workflows/wheel.yml、.github/workflows/build_debug.yml (削除)
- 関連 issue: [[0016-bug-fix-freethreaded-silently-disabled]]
