# CI で lint と typecheck (prek のフック) が実行されていない

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-run-prek-hooks-in-ci
- Polished: {YYYY-MM-DD}

## 目的

[[0027-fmt-fix-lint-typecheck-gate]] で整備した品質ゲート (ruff / ty) は prek の git フックと Makefile のターゲットにしか無く、 CI では一度も実行されない。 CI で prek のフックを実行し、 ローカルと同じ検証を PR で強制する。

## 優先度根拠

- git フックをスキップした変更や、 フック環境を用意していない環境からの push は無検証で merge され得る
- shiguredo-python は CI で `j178/prek-action` を使い、 ローカルと同じ prek.toml のフックをすべて実行することを規定している
- CI に pytest も無い問題は [[0022-test-enable-ci-tests]] が扱っており、 本 issue は lint / typecheck のゲートを対象にする

## 現状

- `.github/workflows/wheel.yml` は `uv sync` と wheel ビルドのみで、 pytest はコメントアウトされている。 ruff / ty を実行する step は無い
- `.github/workflows/build_debug.yml` は workflow_dispatch 専用で、 本リポジトリに存在しないファイルに依存している
- そのため `make lint` / `make typecheck` はローカルでしか実行されず、 CI では検出されない

## 設計方針

- `j178/prek-action` を使うワークフローを追加し、 PR と既定ブランチへの push で prek のフックを実行する
- `astral-sh/setup-uv` を先に用意する (ローカルのフックが `uv` を呼ぶため)
- 実行するフックは prek.toml の定義をそのまま使う (ruff-format / ruff-check / ty / clang-format / tombi / builtin)。 ruff / ty を CI で個別にインストールしない
- prek.toml に pytest のフックは無いため、 CI でのテスト実行は対象外とする ([[0022-test-enable-ci-tests]])
- フック環境の初回準備と Python のダウンロードを見込んで `timeout-minutes` に余裕を持たせる
- 必須ステータスチェック化 (branch protection) はリポジトリ設定のため、 本 issue の対象外とする

## 完了条件

- PR と既定ブランチへの push で CI が prek のフックを実行し、 違反がある場合はワークフローが失敗すること
- ローカルと同じ prek.toml を使っていること
- 既存の wheel.yml / build_debug.yml の挙動を変えないこと
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: .github/workflows/ (新規ワークフロー)、 prek.toml
- 関連 issue: [[0027-fmt-fix-lint-typecheck-gate]]、 [[0022-test-enable-ci-tests]]
