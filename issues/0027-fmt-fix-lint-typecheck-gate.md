# make lint と make typecheck が失敗したまま develop に放置されている

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-lint-typecheck-gate
- Polished: {YYYY-MM-DD}

## 目的

`make lint` (ruff check examples/) が 69 errors、`make typecheck` (ty check examples/) が 2 diagnostics で失敗しており、さらに prek の ruff v0.14.14 と uv.lock の ruff 0.16.5 で規約セットが乖離している (prek では通るが make lint では失敗する二重基準)。品質ゲートを機能させる。

## 優先度根拠

- lint が失敗したままの develop は、次の変更の lint 結果の判断を不能にする
- ruff 0.16 でデフォルト規約セットが変化したため、pyproject.toml に select を明示しないと将来の ruff 更新で再発する

## 現状

- 実測: `uv run ruff check examples/` → 69 errors (BLE001 28、UP045 27、S110 6、UP006 2、I001 2、UP035 1、SIM114 1、PIE810 1、SIM103 1)。tests / src / dev.py を含めると全体 87 件
- 実測: `uv run ty check examples/` → 2 diagnostics (whip.py の disconnect 内で renderer.ctx / renderer.img に None を代入する箇所)。リポジトリ全体では 9 diagnostics
- prek.toml の ruff-pre-commit は v0.14.14 で、uv.lock の ruff は 0.16.5
- Makefile の lint 対象が examples/ のみ (src / tests / dev.py が対象外) で、prek の ty-check は全体と、範囲が不統一

## 設計方針

- pyproject.toml の [tool.ruff.lint] に select を明示し、規約セットを固定する
- ruff 0.16.5 でエラーを 0 にする (fixable なものは ruff --fix、非 fixable は手動修正。except pass 6 箇所の扱いは [[0028-fmt-fix-language-convention-violations]] と調整する)
- ty の diagnostics を 0 にする
- prek.toml の ruff-pre-commit を uv.lock の ruff に合わせて更新する
- Makefile の lint / typecheck の対象範囲と prek の範囲を一致させる (全体チェックに寄せる)

## 完了条件

- `make lint` と `make typecheck` (範囲を統一した上で) が PASS すること
- prek の ruff と uv.lock の ruff が同期していること
- pyproject.toml に ruff の規約セットが明示されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: pyproject.toml ([tool.ruff])、prek.toml、Makefile、examples/ / tests/ / dev.py の lint 違反
- 関連 issue: [[0028-fmt-fix-language-convention-violations]]
