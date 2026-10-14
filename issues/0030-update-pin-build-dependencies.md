# build-system 依存のバージョン指定に上限がない

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/update-pin-build-dependencies
- Polished: {YYYY-MM-DD}

## 目的

pyproject.toml の build-system.requires (nanobind>=2.13.0、scikit-build-core>=1.0.3) と [tool.scikit-build.cmake] version (>=4.3) が上限なしの指定のため、isolated build env で解決される依存の破壊的変更を直接吸い込む。依存ライブラリのバージョン指定はマイナーバージョンまでとするプロジェクト規約に合わせて上限を切る。

## 優先度根拠

- build-system.requires は uv.lock の管理外 (isolated build env) で解決されるため、破壊的変更の影響をビルド時に受ける
- ruff 0.14 → 0.16 でデフォルト規約セットが変化して lint が失敗した事例 ([[0027-fmt-fix-lint-typecheck-gate]]) のように、ツール更新による挙動変化のリスクは現実的

## 現状

- pyproject.toml の [build-system] requires は nanobind>=2.13.0 と scikit-build-core>=1.0.3
- [tool.scikit-build.cmake] version は >=4.3
- いずれもメジャー境界なし

## 設計方針

- nanobind>=2.13,<3.0、scikit-build-core>=1.0,<2.0、cmake>=4.3,<5 のようにメジャー (またはマイナー) で境界を切る
- 境界の値は各プロジェクトのメジャーバージョン互換方針を確認して決める
- CHANGES.md への記載は [UPDATE] (リリースノート的に影響がなければ省略) で判断する

## 完了条件

- 上限付きの指定になっていること
- `make wheel` が成功すること
- uv.lock とビルドの整合が保たれていること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: pyproject.toml ([build-system]、[tool.scikit-build.cmake])
- 関連 issue: [[0027-fmt-fix-lint-typecheck-gate]]
