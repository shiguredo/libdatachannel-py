# Windows 対応表明の 3 重矛盾 (classifiers / README / CMakeLists) を解消する

- Priority: Low
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-windows-platform-metadata
- Polished: {YYYY-MM-DD}

## 目的

pyproject.toml の classifiers は Windows を含み、CMakeLists.txt のプラットフォームチェックのメッセージは「supports ... and Windows」と表明し、README は Windows 未対応 (優先実装として有償対応) と表明している。3 者の矛盾を解消する。

## 優先度根拠

- PyPI の classifiers は配布物の対応プラットフォームの表明であり、誤った表明は利用者の誤解を招く
- Windows 用コード (WIN32 分岐) は CI で一度も検証されていない

## 現状

- pyproject.toml の classifiers に "Operating System :: Microsoft :: Windows" が含まれる
- README.md のプラットフォーム一覧に Windows はなく、優先実装として「Windows 11 対応」を有償で提示している
- CMakeLists.txt は Windows 向け設定 (MSVC_RUNTIME_LIBRARY、/utf-8 /bigobj 等) を持ち、プラットフォームチェックの FATAL_ERROR メッセージに Windows を含む
- wheel.yml は Windows をビルドしない

## 設計方針

- 現在の対応範囲に合わせて、classifiers から Windows を外し、CMakeLists.txt のメッセージから Windows を外す
- WIN32 分岐は「優先実装受注時の準備」であることをコメントで明示する (削除するかは維持コストを考慮して判断する)
- README は現状どおり未対応表明のため変更しない

## 完了条件

- classifiers と CMakeLists.txt の表明が README と一致すること
- WIN32 分岐の位置付けがコメントで明示されていること
- `make wheel` が成功すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: pyproject.toml (classifiers)、CMakeLists.txt (プラットフォームチェック)
