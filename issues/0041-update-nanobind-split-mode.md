# nanobind 3 の分割モードを採用してホイールを削減する

- Priority: Medium
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/nanobind-split-mode
- Polished: {YYYY-MM-DD}

## 目的

nanobind 3 で追加された分割モード (split mode) を採用し、 Python バージョンごとにビルドしているホイールをプラットフォーム単位に削減する。 現在は 1 プラットフォームあたり 4 バージョン (3.12 / 3.13 / 3.14 / 3.14t) のホイールを作っており、 ビルドと配布のコストが大きい。

## 優先度根拠

- 分割モードは frontend が Python 3.10 の stable ABI を対象にするため、 1 プラットフォーム 1 ホイールで 3.10 以降のすべての CPython をカバーできる
- 現在の CI は wheel.yml で 24 leg (ubuntu 4 platform × 4 python + macos 2 platform × 4 python) を回しており、 削減効果が大きい
- nanobind 3 系への更新 ([[0040-update-build-deps]]) が完了しており、 前提が整っている

## 現状

- `CMakeLists.txt` の `nanobind_add_module` に `BACKEND_MODULE` を指定していない (従来モード)
- wheel.yml は Python 3.12 / 3.13 / 3.14 / 3.14t の 4 バージョンでビルドし、 free-threading の leg では `sys._is_gil_enabled()` が偽であることを確認している
- `requires-python = ">=3.12"` (分割モードの frontend は 3.10 stable ABI を対象にするが、 対応バージョンの方針は別途判断する)

## 設計方針

- `CMakeLists.txt` の `nanobind_add_module` に `BACKEND_MODULE nanobind_backend` を指定して分割モードを有効にする
- backend は PyPI の `nanobind-backend` ホイールとして配布され、 frontend は backend を実行時に読み込む。 ランタイム依存の追加方法 (wheel の `Requires-Dist` に `nanobind-backend` を入れる) を scikit-build-core の設定で表現する
- free-threading 版は別の backend が必要になるため、 分割モードで free-threading をどう扱うか (別ホイールにする / 従来モードを併用する) を実測して決める
- CI の leg 削減と auditwheel の扱いを更新する
- 分割モードで生成したホイールを複数の Python バージョンで install してテストする

## 完了条件

- 分割モードでビルドしたホイールが 1 プラットフォーム 1 ファイルになり、 対象の複数 Python バージョンで `import libdatachannel` とテストが PASS する
- free-threading 環境での扱いが確定し、 CI がその方針どおりに動いている
- `prek run --all-files pytest` が PASS する
- CI (wheel.yml / prek.yml) が PASS する
- `CHANGES.md` の `## develop` に変更内容が記録されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## スコープ外 (関連する未解決問題)

- `.freeze()` による型の不変化は、 Python 3.15 未満では効果がない (かつ 3.14 では性能が下がるため nanobind が無視する) うえ、 利用者が型を変更できなくなる挙動変更のため、 本 issue では扱わない
- 恒停問題の根本対応は [[0005-bug-fix-destructor-callback-deadlock]] / [[0039-bug-fix-nanobind-del-not-called]] の範囲とする

## 参考

- nanobind 3.0.0 の分割モード: https://nanobind.readthedocs.io/en/latest/changelog.html (Version 3.0.0 の Split mode 節)
- 関連 issue: [[0040-update-build-deps]]
