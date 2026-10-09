# nanobind 3 の分割モードを採用してホイールを削減する

- Priority: Low
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/update-nanobind-split-mode
- Polished: {YYYY-MM-DD}

## 目的

nanobind 3 で追加された分割モード (split mode) を採用し、 Python バージョンごとにビルドしているホイールをプラットフォーム単位に削減する。 現在は 1 プラットフォームあたり 4 バージョン (3.12 / 3.13 / 3.14 / 3.14t) のホイールを作っており、 ビルドと配布のコストが大きい。

## 優先度根拠

- 分割モードは frontend が Python 3.10 の stable ABI を対象にするため、 1 プラットフォーム 1 ホイールで 3.10 以降のすべての CPython をカバーできる
- 現在の CI は wheel.yml で 24 leg (ubuntu 4 platform × 4 python + macos 2 platform × 4 python) を回しており、 削減効果が大きい
- nanobind 3 系への更新 ([[0040-update-build-deps]]) は完了しており、 前提は整っている
- ただし Free Threading (3.14t) と両立できないため、 Python 3.15 対応まで保留する (実測結果を参照)

## 実測結果 (2026-10-09)

分割モード自体は動作することを実測で確認した。

- `CMakeLists.txt` の `nanobind_add_module` に `BACKEND_MODULE nanobind_backend` を追加し、 `pyproject.toml` に `wheel.py-api = "cp310"` と `dependencies = ["nanobind-backend>=1.0"]` を追加すると、 `cp310-abi3-macosx_26_0_arm64` の 1 ホイールが生成される (`libdatachannel_ext.abi3.so`)
- そのホイール 1 つを Python 3.12 / 3.13 / 3.14 の各 venv に install して全テストが PASS する (84 passed / 12 skipped / 1 deselected、 各 24 秒前後)

**しかし 3.14t (Free Threading) と両立できないため、 現時点では採用しない。**

- 分割モードの Free Threading 対応は **Python 3.15 以降**の `abi3t` (PEP 803) のみで、 3.14t は stable ABI が無く対象外
- `nanobind-backend` 1.0.0 の PyPI wheel 48 件のうち Free Threading 用は `cp315-cp315t-*` の 7 件のみで、 **3.14t 用の wheel は存在しない** (sdist も無い)
- したがって 3.14t 用ホイールに `dependencies = ["nanobind-backend>=1.0"]` を宣言すると、 3.14t 環境では適合する wheel が無く install に失敗する。 1 つの `pyproject.toml` で「3.12〜3.14 は分割モード (abi3)」「3.14t は従来モード (linked)」の 2 種類のメタデータを作り分ける必要があり、 その複雑さに見合う利点が現時点では無い
- Free Threading を落とす選択は取らない (本パッケージは Free Threading 対応をうたっている)

## 再開の条件

- Python 3.15 (および 3.15t) に対応する時点で、 3.15t が `abi3t` + backend wheel を使えるようになるため、 そのときに再度判断する
- その際は 3.12〜3.14 の leg を 1 ホイールに削減できる (24 wheel → 12 wheel 程度になる見込み)

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

- 恒停問題の根本対応は [[0005-bug-fix-destructor-callback-deadlock]] / [[0039-bug-fix-nanobind-del-not-called]] の範囲とする

## 参考

- nanobind の分割モード: https://nanobind.readthedocs.io/en/latest/split_mode.html
- nanobind 3.0.0 の変更点: https://nanobind.readthedocs.io/en/latest/changelog.html
- 関連 issue: [[0040-update-build-deps]]
