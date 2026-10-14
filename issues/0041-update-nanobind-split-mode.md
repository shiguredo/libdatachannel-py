# nanobind 3 の分割モードを採用してホイールを削減する

- Priority: Low
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/update-nanobind-split-mode
- Polished: {YYYY-MM-DD}

## 目的

nanobind 3 で追加された分割モード (split mode) を採用し、 Python バージョンごとにビルドしているホイールをプラットフォーム単位に削減する。 現在は 1 プラットフォームあたり 4 バージョン (3.12 / 3.13 / 3.14 / 3.14t) のホイールを作っており、 ビルドと配布のコストが大きい。

## 優先度根拠

- 分割モードは frontend が stable ABI を対象にするため、 1 プラットフォーム 1 ホイールで複数の CPython をカバーできる (実測: 1 ホイールが 3.12 / 3.13 / 3.14 で動作)
- 現在の CI は wheel.yml で 24 leg (ubuntu 4 platform × 4 python + macos 2 platform × 4 python) を回しており、 削減効果が大きい
- nanobind 3 系への更新 ([[0040-update-build-deps]]) は完了しており、 前提は整っている
- ただし Free Threading (3.14t) と両立できないため、 Python 3.15 対応まで保留する (実測結果を参照)

## 実測結果 (2026-10-09)

分割モード自体は動作することを実測で確認した。

- `CMakeLists.txt` の `nanobind_add_module` に `BACKEND_MODULE nanobind_backend` を追加し、 `pyproject.toml` に `wheel.py-api` と `dependencies = ["nanobind-backend>=1.0"]` を追加すると、 abi3 の 1 ホイールが生成される (`libdatachannel_ext.abi3.so`)
- そのホイール 1 つを Python 3.12 / 3.13 / 3.14 の各 venv に install して全テストが PASS する (84 passed / 12 skipped / 1 deselected、 各 24 秒前後)
- 実験では `py-api = "cp310"` を使ったが、 `requires-python = ">=3.12"` と矛盾するため採用時は `py-api = "cp312"` にする。 未設定だと wheel タグがビルドした Python のものになり削減効果が無い

**しかし 3.14t (Free Threading) と両立できないため、 現時点では採用しない。**

- nanobind の cmake は Python 3.15 未満 + `FREE_THREADED` + 分割モードの組み合わせを configure 時に `FATAL_ERROR` で拒否する (「use a linked mode on Python 3.14t」)。 分割モードの Free Threading 対応は **Python 3.15 以降**の `abi3t` (PEP 803) のみで、 3.14t は stable ABI が無く対象外
- `nanobind-backend` 1.0.0 の PyPI wheel 48 件は cp310〜cp315 と **cp315t の 7 件**で、 **cp314t 用の wheel は無い** (sdist も無い)
- したがって 3.14t 用ホイールに `dependencies = ["nanobind-backend>=1.0"]` を宣言すると、 3.14t 環境では適合する wheel が無く install に失敗する。 1 つの `pyproject.toml` で「3.12〜3.14 は分割モード (abi3)」「3.14t は従来モード (linked)」の 2 種類のメタデータを作り分ける必要があり、 その分離方法 (別ディストリビューション名 / メタデータプラグイン) が未解決
- Free Threading を落とす選択は取らない (本パッケージは Free Threading 対応をうたっている)
- その他、 採用時に追加で確認が必要な点:
  - `nanobind_add_stub` は stubgen でビルド済みモジュールを import するため、 `build-system.requires` にも `nanobind-backend` が必要になる可能性がある (要実測)
  - split mode では `nanobind-static` target が生成されないため、 `CMakeLists.txt` が同 target に設定しているプロパティが Windows で configure エラーになる。 また split mode の Windows は `Development.SABIModule` を要求するが、 現状の `find_package` は `Development.Module` のみ
  - Linux の `auditwheel repair` は libstdc++ を vendoring し得るが、 split mode は C++ ランタイムの同梱を想定していない。 `auditwheel show` と `ldd` での確認が要る

## 再開の条件

- Python 3.15 以降に対応する時点で、 scikit-build-core 1.0 以降の combined tag (`py-api = "cp315.cp315t"`) により `cp315-abi3.abi3t` の 1 ホイールにできる (nanobind-backend にも cp315t wheel がある)。 これが最も筋の良い将来パス
- 3.12〜3.14 と 3.14t を並行して維持する間は、 platform あたり abi3 1 個 + cp314t 1 個の 2 ホイールが上限 (leg は 24 → 12)。 依存 metadata の分離方法が解決できれば、 その範囲で採用できる

## 現状

- `CMakeLists.txt` の `nanobind_add_module` に `BACKEND_MODULE` を指定していない (従来モード)
- wheel.yml は Python 3.12 / 3.13 / 3.14 / 3.14t の 4 バージョンでビルドし、 free-threading の leg では `sys._is_gil_enabled()` が偽であることを確認している
- `requires-python = ">=3.12"` (分割モードの frontend は 3.10 stable ABI を対象にするが、 対応バージョンの方針は別途判断する)

## 設計方針

- `CMakeLists.txt` の `nanobind_add_module` に `BACKEND_MODULE nanobind_backend` を指定して分割モードを有効にし、 `pyproject.toml` に `wheel.py-api = "cp312"` を設定する
- backend は PyPI の `nanobind-backend` ホイールとして配布され、 frontend は backend を実行時に読み込む。 ランタイム依存は `[project] dependencies = ["nanobind-backend>=1.0"]` で宣言する (上限制約は付けない)。 Free Threading の leg とメタデータを分離する方法は未解決のため、 先にそちらを決める
- Free Threading は Python 3.15 以降に `abi3t` で統合する方針とし、 それまでは従来モードを併用する
- CI は abi3 の leg (1 platform 1 wheel を複数の Python で検証) と Free Threading の leg に分け、 leg 数と `_deps` キャッシュキーを見直す
- Linux の `auditwheel` の扱い (libstdc++ の vendoring と backend との二重ロード) を実測して決める

## 完了条件

- abi3 のホイールがプラットフォームごとに 1 ファイルになり、 同一のホイールを Python 3.12 / 3.13 / 3.14 の各環境に install してテストが PASS する
- Free Threading のホイールが `cp314t` の 1 ファイルになり、 `Requires-Dist` に `nanobind-backend` を含まず、 3.14t 環境で install とテストが成功する
- 1 リリースあたりのホイール数が 24 から 12 に減っている
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
