# nanobind を 3 系に、 scikit-build-core を 1.1.1 に更新する

- Priority: High
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/update-build-deps
- Polished: {YYYY-MM-DD}

## 目的

ビルド依存の `nanobind` を 3 系 (最新 3.1.0) に、 `scikit-build-core` を 1.1.1 に更新する。 nanobind 3 系は内部 ABI と API に破壊的変更を含むため、 影響箇所を修正したうえでビルド・テスト・ホイール生成が通ることを確認する。

## 優先度根拠

- nanobind 2 系のままだと 3 系で入った性能改善 (interned string keys / 不要な参照カウント削除 / シーケンス構築の高速化 / 分割モード) を取り込めない
- nanobind 3 系は Python 3.15 以降での interpreter finalization 対応 (PEP 788) を含み、 恒停問題 ([[0005-bug-fix-destructor-callback-deadlock]] / [[0039-bug-fix-nanobind-del-not-called]]) の調査にも影響する
- scikit-build-core 1.1.1 は `minimum-version = "build-system.requires"` で最小バージョンを同期しているため、 更新するとビルド時の検証も最新になる

## 現状

- `pyproject.toml`: `requires = ["nanobind>=2.13.0", "scikit-build-core>=1.0.3"]`、 `[tool.scikit-build] minimum-version = "build-system.requires"`、 `requires-python = ">=3.12"`
- CI の Python は 3.12 / 3.13 / 3.14 / 3.14t (wheel.yml) で、 nanobind 3 の `>=3.10` 要件は満たしている
- nanobind 3.0 の破壊的変更のうち、 本リポジトリに関係するもの:
  - `NB_TRAMPOLINE(Base, Size)` の `Size` が不要になり、 指定すると deprecation warning が出る (`src/bind_libdatachannel.cpp` の `PyMediaHandlerImpl` で `NB_TRAMPOLINE(PyMediaHandler, 5)` を使用)
  - 型 caster の `from_python()` の `flags` が `uint8_t` から `uint32_t` に拡張された (`src/bind_libdatachannel.cpp` の独自 caster で `uint8_t flags` を使用。 旧シグネチャでも動作するが警告が出る可能性がある)
  - `nb::gil_scoped_acquire` が interpreter 停止中に失敗し得るようになり、 `is_valid()` でのガードが推奨される (該当箇所は `close_peer_connection` の timeout 分岐と `close_websocket` の timeout 分岐。 Python 3.15 未満では従来どおり)
  - `nb::none` が wrapper class になり、 条件式で他の wrapper 型と混在できなくなった
  - `rv_policy` は compile-time tag になり、 実行時に計算した値を渡せなくなった
- 参考: nanobind 3.0.0 は Python 3.9 互換を誤って宣言していたため yank され、 3.0.1 が後継。 最新は 3.1.0

## 設計方針

- `pyproject.toml` の `requires` を `nanobind>=3.1.0` と `scikit-build-core>=1.1.1` に更新する (`minimum-version = "build-system.requires"` により scikit-build-core の最小バージョンも同期する)
- nanobind 3 で必要なソース修正を行う:
  - `NB_TRAMPOLINE(PyMediaHandler, 5)` を `NB_TRAMPOLINE(PyMediaHandler)` にする
  - 独自型 caster の `from_python()` の `flags` を `uint32_t` に広げる
  - `nb::gil_scoped_acquire` の `is_valid()` ガードは、 Python 3.15 未満で挙動が変わらないことと、 ガードしない場合の実害 (停止中の acquire で恒停し得る) を確認したうえで要否を決める
  - その他、 ビルドエラー・警告が出た箇所を修正する
- ビルドは `make develop` (フルビルド) で確認し、 ホイール生成 (`uv build --wheel`) も確認する
- `uv.lock` や CI のビルドステップにビルド依存のバージョン固定があれば合わせて更新する

## 完了条件

- `make develop` が通る (nanobind 3 系 / scikit-build-core 1.1.1 でコンパイルエラー・警告なし)
- `uv build --wheel` が通り、 生成したホイールで `uv run --no-sync python -m pytest tests/ -v --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close` が PASS する
- `prek run --all-files pytest` (prek.toml の pytest フック) が PASS する
- CI (wheel.yml の 24 leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に `[UPDATE]` エントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## スコープ外 (関連する未解決問題)

- nanobind 3 の分割モード (split mode / `BACKEND_MODULE`) の採用は行わない (ホイール配布戦略の変更になるため別途判断する)
- 恒停問題の根本対応は [[0005-bug-fix-destructor-callback-deadlock]] / [[0039-bug-fix-nanobind-del-not-called]] の範囲とする
- libdatachannel 本体や他の依存の更新は対象外

## 参考

- nanobind changelog: https://nanobind.readthedocs.io/en/latest/changelog.html
- nanobind 3.0.0 の API break: `NB_TRAMPOLINE` の Size 廃止 / `rv_policy` の tag 化 / `nb::none` の wrapper 化 / 型 caster の `flags` 拡張 / `nb::gil_scoped_acquire::is_valid()` / Python 3.10 以上必須 / `nb::ndarray_traits` 削除
- scikit-build-core 1.1.1: https://github.com/scikit-build/scikit-build-core/releases
- 関連 issue: [[0039-bug-fix-nanobind-del-not-called]] / [[0005-bug-fix-destructor-callback-deadlock]]
