# CI で lint と typecheck (prek のフック) が実行されていない

- Priority: High
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-run-prek-hooks-in-ci
- Polished: 2026-10-09

## 目的

[[0027-fmt-fix-lint-typecheck-gate]] で整備した品質ゲート (ruff / ty) は prek の git フックと Makefile のターゲットにしか無く、 CI では一度も実行されない。 CI で prek のフックを実行し、 ローカルと同じ検証を PR で行う (必須ステータスチェック化 (branch protection) はリポジトリ設定のため対象外)。

## 優先度根拠

- git フックをスキップした変更や、 フック環境を用意していない環境からの push は無検証で merge され得る
- shiguredo-python は CI で `j178/prek-action` を使い、 ローカルと同じ prek.toml のフックをすべて実行することを規定している
- ty は生成スタブ (`src/libdatachannel/__init__.pyi`) を必要とし、 素の checkout では 92 diagnostics で必ず失敗する (実測)。 CI で ty を実行するにはビルドを伴うジョブが要る
- CI に pytest も無い問題は [[0022-test-enable-ci-tests]] が扱っており、 本 issue は lint / typecheck のゲートを対象にする

## 現状

- `.github/workflows/wheel.yml` は `uv build --wheel` と `uvx auditwheel` で wheel を作るだけで、 pytest はコメントアウトされている (`# - run: uv run pytest tests/ -v`)。 ruff / ty を実行する step は無い
- `.github/workflows/build_debug.yml` は workflow_dispatch 専用で、 本リポジトリに存在しないファイル (DEPS / run.py / src/libdatachannel/py.typed / src/libdatachannel/libdatachannel_ext.pyi) に依存している
- そのため `make lint` / `make typecheck` はローカルでしか実行されず、 CI では検出されない
- **ty は生成スタブを必要とする**: `*.pyi` は `.gitignore` の対象で git 管理外であり、 `nanobind_add_stub` (CMakeLists.txt) がビルド時に生成し、 `make develop` が `_build/__init__.pyi` をコピーする。 スタブが無い状態で ty を実行すると `libdatachannel` を解決できず 92 diagnostics で失敗する (実測: クリーンな checkout で `prek run --all-files` を実行すると 9 フックは Passed、 ty のみ Failed)。 Makefile にも「生成スタブが無いと libdatachannel を解決できないため、 事前に make develop を実行しておく」と書かれている
- **clang-format フックは runner に実体が必要**: prek.toml の clang-format は `language = "system"` / `entry = "clang-format -i"` で、 PATH 上の `clang-format` を実行する。 ubuntu-slim には入っていないため、 導入しないと prek が `Failed to run hook 'clang-format'` で exit 2 になる (実測)
- 実行対象のフック (prek.toml): builtin 4 件 (trailing-whitespace / end-of-file-fixer / check-toml / check-yaml)、 ruff-format、 ruff-check (`--fix`)、 tombi-lint、 tombi-format、 ty (`--isolated`)、 clang-format
- 既定ブランチは develop で、 既存ワークフローに `pull_request` トリガは無い (wheel.yml は push / schedule / workflow_dispatch)

## 設計方針

- `.github/workflows/prek.yml` を新規に追加し、 PR と既定ブランチ (develop) への push で実行する。 既存の wheel.yml / build_debug.yml は変更しない
- ビルドが要る ty と、 それ以外のフックをジョブに分ける (lint のフィードバックを待たせない)
  - `prek` ジョブ (runner: ubuntu-slim、 `timeout-minutes: 15`): checkout → setup-uv → clang-format を導入 → prek-action で `--all-files --skip ty` を実行する
  - `ty` ジョブ (runner: ubuntu-24.04、 `timeout-minutes: 45`): checkout → setup-uv → `_deps` をキャッシュ → `uv build --wheel` → `cp _build/__init__.pyi src/libdatachannel/` → prek-action で `--all-files ty` を実行する (wheel.yml の build_ubuntu と同じ手順で、 追加の apt 依存はない)
    - ubuntu-slim を使わない理由: ubuntu-slim は 1 vCPU / 5 GB でジョブの実行時間が 15 分に制限されるため、 wheel ビルドを伴う ty ジョブは完走の保証が無い (キャッシュミス時のビルドは wheel.yml でも 30 分を見込んでいる)。 この理由はワークフローのコメントにも残す
- clang-format は runner に入っていないため導入する。 ローカルと同じバージョン (23.1.3) を `uv tool install` で固定する (バージョン差による偽陽性を避ける)
- フックは prek.toml の定義をそのまま使う (ruff / ty / tombi を CI で個別にインストールしない)
- action は既存ワークフローと同じ「コミットハッシュ固定 + バージョンコメント」で書く (`actions/checkout` / `astral-sh/setup-uv` は既存と同じ pin、 `j178/prek-action` は使用するバージョンのハッシュ)。 `permissions: contents: read` を明示する (既存ワークフローの `contents: write` はコピーしない)
- pytest は prek.toml にフックが無いため対象外とする ([[0022-test-enable-ci-tests]])。 ただし 0022 で pytest のフックが入る場合は `ty` ジョブと同じ「ビルドしてからフックを実行する」形になる
- 必須ステータスチェック化 (branch protection) はリポジトリ設定のため、 本 issue の対象外とする

## 完了条件

- PR と既定ブランチ (develop) への push でワークフローが動き、 クリーンな checkout で全フック (ty を含む) が PASS すること
- 意図的な違反 (未整形のソースや型エラー) を入れた PR でワークフローが失敗し、 修正すると PASS することを実際に確認すること
- ローカルと同じ prek.toml を使っていること
- 既存の wheel.yml / build_debug.yml の挙動を変えないこと
- `timeout-minutes` の範囲で完走すること
- `CHANGES.md` の `### misc` に `[FIX]` エントリを追加すること (CI の変更は利用者に見える API / 挙動の変更ではないため misc。 既存の CI 変更と同じ扱い)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: `.github/workflows/prek.yml` (新規)
- 対象シンボル: prek.toml のフック (builtin / ruff-format / ruff-check / tombi-lint / tombi-format / ty / clang-format)
- 関連 issue: [[0022-test-enable-ci-tests]] (CI での pytest 実行)、 [[0027-fmt-fix-lint-typecheck-gate]] (品質ゲートの整備。 ty は生成スタブを配置した作業ツリーで判定する)
