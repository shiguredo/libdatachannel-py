# CI でテストが一度も実行されていない

- Priority: High
- Created: 2026-08-30
- Completed: 2026-10-09
- Branch: feature/fix-enable-ci-tests
- Polished: 2026-10-09

## 目的

wheel.yml は pytest をコメントアウトしており、 テストするはずの build_debug.yml は存在しないファイルに依存して手動実行でも失敗する。 push / tag / schedule のどれでも CI で pytest が動いておらず、 リリースタグでは無検証の wheel が PyPI に publish される。 CI でテストを実行し、 公開される wheel を検証する。 あわせて prek.toml に pytest のフックを追加し、 ローカルと CI の両方で実行できるようにする。

## 優先度根拠

- shiguredo-git は「全てのテストが通らない限りコミットしないこと」を規定するが、 テストについては CI に検証がなく規律のみに依存している (lint / typecheck は [[0035-test-run-prek-hooks-in-ci]] で CI 検証済み)
- リリース物 (PyPI publish) の検証がゼロであることは、 配布品質の問題として最優先
- 公開される wheel は ubuntu と macOS で 24 個あり、 どちらも検証されていない

## 現状

- wheel.yml の build_ubuntu / build_macos は `# - run: uv run pytest tests/ -v` がコメントアウトされている。 有効な pytest step は無い
- build_debug.yml は workflow_dispatch 専用で、 DEPS / run.py / src/libdatachannel/py.typed / src/libdatachannel/libdatachannel_ext.pyi に依存している。 4 つとも本リポジトリに存在せず、 最初の step (`grep '^OPENH264_VERSION=' DEPS`) で失敗する
- build_debug.yml への参照はワークフロー / Makefile / ドキュメントに無い (README の Actions バッジは存在しない `build` ワークフローを指しており、 これとは別に壊れている)
- テストは外部依存 (OpenH264 等) を持たないが、 fresh な環境では tests/conftest.py の `from aiohttp import web` と tests/test_peerconnection.py の `@pytest.mark.timeout` のため aiohttp と pytest-timeout が要る
- コメントアウトされた `uv run pytest tests/ -v` はそのままでは動かない (`[tool.uv] package = false` のため `uv run` では `libdatachannel` が install されず `ModuleNotFoundError` になる)
- **既知の恒停テストがある**: tests/test_peerconnection.py::test_destruct_without_explicit_close は GIL と libdatachannel の mutex のデッドロックで停止し、 pytest-timeout (120 秒) も発火しない (実測: 180 秒で外部から kill)。 CI でそのまま実行すると job の `timeout-minutes` まで止まる ([[0005-bug-fix-destructor-callback-deadlock]] が扱う構造で、 優先度 Low のため pending)
- 3.14t (free-threading) は CMakeLists.txt の `FREE_THREADED` と nanobind の ABI タグ判定により cp314t でビルドされるため、 tests/test_free_threading.py は現状でも skip されずに実行される (実測: 3.14t の wheel を install して 92 passed / skip 0、 `sys._is_gil_enabled()` は False)。 ただし skipif で静かに skip しても CI は緑になるため、 検証方法を決める必要がある
- 修復済み wheel の置き場所は `wheelhouse/` (ubuntu は `uvx auditwheel repair dist/*.whl --strip --only-plat` の既定出力、 macOS は `uv build --wheel --out-dir wheelhouse`)
- prek.toml に pytest のフックが無い (shiguredo-python は ruff / ty / tombi に加えて pytest もフックすることを規定している)

## 設計方針

- build_debug.yml は削除する (本リポジトリに存在しないファイルに依存しており、 手動実行でも動かない。 参照も無い)
- wheel.yml の build_ubuntu / build_macos の両方で、 ビルドした wheel を fresh な環境に install して pytest を実行する step を追加する (コメントアウトされた行を置き換える)
  - 対象は ubuntu と macOS の両方にする (公開される wheel は 24 個あり、 macOS だけ無検証にしない。 テストの aiohttp サーバーは port 0 を使うため runner の差異は問題にならない)
  - インストールするのは修復済みの wheel (`wheelhouse/*.whl`)。 ubuntu で `dist/*.whl` を入れると未修復の wheel をテストすることになる
  - 手順は `uv venv --python ${{ matrix.python_version }} .venv-test` → `uv pip install --python .venv-test/bin/python --group test wheelhouse/*.whl` → `.venv-test/bin/python -m pytest tests/ -v` とする (`--group test` で aiohttp / pytest-timeout を含むテスト依存を入れる。 `uv sync` / `uv run` ではプロジェクトが install されないため使わない)
  - 恒停テストは `--deselect tests/test_peerconnection.py::test_destruct_without_explicit_close` で除外する。 除外する理由 (GIL と libdatachannel の mutex のデッドロックで停止し、 pytest-timeout が発火しないため job の timeout まで止まる) をコメントに残す
  - 3.14t の leg ではテスト step とは別の step で `.venv-test/bin/python -c "import sys; assert not sys._is_gil_enabled()"` を実行し、 free-threading ビルドで動いていることを機械的に確認する (skip のままでも CI が緑になるため。 実行する python を venv のものに固定する)
- prek.toml に pytest のフックを追加し、 prek.yml のビルドを伴うジョブで ty と一緒に実行する (shiguredo-python の規定と、 [[0035-test-run-prek-hooks-in-ci]] の「ビルドしてからフックを実行する」形に合わせる)
  - フック定義は shiguredo-python の参考定義に合わせ、 `id = "pytest"` / `name = "pytest"` / `entry = "uv run pytest"` / `args` に恒停テストの `--deselect` / `language = "system"` / `types = ["python"]` / `pass_filenames = false` / `priority = 30` とする。 `pass_filenames = false` にしないと prek が対象ファイルを引数に渡すため、 コミット時は変更したテストファイルの分しか実行されず (変更ファイルにテストが無い場合は pytest が exit 5 で失敗する)、 CI の `--all-files` でもファイルが複数バッチに分割されて pytest が複数回起動する。 テストスイート全体を 1 回で実行させるため `pass_filenames = false` にする
  - prek.yml の非ビルドの `prek` ジョブは `--all-files --skip ty --skip pytest` にする (`--skip` は繰り返し指定する。 `--skip "ty,pytest"` はどのフックにも一致せず pytest が実行されてしまう)。 このジョブには wheel もテスト依存も無いため、 pytest を実行すると必ず失敗する
  - prek.yml のビルドを伴う `ty` ジョブは `--all-files ty pytest` にし、 `uv sync` でテスト依存を入れたあと `uv pip install --python .venv/bin/python dist/*.whl` で wheel を入れる (`uv run` が使うのは `.venv`。 wheel を入れた後に `uv sync` を実行すると wheel が削除されるため順序を守る)。 このジョブのビルドは `uv build --wheel` で出力先は `dist/` (`wheelhouse/` は wheel.yml の auditwheel 修復後の出力先で、 このジョブには無い)
  - ローカルでも `make develop` 相当で wheel (拡張モジュール) を入れるまで pytest フックは失敗するため、 その旨を prek.toml のコメントに残す
  - prek.toml の local ブロックは既に `hooks = [...]` を持つため、 新しい `[[repos.hooks]]` を足すのではなくこの配列に 1 エントリ追加する (別の `[[repos.hooks]]` にすると duplicate key で設定が読めない)
  - ワークフロー / prek.toml のコメントには issue 番号を書かず、 除外・再実行の理由そのものを書く
  - `.venv-test` は `.gitignore` の `.venv/` に一致しないため、 ローカル再現で未追跡ファイルが残らないよう `.venv-test/` を `.gitignore` に追加する
- 3.14t の leg でのみ実行される tests/test_free_threading.py のテストのうち、 [[0021-test-fix-flaky-concurrent-datachannel]] が扱う flaky テスト (test_concurrent_datachannel_creation) は断続的に失敗し得る。 除外はせず、 失敗した場合は再実行で判断することをワークフローのコメントに残す (根本対応は 0021)
- callback 内 print が残る経路 ([[0025-test-remove-callback-prints]]) は恒停を踏み得るため、 同じくコメントに残す ([[0005-bug-fix-destructor-callback-deadlock]] の構造)
- `CHANGES.md` の `### misc` に `[FIX]` エントリを追加する (CI の変更は利用者に見える API / 挙動の変更ではないため)

## 完了条件

- CI (push / tag / schedule / workflow_dispatch) で pytest が実行され PASS すること。 恒停テストは除外し、 除外理由がワークフローのコメントに残っていること
- build_ubuntu / build_macos の両方で修復済み wheel を install してテストしていること (公開される wheel がすべて検証されること)
- 3.14t の leg で free-threading ビルドであることを機械的に確認していること (skip のまま CI が緑にならないこと)
- prek.toml に pytest のフックが追加され、 prek.yml のビルドを伴うジョブで実行されること (非ビルドの `prek` ジョブでは `--skip pytest` で除外されていること)
- 3.14t の leg での flaky と callback 内 print の経路について、 再実行で判断する旨がワークフローのコメントに残っていること
- build_debug.yml が削除されていること
- `CHANGES.md` の `### misc` に `[FIX]` エントリが追加されていること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `.github/workflows/wheel.yml` の build_ubuntu / build_macos で、 ビルドした wheel を fresh な環境に install して pytest を実行する step を追加した
  - `uv venv --python <matrix>` → `uv pip install --python .venv-test/bin/python --group test wheelhouse/*.whl` → `.venv-test/bin/python -m pytest tests/ -v` の手順 (`uv sync` / `uv run` ではプロジェクトが install されないため wheel を明示的に install する)
  - ubuntu は auditwheel 修復後の wheel、 macOS は `--out-dir wheelhouse` の出力を対象にするため、 公開される 24 個の wheel すべてが検証される
  - 恒停する tests/test_peerconnection.py::test_destruct_without_explicit_close は `--deselect` で除外し、 理由をコメントに残した
- 断続的に失敗し得るテストがあるため、 失敗した場合はテストスイート全体を 1 回だけやり直す (2 回目も失敗した場合は step が失敗する。 再実行したことは警告アノテーションで示す)
- 3.14t の leg では、 モジュールを import した後に GIL が無効であることを確認する step を追加した (テストが skip されたまま緑になるのを防ぐ)
- `prek.toml` に pytest のフックを追加し、 `.github/workflows/prek.yml` のビルドを伴うジョブで ty と一緒に実行するようにした (ビルドを伴わないジョブでは `--skip pytest`)。 prek の pytest も断続的な失敗に備えて 1 回だけやり直す
- 動かない `.github/workflows/build_debug.yml` と、 参照されていない `.github/actions/download` を削除した
- `.venv-test/` を `.gitignore` に追加し、 `CHANGES.md` の `### misc` に `[FIX]` エントリを追加した
- CI の実測: PR の CI で wheel の 24 leg と prek の 2 ジョブが成功した。 1 回目の CI では WebSocketServer 系の native crash で 1 leg が失敗し、 再試行で成功することを確認した (原因の追跡は別 issue)

## 参考

- 対象: .github/workflows/wheel.yml、 .github/workflows/build_debug.yml (削除)、 .github/workflows/prek.yml、 prek.toml、 .gitignore、 CHANGES.md
- 関連 issue: [[0035-test-run-prek-hooks-in-ci]] (CI の prek フック。 pytest フックを入れる場合はビルドしてから実行する形になる)、 [[0005-bug-fix-destructor-callback-deadlock]] (恒停テストの根本対応)、 [[0016-bug-fix-freethreaded-silently-disabled]] (free-threading のビルド整合)、 [[0021-test-fix-flaky-concurrent-datachannel]] (free-threading 環境の flaky)、 [[0023-test-set-pytest-default-timeout]] (pytest-timeout の既定値)、 [[0025-test-remove-callback-prints]] (callback 内 print の削除)
