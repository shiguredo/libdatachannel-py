# make lint と make typecheck が失敗したまま develop に放置されている

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-lint-typecheck-gate
- Polished: 2026-10-08

## 目的

`make lint` (`ruff check examples/`) が 69 errors、 `make typecheck` (`ty check examples/`) が 2 diagnostics で失敗しており、 リポジトリ全体では ruff が 87 errors、 ty が 9 diagnostics で失敗している。 さらに ruff の規約セットが pyproject.toml に明示されていないため ruff の更新で検出結果が予告なく変わる。 品質ゲートを機能させる。

## 優先度根拠

- lint / typecheck が失敗したままの develop は、 次の変更の lint 結果の判断を不能にする
- ruff 0.16 でデフォルト規約セットが 59 件から 413 件に拡大しており、 select を明示しない限り将来の ruff 更新でも検出結果が予告なく変わる
- ローカルの ruff / ty と prek のフックのバージョンが二重管理になっており、 `uv sync --upgrade` のたびに乖離し得る (実際に 2026-08-29 の更新で prek の rev が v0.14.14 のまま uv.lock の ruff だけが上がり、 検出結果が食い違った)

## 現状

- 実測: `uv run ruff check .` → 87 errors (BLE001 31、 UP045 29、 S110 6、 PERF102 5、 RUF100 7、 UP006 2、 I001 2、 UP035 1、 SIM114 1、 SIM103 1、 RUF059 1、 PIE810 1)
  - ファイル別: examples/whip.py 39、 examples/whep.py 30、 tests/test_free_threading.py 8、 src/libdatachannel/__init__.py 7、 dev.py 3
- 実測: `uv run ty check` → 9 diagnostics (examples/whip.py 2、 tests/test_peerconnection.py 4、 tests/test_websocket.py 1、 tests/test_websocketserver.py 2)
  - examples/whip.py:1397-1398 は `disconnect()` 内で `self.renderer.ctx` / `self.renderer.img` に None を代入している箇所
  - tests の 7 件は `Optional[...]` の絞り込み漏れによる attribute 未定義の指摘
- pyproject.toml の `[tool.ruff]` は target-version と line-length のみで `[tool.ruff.lint] select` が無く、 ruff 0.16.10 の既定 413 ルールが暗黙に適用されている
- Makefile の lint / typecheck の対象が examples/ のみで、 prek の ty-check はリポジトリ全体と、 範囲が不統一
- ruff / ty が `[dependency-groups] lint` と prek.toml の rev (および ty はローカルフック `uv run ty check`) の二重管理になっている

## 設計方針

- ruff / ty のバージョンは prek.toml の rev を唯一の正とする
  - `[dependency-groups]` の `lint = ["ruff", "ty"]` を削除し、 `dev` の `{ include-group = "lint" }` も削除する (group を消して include を残すと `uv sync` が "Failed to find group `lint` included by `dev`" で失敗する)
  - prek.toml の ty フックをローカルの `uv run ty check` から `astral-sh/ty-pre-commit` (rev v0.0.85、 `args = ["--isolated"]`) に置き換える
  - ruff-pre-commit の rev (v0.16.10) はそのまま使う
- pyproject.toml の `[tool.ruff.lint]` に select を明示して規約セットを固定する
  - ruff 0.16.10 の既定 413 ルールを列挙する。 バージョン更新時は `uvx ruff@<prek.toml の ruff-pre-commit の rev と同じバージョン> check --isolated --show-settings <file>` の `linter.rules.enabled` を見て差分を反映する (`--isolated` を付けないと設定済みの select がそのまま返り、 既定セットの差分を検出できない)
  - `src/libdatachannel/__init__.py` の star import とその別名定義に必要な F403 / F405 を加える (これにより同ファイルの不要な noqa 6 件の指摘が解消し、 修正対象は 81 件になる)
  - `ignore` は空にする
- Makefile の lint / typecheck / format を prek のフック経由に統一し、 対象をリポジトリ全体にする
  - `lint: prek run --all-files ruff-check`。 ruff-check フックは `args = ["--fix"]` を持つため、 自動修正できる違反は `make lint` が修正する
  - `typecheck: prek run --all-files ty`
  - `format:` は現行の `clang-format -i src/*.cpp` を残し、 `uv run ruff format src/ examples/ tests/` を `prek run --all-files ruff-format` に置き換える。 ruff-format フックは Markdown も整形対象にする
- `[tool.ty.src] exclude = ["src/libdatachannel/__init__.py"]` は維持する (star import の別名モジュールは型検査の対象外)
- ruff の 81 errors を 0 にする
  - `ruff check --fix` で自動修正できる 35 件は自動修正する
  - BLE001 31 件 (examples 28 / tests 3) は例外の種類を絞れる箇所は絞り、 意図的に広く捕捉する箇所は `# noqa: BLE001` に理由コメントを付けて残す
  - S110 6 件 (`except ...: pass`) は debug ログを残すか、 握り潰す理由をコメントで明記する。 この 6 箇所は [[0028-fmt-fix-language-convention-violations]] の現状にも挙がっているため、 本 issue で修正した旨を 0028 側へ反映する
  - UP045 / UP006 / UP035 / PERF102 / I001 / SIM103 / SIM114 / PIE810 / RUF059 / RUF100 は修正する
- ty の 9 diagnostics を 0 にする
  - examples/whip.py の `disconnect()` は `self.renderer.ctx = None` / `self.renderer.img = None` の 2 行が `invalid-assignment` になる (`ctx` / `img` は None を許さない型)。 直後に `self.renderer = None` として参照を切るため、 この 2 行を削除する
  - tests の 7 件は None チェック (assert) またはローカル変数で絞り込む

## 完了条件

- `make lint` と `make typecheck` が PASS すること
- `prek validate-config prek.toml` と `prek run --all-files` が PASS すること
  - 判定は `make develop` で生成スタブ (`src/libdatachannel/__init__.pyi`) を配置した作業ツリーで行う (未生成の状態では libdatachannel を解決できず diagnostics が大幅に増える)
  - `prek run --all-files` には ruff-format による Markdown の整形が含まれる。 現状 ruff-format は `issues/0008-bug-fix-non-owning-reference-lifetime.md` / `issues/0014-bug-fix-candidate-hash-inconsistency.md` / `issues/closed/0001-bug-fix-peer-connection-destructor-gil-release.md` の 3 ファイルを整形するため、 これも本 issue に含めてコミットする
- pyproject.toml の `[tool.ruff.lint]` に select が明示されていること
- ruff / ty が prek.toml の rev で単一管理され、 `[dependency-groups]` に含まれていないこと
- `uv sync && make test` で全テストが PASS すること。 ただし `tests/test_peerconnection.py::test_destruct_without_explicit_close` は [[0005-bug-fix-destructor-callback-deadlock]] が扱う既知の恒停を持つため、 本 issue の完了判定では同テストの恒停の有無を問わない
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: pyproject.toml (`[tool.ruff.lint]` / `[dependency-groups]` / `[tool.ty.src]`)、 prek.toml、 Makefile、 examples/ / tests/ / src/libdatachannel/__init__.py / dev.py の lint 違反、 ruff-format が整形する issues/ の Markdown 3 ファイル
- 関連 issue: [[0028-fmt-fix-language-convention-violations]]、 [[0022-test-enable-ci-tests]]、 [[0023-test-set-pytest-default-timeout]]
- 同じ方針の実装例: 兄弟プロジェクト webtransport-py の pyproject.toml (`[tool.ruff.lint] select` に既定ルールを列挙) と prek.toml (`astral-sh/ty-pre-commit` を使用)
