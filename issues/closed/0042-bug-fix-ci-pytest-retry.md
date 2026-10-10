# CI の pytest リトライが job を救済できていない

- Priority: High
- Created: 2026-10-09
- Completed: 2026-10-09
- Branch: feature/fix-ci-pytest-retry
- Polished: {YYYY-MM-DD}

## 目的

prek.yml の `ty` ジョブにある pytest のリトライが、 1 回目の失敗で job を失敗扱いにしてしまい機能していない。 断続的に失敗するテスト ([[0038-bug-fix-websocketserver-native-crash]] の native crash 等) でリトライが成功しても job が赤くなるため、 リトライの仕組みとして成立するように直す。

## 優先度根拠

- 実際に PR #41 で発生した。 `tests/test_websocketserver.py::test_websocketserver` が `Fatal Python error: Aborted` で落ち、 続く `Retry pytest` は Passed になったが、 job の結論は failure のままだった (リトライが無意味になっている)
- 断続的な失敗は既知で、 リトライは [[0022-test-enable-ci-tests-in-ci]] で導入した回避策であるため、 動かないままにすると同じ再実行の手間が繰り返し発生する
- CI が赤いままでは auto-resolve のフロー (CI 通過後に squash merge) が止まる

## 現状

- `.github/workflows/prek.yml` の `ty` ジョブは `Run ty and pytest` (id: `prek`) と `Retry pytest` (`if: failure() && steps.prek.outcome == 'failure'`) の 2 ステップで構成している
- GitHub Actions では、 あるステップが失敗すると `continue-on-error: true` が無い限り job の結論は failure になる。 2 番目のステップが成功しても 1 番目の失敗は取り消されない
- 実測 (PR #41 の run 37873395315): `Run ty and pytest` が failure (`test_websocketserver` の Aborted)、 `Retry pytest` が success (pytest Passed)、 それでも job の結論は failure だった
- `.github/workflows/wheel.yml` の `Test wheel` は 1 つの bash ステップ内で `run_pytest || { echo "::warning::..."; run_pytest; }` としており、 こちらは正しく救済できる

## 設計方針

- `prek.yml` の `ty` ジョブのステップを `Run ty` と `Run pytest` に分ける。 1 つのステップで ty と pytest をまとめて実行すると、 `continue-on-error` により ty の失敗までリトライで救済されて型検査のゲートが失われるため
- `Run pytest` に `continue-on-error: true` を付ける (ステップの結論は success になるが `steps.pytest.outcome` は failure のまま残る)
- `Retry pytest` の条件を `steps.pytest.outcome == 'failure'` にする (継続時は job が失敗状態ではないため `failure()` は使えない)。 `Run ty` には `continue-on-error` を付けないため、 ty が失敗すれば job は失敗する
- これにより 1 回目失敗 + 2 回目成功なら job は success、 2 回目も失敗なら job は failure になる
- リトライであることが分かるように `::warning::` の注記を出す (wheel.yml と揃える)

## 完了条件

- 1 回目の pytest が失敗し 2 回目が成功する状況で job が success になり、 `::warning::` がログに出ること (断続的な失敗のため、 発生時に CI で確認する)
- リトライも失敗した場合は job が failure のままであること
- `ty` が失敗した場合はリトライされず job が failure になること
- `CHANGES.md` の `### misc` にエントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `.github/workflows/prek.yml` の `ty` ジョブのステップを `Run ty` と `Run pytest` に分けた。 1 つのステップで ty と pytest をまとめて実行すると、 `continue-on-error` により ty の失敗までリトライで救済されて型検査のゲートが失われるため
- `Run pytest` に `continue-on-error: true` を付け、 `Warn pytest retry` と `Retry pytest` の条件を `steps.pytest.outcome == 'failure'` にした。 これにより 1 回目失敗 + 2 回目成功なら job は success、 2 回目も失敗なら job は failure、 ty が失敗した場合はリトライされず job は failure になる
- リトライ時に `::warning::pytest failed; retrying the test suite once` を出し、 wheel.yml と同じ文言にした
- `CHANGES.md` の `### misc` に `[FIX]` エントリを追加した

## スコープ外 (関連する未解決問題)

- 断続的な失敗そのものの根本対応 (WebSocketServer の native crash) は [[0038-bug-fix-websocketserver-native-crash]] の範囲とする
- wheel.yml 側のリトライは正しく動いているため変更しない

## 参考

- GitHub Actions の `continue-on-error` と `steps.<id>.outcome` / `steps.<id>.conclusion` の関係
- 関連 issue: [[0038-bug-fix-websocketserver-native-crash]] / [[0022-test-enable-ci-tests-in-ci]]
