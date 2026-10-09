# FREE_THREADED が GIL あり Python で黙って無効化され、Free-Threading 対応が検証されない

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-freethreaded-silently-disabled
- Polished: 2026-10-09

## 目的

CMakeLists.txt は FREE_THREADED を無条件に指定しているが、 nanobind は GIL あり Python でビルドした場合にエラーも警告もなく無効化する。 そのため「FREE_THREADED を指定しているのに GIL ありビルドができる」状態が黙って成立し、 配布する wheel の ABI がビルド環境任せになっている。 Free-Threading 対応の Python かどうかを判定して明示的に切り替え、 方針と食い違わないようにする。

## 優先度根拠

- 実測 (ローカルの GIL あり Python 3.12): `_build/CMakeCache.txt` で `NB_FREE_THREADED:INTERNAL=0`、 生成物は `libdatachannel_ext.cpython-312-darwin.so` と GIL あり wheel (`cp312`)
- nanobind の CMake は GIL ありインタプリタで `set(ARG_FREE_THREADED FALSE)` とするだけで、 警告もメッセージも出さない (nanobind 3.1.0 の `cmake/nanobind-config.cmake`)
- ソースからビルドした場合、 指定と実際の ABI が食い違っても気付けない (現行の配布は wheel のみで sdist は無い)
- Free-Threading の並行性テストは [[0022-test-enable-ci-tests]] の対応で CI の 3.14t leg で実行されるようになった (実測: `tests/test_free_threading.py` の 12 テストが skip されず PASSED、 全体 116 passed)

## 現状

- pyproject.toml は requires-python >= 3.12 で、 classifiers に 3.12 / 3.13 / 3.14 と `Free Threading :: 2 - Beta` を列挙している (GIL あり版と Free-Threading 版の両方を配布する前提)
- wheel.yml の matrix は 3.12 / 3.13 / 3.14 (GIL あり) と 3.14t で、 3.13t 向けの wheel は配布していない。 3.14t leg には wheel を install して pytest を実行する step と、 import 後に GIL が無効であることを確認する step がある ([[0022-test-enable-ci-tests]] で対応済み)
- tests/test_free_threading.py の skipif は実行時の `sys._is_gil_enabled()` だけを見ており、 モジュールが Free-Threading 対応でビルドされたか (`NB_FREE_THREADED`) は検証しない (GIL あり wheel を Free-Threading の Python に入れた場合は逆に実行される)

## 設計方針

- Free-Threading 対応の Python かどうかを CMake で判定し、 `FREE_THREADED` を条件付きで指定する (採用)
  - 判定は nanobind と同じく Python の ABI が `[0-9]t` かどうかで行う (`NB_FREE_THREADED` / `Python_SOABI` / `SKBUILD_SOABI`)
  - `message(FATAL_ERROR)` は入れない。 GIL あり leg (3.12 / 3.13 / 3.14) とローカルの `make develop` は GIL ありビルドが正規の構成であり、 落とすと開発と CI が成立しないため
  - 代わりに判定と実際のビルドが食い違った場合に `message(WARNING)` を出し、 nanobind の暗黙の上書きを検出する
- 配布方針をコメントと issue に明記する: Free-Threading 版の wheel は 3.14t のみ、 GIL あり版は 3.12 / 3.13 / 3.14、 3.13t 向けは配布しない
- 3.14t leg で `tests/test_free_threading.py` が実行され続けることを回帰防止として確認する ([[0022-test-enable-ci-tests]] で対応済み)

## 完了条件

- Free-Threading 対応の Python では `FREE_THREADED` が指定され `NB_FREE_THREADED=1` の wheel (`cp314t`) になり、 GIL ありの Python では指定されず `NB_FREE_THREADED=0` の wheel になること (`_build/CMakeCache.txt` と wheel タグで確認する)
- 判定とビルドが食い違った場合に `message(WARNING)` が出ること (nanobind の暗黙の上書きの検出)
- 3.14t leg で `tests/test_free_threading.py` が skip されず PASSED のままであること (回帰防止)
- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の全 leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に変更内容が記録されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: CMakeLists.txt (`nanobind_add_module`)、pyproject.toml (requires-python / classifiers)
- nanobind: nanobind-config.cmake の ARG_FREE_THREADED 処理
- 関連 issue: [[0022-test-enable-ci-tests]]
