# FREE_THREADED が GIL あり Python で黙って無効化され、Free-Threading 対応が検証されない

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-freethreaded-silently-disabled
- Polished: {YYYY-MM-DD}

## 目的

CMakeLists.txt は FREE_THREADED を指定しているが、nanobind は GIL あり Python でビルドした場合に、エラーも警告もなく FREE_THREADED を無効化する。ローカルの make wheel / make test は常に GIL ありビルドになり、tests/test_free_threading.py はどの環境でも一度も実行されていない。CHANGES.md の [ADD] Python 3.14t に対応する (Free Threading 対応) と実態が乖離している。

## 優先度根拠

- 実測: `_build/CMakeCache.txt` で `NB_FREE_THREADED:INTERNAL=0`。ローカルで生成される wheel は GIL あり版 (`libdatachannel_ext.cpython-312-darwin.so` が実在)
- Free-Threading の並行性テストがどの CI / ローカルでも走っておらず、対応の検証がゼロ
- GIL ありユーザーが sdist からビルドした場合、プロジェクト方針に反するビルドが配られる

## 現状

- nanobind の CMake config は Python ABI が free-threading でない場合、ARG_FREE_THREADED を FALSE に上書きする (エラーも警告も出ない)
- pyproject.toml は requires-python >= 3.12 で、classifiers に 3.12 / 3.13 (GIL あり) を列挙している
- wheel.yml は 3.14t で wheel をビルドするがテストは実行しない ([[0022-test-enable-ci-tests]] で対応)
- tests/test_free_threading.py は free-threading ビルドでのみ実行される skipif 付き

## 設計方針

- CMakeLists.txt で Python の free-threading 状態を検証し、FREE_THREADED 指定と不整合の場合に `message(FATAL_ERROR)` で落とす
- pyproject.toml の requires-python / classifiers と、実際に配布する wheel の ABI の整合を確認し、方針を明記する (GIL あり wheel も配布するのかを決める)
- 3.14t ビルド + テスト実行を必須化する (CI 実行自体は [[0022-test-enable-ci-tests]])

## 完了条件

- FREE_THREADED 指定時に GIL あり Python でビルドすると失敗すること (または、明示的な警告と意図のコメントで方針が固定されていること)
- 3.14t ビルドで tests/test_free_threading.py が実際に実行され PASS すること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: CMakeLists.txt (`nanobind_add_module`)、pyproject.toml (requires-python / classifiers)
- nanobind: nanobind-config.cmake の ARG_FREE_THREADED 処理
- 関連 issue: [[0022-test-enable-ci-tests]]
