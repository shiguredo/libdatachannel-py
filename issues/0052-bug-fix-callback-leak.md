# callback が wrapper を捕捉すると close() を呼ぶまでインスタンスが解放されない

- Priority: High
- Created: 2026-10-10
- Completed: 2026-10-10
- Branch: develop (直接コミット)
- Polished: 2026-10-10

## 目的

callback が wrapper 自身を捕捉している (閉包や束縛メソッド) 場合、 `close()` を呼ばない限り `PeerConnection` などのインスタンスが解放されず、 長時間動かす利用者でメモリが増え続ける。 原因を記録し、 回帰テストで「close() で解放される」ことを固定する。

## 優先度根拠

- 実測: `pc.on_state_change(lambda state: (pc, state))` のように wrapper を捕捉する callback を登録し、 `del pc` + `gc.collect()` しても解放されず、 終了時に `nanobind: leaked 1 instances!` (付随して `leaked N types / functions`) が出る
- 実測: `close()` または `reset_callbacks()` を呼ぶと解放される (参照カウント 登録前 2 → 登録後 3 → close 後 2)
- nanobind の `std::function` caster (`nanobind/stl/function.h`) が callable を `Py_INCREF` で強参照するため、 `nb_inst → C++ → std::function → callable → nb_inst` の循環ができ、 C++ 側が GC の対象外であるために回収されない
- `nanobind/src/nb_internals.cpp` のコメントどおり、 `leaked types / functions / keep_alive records` はインスタンスリークの付随報告で、 実体はインスタンス 1 個

## 現状

- テストが `close()` を呼ばずに callback を登録したまま終了すると pytest の終了時に警告が出る
- `keep_alive` は無関係 (`keep_alive<1, 2>` が 1 箇所と `rv_policy::reference_internal` のみ)。 `__del__` は循環があると呼ばれないため (issue 0039) 対処にならない

## 設計方針

- callback を登録したオブジェクトは `close()` で解放することを利用者向けに明記する
- 回帰テストとして、 wrapper を捕捉する callback を登録して `close()` を呼ぶ子プロセスを実行し、 `nanobind: leaked` が stderr に出ないことを検証する
- 恒久対処 (caster を弱参照にして循環の辺を消す) は挙動変化 (参照が切れた callback が呼ばれなくなる) を伴うため、 本 issue では実施せず別途判断する

## 完了条件

- 回帰テストが追加され、 `close()` を呼べばリークしないことが検証されていること
- 恒久対処の可否を判断できる材料 (原因・副作用) が issue に記録されていること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `tests/leak_reproduction_callbacks.py` (新規)
  - wrapper を捕捉する callback (`lambda state: (pc, state)`) を登録し、 `close()` してから破棄する子プロセス用スクリプト
- `tests/test_leak.py` (新規)
  - 上記を subprocess で実行し、 終了コード 0 と `nanobind: leaked` が stderr に出ないことを検証する
- 検証
  - `tests/test_leak.py` が pass する (close を外すと `nanobind: leaked` で失敗することを確認する)
  - 全体 173 passed / 12 skipped / 2 deselected
  - `/review-diff-code` の致命的 / 重要指摘が 0 件
- 恒久対処 (弱参照 caster) は挙動変化を伴うため未実施。 実施するかは別途判断する

## 参考

- 実測スクリプト: /tmp/leak_investigation.md (調査時の再現表)
- 関連: [[0039-bug-fix-nanobind-del-not-called]]
