# callback の参照循環を解消する (C++ 側を弱参照、 Python 側で強参照)

- Priority: High
- Created: 2026-10-10
- Completed: 2026-10-10
- Branch: develop (直接コミット)
- Polished: 2026-10-10

## 目的

[[0052-bug-fix-callback-leak]] で「callback が wrapper 自身を捕捉すると `close()` を呼ぶまで解放されない」ことを記録した。 利用者が `close()` を呼び忘れると確実にメモリが漏れるため、 参照循環そのものを解消する。 互換性よりも安定性を優先する。

## 優先度根拠

- 実測: `pc.on_state_change(lambda state: (pc, state))` のように wrapper を捕捉する callback を登録し、 `del pc` + `gc.collect()` しても解放されず `nanobind: leaked` が出る。 `gc.collect()` では回収できない
- nanobind の `std::function` caster が callable を `Py_INCREF` するため `nb_inst → C++ → std::function → callable → nb_inst` の循環ができ、 C++ 側が Python の GC の対象外であるために回収されない
- 利用者に `close()` の呼び出しを必須にするのは、 忘れると静かに漏れる設計で不安定

## 現状

- `src/bind_libdatachannel.cpp` の callback 登録 17 箇所が `std::function` をそのまま受け取るため、 C++ 側に強参照が残る
- `close()` / `reset_callbacks()` を呼べば解放される (参照カウント 登録前 2 → 登録後 3 → close 後 2)

## 設計方針

- callback を **Python 側 (インスタンスの `__dict__`) で強参照**し、 **C++ 側では弱参照 (`weakref.ref` 経由) で持つ**
  - Python 側に強参照があるため、 inline の lambda や束縛メソッドを渡しても従来どおり呼ばれる
  - C++ 側は弱参照なので参照カウントに寄与せず、 循環は GC から見える形 (instance → `__dict__` → callable → instance) になり回収される
- 恒停時の挙動: callback が既に解放されていれば呼ばない (weakref が切れている場合は何もしない)
- 互換性の変化: C++ 側が弱参照になるため、 「登録した callback を利用者が一切参照しない」場合でも Python 側で保持される (従来と同じく呼ばれる)。 内部で `_rdc_callback_` で始まるキーを `__dict__` に置く

## 完了条件

- wrapper を捕捉する callback を登録し、 `close()` を呼ばずに破棄しても `nanobind: leaked` が出ないこと (子プロセスで検証)
- inline の lambda と束縛メソッドが従来どおり呼ばれること (テストで検証)
- 既存の全テストが PASS すること (`prek run --all-files pytest` / `prek run --all-files ty`)
- `CHANGES.md` の `## develop` に記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `src/bind_libdatachannel.cpp`
  - callback を Python 側 (インスタンスの `__dict__`、 キーは `_rdc_callback_` で始まる) で強参照し、 C++ 側は `weakref.ref` 経由の弱参照で持つ `python_callback<R, Args...>()` を追加し、 callback 登録 17 箇所 (18 個の登録) を置き換えた
  - これにより循環が `instance → __dict__ → callable → instance` という GC から見える形になり、 `del` + `gc.collect()` で回収される。 C++ 側は参照カウントに寄与しない
  - `Channel` / `DataChannel` / `Track` / `WebSocket` / `PeerConnection` / `WebSocketServer` に `nb::dynamic_attr()` を追加した (`__dict__` を持つために必須)
  - 弱参照を作れない callable (メソッド記述子など) と `nb::find()` に失敗した場合は、 従来どおり C++ 側で強参照するフォールバックを入れた (callback が呼ばれなくなる事故を防ぐため)
  - 弱参照の後始末は nanobind の `pyfunc_wrapper` と同じ `cleanup_guard` 方式にした (worker thread から GIL 無しで破棄され得るため)
  - `reset_callbacks()` は `__dict__` の `_rdc_callback_` も削除する
- `tests/leak_reproduction_no_close.py` / `tests/test_callback_lifetime.py` (新規)
  - wrapper を捕捉する callback を登録し `close()` を呼ばずに破棄しても `nanobind: leaked` が出ないことを子プロセスで検証する
  - inline の lambda と束縛メソッドが従来どおり呼ばれること、 `on_open` と `on_message` の 2 引数版が実接続で呼ばれることを検証する
- `CHANGES.md`
  - `## develop` に `[FIX]` として記録した
- 検証
  - 全体 176 passed / 12 skipped / 2 deselected、 `nanobind: leaked` の出力なし
  - 修正前の wheel (`dist` の dev5) を別環境で実行すると `nanobind: leaked 1 instances!` が出ることを確認した (新テストが回帰を捕まえる)
  - `prek run --all-files ty` が PASS、 clang-format / ruff も通過
  - `/review-diff-code` の致命的 / 重要指摘が 0 件
- 互換性の変化: 上記 6 クラスが `__dict__` を持つようになる (利用者が属性を設定可能、 インスタンスあたり 8 バイト増)。 callback の呼ばれ方は変わらない

## 参考

- 調査結果: `/tmp/leak_investigation.md` (再現表と一次資料の逐語)
- 関連: [[0052-bug-fix-callback-leak]] / [[0039-bug-fix-nanobind-del-not-called]]
