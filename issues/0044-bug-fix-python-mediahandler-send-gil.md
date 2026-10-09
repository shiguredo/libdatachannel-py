# Python の MediaHandler サブクラスから send() を呼ぶ経路が GIL を保持したまま送信経路に入る

- Priority: Medium
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-python-mediahandler-send-gil
- Polished: {YYYY-MM-DD}

## 目的

`Track.request_keyframe()` / `Track.request_bitrate()` / `Track.send()` の binding は GIL を解放して実行するようになったが、 GIL 解放が効くのは C++ の `MediaHandler` が送信経路に入る場合に限られる。 Python の `MediaHandler` サブクラス (trampoline 経由) が `send` callback を呼ぶと、 callback は `impl()->transportSend` を呼ぶため、 GIL を保持したまま送信経路の内部ロックを取る構造が残る。 [[0032-bug-fix-send-gil-deadlock]] と同じ循環待ちに至り得る。

## 優先度根拠

- [[0033-bug-fix-request-keyframe-gil-release]] の磨き上げで、 trampoline 経由の経路が GIL 解放の対象外であることが判明した
- Python で `MediaHandler` をサブクラス化して送信経路に介入する使い方は公開 API として可能であり、 `PliHandler` を Python で実装する例がある
- binding 側の修正だけでは解消できず、 設計判断が要る

## 現状

- `PyMediaHandlerImpl` (src/bind_libdatachannel.cpp) の `request_keyframe` / `request_bitrate` / `incoming` / `outgoing` は nanobind の trampoline で、 GIL を取得して Python のメソッドを呼ぶ
- `Track.request_keyframe()` を GIL 解放下で実行しても、 chain の途中に Python handler があると、 その handler が `gil_scoped_acquire` してから Python コードを実行する
- Python 側で `send(messages)` を呼ぶと callback は `impl()->transportSend` のため、 GIL を保持したまま `DtlsSrtpTransport::sendMedia` (`sendMutex`) を取る
- 受信経路の Python callback も GIL を必要とするため、 ロック順序が絡むと恒久デッドロックし得る (再現は未確認)

## 設計方針

- 案を比較して決める
  - 案 A: Python handler に渡す `send` callback を、 GIL を解放してから呼ぶラッパーにする (`nb::gil_scoped_release` を持つ C++ ラムダで包む)
  - 案 B: `MediaHandler` の trampoline 全体を GIL 解放下で実行する。 Python のオーバーライドを呼ぶ時点で GIL を再取得する必要があるため、 実現可能性を確認する
  - 案 C: Python handler 経由の送信は GIL 保持のままとする制約を明記し、 利用者に C++ handler の利用を求める
- どの案を採るかは、 `send` callback が Python から呼ばれる頻度と、 callback 内で Python オブジェクトに触れる必要があるかを実測してから決める
- 再現手順 (タイミング依存のため、 送信中フラグで callback の実行区間を判定する方式) を tests/ に追加できるか検討する

## 完了条件

- 採用した案に応じて、 Python handler 経由の送信で GIL を保持したまま送信経路に入らないことが検証できる
- 既存の `PyMediaHandler` の動作 (callback の呼び出し順序、 戻り値) が変わらないこと
- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に変更内容が記録されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- libdatachannel v0.24.0: `source/src/mediahandler.cpp`、 `source/src/impl/track.cpp` (`transportSend`)、 `source/src/impl/dtlssrtptransport.cpp` (`sendMedia` の `sendMutex`)
- nanobind: `include/nanobind/stl/function.h` (`pyfunc_wrapper_t::operator()` が `gil_scoped_acquire` する)
- 関連 issue: [[0032-bug-fix-send-gil-deadlock]] / [[0033-bug-fix-request-keyframe-gil-release]]
