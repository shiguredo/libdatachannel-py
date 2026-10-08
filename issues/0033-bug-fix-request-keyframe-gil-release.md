# Track.request_keyframe() と request_bitrate() が GIL を保持したまま送信経路に入る

- Priority: Medium
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-request-keyframe-gil-release
- Polished: {YYYY-MM-DD}

## 目的

`Track.request_keyframe()` / `Track.request_bitrate()` も送信経路に入る API だが、 GIL を保持したまま実行される。 [[0032-bug-fix-send-gil-deadlock]] で GIL 解放を付与した send 系と同じ構造を残さないようにする。

## 優先度根拠

- [[0032-bug-fix-send-gil-deadlock]] で `send` / `send_frame` に `nb::call_guard<nb::gil_scoped_release>()` を付与したが、 同じ送信経路に入る `request_keyframe` / `request_bitrate` は未対応のまま残っている
- 送信経路のロックを保持したまま GIL 待ちに入ると、 受信経路の callback が GIL を取得できず恒久デッドロックに至り得る (再現は未確認)

## 現状

- `bind_track` の `.def("request_keyframe", &Track::requestKeyframe)` と `.def("request_bitrate", &Track::requestBitrate, "bitrate"_a)` に call_guard が付いていない
- `Track::requestKeyframe()` は `impl()->transportSend(m)` を callback として渡す (`_deps/libdatachannel/v0.24.0/source/src/track.cpp`)。 `impl::Track::transportSend` (`source/src/impl/track.cpp`) は `DtlsSrtpTransport::sendMedia` (`source/src/impl/dtlssrtptransport.cpp`、 `sendMutex` を取得) 経由で ICE / UDP の送信経路に入る
- `Track::requestBitrate()` も同じ経路を使う
- この経路は GIL を保持したまま実行されるため、 受信経路の Python callback とロック順序が絡むとデッドロックし得る

## 設計方針

- `request_keyframe` / `request_bitrate` に `nb::call_guard<nb::gil_scoped_release>()` を付与する
- 理由コメントは `bind_channel` 直前の既存コメントを参照できるようにする
- [[0032-bug-fix-send-gil-deadlock]] で追加した regression テストと同じループバック構成を使い、 `request_keyframe()` の実行中に PLI の callback が実行されることを検証するテストを追加する

## 完了条件

- `Track.request_keyframe()` / `Track.request_bitrate()` が GIL 解放下で実行されること
- ループバック構成で、 `request_keyframe()` の実行中に受信側の callback が実行されることを検証するテストを追加し PASS すること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_track` の `request_keyframe` / `request_bitrate` (src/bind_libdatachannel.cpp)
- 関連 issue: [[0032-bug-fix-send-gil-deadlock]]、 [[0005-bug-fix-destructor-callback-deadlock]]
