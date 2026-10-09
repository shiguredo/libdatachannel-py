# Track.request_keyframe() と request_bitrate() が GIL を保持したまま送信経路に入る

- Priority: Medium
- Created: 2026-10-08
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-request-keyframe-gil-release
- Polished: 2026-10-09

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
- [[0032-bug-fix-send-gil-deadlock]] で追加したループバック構成を `tests/test_peerconnection.py` の共通ヘルパー (`make_loopback_with_pli`) に切り出し、 `request_keyframe()` の実行中に PLI の callback が実行されることを検証するテストを追加する
  - `Track.request_keyframe()` は media handler chain の `RtcpReceivingSession` などの handler が `send` callback (= `impl()->transportSend`) を呼ぶことで送信経路に入る (`PliHandler` は PLI の受信側で、 送信は行わない)
  - 送った PLI は送信側 (pc1) の `PliHandler` が受信し、 worker thread が GIL を取得して callback を実行する。 タイミング依存のため、 送信中フラグで「実行区間中に callback が動いたか」を判定する ([[0032-bug-fix-send-gil-deadlock]] と同じ方式)
  - 測定区間で呼ぶのは `RtcpReceivingSession` を持つ受信側トラックの `request_keyframe()` で、 真が返ることが送信経路を通った証拠になる (送信側トラックの chain には `requestKeyframe` を実装する handler が無く、 既定実装は false を返すだけである)
- GIL 解放が効くのは C++ の handler が送信経路に入る場合に限る。 Python の `MediaHandler` サブクラスの trampoline から `send()` を呼ぶ経路は、 GIL を保持したまま送信経路に入るため [[0044-bug-fix-python-mediahandler-send-gil]] で扱う

## 完了条件

- `Track.request_keyframe()` / `Track.request_bitrate()` が GIL 解放下で実行されること
- ループバック構成で、 `request_keyframe()` の実行中に PLI を受信した pc1 の `PliHandler` callback が実行されることを検証するテストを追加し PASS すること
- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_track` の `request_keyframe` / `request_bitrate` (src/bind_libdatachannel.cpp)
- 関連 issue: [[0032-bug-fix-send-gil-deadlock]]、 [[0005-bug-fix-destructor-callback-deadlock]]
