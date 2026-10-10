# DataChannel.close() / Track.close() が GIL を保持したまま送信経路と同じ mutex を取る

- Priority: Medium
- Created: 2026-10-09
- Completed: 2026-10-10
- Branch: feature/fix-close-gil-release
- Polished: 2026-10-10

## 目的

`WebSocket.close()` には GIL 解放 (`nb::call_guard<nb::gil_scoped_release>()`) が付いているが、 `DataChannel.close()` と `Track.close()` には付いていない。 `DataChannel::close()` は `SctpTransport::closeStream()` 経由で送信側と同じ mutex を取るため、 [[0032-bug-fix-send-gil-deadlock]] / [[0033-bug-fix-request-keyframe-gil-release]] と同じ循環待ちが残り得る。

## 優先度根拠

- [[0033-bug-fix-request-keyframe-gil-release]] のレビューで、 送信経路の call_guard 全数調査から漏れとして検出された
- `_deps/libdatachannel/v0.24.0/source/src/impl/sctptransport.cpp` の `closeStream()` は `mSendMutex` を取って Reset を enqueue する (送信経路と同じ mutex)
- `WebSocket.close()` ([[0002-bug-fix-websocket-destructor-gil-release]]) には付いており、 扱いが不整合

## 現状

- `bind_datachannel` の `.def("close", ...)` と `bind_track` の `.def("close", ...)` に call_guard が無い
- `DataChannel::close()` → `impl()->close()` → `SctpTransport::closeStream()` (`mSendMutex` を取得)
- `Track::close()` は送信経路の mutex は取らないが、 `resetCallbacks()` で callback の mutex を取る。 callback の実行中は同じ mutex が保持されるため、 こちらも同じ循環待ちになり得る (実装は `source/src/impl/track.cpp` の `Track::close()`)
- 恒久デッドロックの再現は未確認

## 設計方針

- `Track::close()` / `DataChannel::close()` の実装を確認し、 送信経路と同じ mutex を取る場合に `nb::call_guard<nb::gil_scoped_release>()` を付与する
- 付与した場合、 close 完了を待つ処理 (WebSocket の close のように状態遷移を待つ実装) があるかどうかを確認し、 GIL 解放中の待機が安全かを検証する
- 理由コメントは `bind_channel` 直前の既存コメントを参照する

## 完了条件

- 対象となる close 系 binding が GIL 解放下で実行されること (または対象外である根拠が実装で示されていること)
- close の完了待ちが必要な場合は、 [[0002-bug-fix-websocket-destructor-gil-release]] と同じ扱い (状態が Closed になるまで待つ、 タイムアウトで `RuntimeWarning`) を検討する
- `make develop` で拡張モジュールをインストールしたうえで、 `prek run --all-files pytest` が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `DataChannel.close()` に `nb::call_guard<nb::gil_scoped_release>()` を付けた。 `SctpTransport::closeStream()` が送信経路と同じ mutex を取るため、 GIL を保持したまま待つと、 送信経路の Python callback が GIL を取れずに循環待ちになる
  - `Track.close()` にも同じ call_guard を付けた。 送信経路の mutex は取らないが、 `resetCallbacks()` で callback の mutex を取り、 callback の実行中は同じ mutex が保持されるため、 こちらも同じ循環待ちになり得る (実装は `source/src/impl/track.cpp` の `Track::close()`)
- `tests/test_peerconnection.py`
  - `test_data_channel_close_releases_gil` を追加した。 GIL を待つ thread が `close()` の呼び出し中に進行するかで解放を判定する (`test_request_media_control_releases_gil` と同じ方式)。 解放窓は µs 程度のため、 50 ms のあいだ呼び続けて判定する
- `CHANGES.md`
  - `## develop` に `[FIX]` として記録した
- 検証
  - `test_data_channel_close_releases_gil` が 3 回連続で pass (1.14s / 3.14s / 2.13s)
  - 全体 175 passed / 12 skipped / 1 deselected、 `prek run --all-files ty` が PASS
  - `/review-diff-code` の致命的 / 重要指摘が 0 件
- 補足: テストの後始末 (`gc.collect()`) を入れないと、 閉じた DataChannel が残って nanobind のリーク警告がインタプリタ終了時に出る

## 参考

- libdatachannel v0.24.0: `source/src/impl/sctptransport.cpp` (`closeStream`)、 `source/src/impl/track.cpp`、 `source/src/channel.cpp`
- 関連 issue: [[0032-bug-fix-send-gil-deadlock]] / [[0033-bug-fix-request-keyframe-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]]
