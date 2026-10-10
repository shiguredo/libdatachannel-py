# test_concurrent_datachannel_creation が signaling state の競合で断続的に失敗する

- Priority: Medium
- Created: 2026-08-30
- Completed: 2026-10-10
- Branch: feature/fix-flaky-concurrent-datachannel
- Polished: 2026-10-10

## 目的

tests/test_free_threading.py の test_concurrent_datachannel_creation は 4 スレッドから同一 PeerConnection へ create_data_channel を実行するが、libdatachannel の auto negotiation が setLocalDescription(Offer) を呼ぶため、2 番手以降が std::logic_error (Wrong signaling state) で失敗し得る。テストとして確定的に失敗する可能性のある設計を修正する。

## 優先度根拠

- 失敗が仕様として起きうる (スレッドの競合タイミング次第)。free-threading 環境での flaky テストは CI テスト実行 ([[0022-test-enable-ci-tests]]) の邪魔になる
- このテストはスレッド内例外を収集しないため、失敗時に原因が分からない

## 現状

- このテストは barrier で 4 スレッドを同期させ、同一 PeerConnection に同時 create_data_channel する
- libdatachannel の `PeerConnection::createDataChannel` (`src/peerconnection.cpp`) は auto negotiation 有効かつ signalingState == Stable で negotiationNeeded() を満たすと setLocalDescription(Offer) を呼ぶ
- setLocalDescription は signalingMutex で直列化されるが、2 番手以降は signalingState != Stable で std::logic_error を投げる
- このテストはスレッド内例外を errors に収集しないため、例外は results のカウント不足として現れる (原因が確認できない)

## 設計方針

- テストの目的 (並列 create_data_channel が binding を壊さないこと) を保ちつつ、disable_auto_negotiation = True の PeerConnection を使い、setLocalDescription 経路を排除する
- スレッド内例外を errors に収集するパターンに統一する (tests/test_free_threading.py の他テストとの統一)
- 同一オブジェクトへの並列操作の検証の拡充は [[0026-test-add-missing-binding-tests]] で行う

## 完了条件

- テストが断続的に失敗しないこと
- 失敗時に原因 (例外) が確認できること
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること (`uv sync` は `make develop` で入れた拡張モジュールを削除するため使わない)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `tests/test_free_threading.py`
  - `test_concurrent_datachannel_creation` の `PeerConnection` を `disable_auto_negotiation = True` で作るようにした。 auto negotiation が有効なままだと `create_data_channel()` が `setLocalDescription(Offer)` を呼び、 2 番手以降が signaling state の競合で `std::logic_error` になるため
  - スレッド内の例外を `errors` に集めてテストで検証するようにした (同ファイルの他のテストと同じ書き方)。 これまでは例外が `results` の件数不足として現れるだけで、 原因が分からなかった
- 検証
  - `tests/test_free_threading.py` は free-threading ビルドでのみ実行されるため、 通常ビルドでは skip される (12 skipped)。 実行結果は CI の free-threading leg で確認する
  - `prek run --all-files pytest` と `prek run --all-files ty` が PASS
  - `/review-diff-code` の致命的 / 重要指摘が 0 件

## 参考

- 対象シンボル: `test_concurrent_datachannel_creation` (tests/test_free_threading.py)
- libdatachannel v0.24.0: `src/peerconnection.cpp` (`PeerConnection::createDataChannel`、`PeerConnection::setLocalDescription`)
- 関連 issue: [[0022-test-enable-ci-tests]]、[[0026-test-add-missing-binding-tests]]
