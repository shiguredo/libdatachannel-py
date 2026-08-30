# 公開 binding API のうちテストが 0 件の領域にテストを追加する

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-add-missing-binding-tests
- Polished: {YYYY-MM-DD}

## 目的

binding で公開されている API のうち、送受信系・MediaHandler 系・Packetizer 系を中心にテストが 0 件の領域が多い。主要な領域にテストを追加し、回帰を検知できる状態にする。

## 優先度根拠

- バグ修正 ([[0006-bug-fix-send-size-out-of-bounds-read]]、[[0008-bug-fix-non-owning-reference-lifetime]] 等) の検証にはこれらの領域のテストが前提になる
- CI テスト実行 ([[0022-test-enable-ci-tests]]) が有効になった際に効果が出る

## 現状

- テストが 0 件の主要領域:
  - Channel 系: receive / peek / available_amount / on_available / on_buffered_amount_low / set_buffered_amount_low_threshold / buffered_amount / on_message 2 引数版
  - 送受信系: Track.send_frame (両オーバーロード)、Track.on_frame、Track.request_keyframe / request_bitrate
  - MediaHandler 系: PacingHandler / RembHandler / RtcpNackResponder (単体) / RtcpReceivingSession (incoming / request_keyframe / request_bitrate / get_sync_timestamps) / RtcpSrReporter (last_reported_timestamp) / chain_media_handler / get_media_handler
  - Packetizer / Depacketizer 系: AV1RtpPacketizer / H264RtpPacketizer / H265RtpPacketizer / RtpDepacketizer / H264RtpDepacketizer / H265RtpDepacketizer
  - Message 系: make_message_with_frame_info / message_size / dscp
  - PeerConnection 系: bytes_sent / bytes_received / rtt / clear_stats / get_selected_candidate_pair / remote_fingerprint / negotiation_needed / has_media / on_data_channel / create_offer / create_answer
  - IceUdpMuxListener (全体)
- DependencyDescriptorWriter のテストも 0 件 ([[0009-bug-fix-dependency-descriptor-writer-dangling]] で追加)
- モック / スタブ禁止の規約があるため、実 C++ オブジェクトでの検証 (PeerConnection 間接続等) で構成する

## 設計方針

- 優先度順に (1) Channel 系の送受信 (2) Track の send_frame / on_frame (3) RtcpReceivingSession / MediaHandler 系 (4) Packetizer / Depacketizer 系の 4 バッチで追加する
- 実オブジェクトを使うテストとして、PeerConnection 間の track 接続 (tests/test_peerconnection.py のヘルパー化 [[0024-refactor-deduplicate-peerconnection-tests]] を再利用) を前提にする
- テストファイルの命名は既存の test_<module>.py に従う
- hypothesis を使った PBT は本 issue のスコープ外 (SDP round-trip、NalUnit ヘッダのビット操作等が候補であり、必要になった時点で別 issue として起票する)

## 完了条件

- Channel 系・Track 系・MediaHandler 系・Packetizer / Depacketizer 系のテストが追加されていること
- 全テストが PASS すること
- 追加テストがモック / スタブを使っていないこと
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: tests/ 配下
- 関連 issue: [[0006-bug-fix-send-size-out-of-bounds-read]]、[[0008-bug-fix-non-owning-reference-lifetime]]、[[0009-bug-fix-dependency-descriptor-writer-dangling]]、[[0022-test-enable-ci-tests]]、[[0024-refactor-deduplicate-peerconnection-tests]]
