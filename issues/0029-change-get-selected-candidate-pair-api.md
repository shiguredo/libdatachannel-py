# get_selected_candidate_pair を Python の慣習に沿った API に変更する

- Priority: Low
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/change-get-selected-candidate-pair
- Polished: {YYYY-MM-DD}

## 目的

PeerConnection.get_selected_candidate_pair は C++ の out パラメータ API (`bool getSelectedCandidatePair(Candidate*, Candidate*)`) をそのまま公開しており、渡した 2 つの Candidate の内容が破壊的に書き換わる。型スタブから挙動が読めず、テストも 0 件。Python の慣習 (入力は引数、出力は戻り値) に沿った API に変更する。

## 優先度根拠

- 後方互換のない変更 (change) になるため、利用者の移行考慮が必要。機能不全ではないため Low
- 現状の API はテスト 0 件で、挙動が仕様として固定されていない

## 現状

- `bind_peerconnection` は libdatachannel の `PeerConnection::getSelectedCandidatePair(Candidate*, Candidate*)` を直接バインドする
- 実装は引数の指し先を書き換える out パラメータ (`src/impl/icetransport.cpp` 内で `*local = ...` の形で代入)
- 型スタブは `def get_selected_candidate_pair(self, arg0: Candidate, arg1: Candidate, /) -> bool` のみで、out パラメータであることが型情報から分からない
- テスト 0 件

## 設計方針

- ラムダでラップし、内部で Candidate を確保して `Optional[tuple[Candidate, Candidate]]` を返す形に変更する (未選択の場合は None)
- 後方互換のない変更のため CHANGES.md は [CHANGE] で記載する
- 接続確立後の selected pair を検証するテストを追加する
- 型スタブを再生成する

## 完了条件

- 新 API が動作し、テストで挙動が固定されていること
- 型スタブが再生成されていること
- CHANGES.md に [CHANGE] として記載されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_peerconnection` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/peerconnection.hpp` (`PeerConnection::getSelectedCandidatePair`)
