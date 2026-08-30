# 言語・表記のプロジェクト規約違反を一括で解消する

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/refactor-fix-language-conventions
- Polished: {YYYY-MM-DD}

## 目的

AGENTS.md の「コメントは全て日本語」「テストメッセージは全て日本語」「全角と半角の間には半角スペース」「例外を握り潰さない」等の規約違反を一括で解消する。

## 優先度根拠

- 規約違反はプロジェクト全体の品質基準を下げる。修正自体は機械的だが複数ファイルにまたがるため、issue として追跡する

## 現状

- テストメッセージが英語: tests/test_free_threading.py の skipif reason と assert メッセージ (GIL should be disabled 等)。一方 test_peerconnection.py の assert メッセージは日本語で、ファイル間でも不統一
- 英語コメント: tests/test_peerconnection.py (Test opening a track 等複数)、tests/test_description.py (Depending on implementation)、tests/test_packetizationconfig.py (static versions 等)、examples/whip.py (Parse Link header 等 14 箇所以上)、examples/whep.py (Running flag 等)
- 全角半角間スペース違反: examples/whip.py (7セグメント風の数字を描画、多くのマイクは1ch、1秒ごとに出力 等 5 箇所)、tests/conftest.py (WebSocketエコー)
- except Exception: pass 6 箇所: examples/whip.py (エンコーダの flush / close の握り潰し)、examples/whep.py (デコーダ close の握り潰し)
- issue 番号がソースに残留: tests/test_peerconnection.py の test_destruct_without_explicit_close のコメント (issues/pending/0005 への言及)。issue 番号の許容される置き場所は issues/ 配下のファイルとコミットメッセージのみ

## 設計方針

- テストメッセージ (assert メッセージ、skip reason) を日本語に統一する
- コメントを日本語化する (固有名詞・識別子を除く)
- 全角と半角の間に半角スペースを入れる
- except Exception: pass は debug ログを残すか、握り潰しの理由をコメントで明記する ([[0027-fmt-fix-lint-typecheck-gate]] の ruff 規約セットとも整合させる)
- issue 番号への言及は、issue 番号を除去して事実 (callback 内の blocking I/O が close の待機を妨げる旨) だけをコメントに残す
- [[0024-refactor-deduplicate-peerconnection-tests]] / [[0025-test-remove-callback-prints]] と着手順序を調整する (重複解消後に言語修正した方が二度手間がない)

## 完了条件

- 上記の規約違反が 0 件になること (grep で検証)
- `uv sync && make test` で全テストが PASS すること
- [[0027-fmt-fix-lint-typecheck-gate]] の ruff 規約セットで lint が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: tests/ 配下、examples/ 配下、tests/conftest.py
- 関連 issue: [[0024-refactor-deduplicate-peerconnection-tests]]、[[0025-test-remove-callback-prints]]、[[0027-fmt-fix-lint-typecheck-gate]]
