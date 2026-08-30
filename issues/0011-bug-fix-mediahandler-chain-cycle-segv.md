# MediaHandler chain に cycle を作ると即 SEGV する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-mediahandler-chain-cycle-segv
- Polished: {YYYY-MM-DD}

## 目的

add_to_chain は cycle を阻止しないため、`h2.add_to_chain(h1)` や `h.add_to_chain(h)` という 1 行の誤用で `MediaHandler::last()` の無限再帰に到達し、プロセスが stack overflow で落ちる。さらに shared_ptr の cycle としてリークも残る。binding 側で cycle を検出して例外にする。

## 優先度根拠

- 実測: 自己参照・相互参照のどちらも add_to_chain 呼び出しの時点で即 SIGSEGV
- tests/test_mediahandler.py は一方向の chain しか試しておらず、この誤用が未検証
- MediaHandler chain は映像送信の標準的な構成要素であり、誤用時に落ちるのは debug 困難

## 現状

再現手順:

```python
from libdatachannel import MediaHandler

h1 = MediaHandler()
h2 = MediaHandler()
h1.add_to_chain(h2)
h2.add_to_chain(h1)  # cycle → SIGSEGV
```

- `bind_mediahandler` の add_to_chain / set_next は `shared_ptr<MediaHandler>` を制約なしで受け取る
- libdatachannel の `MediaHandler::last()` (`src/mediahandler.cpp`) は `next()` の再帰で終端を探すため、cycle があると戻らない
- `MediaHandler::mNext` は shared_ptr 保持のため、cycle はメモリリークとしても残る

## 設計方針

- binding 側の add_to_chain / set_next で「handler == self」および「次チェーンの先頭から自身が到達可能か」をチェックし、cycle を検出したら例外を投げる
- チェーン長の上限を実装して過度な探索を避けるか、到達チェックを堅牢にする
- cycle 検出のテストを追加する

## 完了条件

- cycle を作ろうとすると例外になること (SEGV しないこと)
- cycle 検出のテストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_mediahandler` 内の add_to_chain / set_next (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/mediahandler.cpp` (`MediaHandler::addToChain`、`MediaHandler::last`)
