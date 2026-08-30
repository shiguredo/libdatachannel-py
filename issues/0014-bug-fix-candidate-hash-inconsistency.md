# Candidate の __eq__ と __hash__ が不整合で dict / set で等価扱いされない

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-candidate-hash-inconsistency
- Polished: {YYYY-MM-DD}

## 目的

Candidate は __eq__ と __ne__ を binding しているが __hash__ を定義していないため、`==` が True の 2 つの Candidate が hash 不一致となり、dict の key や set で等価扱いされない。Python のハッシュ規約 (a == b なら hash(a) == hash(b)) に反する静かな誤動作を解消する。

## 優先度根拠

- 実測: `==` が True でも hash が異なり、`{c1: ...}.get(c2)` が None、`len({c1, c2})` が 2 になる
- candidate 文字列で正規化した比較結果と hash が矛盾する状態は、利用者にとって原因の特定が困難

## 現状

再現手順:

```python
from libdatachannel import Candidate

s = "candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host"
c1 = Candidate(s)
c2 = Candidate(s)

assert c1 == c2          # True
assert hash(c1) == hash(c2)  # False → 規約違反
assert len({c1, c2}) == 1    # 2 になる
```

- `bind_candidate` は `nb::self == nb::self` と `nb::self != nb::self` をバインドするが、__hash__ を定義していない
- Python の規約では __eq__ を定義したクラスは __hash__ も定義すべき

## 設計方針

- `candidate()` 文字列ベースで __hash__ を定義する
- 併せて __eq__ の比較対象 (mid を含むか等) の仕様を確認し、テストで固定する

## 完了条件

- `==` が True の Candidate が同一 hash を持つこと (テストで検証)
- dict / set での等価性のテストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_candidate` (src/bind_libdatachannel.cpp)
