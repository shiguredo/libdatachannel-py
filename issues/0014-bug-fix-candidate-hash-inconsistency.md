# Candidate の __eq__ と __hash__ が不整合で dict / set で等価扱いされない

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-candidate-hash-inconsistency
- Polished: 2026-10-10

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

assert c1 == c2  # True
assert hash(c1) == hash(c2)  # False → 規約違反
assert len({c1, c2}) == 1  # 2 になる
```

- `bind_candidate` は `nb::self == nb::self` と `nb::self != nb::self` をバインドするが、__hash__ を定義していない
- Python の規約では __eq__ を定義したクラスは __hash__ も定義すべき

## 設計方針

- `bind_candidate` の `__eq__` を SDP の candidate 行 (`candidate()`) の比較に変え、 `__hash__` を同じ値から作り、 `__ne__` は明示的にバインドしない (Python が `__eq__` の否定として導出する)
  - 現在の `nb::self == nb::self` は libdatachannel の `Candidate::operator==` (foundation / service / node の比較) をそのまま使っており、 candidate 行で表される値 (priority や type) が違っても等しいと判定される。 `__hash__` を candidate 行から作ると「等しいのに hash が違う」状態が残るため、 比較と hash の対象を candidate 行に揃える
  - `Operator!=` は foundation のみを比較しており `==` と非対称 (同じ foundation で node が違うと `==` も `!=` も False)。 明示的な `__ne__` のバインドを外して Python の導出に任せる
- `__eq__` は `Candidate` 以外の object と比較されたときに例外を投げず False を返す
- テストで固定する内容: 同じ candidate 行の 2 つが `==` かつ同一 hash であること、 `!=` が `==` の否定であること、 dict / set で 1 つに畳まれること、 candidate 行が違えば等しくないこと

## 完了条件

- `==` が True の Candidate が同一 hash を持つこと (テストで検証)
- `!=` が `==` の否定になっていること (同じ foundation で node だけ違う場合を含む)
- dict / set での等価性のテストが追加されていること
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること (`uv sync` は `make develop` で入れた拡張モジュールを削除するため使わない)
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_candidate` (src/bind_libdatachannel.cpp)
