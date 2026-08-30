# DependencyDescriptorWriter が context の内部メンバへの参照のみを保持する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-dependency-descriptor-writer-dangling
- Polished: {YYYY-MM-DD}

## 目的

DependencyDescriptorWriter は DependencyDescriptorContext 自体ではなく、context のメンバ (structure / descriptor) への const 参照を保持する。binding が keep_alive を持たないため、context が破棄されると writer が dangling になり、get_size() / write_to() が壊れた値を読む。実測で再現する。

## 優先度根拠

- 実測: `DependencyDescriptorWriter(DependencyDescriptorContext())` のように一時 context を渡した場合、context 破棄後の get_size() が `RuntimeError: No matching template found` を返し、破壊された structure を読んでいる
- 本リポジトリの公開 API として到達可能であり、テストが 1 件もない

## 現状

再現手順:

```python
from libdatachannel import DependencyDescriptorContext, DependencyDescriptorWriter

writer = DependencyDescriptorWriter(DependencyDescriptorContext())
# context は一時オブジェクトで即破棄される
writer.get_size()  # 壊れた structure を読む
```

- `bind_dependencydescriptor` の DependencyDescriptorWriter は `nb::init<const DependencyDescriptorContext&>()` で context を受け、keep_alive を付けていない
- libdatachannel の `DependencyDescriptorWriter` (`include/rtc/dependencydescriptor.hpp`) は `const FrameDependencyStructure &mStructure` と `const DependencyDescriptor &mDescriptor` を保持する (context 自体は保持しない)

## 設計方針

- keep_alive で context を保持させる、または binding 側のラッパーで context を shared_ptr 保持する
- どちらを採用するかは、writer の利用パターン (context を後から差し替える用途があるか) を確認して決める

## 完了条件

- context を先に破棄しても writer が有効に動作すること
- get_size_bits / get_size / write_to のテストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_dependencydescriptor` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/dependencydescriptor.hpp` (`DependencyDescriptorWriter`)
