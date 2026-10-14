# DependencyDescriptorWriter が context の内部メンバへの参照のみを保持する

- Priority: High
- Created: 2026-08-30
- Completed: 2026-10-09
- Branch: feature/fix-dependency-descriptor-writer-dangling
- Polished: 2026-10-09

## 目的

DependencyDescriptorWriter は DependencyDescriptorContext 自体ではなく、context のメンバ (structure / descriptor) への const 参照を保持する。binding が keep_alive を持たないため、context が破棄されると writer が dangling になり、get_size() / write_to() が壊れた値を読む。実測で再現する。

## 優先度根拠

- 実測: 有効な context を作る一時関数から writer だけを返した場合、 context 破棄後の `get_size_bits()` は `RuntimeError: No matching template found` になる。 ただし use-after-free のため結果は不定で、 例外にならず誤った値になる場合もある
- なお、 既定の空の context でも同じ例外になる (テンプレートが見つからないため)。 寿命の問題を再現するには structure / descriptor を設定した有効な context が要る
- 本リポジトリの公開 API として到達可能であり、テストが 1 件もない

## 現状

再現手順:

```python
from libdatachannel import (
    DecodeTargetIndication,
    DependencyDescriptorContext,
    DependencyDescriptorWriter,
    FrameDependencyTemplate,
)


def make_writer():
    context = DependencyDescriptorContext()
    context.structure.decode_target_count = 1
    context.structure.chain_count = 1
    context.structure.decode_target_protected_by = [0]
    template = FrameDependencyTemplate()
    template.decode_target_indications = [DecodeTargetIndication.Required]
    context.structure.templates = [template]
    context.descriptor.dependency_template = template
    return DependencyDescriptorWriter(context)


writer = make_writer()  # context は関数の終了時に破棄される
writer.get_size_bits()  # 解放済みメモリを読む → RuntimeError
```

- `bind_dependencydescriptor` の DependencyDescriptorWriter は `nb::init<const DependencyDescriptorContext&>()` で context を受け、keep_alive を付けていない
- libdatachannel の `DependencyDescriptorWriter` (`include/rtc/dependencydescriptor.hpp`) は `const FrameDependencyStructure &mStructure` と `const DependencyDescriptor &mDescriptor` を保持する (context 自体は保持しない)。 `context.activeChains` だけは構築時に値コピーされる

## 設計方針

- コンストラクタに `nb::keep_alive<1, 2>()` を付け、 writer が context を生存させる (採用)
  - writer は const 参照のみを保持する不変オブジェクトで、 context を後から差し替える用途が無いため、 binding 側で shared_ptr を持つ必要はない
  - `DependencyDescriptorContext` のメンバを書き換えても参照自体は有効なままで、 寿命の問題は context の破棄だけが原因
- context を生存させても、 writer を破棄した後に context が残る (リーク) ことが無いことを確認する

## 解決方法

- `src/bind_libdatachannel.cpp` の `DependencyDescriptorWriter.__init__` に `nb::keep_alive<1, 2>()` を追加し、 writer が context を生存させるようにした
  - writer は const 参照のみを保持する不変オブジェクトで、 context を後から差し替える用途が無いため、 binding 側で shared_ptr を持つ必要はない
- テスト `tests/test_dependencydescriptor.py` を追加した (このモジュールにはテストが 1 件も無かった)
  - 正常系の期待値 (24 bit / 3 byte / `c00001`)、 テンプレートが無い場合の `RuntimeError`、 context 破棄後も正しい値になること、 writer が context を生存させること (参照カウント) を検証する
- `CHANGES.md` の `## develop` に `[FIX]` エントリを追加した
- 実測: 修正前は context 破棄後に `RuntimeError`。 修正後は `tests/test_dependencydescriptor.py` の 4 テストが PASS、 全体で 98 passed / 12 skipped / 1 deselected

## 完了条件

- context を先に破棄しても writer の get_size_bits / get_size / write_to が正しい値を返すこと
- writer が context を生存させること (writer を破棄すれば context の参照カウントが元に戻ること)
- テストが追加されていること (`tests/test_dependencydescriptor.py`: 正常系の期待値 / context 破棄後 / 参照カウント)
- `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_dependencydescriptor` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/dependencydescriptor.hpp` (`DependencyDescriptorWriter`)
