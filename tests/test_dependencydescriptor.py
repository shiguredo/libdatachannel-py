"""DependencyDescriptorWriter のテスト"""

import gc
import sys

import pytest

from libdatachannel import (
    DecodeTargetIndication,
    DependencyDescriptorContext,
    DependencyDescriptorWriter,
    FrameDependencyTemplate,
)


def make_context() -> DependencyDescriptorContext:
    """writer が扱える最小の context を作る

    同じ内容の FrameDependencyTemplate を structure と descriptor の両方に設定する
    ことで、 find_best_template がテンプレートを見つけられるようにする。
    """
    context = DependencyDescriptorContext()
    context.structure.decode_target_count = 1
    context.structure.chain_count = 1
    context.structure.decode_target_protected_by = [0]

    template = FrameDependencyTemplate()
    template.decode_target_indications = [DecodeTargetIndication.Required]
    context.structure.templates = [template]

    context.descriptor.dependency_template = template
    context.descriptor.frame_number = 1
    return context


def test_writer_size_and_write_to() -> None:
    """get_size_bits / get_size / write_to が同じ内容を返すこと

    start_of_frame と end_of_frame が真、 template id が 0、 frame number が 1 の
    24 bit (3 byte) の descriptor になる。
    """
    context = make_context()
    writer = DependencyDescriptorWriter(context)

    assert writer.get_size_bits() == 24
    assert writer.get_size() == 3
    assert bytes(writer.write_to()).hex() == "c00001"


def test_writer_raises_without_matching_template() -> None:
    """テンプレートが無い context では RuntimeError になること

    libdatachannel の find_best_template が失敗するため。 寿命の問題を再現するには
    テンプレートを持つ有効な context が要る (このテストはその対照)。
    """
    writer = DependencyDescriptorWriter(DependencyDescriptorContext())

    with pytest.raises(RuntimeError):
        writer.get_size_bits()


def test_writer_outlives_context() -> None:
    """context を破棄しても writer が正しく動くこと

    writer は context 自体ではなく context のメンバへの参照を保持するため、
    context を生存させないと解放済みメモリを読む。 解放済みメモリの内容は保証されず
    例外にならない場合もあるため、 決定的な検出は
    test_writer_keeps_context_alive の参照カウントで行う。
    """
    writer = DependencyDescriptorWriter(make_context())
    gc.collect()

    assert writer.get_size_bits() == 24
    assert writer.get_size() == 3
    assert bytes(writer.write_to()).hex() == "c00001"


def test_writer_keeps_context_alive() -> None:
    """writer が context を生存させ、 writer の破棄で解放されること

    参照カウントで確認する。 nanobind のクラスは `nb::is_weak_referenceable()` を
    付けていないため weakref を作成できない。 単一スレッドのテストなので参照カウントは
    即時に反映される。
    """
    context = make_context()
    baseline = sys.getrefcount(context)
    writer = DependencyDescriptorWriter(context)
    gc.collect()
    assert sys.getrefcount(context) == baseline + 1

    del writer
    gc.collect()
    # 循環参照になっていなければ参照カウントが元に戻る
    assert sys.getrefcount(context) == baseline
