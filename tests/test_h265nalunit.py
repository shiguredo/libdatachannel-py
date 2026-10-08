import pytest

from libdatachannel import (
    H265NalUnit,
    H265NalUnitFragment,
    H265NalUnitFragmentHeader,
    H265NalUnitHeader,
)


def test_h265_nalu_header_bits():
    h = H265NalUnitHeader()
    h.set_forbidden_bit(True)
    h.set_unit_type(33)
    h.set_nuh_layer_id(3)
    h.set_nuh_temp_id_plus1(5)

    assert h.forbidden_bit() is True
    assert h.unit_type() == 33
    assert h.nuh_layer_id() == 3
    assert h.nuh_temp_id_plus1() == 5


def test_h265_nalu_fragment_header_bits():
    h = H265NalUnitFragmentHeader()
    h.set_start(True)
    h.set_end(True)
    h.set_unit_type(49)

    assert h.is_start() is True
    assert h.is_end() is True
    assert h.unit_type() == 49


def test_h265_nalu_basic_payload_handling():
    nalu = H265NalUnit(16)
    nalu.set_forbidden_bit(True)
    nalu.set_unit_type(32)
    nalu.set_nuh_layer_id(2)
    nalu.set_nuh_temp_id_plus1(4)

    assert nalu.forbidden_bit() is True
    assert nalu.unit_type() == 32
    assert nalu.nuh_layer_id() == 2
    assert nalu.nuh_temp_id_plus1() == 4

    data = b"\x01\x02\x03\x04"
    nalu.set_payload(data)
    assert nalu.payload() == data


def test_h265_fragment_instance_behavior():
    frag = H265NalUnitFragment(H265NalUnitFragment.FragmentType.Start, True, 1, 1, 32, b"\xaa\xbb")

    assert frag.type() == H265NalUnitFragment.FragmentType.Start
    assert frag.unit_type() == 32
    assert frag.payload() == b"\xaa\xbb"

    frag.set_fragment_type(H265NalUnitFragment.FragmentType.End)
    frag.set_unit_type(34)
    frag.set_payload(b"\xcc\xdd")
    assert frag.type() == H265NalUnitFragment.FragmentType.End
    assert frag.unit_type() == 34
    assert frag.payload() == b"\xcc\xdd"


def test_h265_nalu_rejects_short_buffer():
    """ヘッダサイズ (2 バイト) 未満のバッファでは H265NalUnit を生成できないこと

    生成できてしまうと、 ヘッダの 2 バイト目を読む nuh_layer_id() などが範囲外を
    読み書きする。
    """
    # size 版 (including_header=True) は 2 バイト必要
    with pytest.raises(ValueError):
        H265NalUnit(0)
    with pytest.raises(ValueError):
        H265NalUnit(1)

    # bytes 版も同じ下限
    with pytest.raises(ValueError):
        H265NalUnit(b"")
    with pytest.raises(ValueError):
        H265NalUnit(b"\x00")


def test_h265_nalu_accepts_header_only_buffer():
    """including_header=False はヘッダサイズが加算されるため size 0 でも生成できること"""
    # 2 バイトのヘッダのみが確保される
    assert H265NalUnit(0, False).payload() == b""


def test_h265_nalu_rejects_size_overflow():
    """including_header=False でヘッダサイズの加算が桁あふれする size を拒否すること

    桁あふれすると空のバッファが確保され、 ヘッダアクセスで SIGSEGV する。
    """
    with pytest.raises(ValueError):
        H265NalUnit(2**64 - 1, False)


def test_h265_nalu_bytes_and_set_payload():
    """bytes 版の生成と、 最小バッファでの set_payload() が動作すること"""
    # bytes 版は data をそのままバッファにする (ヘッダ 2 バイトを除いた部分が payload)
    # ヘッダサイズちょうどの data でも生成できる
    assert H265NalUnit(b"\x00\x01").payload() == b""
    assert H265NalUnit(b"abc").payload() == b"c"

    # ヘッダのみのバッファでも set_payload() が範囲外を読み書きしない
    unit = H265NalUnit(2)
    unit.set_payload(b"a")
    assert unit.payload() == b"a"


def test_h265_nalu_header_accessors_with_minimum_buffer():
    """ヘッダサイズちょうどのバッファでヘッダアクセスが範囲外を読み書きしないこと

    ヘッダ 2 バイトのうち 2 バイト目を読み書きするメソッドが本来の範囲外アクセスの
    経路だった。
    """
    unit = H265NalUnit(2)

    # 2 バイト目を読むメソッドが範囲外を読まない
    assert unit.forbidden_bit() is False
    assert unit.unit_type() == 0
    assert unit.nuh_layer_id() == 0
    assert unit.nuh_temp_id_plus1() == 0

    # 2 バイト目に書くメソッドが範囲外を書かない
    unit.set_nuh_layer_id(3)
    unit.set_nuh_temp_id_plus1(5)

    assert unit.nuh_layer_id() == 3
    assert unit.nuh_temp_id_plus1() == 5
