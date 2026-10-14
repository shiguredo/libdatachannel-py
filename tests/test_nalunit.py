import pytest

from libdatachannel import (
    NalUnit,
    NalUnitFragmentA,
    NalUnitFragmentHeader,
    NalUnitHeader,
    NalUnitStartSequenceMatch,
)


def test_nal_unit_header_bits():
    h = NalUnitHeader()
    h.set_forbidden_bit(True)
    h.set_nri(2)
    h.set_unit_type(5)

    assert h.forbidden_bit() is True
    assert h.nri() == 2
    assert h.unit_type() == 5


def test_nal_unit_fragment_header_bits():
    h = NalUnitFragmentHeader()
    h.set_start(True)
    h.set_end(True)
    h.set_reserved_bit6(True)
    h.set_unit_type(27)

    assert h.is_start() is True
    assert h.is_end() is True
    assert h.reserved_bit6() is True
    assert h.unit_type() == 27


def test_nal_unit_basic_fields():
    n = NalUnit(10)  # 10 bytes
    n.set_forbidden_bit(True)
    n.set_nri(3)
    n.set_unit_type(7)

    assert n.forbidden_bit() is True
    assert n.nri() == 3
    assert n.unit_type() == 7

    payload = b"\x11\x22\x33"
    n.set_payload(payload)
    assert n.payload() == payload


def test_nal_fragment_creation_and_fields():
    payload = b"\x01\x02\x03\x04"
    frag = NalUnitFragmentA(NalUnitFragmentA.FragmentType.Start, True, 2, 7, payload)

    assert frag.type() == NalUnitFragmentA.FragmentType.Start
    assert frag.unit_type() == 7
    assert frag.payload() == payload

    frag.set_unit_type(5)
    frag.set_fragment_type(NalUnitFragmentA.FragmentType.End)
    frag.set_payload(b"\xaa\xbb")

    assert frag.unit_type() == 5
    assert frag.type() == NalUnitFragmentA.FragmentType.End
    assert frag.payload() == b"\xaa\xbb"


def test_start_sequence_match_succ():
    result = NalUnit.start_sequence_match_succ(
        NalUnitStartSequenceMatch.FirstZero, b"\x00"[0], NalUnit.Separator.ShortStartSequence
    )
    assert isinstance(result, NalUnitStartSequenceMatch)


def test_nal_unit_rejects_short_buffer():
    """ヘッダサイズ (1 バイト) 未満のバッファでは NalUnit を生成できないこと

    生成できてしまうと、 ヘッダアクセスや set_payload() が範囲外を読み書きして
    SIGSEGV する (Release ビルドでは assert が消えるため libdatachannel 本体の
    防御が働かない)。
    """
    # size 版 (including_header=True) は 1 バイト必要
    with pytest.raises(ValueError):
        NalUnit(0)
    with pytest.raises(ValueError):
        NalUnit(0, True, NalUnit.Type.H265)

    # bytes 版も同じ下限
    with pytest.raises(ValueError):
        NalUnit(b"")


def test_nal_unit_accepts_header_only_buffer():
    """including_header=False はヘッダサイズが加算されるため size 0 でも生成できること"""
    # H264 は 1 バイト、 H265 は 2 バイトが確保される
    assert NalUnit(0, False).payload() == b""
    assert NalUnit(0, False, NalUnit.Type.H265).payload() == b"\x00"


def test_nal_unit_rejects_size_overflow():
    """including_header=False でヘッダサイズの加算が桁あふれする size を拒否すること

    桁あふれすると空のバッファが確保され、 ヘッダアクセスで SIGSEGV する。
    """
    with pytest.raises(ValueError):
        NalUnit(2**64 - 1, False)
    with pytest.raises(ValueError):
        NalUnit(2**64 - 2, False, NalUnit.Type.H265)


def test_nal_unit_with_h265_type_needs_one_byte():
    """NalUnit の type に H265 を渡しても必要な下限は NalUnit のヘッダサイズ (1 バイト) であること

    NalUnit のヘッダアクセスは 1 バイトしか読まないため、 1 バイトあれば落ちない。
    """
    unit = NalUnit(1, True, NalUnit.Type.H265)

    assert unit.forbidden_bit() is False
    assert unit.payload() == b""


def test_nal_unit_bytes_and_set_payload():
    """bytes 版の生成と、 最小バッファでの set_payload() が動作すること"""
    # bytes 版は data をそのままバッファにする (ヘッダ 1 バイトを除いた部分が payload)
    # ヘッダサイズちょうどの data でも生成できる
    assert NalUnit(b"\x00").payload() == b""
    assert NalUnit(b"abc").payload() == b"bc"

    # ヘッダのみのバッファでも set_payload() が範囲外を読み書きしない
    unit = NalUnit(1)
    unit.set_payload(b"a")
    assert unit.payload() == b"a"

    # including_header=False でヘッダサイズだけ確保した場合も同じ
    h265 = NalUnit(0, False, NalUnit.Type.H265)
    h265.set_payload(b"a")
    assert h265.payload() == b"a"
