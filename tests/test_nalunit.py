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
