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
