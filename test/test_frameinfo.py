import libdatachannel


def test_frame_info_creation():
    info = libdatachannel.FrameInfo(100, 654321)
    assert info.payload_type == 100
    assert info.timestamp == 654321
    info.payload_type = 127
    info.timestamp = 999999
    assert info.payload_type == 127
    assert info.timestamp == 999999
