import libdatachannel


def test_frame_info_creation_with_timestamp():
    info = libdatachannel.FrameInfo(654321)
    assert info.timestamp == 654321
    info.timestamp = 999999
    assert info.timestamp == 999999


def test_frame_info_payload_type():
    info = libdatachannel.FrameInfo(654321)
    info.payload_type = 127
    assert info.payload_type == 127
