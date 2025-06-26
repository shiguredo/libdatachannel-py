import libdatachannel


def test_frame_info_creation():
    # 新しい API を使用（timestamp のみ）
    info = libdatachannel.FrameInfo(654321)
    assert info.timestamp == 654321
    info.payload_type = 100  # payload_type は別途設定
    info.timestamp = 999999
    assert info.payload_type == 100
    assert info.timestamp == 999999
