from libdatachannel import (
    FrameInfo,
    Message,
    Reliability,
    make_message,
    make_message_from_data,
    make_message_from_variant,
    to_variant,
)


def test_create_message_and_access():
    m = Message(5)
    assert len(m) == 5
    m[0] = 65
    m[1] = 66
    assert m[0] == 65
    assert m[1] == 66


def test_make_message_with_metadata():
    r = Reliability()
    r.max_retransmits = 5
    f = FrameInfo(96, 12345)

    m = make_message(10, type=Message.Type.String, stream=2, reliability=r)
    assert m.type == Message.Type.String
    assert m.stream == 2
    m.reliability = r
    m.frame_info = f

    assert m.reliability.max_retransmits == 5
    assert m.frame_info.payload_type == 96


def test_to_variant_string():
    m = Message(4, Message.Type.String)
    m[0] = ord("t")
    m[1] = ord("e")
    m[2] = ord("s")
    m[3] = ord("t")

    variant = to_variant(m)
    assert isinstance(variant, str)
    assert variant == "test"


def test_to_variant_binary():
    m = Message(3, Message.Type.Binary)
    m[0] = 1
    m[1] = 2
    m[2] = 3

    variant = to_variant(m)
    assert isinstance(variant, bytes)
    assert variant == b"\x01\x02\x03"


def test_make_message_from_data_bytes():
    data = b"\x10\x20\x30"
    msg = make_message_from_data(
        data,
        type=Message.Type.Binary,
        stream=5,
        reliability=None,
        frame_info=None,
    )

    assert isinstance(msg, Message)
    assert msg.stream == 5
    assert msg.type == Message.Type.Binary
    assert msg.to_bytes() == data


def test_make_message_from_variant():
    binary = b"\x01\x02\x03"
    msg = make_message_from_variant(binary)
    assert isinstance(msg, Message)
    assert msg.type == Message.Type.Binary
    assert msg.to_bytes() == binary
    string = "hello"
    msg = make_message_from_variant(string)
    assert isinstance(msg, Message)
    assert msg.type == Message.Type.String
    assert msg.to_str() == string
