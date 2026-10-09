"""max_fragment_size を小さくしたときの outgoing を子プロセスで実行するスクリプト

恒停し得る呼び出しを pytest のプロセスから分離するために使う。 引数は
<codec> <max_fragment_size> <input_size> で、 outgoing が戻れば exit 0 で終わる。
戻らなければ親プロセスの timeout で検出する。
"""

import sys

from libdatachannel import (
    AV1RtpPacketizer,
    H264RtpPacketizer,
    H265RtpPacketizer,
    Message,
    NalUnit,
    RtpPacketizationConfig,
)


def make_message(data: bytes) -> Message:
    """バイト列をそのまま持つ Message を作る"""
    message = Message(len(data))
    for index, value in enumerate(data):
        message[index] = value
    return message


def make_nal_message(nal_unit: bytes) -> Message:
    """NalUnit.Separator.Length 用に NAL 長 4 バイトを前置した Message を作る"""
    return make_message(len(nal_unit).to_bytes(4, "big") + nal_unit)


def main() -> int:
    codec = sys.argv[1]
    max_fragment_size = int(sys.argv[2])
    input_size = int(sys.argv[3])

    if codec == "h264":
        config = RtpPacketizationConfig(1234, "stream1", 96, H264RtpPacketizer.CLOCK_RATE)
        packetizer = H264RtpPacketizer(NalUnit.Separator.Length, config, max_fragment_size)
        # 0x65 は NAL ヘッダ (forbidden_zero_bit 0 / nal_ref_idc 3 / unit type 5)
        message = make_nal_message(bytes([0x65]) + bytes(input_size - 1))
    elif codec == "h265":
        config = RtpPacketizationConfig(1234, "stream1", 96, H265RtpPacketizer.CLOCK_RATE)
        packetizer = H265RtpPacketizer(NalUnit.Separator.Length, config, max_fragment_size)
        # 0x42 0x01 は H265 の NAL ヘッダ (unit type 32 = VPS)
        message = make_nal_message(bytes([0x42, 0x01]) + bytes(input_size - 2))
    elif codec == "av1":
        config = RtpPacketizationConfig(1234, "stream1", 96, AV1RtpPacketizer.CLOCK_RATE)
        packetizer = AV1RtpPacketizer(AV1RtpPacketizer.Packetization.Obu, config, max_fragment_size)
        # 先頭バイト 0x32 は OBU ヘッダ (type 6 = OBU_FRAME)。 SequenceHeader (type 1) では
        # ないため、 この呼び出しで SequenceHeader はキャッシュされない
        message = make_message(bytes([0x32, 0x00, 0x01, 0x02, 0x03, 0x04]))
    else:
        print(f"unknown codec: {codec}", file=sys.stderr)
        return 2

    # outgoing は引数のメッセージ列を RTP パケットに置き換えるだけで send を呼ばないため、
    # 固定した入力では結果を観測できない。 ここでは呼び出しが戻ることを確認する
    packetizer.outgoing([message], lambda out: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
