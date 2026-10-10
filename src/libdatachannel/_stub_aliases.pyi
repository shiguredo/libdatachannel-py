# __init__.py が定義するエイリアス
#
# __init__.py と同じく star import の名前を参照する断片なので、 ruff の F821 と
# ty の検査対象外にしている (pyproject.toml)。
#
# 型チェッカーは __init__.py より __init__.pyi を優先するため、 生成されるスタブにも
# 同じエイリアスが必要になる (cmake/append_stub_aliases.cmake が __init__.pyi に追記する)

# Audio RTP Packetizers
AACRtpPacketizer = OpusRtpPacketizer  # noqa: F821
PCMURtpPacketizer = PCMARtpPacketizer  # noqa: F821
G722RtpPacketizer = PCMARtpPacketizer  # noqa: F821

# Audio RTP Depacketizers
AACRtpDepacketizer = OpusRtpDepacketizer  # noqa: F821
PCMURtpDepacketizer = PCMARtpDepacketizer  # noqa: F821
G722RtpDepacketizer = PCMARtpDepacketizer  # noqa: F821
