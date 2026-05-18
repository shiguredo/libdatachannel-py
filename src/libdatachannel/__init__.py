from .libdatachannel_ext import *  # noqa: F401,F403

# native を後勝ち上書きするため private 名で退避する。
from .libdatachannel_ext import PeerConnection as _PeerConnection


class PeerConnection(_PeerConnection):  # type: ignore[misc]
    """PeerConnection の Python wrapper。

    明示的に ``close()`` を呼ぶことを推奨する。 ``__del__`` 内で ``close()``
    を呼ぶセーフティネットを備えるが、 ``__del__`` 経由では timeout 警告も
    含めた全例外が握り潰される silent failure になる。 例外や警告を観測したい
    場合は明示 ``close()`` を呼ぶこと。
    """

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Audio RTP Packetizers
# OpusRtpPacketizer と AACRtpPacketizer は同じ型 (AudioRtpPacketizer<48000>)
AACRtpPacketizer = OpusRtpPacketizer  # noqa: F405

# PCMARtpPacketizer, PCMURtpPacketizer, G722RtpPacketizer は同じ型 (AudioRtpPacketizer<8000>)
PCMURtpPacketizer = PCMARtpPacketizer  # noqa: F405
G722RtpPacketizer = PCMARtpPacketizer  # noqa: F405

# Audio RTP Depacketizers
# OpusRtpDepacketizer と AACRtpDepacketizer は同じ型 (AudioRtpDepacketizer<48000>)
AACRtpDepacketizer = OpusRtpDepacketizer  # noqa: F405

# PCMARtpDepacketizer, PCMURtpDepacketizer, G722RtpDepacketizer は同じ型 (AudioRtpDepacketizer<8000>)
PCMURtpDepacketizer = PCMARtpDepacketizer  # noqa: F405
G722RtpDepacketizer = PCMARtpDepacketizer  # noqa: F405
