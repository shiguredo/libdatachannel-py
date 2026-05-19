import warnings

from .libdatachannel_ext import *  # noqa: F401,F403

# native を完全修飾で参照するため module を import する (後勝ち上書きで wrapper が公開名を取る)。
from . import libdatachannel_ext


class PeerConnection(libdatachannel_ext.PeerConnection):
    """PeerConnection の Python wrapper。

    明示的に ``close()`` を呼ぶことを推奨する。 ``__del__`` 内で ``close()``
    を呼ぶセーフティネットを備えるが、 ``__del__`` 経由で発生した例外は
    ``RuntimeWarning`` としてしか観測できない。 例外を直接捕捉したい場合は
    明示 ``close()`` を呼ぶこと。
    """

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            # Python では __del__ 内から例外を上に伝播させても呼び出し側
            # (利用者コード) では捕捉できず、 destructor の cleanup が中断する
            # だけで利益が無い。 一方で完全 silent はリグレッション検知を
            # 困難にするため、 warnings 経由で RuntimeWarning として記録する。
            # warnings.warn 自体が interpreter shutdown 中に失敗する可能性は
            # 内側の try/except で握り潰し、 destructor が落ちないようにする。
            try:
                warnings.warn(
                    f"PeerConnection.__del__: close() raised {e!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
