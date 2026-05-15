from .libdatachannel_ext import *  # noqa: F401,F403
from .libdatachannel_ext import PeerConnection as _PeerConnection

# PeerConnection の Python wrapper
# 目的: pc = None など明示的な close() を伴わない destruct が走った場合に、
# libdatachannel 本体の ~PeerConnection() 内 mProcessor.join() が GIL を
# 保持したまま長時間 blocking する事象を回避する。
# __del__ で先に close() を呼ぶことで非同期処理の完了まで待った状態で
# C++ destructor に進ませ、 mProcessor.join() を即時 return させる。
# close() の binding 側 (bind_libdatachannel.cpp) で GIL を release しつつ
# state==Closed まで polling で待つ実装になっているため、 __del__ 中でも
# 他スレッド (webhook サーバー等) が動ける。
class PeerConnection(_PeerConnection):  # type: ignore[misc]
    """PeerConnection の Python wrapper。

    リソースの確実な解放のため、 利用後は明示的に ``close()`` を呼ぶことを
    推奨する。 ``close()`` を呼び忘れた場合のセーフティネットとして、
    ``__del__`` 内で ``close()`` を呼んで destructor の hang を回避する。
    ただし ``__del__`` は GC タイミングに依存するため、 close 完了時刻が
    予測しにくい。
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
