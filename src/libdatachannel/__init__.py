from .libdatachannel_ext import *  # noqa: F401,F403
from .libdatachannel_ext import PeerConnection as _PeerConnection
from .libdatachannel_ext import WebSocket as _WebSocket
from .libdatachannel_ext import WebSocketServer as _WebSocketServer

# PeerConnection / WebSocket / WebSocketServer の Python wrapper
# 目的: 明示的な close()/stop() を伴わない destruct が走った場合に、
# libdatachannel 本体の destructor 内 blocking 処理 (mProcessor.join() /
# mThread.join() 等) が GIL を保持したまま長時間 hang する事象を回避する。
# __del__ で先に close()/stop() を呼んでおくことで destructor を即時
# 完了させる。 binding 側 (bind_libdatachannel.cpp) で GIL を release
# しつつ閉鎖完了を待つ実装になっているため、 __del__ 中でも他スレッドが
# 動ける。


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


class WebSocket(_WebSocket):  # type: ignore[misc]
    """WebSocket の Python wrapper。

    リソースの確実な解放のため、 利用後は明示的に ``close()`` を呼ぶことを
    推奨する。 ``close()`` を呼び忘れた場合のセーフティネットとして、
    ``__del__`` 内で ``close()`` を呼んで destructor の hang を回避する。
    """

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class WebSocketServer(_WebSocketServer):  # type: ignore[misc]
    """WebSocketServer の Python wrapper。

    リソースの確実な解放のため、 利用後は明示的に ``stop()`` を呼ぶことを
    推奨する。 ``stop()`` を呼び忘れた場合のセーフティネットとして、
    ``__del__`` 内で ``stop()`` を呼んで destructor の hang を回避する。
    """

    def __del__(self):
        try:
            self.stop()
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
