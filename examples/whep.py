"""
WHEP (WebRTC-HTTP Egress Protocol) クライアント

webcodecs-py でデコードして libdatachannel-py で WHEP 受信します。

使い方:
    # 映像を受信（表示なし）
    uv run python examples/whep.py --url https://example.com/whep/channel

    # 映像を受信して表示
    uv run python examples/whep.py --url https://example.com/whep/channel --display
"""

import argparse
import logging
import queue
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
from wish import get_nal_type_name, handle_error, parse_link_header

# webcodecs-py
from webcodecs import (
    EncodedVideoChunk,
    EncodedVideoChunkInit,
    EncodedVideoChunkType,
    VideoDecoder,
    VideoDecoderConfig,
    VideoFrame,
    VideoPixelFormat,
)

# libdatachannel-py
from libdatachannel import (
    Configuration,
    Description,
    H264RtpDepacketizer,
    NalUnit,
    OpusRtpDepacketizer,
    PeerConnection,
    Track,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WHEPClient:
    """WHEP クライアント（webcodecs-py ベース）"""

    def __init__(
        self,
        whep_url: str,
        bearer_token: Optional[str] = None,
        display_video: bool = False,
    ):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.display_video = display_video

        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Track counters
        self.video_frame_count = 0
        self.audio_frame_count = 0
        self.decoded_frame_count = 0

        # webcodecs Decoder
        self.video_decoder: Optional[VideoDecoder] = None
        self.decoder_configured = False

        # OpenCV display
        self.window_name = "WHEP Video"
        self.frame_queue: Optional[queue.Queue] = queue.Queue(maxsize=10) if display_video else None

        # Running flag
        self.running = True

    def connect(self) -> None:
        """WHEP サーバーに接続"""
        logger.info(f"Connecting to WHEP endpoint: {self.whep_url}")

        # PeerConnection を作成
        config = Configuration()
        config.ice_servers = []
        self.pc = PeerConnection(config)

        # オーディオトラックを追加（RecvOnly）
        audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # ビデオトラックを追加（RecvOnly）
        video_desc = Description.Video("video", Description.Direction.RecvOnly)
        video_desc.add_h264_codec(96)
        self.video_track = self.pc.add_track(video_desc)

        logger.info("Audio track added with Opus codec (PT=111)")
        logger.info("Video track added with H.264 codec (PT=96)")

        # デパケッタイザーとハンドラーをセットアップ
        self._setup_video_depacketizer()
        self._setup_audio_depacketizer()

        # デコーダーをセットアップ
        if self.display_video:
            self._setup_video_decoder()

        # SDP オファーを生成
        self.pc.set_local_description()
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise RuntimeError("Failed to create offer")

        # WHEP サーバーにオファーを送信
        logger.info("Sending offer to WHEP server...")
        with httpx.Client(timeout=10.0) as client:
            headers = {"Content-Type": "application/sdp"}
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"

            response = client.post(
                self.whep_url,
                content=str(local_sdp),
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 201:
                raise RuntimeError(f"WHEP server returned {response.status_code}: {response.text}")

            # セッション URL を取得
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whep_url, self.session_url)

            # Link ヘッダーから ICE サーバーを取得
            link_header = response.headers.get("Link")
            if link_header:
                ice_servers = parse_link_header(link_header)
                if ice_servers:
                    logger.info(f"Found {len(ice_servers)} ICE server(s) in Link header")

            # リモート SDP を設定
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)

        logger.info("Connected to WHEP server")

    def _setup_video_decoder(self) -> None:
        """webcodecs-py ビデオデコーダーをセットアップ"""

        def on_output(frame: VideoFrame) -> None:
            self.decoded_frame_count += 1
            if self.decoded_frame_count % 30 == 0:
                logger.info(
                    f"Decoded frame #{self.decoded_frame_count}: "
                    f"{frame.coded_width}x{frame.coded_height}"
                )

            # 表示用にキューに追加
            if self.frame_queue:
                try:
                    # I420 → RGB 変換
                    width = frame.coded_width
                    height = frame.coded_height

                    # RGB バッファを作成
                    rgb_size = width * height * 3
                    rgb_buffer = np.zeros(rgb_size, dtype=np.uint8)

                    # copy_to で RGB に変換
                    frame.copy_to(rgb_buffer, {"format": VideoPixelFormat.RGB})

                    # 形状を変更
                    rgb_frame = rgb_buffer.reshape((height, width, 3))

                    # BGR に変換（OpenCV 用）
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                    self.frame_queue.put_nowait(bgr_frame)
                except queue.Full:
                    pass  # Drop frame
                except Exception as e:
                    if self.decoded_frame_count <= 5:
                        logger.error(f"Error converting frame: {e}")

            frame.close()

        def on_error(error: str) -> None:
            logger.error(f"Video decoder error: {error}")

        self.video_decoder = VideoDecoder(on_output, on_error)
        logger.info("Video decoder created (will configure on first frame)")

    def _configure_decoder(self, data: bytes) -> None:
        """デコーダーを設定（SPS/PPS から）"""
        if self.decoder_configured or not self.video_decoder:
            return

        # SPS/PPS を探す
        sps_data = None
        pps_data = None

        offset = 0
        while offset < len(data):
            # スタートコードを探す
            if offset + 4 <= len(data) and data[offset:offset + 4] == b"\x00\x00\x00\x01":
                start_code_len = 4
            elif offset + 3 <= len(data) and data[offset:offset + 3] == b"\x00\x00\x01":
                start_code_len = 3
            else:
                offset += 1
                continue

            nal_start = offset + start_code_len
            if nal_start >= len(data):
                break

            # NAL タイプを取得
            nal_type = data[nal_start] & 0x1F

            # 次のスタートコードまでを取得
            next_offset = nal_start + 1
            while next_offset < len(data):
                if (next_offset + 4 <= len(data) and data[next_offset:next_offset + 4] == b"\x00\x00\x00\x01") or \
                   (next_offset + 3 <= len(data) and data[next_offset:next_offset + 3] == b"\x00\x00\x01"):
                    break
                next_offset += 1

            nal_data = data[nal_start:next_offset]

            if nal_type == 7:  # SPS
                sps_data = nal_data
                logger.info(f"Found SPS: {len(sps_data)} bytes")
            elif nal_type == 8:  # PPS
                pps_data = nal_data
                logger.info(f"Found PPS: {len(pps_data)} bytes")

            offset = next_offset

        if sps_data and pps_data:
            # SPS から解像度を取得（簡易版）
            # デフォルト値を使用
            width = 1920
            height = 1080

            config: VideoDecoderConfig = {
                "codec": "avc1.64001F",  # H.264 High Profile
                "coded_width": width,
                "coded_height": height,
            }

            try:
                self.video_decoder.configure(config)
                self.decoder_configured = True
                logger.info(f"Video decoder configured: {width}x{height}")
            except Exception as e:
                logger.error(f"Failed to configure decoder: {e}")

    def _on_video_frame(self, data: bytes, frame_info) -> None:
        """ビデオフレームを受信"""
        self.video_frame_count += 1

        # NAL ユニットを解析
        nal_units = []
        has_keyframe = False

        offset = 0
        while offset < len(data):
            if offset + 4 <= len(data) and data[offset:offset + 4] == b"\x00\x00\x00\x01":
                start_code_len = 4
            elif offset + 3 <= len(data) and data[offset:offset + 3] == b"\x00\x00\x01":
                start_code_len = 3
            else:
                offset += 1
                continue

            nal_start = offset + start_code_len
            if nal_start >= len(data):
                break

            # NAL タイプを取得
            nal_header = data[nal_start]
            nal_type = nal_header & 0x1F

            if nal_type == 5:  # IDR
                has_keyframe = True
            elif nal_type in [7, 8]:  # SPS, PPS
                has_keyframe = True

            nal_units.append({
                "type": nal_type,
                "type_name": get_nal_type_name(nal_type),
            })

            # 次のスタートコードへ
            next_offset = nal_start + 1
            while next_offset < len(data):
                if (next_offset + 4 <= len(data) and data[next_offset:next_offset + 4] == b"\x00\x00\x00\x01") or \
                   (next_offset + 3 <= len(data) and data[next_offset:next_offset + 3] == b"\x00\x00\x01"):
                    break
                next_offset += 1
            offset = next_offset

        # デコーダーを設定
        if not self.decoder_configured:
            self._configure_decoder(data)

        # デコード
        if self.video_decoder and self.decoder_configured:
            try:
                chunk_type = EncodedVideoChunkType.KEY if has_keyframe else EncodedVideoChunkType.DELTA

                init: EncodedVideoChunkInit = {
                    "type": chunk_type,
                    "timestamp": frame_info.timestamp,
                    "data": np.frombuffer(data, dtype=np.uint8),
                }
                chunk = EncodedVideoChunk(init)
                self.video_decoder.decode(chunk)
            except Exception as e:
                if self.video_frame_count <= 5:
                    logger.error(f"Decode error: {e}")

        # ログ出力
        if self.video_frame_count % 30 == 0 or has_keyframe:
            nal_types = [f"{u['type']}({u['type_name']})" for u in nal_units[:3]]
            logger.info(
                f"Video frame #{self.video_frame_count}: "
                f"size={len(data)}, NALs={len(nal_units)} [{', '.join(nal_types)}...]"
            )

    def _setup_video_depacketizer(self) -> None:
        """H.264 RTP デパケッタイザーをセットアップ"""
        if self.video_track:
            h264_depacketizer = H264RtpDepacketizer()
            self.video_track.on_frame(self._on_video_frame)
            self.video_track.set_media_handler(h264_depacketizer)
            logger.info("H.264 depacketizer set for video track")

    def _on_audio_frame(self, data: bytes, frame_info) -> None:
        """オーディオフレームを受信"""
        self.audio_frame_count += 1
        if self.audio_frame_count % 50 == 0:
            logger.info(
                f"Audio frame #{self.audio_frame_count}: "
                f"size={len(data)}, timestamp={frame_info.timestamp}"
            )

    def _setup_audio_depacketizer(self) -> None:
        """Opus RTP デパケッタイザーをセットアップ"""
        if self.audio_track:
            opus_depacketizer = OpusRtpDepacketizer()
            self.audio_track.on_frame(self._on_audio_frame)
            self.audio_track.set_media_handler(opus_depacketizer)
            logger.info("Opus depacketizer set for audio track")

    def receive_frames(self, duration: Optional[int] = None) -> None:
        """フレームを受信"""
        if not self.pc:
            raise RuntimeError("PeerConnection not initialized")

        # 接続を待機
        timeout = 10.0
        start_time = time.time()
        while self.pc.state() != PeerConnection.State.Connected:
            if time.time() - start_time > timeout:
                raise RuntimeError("Connection timeout")
            time.sleep(0.1)

        logger.info("Connection established")

        start_time = time.time()
        last_video_count = 0
        last_audio_count = 0

        try:
            while self.running:
                current_time = time.time()

                if duration and current_time - start_time >= duration:
                    break

                # 統計をログ出力（毎秒）
                elapsed = int(current_time - start_time)
                if elapsed > 0 and elapsed != int(current_time - start_time - 0.1):
                    video_fps = self.video_frame_count - last_video_count
                    audio_fps = self.audio_frame_count - last_audio_count
                    last_video_count = self.video_frame_count
                    last_audio_count = self.audio_frame_count

                    if video_fps > 0 or audio_fps > 0:
                        logger.info(
                            f"Stats: Video {video_fps} fps, Audio {audio_fps} fps, "
                            f"Decoded: {self.decoded_frame_count}"
                        )

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        logger.info(
            f"Receive completed. Video: {self.video_frame_count}, "
            f"Audio: {self.audio_frame_count}, Decoded: {self.decoded_frame_count}"
        )

    def disconnect(self) -> None:
        """切断処理"""
        logger.info("Starting graceful shutdown...")

        # WHEP セッションを終了
        if self.session_url:
            logger.info("Sending DELETE request to WHEP server...")
            try:
                with httpx.Client(timeout=5.0) as client:
                    headers = {}
                    if self.bearer_token:
                        headers["Authorization"] = f"Bearer {self.bearer_token}"

                    response = client.delete(self.session_url, headers=headers)
                    if response.status_code in [200, 204]:
                        logger.info("WHEP session terminated successfully")
                    else:
                        logger.warning(f"DELETE request returned status {response.status_code}")
            except Exception as e:
                handle_error("terminating WHEP session", e)

        time.sleep(0.5)

        # デコーダーをクリーンアップ
        if self.video_decoder:
            try:
                self.video_decoder.close()
            except Exception:
                pass
            finally:
                self.video_decoder = None

        # トラックをクリーンアップ
        self.video_track = None
        self.audio_track = None

        # PeerConnection をクローズ
        if self.pc:
            try:
                self.pc.close()
            except Exception as e:
                handle_error("closing PeerConnection", e)
            finally:
                self.pc = None

        logger.info("Graceful shutdown completed")


def display_frames(client: WHEPClient) -> bool:
    """フレームを表示（メインスレッドで実行）"""
    logger.info("Starting video display...")

    cv2.namedWindow(client.window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(client.window_name, 100, 100)
    logger.info("Window created (Press 'q' or ESC to close)")

    frame_count = 0
    window_closed = False

    while client.running:
        try:
            frame = client.frame_queue.get(timeout=0.1)
            frame_count += 1

            if frame_count == 1:
                logger.info(f"First frame: shape={frame.shape}")

            cv2.imshow(client.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User pressed 'q' or ESC")
                break

            try:
                if cv2.getWindowProperty(client.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed by user")
                    window_closed = True
                    break
            except cv2.error:
                window_closed = True
                break

        except queue.Empty:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            try:
                if cv2.getWindowProperty(client.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    window_closed = True
                    break
            except cv2.error:
                window_closed = True
                break

    logger.info(f"Display finished. Frames displayed: {frame_count}")
    if not window_closed:
        cv2.destroyAllWindows()

    return window_closed


def main():
    parser = argparse.ArgumentParser(description="WHEP クライアント（webcodecs-py ベース）")
    parser.add_argument("--url", required=True, help="WHEP エンドポイント URL")
    parser.add_argument("--token", help="Bearer トークン（認証用）")
    parser.add_argument("--duration", type=int, help="受信時間（秒）")
    parser.add_argument("--display", action="store_true", help="映像を表示")

    args = parser.parse_args()

    logger.info(f"WHEP endpoint: {args.url}")
    logger.info(f"Display: {args.display}")

    client = WHEPClient(args.url, args.token, args.display)

    try:
        client.connect()

        if args.display:
            # 受信スレッドを開始
            def receive_thread():
                try:
                    client.receive_frames(args.duration)
                except Exception as e:
                    handle_error("receiving frames", e)
                finally:
                    client.running = False

            receiver = threading.Thread(target=receive_thread, daemon=True)
            receiver.start()

            # メインスレッドで表示
            try:
                window_closed = display_frames(client)
                if window_closed:
                    logger.info("Exiting because window was closed")
            finally:
                client.running = False

            receiver.join(timeout=2.0)
        else:
            client.receive_frames(args.duration)

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        handle_error("running WHEP client", e)
    finally:
        try:
            client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()
