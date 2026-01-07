"""
WHEP (WebRTC-HTTP Egress Protocol) クライアント

webcodecs-py でデコードして libdatachannel-py で WHEP 受信します。

使い方:
    # 映像を受信（表示なし）
    uv run python examples/whep.py --url https://example.com/whep/channel

    # 映像と音声を受信して表示・再生
    uv run python examples/whep.py --url https://example.com/whep/channel --display
"""

import argparse
import logging
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import httpx
import numpy as np
import structlog
from whip import (
    find_nal_units,
    get_h265_nal_type_name,
    get_nal_type_name,
    handle_error,
    parse_link_header,
)

# raw-player
from raw_player import AudioPlayer, VideoPlayer

# webcodecs-py
from webcodecs import (
    AudioData,
    AudioDecoder,
    AudioDecoderConfig,
    AudioSampleFormat,
    EncodedAudioChunk,
    EncodedAudioChunkInit,
    EncodedAudioChunkType,
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
    H265RtpDepacketizer,
    OpusRtpDepacketizer,
    PeerConnection,
    Track,
)

logger = structlog.get_logger(__name__)


class WHEPClient:
    """WHEP クライアント（webcodecs-py ベース）"""

    def __init__(
        self,
        whep_url: str,
        bearer_token: Optional[str] = None,
        display_video: bool = False,
        preferred_codec: Optional[str] = None,
    ):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.display_video = display_video
        self.preferred_codec = preferred_codec  # "h264" or "h265"

        # 使用するコーデック
        self.video_codec: str = preferred_codec or "h264"

        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Track counters
        self.video_frame_count = 0
        self.audio_frame_count = 0
        self.decoded_video_count = 0
        self.decoded_audio_count = 0

        # タイムスタンプ用（ローカル時間ベース）
        self.playback_start_time: Optional[float] = None

        # webcodecs Decoder
        self.video_decoder: Optional[VideoDecoder] = None
        self.audio_decoder: Optional[AudioDecoder] = None
        self.video_decoder_configured = False
        self.audio_decoder_configured = False

        # Video settings (デフォルト 540p)
        self.video_width = 960
        self.video_height = 540

        # Audio settings
        self.audio_sample_rate = 48000
        self.audio_channels = 2  # Opus はステレオ

        # raw-player display
        self.player: Optional[VideoPlayer] = None
        self.audio_player: Optional[AudioPlayer] = None

        # Running flag
        self.running = True

    def connect(self) -> None:
        """WHEP サーバーに接続"""
        logger.info("Connecting to WHEP endpoint", url=self.whep_url)

        # PeerConnection を作成（自動 gathering を無効化）
        config = Configuration()
        config.ice_servers = []
        config.disable_auto_gathering = True
        config.force_media_transport = True
        self.pc = PeerConnection(config)

        # オーディオトラックを追加（RecvOnly）
        audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)
        logger.info("Audio track added with Opus codec (PT=111)")

        # ビデオトラックを追加（RecvOnly）
        video_desc = Description.Video("video", Description.Direction.RecvOnly)
        if self.preferred_codec == "h265":
            video_desc.add_h265_codec(97)
            logger.info("Video track added with H.265 codec")
        else:
            video_desc.add_h264_codec(96)
            logger.info("Video track added with H.264 codec")
        self.video_track = self.pc.add_track(video_desc)

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

            # セッション URL を取得（201 と 406 の両方で返される）
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whep_url, self.session_url)

            if response.status_code == 201:
                # サーバーがクライアントのオファーを受け入れた場合
                logger.info("Server accepted client offer (201 Created)")

                # Link ヘッダーから ICE サーバーを取得
                ice_servers = []
                link_header = response.headers.get("Link")
                if link_header:
                    ice_servers = parse_link_header(link_header)
                    if ice_servers:
                        logger.info("Found ICE servers in Link header", count=len(ice_servers))

                # リモート SDP を設定
                answer = Description(response.text, Description.Type.Answer)
                self.pc.set_remote_description(answer)

                # ICE サーバーがある場合、gathering を実行
                if ice_servers:
                    logger.info("Starting ICE gathering with TURN servers...")
                    self.pc.gather_local_candidates(ice_servers)

            elif response.status_code == 406:
                # サーバーがカウンターオファーを送信した場合
                logger.info("Server sent counter-offer (406 Not Acceptable)")

                # Link ヘッダーから ICE サーバーを取得
                ice_servers = []
                link_header = response.headers.get("Link")
                if link_header:
                    ice_servers = parse_link_header(link_header)
                    if ice_servers:
                        logger.info("Found ICE servers in Link header", count=len(ice_servers))

                # サーバーのオファーを処理
                server_offer = Description(response.text, Description.Type.Offer)
                self.pc.set_remote_description(server_offer)

                # アンサーを生成
                self.pc.set_local_description(Description.Type.Answer)
                local_answer = self.pc.local_description()
                if not local_answer:
                    raise RuntimeError("Failed to create answer for counter-offer")

                # HTTP PATCH でアンサーを送信
                logger.info("Sending answer via PATCH", url=self.session_url)
                patch_response = client.patch(
                    self.session_url,
                    content=str(local_answer),
                    headers=headers,
                )

                if patch_response.status_code != 204:
                    raise RuntimeError(
                        f"PATCH request failed with status {patch_response.status_code}: {patch_response.text}"
                    )

                logger.info("Counter-offer exchange completed (204 No Content)")

                # ICE サーバーがある場合、gathering を実行
                if ice_servers:
                    logger.info("Starting ICE gathering with TURN servers...")
                    self.pc.gather_local_candidates(ice_servers)

            else:
                raise RuntimeError(f"WHEP server returned {response.status_code}: {response.text}")

        # コーデック決定後にデパケッタイザーとデコーダーをセットアップ
        self._setup_video_depacketizer()
        self._setup_audio_depacketizer()

        if self.display_video:
            self._setup_video_decoder()
            self._setup_audio_decoder()

        logger.info("Connected to WHEP server")

    def _setup_video_decoder(self) -> None:
        """webcodecs-py ビデオデコーダーをセットアップ"""

        def on_output(frame: VideoFrame) -> None:
            self.decoded_video_count += 1
            if self.decoded_video_count % 30 == 0:
                logger.info(
                    "Decoded video frame",
                    count=self.decoded_video_count,
                    width=frame.coded_width,
                    height=frame.coded_height,
                )

            # 表示用に VideoPlayer に enqueue
            if self.player:
                try:
                    # ローカル時間ベースのタイムスタンプを使用
                    if self.playback_start_time is None:
                        self.playback_start_time = time.time()
                    pts_us = int((time.time() - self.playback_start_time) * 1_000_000)

                    if frame.format == VideoPixelFormat.NV12:
                        y_data = frame.plane(0)
                        uv_data = frame.plane(1)
                        self.player.enqueue_video_nv12(y_data, uv_data, pts_us)
                    else:
                        # I420
                        y_data, u_data, v_data = frame.planes()
                        self.player.enqueue_video_i420(y_data, u_data, v_data, pts_us)
                except Exception as e:
                    if self.decoded_video_count <= 5:
                        logger.error("Error enqueuing video frame", error=str(e))

            frame.close()

        def on_error(error: str) -> None:
            logger.error("Video decoder error", error=error)

        self.video_decoder = VideoDecoder(on_output, on_error)
        logger.info("Video decoder created (will configure on first frame)")

    def _setup_audio_decoder(self) -> None:
        """webcodecs-py オーディオデコーダーをセットアップ"""

        def on_output(audio_data: AudioData) -> None:
            self.decoded_audio_count += 1
            if self.decoded_audio_count % 50 == 0:
                logger.info(
                    "Decoded audio frame",
                    count=self.decoded_audio_count,
                    frames=audio_data.number_of_frames,
                    channels=audio_data.number_of_channels,
                )

            # AudioPlayer に enqueue
            if self.audio_player:
                try:
                    # ローカル時間ベースのタイムスタンプを使用
                    if self.playback_start_time is None:
                        self.playback_start_time = time.time()
                    pts_us = int((time.time() - self.playback_start_time) * 1_000_000)

                    # AudioData から float32 データを取得
                    num_frames = audio_data.number_of_frames
                    num_channels = audio_data.number_of_channels
                    sample_rate = audio_data.sample_rate

                    # 出力バッファを確保
                    buffer = np.zeros((num_frames, num_channels), dtype=np.float32)

                    # copy_to で F32 フォーマットとしてコピー
                    audio_data.copy_to(buffer, {"plane_index": 0, "format": AudioSampleFormat.F32})

                    self.audio_player.enqueue_audio(buffer, pts_us, sample_rate)
                except Exception as e:
                    if self.decoded_audio_count <= 5:
                        logger.error("Error enqueuing audio frame", error=str(e))

            audio_data.close()

        def on_error(error: str) -> None:
            logger.error("Audio decoder error", error=error)

        self.audio_decoder = AudioDecoder(on_output, on_error)

        # Opus デコーダーを設定
        decoder_config: AudioDecoderConfig = {
            "codec": "opus",
            "sample_rate": self.audio_sample_rate,
            "number_of_channels": self.audio_channels,
        }
        self.audio_decoder.configure(decoder_config)
        self.audio_decoder_configured = True
        logger.info(
            "Audio decoder configured",
            codec="opus",
            sample_rate=self.audio_sample_rate,
            channels=self.audio_channels,
        )

    def _configure_video_decoder(self, data: bytes) -> None:
        """デコーダーを設定（SPS/PPS または VPS/SPS/PPS から）"""
        if self.video_decoder_configured or not self.video_decoder:
            return

        # NAL ユニットを検索
        nal_positions = find_nal_units(data)
        h264_has_sps = False
        h264_has_pps = False
        h265_has_vps = False
        h265_has_sps = False
        h265_has_pps = False

        for i, (_, nal_start, _) in enumerate(nal_positions):
            if nal_start >= len(data):
                continue

            if self.video_codec == "h265":
                nal_type = (data[nal_start] >> 1) & 0x3F
                if nal_type == 32:
                    h265_has_vps = True
                elif nal_type == 33:
                    h265_has_sps = True
                elif nal_type == 34:
                    h265_has_pps = True
            else:
                nal_type = data[nal_start] & 0x1F
                if nal_type == 7:
                    h264_has_sps = True
                elif nal_type == 8:
                    h264_has_pps = True

        # デフォルト解像度
        width = 960
        height = 540

        if self.video_codec == "h265" and h265_has_vps and h265_has_sps and h265_has_pps:
            self.video_width = width
            self.video_height = height
            config: VideoDecoderConfig = {
                "codec": "hev1.1.6.L93.B0",
                "coded_width": width,
                "coded_height": height,
            }
            try:
                self.video_decoder.configure(config)
                self.video_decoder_configured = True
                logger.info(
                    "Video decoder configured (H.265)",
                    codec="hev1.1.6.L93.B0",
                    width=width,
                    height=height,
                )
            except Exception as e:
                logger.error("Failed to configure H.265 video decoder", error=str(e))

        elif self.video_codec == "h264" and h264_has_sps and h264_has_pps:
            self.video_width = width
            self.video_height = height
            h264_config: VideoDecoderConfig = {
                "codec": "avc1.64001F",
                "coded_width": width,
                "coded_height": height,
            }
            try:
                self.video_decoder.configure(h264_config)
                self.video_decoder_configured = True
                logger.info(
                    "Video decoder configured (H.264)",
                    codec="avc1.64001F",
                    width=width,
                    height=height,
                )
            except Exception as e:
                logger.error("Failed to configure H.264 video decoder", error=str(e))

    def _on_video_frame(self, data: bytes, frame_info) -> None:
        """ビデオフレームを受信"""
        self.video_frame_count += 1

        # NAL ユニットを解析
        nal_positions = find_nal_units(data)
        nal_units = []
        has_keyframe = False

        for _, nal_start, _ in nal_positions:
            if nal_start >= len(data):
                continue

            nal_header = data[nal_start]
            if self.video_codec == "h265":
                nal_type = (nal_header >> 1) & 0x3F
                # IDR_W_RADL=19, IDR_N_LP=20, CRA_NUT=21, VPS=32, SPS=33, PPS=34
                if nal_type in [19, 20, 21, 32, 33, 34]:
                    has_keyframe = True
                type_name = get_h265_nal_type_name(nal_type)
            else:
                nal_type = nal_header & 0x1F
                if nal_type == 5:  # IDR
                    has_keyframe = True
                elif nal_type in [7, 8]:  # SPS, PPS
                    has_keyframe = True
                type_name = get_nal_type_name(nal_type)

            nal_units.append({"type": nal_type, "type_name": type_name})

        # デコーダーを設定
        if not self.video_decoder_configured:
            self._configure_video_decoder(data)

        # デコード
        if self.video_decoder and self.video_decoder_configured:
            try:
                chunk_type = (
                    EncodedVideoChunkType.KEY if has_keyframe else EncodedVideoChunkType.DELTA
                )

                init: EncodedVideoChunkInit = {
                    "type": chunk_type,
                    "timestamp": frame_info.timestamp,
                    "data": data,  # bytes をそのまま渡す
                }
                chunk = EncodedVideoChunk(init)
                self.video_decoder.decode(chunk)
            except Exception as e:
                if self.video_frame_count <= 5:
                    logger.error("Video decode error", error=str(e))

        # ログ出力
        if self.video_frame_count % 30 == 0 or has_keyframe:
            nal_types = [f"{u['type']}({u['type_name']})" for u in nal_units[:3]]
            logger.info(
                "Video frame received",
                count=self.video_frame_count,
                size=len(data),
                nals=len(nal_units),
                types=nal_types,
            )

    def _setup_video_depacketizer(self) -> None:
        """RTP デパケッタイザーをセットアップ"""
        if self.video_track:
            if self.video_codec == "h265":
                depacketizer = H265RtpDepacketizer()
                logger.info("H.265 depacketizer set for video track")
            else:
                depacketizer = H264RtpDepacketizer()
                logger.info("H.264 depacketizer set for video track")
            self.video_track.on_frame(self._on_video_frame)
            self.video_track.set_media_handler(depacketizer)

    def _on_audio_frame(self, data: bytes, frame_info) -> None:
        """オーディオフレームを受信"""
        self.audio_frame_count += 1

        # デコード
        if self.audio_decoder and self.audio_decoder_configured:
            try:
                init: EncodedAudioChunkInit = {
                    "type": EncodedAudioChunkType.KEY,
                    "timestamp": frame_info.timestamp,
                    "data": data,  # bytes をそのまま渡す
                }
                chunk = EncodedAudioChunk(init)
                self.audio_decoder.decode(chunk)
            except Exception as e:
                if self.audio_frame_count <= 5:
                    logger.error("Audio decode error", error=str(e))

        if self.audio_frame_count % 50 == 0:
            logger.info(
                "Audio frame received",
                count=self.audio_frame_count,
                size=len(data),
                timestamp=frame_info.timestamp,
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
                            "Stats",
                            video_fps=video_fps,
                            audio_fps=audio_fps,
                            decoded_video=self.decoded_video_count,
                            decoded_audio=self.decoded_audio_count,
                        )

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        logger.info(
            "Receive completed",
            video_frames=self.video_frame_count,
            audio_frames=self.audio_frame_count,
            decoded_video=self.decoded_video_count,
            decoded_audio=self.decoded_audio_count,
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
                        logger.warning(
                            "DELETE request returned unexpected status",
                            status=response.status_code,
                        )
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

        if self.audio_decoder:
            try:
                self.audio_decoder.close()
            except Exception:
                pass
            finally:
                self.audio_decoder = None

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
    """フレームを表示（メインスレッドで実行、raw-player 使用）"""
    logger.info("Starting video/audio display...")

    # VideoPlayer を作成
    client.player = VideoPlayer(
        width=client.video_width,
        height=client.video_height,
        title=f"WHEP Video ({client.video_width}x{client.video_height})",
    )

    # AudioPlayer を作成
    client.audio_player = AudioPlayer()

    # キーコールバックを設定 (ESC または q で終了)
    def on_key(key: int) -> bool:
        if key == 27 or key == 113:  # ESC or 'q'
            return False
        return True

    client.player.set_key_callback(on_key)
    client.player.play()
    client.audio_player.play()

    logger.info("GPU Renderer", name=client.player.renderer_name)
    logger.info("Window created (Press 'q' or ESC to close)")

    window_closed = False

    while client.running and client.player.is_open:
        if not client.player.poll_events():
            logger.info("Window closed by user")
            window_closed = True
            break
        time.sleep(0.001)

    logger.info(
        "Display finished",
        decoded_video=client.decoded_video_count,
        decoded_audio=client.decoded_audio_count,
    )

    # プレイヤー統計を表示
    if client.player:
        stats = client.player.stats()
        logger.info("Video player stats", **stats)
        client.player.close()
        client.player = None

    if client.audio_player:
        audio_stats = client.audio_player.stats()
        logger.info("Audio player stats", **audio_stats)
        client.audio_player.stop()
        client.audio_player = None

    return window_closed


def main():
    parser = argparse.ArgumentParser(description="WHEP クライアント（webcodecs-py ベース）")
    parser.add_argument("--url", required=True, help="WHEP エンドポイント URL")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグログを出力",
    )
    parser.add_argument("--token", help="Bearer トークン（認証用）")
    parser.add_argument("--duration", type=int, help="受信時間（秒）")
    parser.add_argument(
        "--video-codec-type",
        choices=["h264", "h265"],
        required=True,
        help="映像コーデック (h264, h265)",
    )
    parser.add_argument("--display", action="store_true", help="映像と音声を表示・再生")

    args = parser.parse_args()

    # structlog 設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(pad_level=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    logger.info("WHEP endpoint", url=args.url)
    logger.info("Display", enabled=args.display)
    if args.video_codec_type:
        logger.info("Video codec type", codec=args.video_codec_type)

    client = WHEPClient(args.url, args.token, args.display, args.video_codec_type)

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
            logger.error("Error during disconnect", error=str(e))


if __name__ == "__main__":
    main()
