"""
WHIP (WebRTC-HTTP Ingestion Protocol) クライアント

webcodecs-py でエンコードして libdatachannel-py で WHIP 配信します。

使い方:
    # テストパターンで配信
    uv run python examples/whip.py --url https://example.com/whip/channel

    # カメラとマイクを使用
    uv run python examples/whip.py --url https://example.com/whip/channel --camera --microphone

    # H.265 で配信
    uv run python examples/whip.py --url https://example.com/whip/channel --codec h265
"""

import argparse
import logging
import queue
import random
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
import sounddevice as sd
from wish import handle_error, parse_link_header

# webcodecs-py
from webcodecs import (
    AudioData,
    AudioDataInit,
    AudioEncoder,
    AudioEncoderConfig,
    AudioSampleFormat,
    EncodedAudioChunk,
    EncodedVideoChunk,
    HardwareAccelerationEngine,
    LatencyMode,
    VideoEncoder,
    VideoEncoderConfig,
    VideoFrame,
    VideoFrameBufferInit,
    VideoPixelFormat,
)

# libdatachannel-py
from libdatachannel import (
    AV1RtpPacketizer,
    Configuration,
    Description,
    H264RtpPacketizer,
    H265RtpPacketizer,
    NalUnit,
    OpusRtpPacketizer,
    PeerConnection,
    PliHandler,
    RtcpNackResponder,
    RtcpSrReporter,
    RtpPacketizationConfig,
    Track,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WHIPClient:
    """WHIP クライアント（webcodecs-py ベース）"""

    def __init__(
        self,
        whip_url: str,
        bearer_token: Optional[str] = None,
        codec: str = "h264",
        use_camera: bool = False,
        use_microphone: bool = False,
    ):
        self.whip_url = whip_url
        self.bearer_token = bearer_token
        self.codec = codec.lower()
        self.use_camera = use_camera
        self.use_microphone = use_microphone

        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # webcodecs Encoders
        self.video_encoder: Optional[VideoEncoder] = None
        self.audio_encoder: Optional[AudioEncoder] = None

        # RTP components
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_sr_reporter = None
        self.audio_sr_reporter = None
        self.pli_handler = None
        self.nack_responder = None
        self.video_config: Optional[RtpPacketizationConfig] = None
        self.audio_config: Optional[RtpPacketizationConfig] = None

        # Frame counters
        self.video_frame_number = 0
        self.audio_frame_number = 0
        self.encoded_video_count = 0
        self.encoded_audio_count = 0

        # Video settings
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = 30

        # Audio settings
        self.audio_sample_rate = 48000
        self.audio_channels = 1  # モノラル（多くのマイクは1ch）
        self.audio_frame_size = 960  # 20ms @ 48kHz

        # Key frame interval
        self.key_frame_interval_frames = self.video_fps * 2  # 2秒ごと

        # Camera capture
        self.camera = None
        self.camera_thread = None
        self.video_queue: queue.Queue = queue.Queue(maxsize=30)
        self.capture_active = False

        # Audio capture
        self.audio_stream = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=50)

        # Test pattern state
        self.pattern_seed = 0

    def connect(self) -> None:
        """WHIP サーバーに接続"""
        logger.info(f"Connecting to WHIP endpoint: {self.whip_url}")

        # PeerConnection を作成
        config = Configuration()
        config.ice_servers = []
        self.pc = PeerConnection(config)

        # オーディオトラックを追加
        audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # ビデオトラックを追加
        video_desc = Description.Video("video", Description.Direction.SendOnly)
        if self.codec == "av1":
            video_desc.add_av1_codec(35)
        elif self.codec == "h265":
            video_desc.add_h265_codec(97)
        else:
            video_desc.add_h264_codec(96)
        self.video_track = self.pc.add_track(video_desc)

        # エンコーダーをセットアップ
        self._setup_video_encoder()
        self._setup_audio_encoder()

        # SDP オファーを生成
        self.pc.set_local_description()
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise RuntimeError("Failed to create offer")

        # WHIP サーバーにオファーを送信
        logger.info("Sending offer to WHIP server...")
        with httpx.Client(timeout=10.0) as client:
            headers = {"Content-Type": "application/sdp"}
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"

            response = client.post(
                self.whip_url,
                content=str(local_sdp),
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 201:
                raise RuntimeError(f"WHIP server returned {response.status_code}: {response.text}")

            # セッション URL を取得
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whip_url, self.session_url)

            # Link ヘッダーから ICE サーバーを取得
            link_header = response.headers.get("Link")
            if link_header:
                ice_servers = parse_link_header(link_header)
                if ice_servers:
                    logger.info(f"Found {len(ice_servers)} ICE server(s) in Link header")

            # リモート SDP を設定
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)

        logger.info("Connected to WHIP server")

    def _setup_video_encoder(self) -> None:
        """webcodecs-py ビデオエンコーダーをセットアップ"""

        def on_output(chunk: EncodedVideoChunk) -> None:
            if self.video_track and self.video_track.is_open():
                try:
                    data = np.zeros(chunk.byte_length, dtype=np.uint8)
                    chunk.copy_to(data)
                    self.video_track.send(bytes(data))
                    self.encoded_video_count += 1
                    if self.encoded_video_count % 60 == 0:
                        logger.debug(f"Sent encoded video frame #{self.encoded_video_count}")
                except Exception as e:
                    handle_error("sending encoded video", e)

        def on_error(error: str) -> None:
            logger.error(f"Video encoder error: {error}")

        self.video_encoder = VideoEncoder(on_output, on_error)

        # エンコーダーを設定
        if self.codec == "av1":
            codec_string = "av01.0.04M.08"
        elif self.codec == "h265":
            codec_string = "hev1.1.6.L120.B0"
        else:
            codec_string = "avc1.64001F"

        encoder_config: VideoEncoderConfig = {
            "codec": codec_string,
            "width": self.video_width,
            "height": self.video_height,
            "bitrate": 10_000_000,
            "framerate": float(self.video_fps),
            "latency_mode": LatencyMode.REALTIME,
        }

        # H.264/H.265 の場合は VideoToolbox を使用
        if self.codec == "h264":
            encoder_config["avc"] = {"format": "annexb"}
            encoder_config["hardware_acceleration_engine"] = (
                HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX
            )
        elif self.codec == "h265":
            encoder_config["hevc"] = {"format": "annexb"}
            encoder_config["hardware_acceleration_engine"] = (
                HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX
            )

        self.video_encoder.configure(encoder_config)
        logger.info(f"Video encoder configured: {codec_string}")

        # RTP パケッタイザーをセットアップ
        if self.codec == "av1":
            payload_type = 35
        elif self.codec == "h265":
            payload_type = 97
        else:
            payload_type = 96

        self.video_config = RtpPacketizationConfig(
            ssrc=random.randint(1, 0xFFFFFFFF),
            cname="video-stream",
            payload_type=payload_type,
            clock_rate=90000,
        )
        self.video_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        self.video_config.timestamp = self.video_config.start_timestamp
        self.video_config.sequence_number = random.randint(0, 0xFFFF)

        if self.codec == "av1":
            self.video_packetizer = AV1RtpPacketizer(
                AV1RtpPacketizer.Packetization.TemporalUnit, self.video_config, 1200
            )
        elif self.codec == "h265":
            self.video_packetizer = H265RtpPacketizer(
                NalUnit.Separator.LongStartSequence,
                self.video_config,
                1200,
            )
        else:
            self.video_packetizer = H264RtpPacketizer(
                NalUnit.Separator.LongStartSequence,
                self.video_config,
                1200,
            )

        # RTCP SR reporter
        self.video_sr_reporter = RtcpSrReporter(self.video_config)
        self.video_packetizer.add_to_chain(self.video_sr_reporter)

        # PLI handler
        def on_pli():
            logger.info("PLI received - requesting keyframe")

        self.pli_handler = PliHandler(on_pli)
        self.video_packetizer.add_to_chain(self.pli_handler)

        # NACK responder
        self.nack_responder = RtcpNackResponder()
        self.video_packetizer.add_to_chain(self.nack_responder)

        if self.video_track:
            self.video_track.set_media_handler(self.video_packetizer)

    def _setup_audio_encoder(self) -> None:
        """webcodecs-py オーディオエンコーダーをセットアップ"""

        def on_output(chunk: EncodedAudioChunk) -> None:
            if self.audio_track and self.audio_track.is_open():
                try:
                    data = np.zeros(chunk.byte_length, dtype=np.uint8)
                    chunk.copy_to(data)
                    self.audio_track.send(bytes(data))
                    self.encoded_audio_count += 1
                    if self.encoded_audio_count % 100 == 0:
                        logger.debug(f"Sent encoded audio frame #{self.encoded_audio_count}")
                except Exception as e:
                    handle_error("sending encoded audio", e)

        def on_error(error: str) -> None:
            logger.error(f"Audio encoder error: {error}")

        self.audio_encoder = AudioEncoder(on_output, on_error)

        encoder_config: AudioEncoderConfig = {
            "codec": "opus",
            "sample_rate": self.audio_sample_rate,
            "number_of_channels": self.audio_channels,
            "bitrate": 128000,
        }
        self.audio_encoder.configure(encoder_config)
        logger.info("Audio encoder configured: Opus")

        # RTP パケッタイザーをセットアップ
        self.audio_config = RtpPacketizationConfig(
            ssrc=random.randint(1, 0xFFFFFFFF),
            cname="audio-stream",
            payload_type=111,
            clock_rate=48000,
        )
        self.audio_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        self.audio_config.timestamp = self.audio_config.start_timestamp
        self.audio_config.sequence_number = random.randint(0, 0xFFFF)

        self.audio_packetizer = OpusRtpPacketizer(self.audio_config)

        self.audio_sr_reporter = RtcpSrReporter(self.audio_config)
        self.audio_packetizer.add_to_chain(self.audio_sr_reporter)

        if self.audio_track:
            self.audio_track.set_media_handler(self.audio_packetizer)

    def _start_camera_capture(self) -> None:
        """カメラキャプチャを開始"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.video_fps)

        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")

        # 実際の解像度に合わせる
        if actual_width != self.video_width or actual_height != self.video_height:
            self.video_width = actual_width
            self.video_height = actual_height
            logger.info(f"Adjusted video size to camera: {self.video_width}x{self.video_height}")

        self.capture_active = True

        def capture_thread():
            while self.capture_active:
                ret, frame = self.camera.read()
                if ret:
                    try:
                        self.video_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Drop frame
                time.sleep(0.001)

        self.camera_thread = threading.Thread(target=capture_thread, daemon=True)
        self.camera_thread.start()

    def _start_audio_capture(self) -> None:
        """マイクキャプチャを開始"""

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio capture status: {status}")
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

        self.audio_stream = sd.InputStream(
            samplerate=self.audio_sample_rate,
            channels=self.audio_channels,
            dtype=np.float32,
            blocksize=self.audio_frame_size,
            callback=audio_callback,
        )
        self.audio_stream.start()
        logger.info(f"Audio capture started: {self.audio_sample_rate}Hz, {self.audio_channels}ch")

    def _generate_test_pattern(self) -> np.ndarray:
        """テストパターンを生成（BGRA）- NumPy ベクトル化版"""
        t = self.video_frame_number / self.video_fps

        # meshgrid で座標配列を作成（初回のみ計算してキャッシュ可能だが、シンプルさ優先）
        x = np.arange(self.video_width, dtype=np.float32)
        y = np.arange(self.video_height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # 背景グラデーション（ベクトル化）
        r = (127 + 127 * np.sin(2 * np.pi * (xx / self.video_width + t * 0.1))).astype(np.uint8)
        g = (127 + 127 * np.sin(2 * np.pi * (yy / self.video_height + t * 0.15))).astype(np.uint8)
        b = (127 + 127 * np.sin(2 * np.pi * ((xx + yy) / (self.video_width + self.video_height) + t * 0.2))).astype(np.uint8)

        # BGRA フレームを構築
        frame = np.stack([b, g, r, np.full_like(r, 255)], axis=-1)

        # 動く円を描画
        cx = int(self.video_width / 2 + self.video_width / 4 * np.sin(t * 2))
        cy = int(self.video_height / 2 + self.video_height / 4 * np.cos(t * 2))
        cv2.circle(frame, (cx, cy), 50, (255, 255, 255, 255), -1)

        return frame

    def _generate_test_audio(self) -> np.ndarray:
        """テストオーディオを生成（サイン波）"""
        t = np.linspace(
            self.audio_frame_number * self.audio_frame_size / self.audio_sample_rate,
            (self.audio_frame_number + 1) * self.audio_frame_size / self.audio_sample_rate,
            self.audio_frame_size,
            dtype=np.float32,
        )
        # 440Hz サイン波
        mono = np.sin(2 * np.pi * 440 * t) * 0.3
        if self.audio_channels == 1:
            return mono.reshape(-1, 1)
        else:
            return np.column_stack([mono] * self.audio_channels)

    def send_frames(self, duration: Optional[int] = None) -> None:
        """フレームを送信"""
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

        # キャプチャを開始
        if self.use_camera:
            self._start_camera_capture()
        if self.use_microphone:
            self._start_audio_capture()

        logger.info(
            f"Sending frames: {self.video_width}x{self.video_height} @ {self.video_fps}fps"
        )

        # フレーム送信ループ
        frame_interval = 1.0 / self.video_fps
        audio_interval = self.audio_frame_size / self.audio_sample_rate
        start_time = time.time()
        next_video_time = start_time
        next_audio_time = start_time

        try:
            while True:
                current_time = time.time()

                if duration and current_time - start_time >= duration:
                    break

                # ビデオフレーム
                if current_time >= next_video_time:
                    self._send_video_frame()
                    next_video_time += frame_interval

                # オーディオフレーム
                if current_time >= next_audio_time:
                    self._send_audio_frame()
                    next_audio_time += audio_interval

                # CPU 使用率を抑える
                sleep_time = min(next_video_time, next_audio_time) - time.time()
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    def _send_video_frame(self) -> None:
        """ビデオフレームを送信"""
        if not self.video_encoder:
            return

        # フレームを取得
        if self.use_camera and not self.video_queue.empty():
            try:
                bgr_frame = self.video_queue.get_nowait()
                # BGR → BGRA
                bgra_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
            except queue.Empty:
                return
        else:
            # テストパターン
            bgra_frame = self._generate_test_pattern()

        # VideoFrame を作成
        timestamp_us = int(self.video_frame_number * 1_000_000 / self.video_fps)

        bgra_init: VideoFrameBufferInit = {
            "format": VideoPixelFormat.BGRA,
            "coded_width": self.video_width,
            "coded_height": self.video_height,
            "timestamp": timestamp_us,
        }
        bgra_video_frame = VideoFrame(bgra_frame, bgra_init)

        # BGRA → I420 変換
        i420_size = self.video_width * self.video_height * 3 // 2
        i420_buffer = np.zeros(i420_size, dtype=np.uint8)
        bgra_video_frame.copy_to(i420_buffer, {"format": VideoPixelFormat.I420})
        bgra_video_frame.close()

        # I420 VideoFrame を作成
        i420_init: VideoFrameBufferInit = {
            "format": VideoPixelFormat.I420,
            "coded_width": self.video_width,
            "coded_height": self.video_height,
            "timestamp": timestamp_us,
        }
        frame = VideoFrame(i420_buffer, i420_init)

        # エンコード
        is_keyframe = self.video_frame_number % self.key_frame_interval_frames == 0
        try:
            if is_keyframe:
                self.video_encoder.encode(frame, {"keyFrame": True})
            else:
                self.video_encoder.encode(frame)
        except Exception as e:
            handle_error("encoding video frame", e)

        frame.close()
        self.video_frame_number += 1

        if self.video_frame_number % self.video_fps == 0:
            elapsed = self.video_frame_number / self.video_fps
            logger.info(f"Video progress: {elapsed:.1f}s ({self.video_frame_number} frames)")

    def _send_audio_frame(self) -> None:
        """オーディオフレームを送信"""
        if not self.audio_encoder:
            return

        # オーディオを取得
        if self.use_microphone and not self.audio_queue.empty():
            try:
                audio_samples = self.audio_queue.get_nowait()
            except queue.Empty:
                return
        else:
            # テストトーン
            audio_samples = self._generate_test_audio()

        # AudioData を作成
        timestamp_us = int(self.audio_frame_number * self.audio_frame_size * 1_000_000 / self.audio_sample_rate)

        init: AudioDataInit = {
            "format": AudioSampleFormat.F32,
            "sample_rate": self.audio_sample_rate,
            "number_of_frames": len(audio_samples),
            "number_of_channels": self.audio_channels,
            "timestamp": timestamp_us,
            "data": audio_samples.astype(np.float32),
        }
        audio_data = AudioData(init)

        # エンコード
        try:
            self.audio_encoder.encode(audio_data)
        except Exception as e:
            handle_error("encoding audio frame", e)

        audio_data.close()
        self.audio_frame_number += 1

    def disconnect(self) -> None:
        """切断処理"""
        logger.info("Starting graceful shutdown...")

        # キャプチャを停止
        self.capture_active = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()

        # WHIP セッションを終了
        if self.session_url:
            logger.info("Sending DELETE request to WHIP server...")
            try:
                with httpx.Client(timeout=5.0) as client:
                    headers = {}
                    if self.bearer_token:
                        headers["Authorization"] = f"Bearer {self.bearer_token}"

                    response = client.delete(self.session_url, headers=headers)
                    if response.status_code in [200, 204]:
                        logger.info("WHIP session terminated successfully")
                    else:
                        logger.warning(f"DELETE request returned status {response.status_code}")
            except Exception as e:
                handle_error("terminating WHIP session", e)

        # エンコーダーをフラッシュ
        if self.video_encoder:
            try:
                self.video_encoder.flush()
            except Exception:
                pass
        if self.audio_encoder:
            try:
                self.audio_encoder.flush()
            except Exception:
                pass

        time.sleep(0.5)

        # リソースをクリーンアップ
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_track = None
        self.audio_track = None

        if self.pc:
            try:
                self.pc.close()
            except Exception as e:
                handle_error("closing PeerConnection", e)
            finally:
                self.pc = None

        if self.video_encoder:
            try:
                self.video_encoder.close()
            except Exception:
                pass
            finally:
                self.video_encoder = None

        if self.audio_encoder:
            try:
                self.audio_encoder.close()
            except Exception:
                pass
            finally:
                self.audio_encoder = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="WHIP クライアント（webcodecs-py ベース）")
    parser.add_argument("--url", required=True, help="WHIP エンドポイント URL")
    parser.add_argument("--token", help="Bearer トークン（認証用）")
    parser.add_argument("--duration", type=int, help="配信時間（秒）")
    parser.add_argument(
        "--codec",
        choices=["h264", "h265", "av1"],
        default="h264",
        help="映像コーデック (デフォルト: h264)",
    )
    parser.add_argument("--camera", action="store_true", help="カメラを使用")
    parser.add_argument("--microphone", action="store_true", help="マイクを使用")

    args = parser.parse_args()

    logger.info(f"Video codec: {args.codec}")
    logger.info(f"WHIP endpoint: {args.url}")
    logger.info(f"Camera: {args.camera}, Microphone: {args.microphone}")

    client = WHIPClient(
        args.url,
        args.token,
        args.codec,
        args.camera,
        args.microphone,
    )

    try:
        client.connect()
        client.send_frames(args.duration)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        handle_error("running WHIP client", e)
    finally:
        try:
            client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()
