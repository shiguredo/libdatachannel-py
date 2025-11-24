"""
WHIP (WebRTC-HTTP Ingestion Protocol) クライアント

webcodecs-py でエンコードして libdatachannel-py で WHIP 配信します。

使い方:
    # テストパターンで配信（blend2d + テスト音声）
    uv run python examples/whip.py --url https://example.com/whip/channel --fake-capture-device

    # カメラとマイクを使用（デバイス番号 0）
    uv run python examples/whip.py --url https://example.com/whip/channel --video-input-device 0 --audio-input-device 0

    # H.265 で配信
    uv run python examples/whip.py --url https://example.com/whip/channel --video-codec-type h265 --fake-capture-device
"""

import argparse
import logging
import queue
import random
import threading
import time
from math import pi
from typing import Optional
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
import sounddevice as sd
from wish import handle_error, parse_link_header

# blend2d-py
from blend2d import CompOp, Context, Image, Path

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


# ============================================================================
# Blend2D レンダラー（--fake-capture-device 用）
# ============================================================================


class MovingShape:
    """アニメーションする図形の基底クラス"""

    def __init__(self, x: float, y: float, vx: float, vy: float, r: int, g: int, b: int, alpha: int):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha
        self.vx_noise = random.uniform(-0.2, 0.2)
        self.vy_noise = random.uniform(-0.2, 0.2)

    def update(self, screen_width: int, screen_height: int, frame: int) -> None:
        noise_factor = 0.1 * np.sin(frame * 0.05)
        self.vx += self.vx_noise * noise_factor
        self.vy += self.vy_noise * noise_factor
        max_speed = 10.0
        self.vx = max(-max_speed, min(max_speed, self.vx))
        self.vy = max(-max_speed, min(max_speed, self.vy))
        self.x += self.vx
        self.y += self.vy

    def check_bounds(self, screen_width: int, screen_height: int) -> None:
        pass

    def draw(self, ctx: Context) -> None:
        pass


class MovingRect(MovingShape):
    """アニメーションする四角形"""

    def __init__(self, x: float, y: float, width: float, height: float, vx: float, vy: float, r: int, g: int, b: int, alpha: int):
        super().__init__(x, y, vx, vy, r, g, b, alpha)
        self.width = width
        self.height = height

    def check_bounds(self, screen_width: int, screen_height: int) -> None:
        if self.x <= 0 or self.x + self.width >= screen_width:
            self.vx = -self.vx
            self.x = max(0.0, min(self.x, screen_width - self.width))
        if self.y <= 0 or self.y + self.height >= screen_height:
            self.vy = -self.vy
            self.y = max(0.0, min(self.y, screen_height - self.height))

    def draw(self, ctx: Context) -> None:
        ctx.set_fill_style_rgba(self.r, self.g, self.b, self.alpha)
        ctx.fill_rect(self.x, self.y, self.width, self.height)


class MovingCircle(MovingShape):
    """アニメーションする円"""

    def __init__(self, x: float, y: float, radius: float, vx: float, vy: float, r: int, g: int, b: int, alpha: int):
        super().__init__(x, y, vx, vy, r, g, b, alpha)
        self.radius = radius

    def check_bounds(self, screen_width: int, screen_height: int) -> None:
        if self.x - self.radius <= 0 or self.x + self.radius >= screen_width:
            self.vx = -self.vx
            self.x = max(self.radius, min(self.x, screen_width - self.radius))
        if self.y - self.radius <= 0 or self.y + self.radius >= screen_height:
            self.vy = -self.vy
            self.y = max(self.radius, min(self.y, screen_height - self.radius))

    def draw(self, ctx: Context) -> None:
        ctx.set_fill_style_rgba(self.r, self.g, self.b, self.alpha)
        ctx.fill_circle(self.x, self.y, self.radius)


def draw_7segment(ctx: Context, digit: int, x: float, y: float, w: float, h: float) -> None:
    """7セグメント風の数字を描画"""
    if digit < 0 or digit > 9:
        return

    thickness = w * 0.15
    gap = thickness * 0.2

    segments = [
        [True, True, True, True, True, True, False],    # 0
        [False, True, True, False, False, False, False],  # 1
        [True, True, False, True, True, False, True],   # 2
        [True, True, True, True, False, False, True],   # 3
        [False, True, True, False, False, True, True],  # 4
        [True, False, True, True, False, True, True],   # 5
        [True, False, True, True, True, True, True],    # 6
        [True, True, True, False, False, False, False], # 7
        [True, True, True, True, True, True, True],     # 8
        [True, True, True, True, False, True, True],    # 9
    ]

    def draw_h(sx: float, sy: float) -> None:
        p = Path()
        p.move_to(sx + gap, sy)
        p.line_to(sx + w - gap, sy)
        p.line_to(sx + w - gap - thickness * 0.5, sy + thickness * 0.5)
        p.line_to(sx + w - gap, sy + thickness)
        p.line_to(sx + gap, sy + thickness)
        p.line_to(sx + gap + thickness * 0.5, sy + thickness * 0.5)
        p.close()
        ctx.fill_path(p)

    def draw_v(sx: float, sy: float, sh: float) -> None:
        p = Path()
        p.move_to(sx, sy + gap)
        p.line_to(sx + thickness * 0.5, sy + gap + thickness * 0.5)
        p.line_to(sx + thickness, sy + gap)
        p.line_to(sx + thickness, sy + sh - gap)
        p.line_to(sx + thickness * 0.5, sy + sh - gap - thickness * 0.5)
        p.line_to(sx, sy + sh - gap)
        p.close()
        ctx.fill_path(p)

    on = segments[digit]
    if on[0]: draw_h(x, y)
    if on[1]: draw_v(x + w - thickness, y, h * 0.5)
    if on[2]: draw_v(x + w - thickness, y + h * 0.5, h * 0.5)
    if on[3]: draw_h(x, y + h - thickness)
    if on[4]: draw_v(x, y + h * 0.5, h * 0.5)
    if on[5]: draw_v(x, y, h * 0.5)
    if on[6]: draw_h(x, y + h * 0.5 - thickness * 0.5)


def draw_colon(ctx: Context, x: float, y: float, h: float) -> None:
    dot = h * 0.1
    ctx.fill_circle(x + dot, y + h * 0.3, dot)
    ctx.fill_circle(x + dot, y + h * 0.7, dot)


def draw_digital_clock(ctx: Context, elapsed_ms: int, width: int, height: int) -> None:
    """デジタル時計を描画"""
    hours = (elapsed_ms // (60 * 60 * 1000)) % 10000
    minutes = (elapsed_ms // (60 * 1000)) % 60
    seconds = (elapsed_ms // 1000) % 60
    milliseconds = elapsed_ms % 1000

    clock_x = width * 0.02
    clock_y = height * 0.02
    digit_w = width * 0.025
    digit_h = height * 0.06
    spacing = digit_w * 0.3
    colon_w = digit_w * 0.3

    x = clock_x
    ctx.set_fill_style_rgba(0, 255, 255, 255)

    # HHHH
    for i in range(4):
        draw_7segment(ctx, (hours // (10 ** (3 - i))) % 10, x, clock_y, digit_w, digit_h)
        x += digit_w + spacing
    draw_colon(ctx, x, clock_y, digit_h)
    x += colon_w + spacing

    # MM
    draw_7segment(ctx, (minutes // 10) % 10, x, clock_y, digit_w, digit_h)
    x += digit_w + spacing
    draw_7segment(ctx, minutes % 10, x, clock_y, digit_w, digit_h)
    x += digit_w + spacing
    draw_colon(ctx, x, clock_y, digit_h)
    x += colon_w + spacing

    # SS
    draw_7segment(ctx, (seconds // 10) % 10, x, clock_y, digit_w, digit_h)
    x += digit_w + spacing
    draw_7segment(ctx, seconds % 10, x, clock_y, digit_w, digit_h)
    x += digit_w + spacing

    # .
    ctx.fill_circle(x + colon_w * 0.3, clock_y + digit_h * 0.8, digit_h * 0.05)
    x += colon_w + spacing

    # mmm (smaller)
    ms_w = digit_w * 0.7
    ms_h = digit_h * 0.7
    ctx.set_fill_style_rgba(200, 200, 200, 255)
    y_off = (digit_h - ms_h) / 2
    draw_7segment(ctx, (milliseconds // 100) % 10, x, clock_y + y_off, ms_w, ms_h)
    x += ms_w + spacing * 0.8
    draw_7segment(ctx, (milliseconds // 10) % 10, x, clock_y + y_off, ms_w, ms_h)
    x += ms_w + spacing * 0.8
    draw_7segment(ctx, milliseconds % 10, x, clock_y + y_off, ms_w, ms_h)


class Blend2DRenderer:
    """Blend2D を使用した映像生成"""

    def __init__(self, width: int, height: int, num_shapes: int = 15):
        self.width = width
        self.height = height
        self.img = Image(width, height)
        self.ctx = Context(self.img)
        self.frame_num = 0
        self.start_time = time.perf_counter()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (255, 128, 0), (128, 0, 255), (64, 255, 64), (255, 192, 203),
        ]

        self.shapes: list[MovingShape] = []
        for _ in range(num_shapes):
            x = random.randint(50, width - 150)
            y = random.randint(50, height - 150)
            vx = random.uniform(-6.0, 6.0)
            vy = random.uniform(-6.0, 6.0)
            color = random.choice(colors)
            alpha = random.randint(150, 220)

            if random.random() < 0.5:
                w = random.randint(40, 100)
                h = random.randint(40, 100)
                self.shapes.append(MovingRect(x, y, w, h, vx, vy, *color, alpha))
            else:
                r = random.randint(20, 50)
                self.shapes.append(MovingCircle(x, y, r, vx, vy, *color, alpha))

    def render_frame(self) -> np.ndarray:
        """フレームを描画して BGRA 配列を返す"""
        self.ctx.set_comp_op(CompOp.SRC_COPY)
        self.ctx.set_fill_style_rgba(30, 30, 30, 255)
        self.ctx.fill_all()

        self.ctx.set_comp_op(CompOp.SRC_OVER)
        elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)

        self.ctx.save()
        draw_digital_clock(self.ctx, elapsed_ms, self.width, self.height)
        self.ctx.restore()

        # 回転する円弧
        self.ctx.save()
        self.ctx.translate(self.width * 0.5, self.height * 0.5)
        self.ctx.rotate(-pi / 2)
        self.ctx.set_fill_style_rgba(255, 255, 255, 255)
        self.ctx.fill_pie(0, 0, min(self.width, self.height) * 0.15, 0, 2 * pi)
        self.ctx.set_fill_style_rgba(100, 200, 255, 255)
        sweep = (self.frame_num % 60) / 60.0 * 2 * pi
        self.ctx.fill_pie(0, 0, min(self.width, self.height) * 0.15, 0, sweep)
        self.ctx.restore()

        for shape in self.shapes:
            shape.update(self.width, self.height, self.frame_num)
            shape.check_bounds(self.width, self.height)
            shape.draw(self.ctx)

        self.frame_num += 1
        return self.img.asarray()


# ============================================================================
# WHIP クライアント
# ============================================================================


class WHIPClient:
    """WHIP クライアント（webcodecs-py ベース）"""

    def __init__(
        self,
        whip_url: str,
        bearer_token: Optional[str] = None,
        codec: str = "h264",
        use_fake_capture: bool = False,
        video_input_device: Optional[int] = None,
        audio_input_device: Optional[int] = None,
    ):
        self.whip_url = whip_url
        self.bearer_token = bearer_token
        self.codec = codec.lower()
        self.use_fake_capture = use_fake_capture
        self.video_input_device = video_input_device
        self.audio_input_device = audio_input_device

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

        # Blend2D renderer (for fake capture)
        self.renderer: Optional[Blend2DRenderer] = None
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

        # PeerConnection を作成（自動 gathering を無効化）
        config = Configuration()
        config.ice_servers = []
        config.disable_auto_gathering = True
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
            ice_servers = []
            if link_header:
                ice_servers = parse_link_header(link_header)
                if ice_servers:
                    logger.info(f"Found {len(ice_servers)} ICE server(s) in Link header")

            # リモート SDP を設定
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)

            # ICE サーバーがある場合、gathering を実行
            if ice_servers:
                self.pc.gather_local_candidates(ice_servers)
                logger.info("Gathering local candidates with ICE servers")

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
            codec_string = "avc1.64002A"  # H.264 High Profile Level 4.2

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
        self.camera = cv2.VideoCapture(self.video_input_device)
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
        # デバイス情報を取得してチャンネル数を決定
        device_info = sd.query_devices(self.audio_input_device, "input")
        max_channels = device_info["max_input_channels"]
        if max_channels < 1:
            logger.error(f"Audio device {self.audio_input_device} has no input channels")
            return
        # デバイスの最大チャンネル数を使用（モノラルかステレオ）
        self.audio_channels = min(max_channels, 2)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio capture status: {status}")
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

        self.audio_stream = sd.InputStream(
            device=self.audio_input_device,
            samplerate=self.audio_sample_rate,
            channels=self.audio_channels,
            dtype=np.float32,
            blocksize=self.audio_frame_size,
            callback=audio_callback,
        )
        self.audio_stream.start()
        logger.info(f"Audio capture started: {self.audio_sample_rate}Hz, {self.audio_channels}ch (device: {device_info['name']})")

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
        if self.use_fake_capture:
            # Blend2D レンダラーを初期化
            self.renderer = Blend2DRenderer(self.video_width, self.video_height)
            logger.info(
                f"Blend2D renderer initialized: {self.video_width}x{self.video_height} @ {self.video_fps}fps"
            )
        elif self.video_input_device is not None:
            self._start_camera_capture()
        if self.audio_input_device is not None:
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
        if self.use_fake_capture and self.renderer:
            # Blend2D でフレームを描画
            bgra_frame = self.renderer.render_frame()
        elif self.video_input_device is not None:
            # カメラからフレームを取得
            if self.video_queue.empty():
                return
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
        if self.audio_input_device is not None and not self.audio_queue.empty():
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
        "--video-codec-type",
        choices=["h264", "h265", "av1"],
        default="h264",
        help="映像コーデック (デフォルト: h264)",
    )
    parser.add_argument(
        "--fake-capture-device",
        action="store_true",
        help="テストパターン（blend2d）とテスト音声を使用",
    )
    parser.add_argument(
        "--video-input-device",
        type=int,
        help="ビデオ入力デバイス番号（未指定時はテストパターン）",
    )
    parser.add_argument(
        "--audio-input-device",
        type=int,
        help="オーディオ入力デバイス番号（未指定時はテスト音声）",
    )

    args = parser.parse_args()

    logger.info(f"Video codec: {args.video_codec_type}")
    logger.info(f"WHIP endpoint: {args.url}")
    if args.fake_capture_device:
        logger.info("Using fake capture device (blend2d video + test audio)")
    elif args.video_input_device is not None or args.audio_input_device is not None:
        logger.info(f"Video device: {args.video_input_device}, Audio device: {args.audio_input_device}")

    client = WHIPClient(
        args.url,
        args.token,
        args.video_codec_type,
        args.fake_capture_device,
        args.video_input_device,
        args.audio_input_device,
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
