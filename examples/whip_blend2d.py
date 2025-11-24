"""
blend2d + webcodecs + libdatachannel WHIP 配信サンプル

blend2d-py で映像を生成し、webcodecs-py でエンコードして、
libdatachannel-py で WHIP プロトコルにより配信します。

使い方:
    # H.264 で配信
    uv run python examples/whip_blend2d.py --url https://example.com/whip --token YOUR_TOKEN

    # AV1 で配信
    uv run python examples/whip_blend2d.py --url https://example.com/whip --token YOUR_TOKEN --codec av1

    # 10秒間だけ配信
    uv run python examples/whip_blend2d.py --url https://example.com/whip --duration 10
"""

import argparse
import logging
import random
import time
from math import pi, sin
from typing import Optional
from urllib.parse import urljoin

import httpx
import numpy as np
from wish import handle_error, parse_link_header

# blend2d-py
from blend2d import CompOp, Context, Image, Path

# webcodecs-py
from webcodecs import (
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

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        vx: float,
        vy: float,
        r: int,
        g: int,
        b: int,
        alpha: int,
    ):
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

    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        vx: float,
        vy: float,
        r: int,
        g: int,
        b: int,
        alpha: int,
    ):
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
        [True, True, True, True, True, True, False],  # 0
        [False, True, True, False, False, False, False],  # 1
        [True, True, False, True, True, False, True],  # 2
        [True, True, True, True, False, False, True],  # 3
        [False, True, True, False, False, True, True],  # 4
        [True, False, True, True, False, True, True],  # 5
        [True, False, True, True, True, True, True],  # 6
        [True, True, True, False, False, False, False],  # 7
        [True, True, True, True, True, True, True],  # 8
        [True, True, True, True, False, True, True],  # 9
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
    if on[0]:
        draw_h(x, y)
    if on[1]:
        draw_v(x + w - thickness, y, h * 0.5)
    if on[2]:
        draw_v(x + w - thickness, y + h * 0.5, h * 0.5)
    if on[3]:
        draw_h(x, y + h - thickness)
    if on[4]:
        draw_v(x, y + h * 0.5, h * 0.5)
    if on[5]:
        draw_v(x, y, h * 0.5)
    if on[6]:
        draw_h(x, y + h * 0.5 - thickness * 0.5)


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

    # :
    draw_colon(ctx, x, clock_y, digit_h)
    x += colon_w + spacing

    # MM
    draw_7segment(ctx, (minutes // 10) % 10, x, clock_y, digit_w, digit_h)
    x += digit_w + spacing
    draw_7segment(ctx, minutes % 10, x, clock_y, digit_w, digit_h)
    x += digit_w + spacing

    # :
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

        # 図形の初期化
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (64, 255, 64),
            (255, 192, 203),
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
        # 背景を黒で塗りつぶし
        self.ctx.set_comp_op(CompOp.SRC_COPY)
        self.ctx.set_fill_style_rgba(30, 30, 30, 255)
        self.ctx.fill_all()

        # アルファブレンディングを有効化
        self.ctx.set_comp_op(CompOp.SRC_OVER)

        # 経過時間の計算
        elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)

        # デジタル時計を描画
        self.ctx.save()
        draw_digital_clock(self.ctx, elapsed_ms, self.width, self.height)
        self.ctx.restore()

        # 回転する円弧を描画
        self.ctx.save()
        self.ctx.translate(self.width * 0.5, self.height * 0.5)
        self.ctx.rotate(-pi / 2)
        self.ctx.set_fill_style_rgba(255, 255, 255, 255)
        self.ctx.fill_pie(0, 0, min(self.width, self.height) * 0.15, 0, 2 * pi)
        self.ctx.set_fill_style_rgba(100, 200, 255, 255)
        sweep = (self.frame_num % 60) / 60.0 * 2 * pi
        self.ctx.fill_pie(0, 0, min(self.width, self.height) * 0.15, 0, sweep)
        self.ctx.restore()

        # 動く図形を描画
        for shape in self.shapes:
            shape.update(self.width, self.height, self.frame_num)
            shape.check_bounds(self.width, self.height)
            shape.draw(self.ctx)

        self.frame_num += 1

        # BGRA 配列として返す
        return self.img.asarray()


class WHIPBlend2DClient:
    """blend2d + webcodecs を使用した WHIP クライアント"""

    def __init__(
        self,
        whip_url: str,
        bearer_token: Optional[str] = None,
        codec: str = "h264",
    ):
        self.whip_url = whip_url
        self.bearer_token = bearer_token
        self.codec = codec.lower()
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Renderer
        self.renderer: Optional[Blend2DRenderer] = None

        # webcodecs Encoder
        self.video_encoder: Optional[VideoEncoder] = None

        # RTP components
        self.video_packetizer = None
        self.video_sr_reporter = None
        self.pli_handler = None
        self.nack_responder = None
        self.video_config: Optional[RtpPacketizationConfig] = None

        # Frame counters
        self.video_frame_number = 0
        self.encoded_frame_count = 0

        # Video settings
        self.video_width = 1920
        self.video_height = 1080
        self.video_fps = 60

        # Key frame interval
        self.key_frame_interval_frames = self.video_fps * 2  # 2秒ごと

    def connect(self) -> None:
        """WHIP サーバーに接続"""
        logger.info(f"Connecting to WHIP endpoint: {self.whip_url}")

        # PeerConnection を作成
        config = Configuration()
        config.ice_servers = []
        self.pc = PeerConnection(config)

        # オーディオトラックを追加（Sora は Opus 必須）
        audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # ビデオトラックを追加
        video_desc = Description.Video("video", Description.Direction.SendOnly)
        if self.codec == "av1":
            video_desc.add_av1_codec(35)
            # AV1 RTP には Dependency Descriptor ヘッダー拡張が必要
            dd_ext = Description.Entry.ExtMap(
                1,
                "https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension",
            )
            video_desc.add_ext_map(dd_ext)
        elif self.codec == "h265":
            video_desc.add_h265_codec(97)
        else:
            video_desc.add_h264_codec(96)
        self.video_track = self.pc.add_track(video_desc)

        # エンコーダーをセットアップ
        self._setup_video_encoder()

        # SDP オファーを生成
        self.pc.set_local_description()
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise RuntimeError("Failed to create offer")

        # WHIP サーバーにオファーを送信
        logger.info("Sending offer to WHIP server...")
        logger.debug(f"SDP Offer:\n{local_sdp}")
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
        encoded_chunks: list[EncodedVideoChunk] = []

        def on_output(chunk: EncodedVideoChunk) -> None:
            """エンコード完了時のコールバック"""
            if self.video_track and self.video_track.is_open():
                try:
                    # EncodedVideoChunk からデータを取得
                    data = np.zeros(chunk.byte_length, dtype=np.uint8)
                    chunk.copy_to(data)

                    # AV1 デバッグ: 最初の数フレームの OBU 構造を確認
                    if self.codec == "av1" and self.encoded_frame_count < 5:
                        self._debug_av1_obus(data, self.encoded_frame_count)

                    self.video_track.send(bytes(data))
                    self.encoded_frame_count += 1
                    if self.encoded_frame_count % 30 == 0:
                        logger.debug(f"Sent encoded frame #{self.encoded_frame_count}")
                except Exception as e:
                    handle_error("sending encoded video", e)

        def on_error(error: str) -> None:
            logger.error(f"Video encoder error: {error}")

        self.video_encoder = VideoEncoder(on_output, on_error)

        # エンコーダーを設定
        if self.codec == "av1":
            codec_string = "av01.0.04M.08"
        elif self.codec == "h265":
            codec_string = "hev1.1.6.L120.B0"  # H.265 Main Profile Level 4.0
        else:
            codec_string = "avc1.64001F"  # H.264 High Profile Level 3.1

        encoder_config: VideoEncoderConfig = {
            "codec": codec_string,
            "width": self.video_width,
            "height": self.video_height,
            "bitrate": 10_000_000,  # 10 Mbps
            "framerate": float(self.video_fps),
            "latency_mode": LatencyMode.REALTIME,
        }

        preferred_engine = None
        # H.264/H.265 の場合は Annex B 形式と VideoToolbox を指定
        if self.codec == "h264":
            encoder_config["avc"] = {"format": "annexb"}
            preferred_engine = HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX
            encoder_config["hardware_acceleration_engine"] = preferred_engine
        elif self.codec == "h265":
            encoder_config["hevc"] = {"format": "annexb"}
            preferred_engine = HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX
            encoder_config["hardware_acceleration_engine"] = preferred_engine

        def configure_with_fallback(config: VideoEncoderConfig) -> HardwareAccelerationEngine:
            try:
                self.video_encoder.configure(config)
                return config.get(
                    "hardware_acceleration_engine", HardwareAccelerationEngine.NONE
                )
            except RuntimeError as e:
                if preferred_engine:
                    raise RuntimeError(
                        "macOS で H.264 を利用するには VideoToolbox が必要です。"
                        "利用できない場合は `--codec av1` を指定してください。"
                    ) from e
                raise

        active_engine = configure_with_fallback(encoder_config)
        logger.info(f"Video encoder configured: {codec_string} (engine: {active_engine.value})")

        # RTP パケッタイザーをセットアップ
        if self.codec == "av1":
            payload_type = 35
        elif self.codec == "h265":
            payload_type = 97
        else:
            payload_type = 96
        clock_rate = 90000

        self.video_config = RtpPacketizationConfig(
            ssrc=random.randint(1, 0xFFFFFFFF),
            cname="video-stream",
            payload_type=payload_type,
            clock_rate=clock_rate,
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
                1200,  # max_fragment_size
            )
        else:
            self.video_packetizer = H264RtpPacketizer(
                NalUnit.Separator.LongStartSequence,
                self.video_config,
                1200,  # max_fragment_size
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

        # パケッタイザーをトラックに設定
        if self.video_track:
            self.video_track.set_media_handler(self.video_packetizer)

    def send_frames(self, duration: Optional[int] = None) -> None:
        """blend2d で生成したフレームを送信"""
        if not self.pc:
            raise RuntimeError("PeerConnection not initialized. Call connect() first.")

        # 接続を待機
        timeout = 10.0
        start_time = time.time()
        while self.pc.state() != PeerConnection.State.Connected:
            if time.time() - start_time > timeout:
                raise RuntimeError("Connection timeout")
            time.sleep(0.1)

        logger.info("Connection established")

        # レンダラーを初期化
        self.renderer = Blend2DRenderer(self.video_width, self.video_height)
        logger.info(
            f"Blend2D renderer initialized: {self.video_width}x{self.video_height} @ {self.video_fps}fps"
        )

        # フレーム送信ループ
        frame_interval = 1.0 / self.video_fps
        start_time = time.time()
        next_frame_time = start_time

        try:
            while True:
                current_time = time.time()

                # 終了判定
                if duration and current_time - start_time >= duration:
                    break

                # フレーム時間まで待機
                if current_time >= next_frame_time:
                    self._send_video_frame()
                    next_frame_time += frame_interval

                # CPU 使用率を抑える
                sleep_time = max(0, next_frame_time - time.time())
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    def _send_video_frame(self) -> None:
        """ビデオフレームを送信"""
        if not self.video_encoder or not self.renderer:
            return

        # blend2d でフレームを描画
        bgra_frame = self.renderer.render_frame()

        # VideoFrame を作成 (BGRA)
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

        # キーフレーム判定
        is_keyframe = self.video_frame_number % self.key_frame_interval_frames == 0

        # エンコード
        try:
            if is_keyframe:
                self.video_encoder.encode(frame, {"keyFrame": True})
            else:
                self.video_encoder.encode(frame)
        except Exception as e:
            handle_error("encoding video frame", e)

        frame.close()
        self.video_frame_number += 1

        # 進捗ログ
        if self.video_frame_number % self.video_fps == 0:
            elapsed = self.video_frame_number / self.video_fps
            logger.info(f"Video progress: {elapsed:.1f}s ({self.video_frame_number} frames)")

    def _debug_av1_obus(self, data: np.ndarray, frame_num: int) -> None:
        """AV1 OBU 構造をデバッグ出力"""
        OBU_TYPES = {
            0: "RESERVED", 1: "SEQUENCE_HEADER", 2: "TEMPORAL_DELIMITER",
            3: "FRAME_HEADER", 4: "TILE_GROUP", 5: "METADATA",
            6: "FRAME", 7: "REDUNDANT_FRAME_HEADER", 8: "TILE_LIST", 15: "PADDING",
        }

        logger.info(f"=== AV1 Frame #{frame_num} ({len(data)} bytes) ===")
        logger.info(f"  First 16 bytes: {' '.join(f'{b:02x}' for b in data[:16])}")

        idx = 0
        # Temporal Unit Delimiter チェック
        if len(data) >= 2 and data[0] == 0x12 and data[1] == 0x00:
            logger.info("  [TU] Temporal Unit Delimiter: FOUND (0x12 0x00)")
            idx = 2
        else:
            logger.info(f"  [TU] Temporal Unit Delimiter: NOT FOUND (first: 0x{data[0]:02x} 0x{data[1]:02x})")

        obu_num = 0
        while idx < len(data) and obu_num < 10:
            obu_header = data[idx]
            obu_type = (obu_header >> 3) & 0x0F
            has_extension = (obu_header >> 2) & 0x01
            has_size = (obu_header >> 1) & 0x01
            type_name = OBU_TYPES.get(obu_type, f"UNKNOWN({obu_type})")

            ext_size = 1 if has_extension else 0

            if not has_size:
                remaining = len(data) - idx - 1 - ext_size
                logger.info(
                    f"  [OBU {obu_num}] type={type_name}, has_size=0 (implicit), "
                    f"header=0x{obu_header:02x}, remaining={remaining} bytes"
                )
                break

            # LEB128 サイズを読む
            size_idx = idx + 1 + ext_size
            obu_size = 0
            leb128_bytes = 0
            while size_idx + leb128_bytes < len(data) and leb128_bytes < 8:
                b = int(data[size_idx + leb128_bytes])
                obu_size |= (b & 0x7F) << (leb128_bytes * 7)
                leb128_bytes += 1
                if not (b & 0x80):
                    break

            header_size = 1 + ext_size + leb128_bytes
            total_size = header_size + obu_size
            logger.info(
                f"  [OBU {obu_num}] type={type_name}, has_size=1, header=0x{obu_header:02x}, "
                f"payload={obu_size}, total={total_size}"
            )

            idx += total_size
            obu_num += 1

    def disconnect(self) -> None:
        """切断処理"""
        logger.info("Starting graceful shutdown...")

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
            except Exception as e:
                handle_error("flushing video encoder", e)

        time.sleep(0.5)

        # リソースをクリーンアップ
        self.video_packetizer = None
        self.video_sr_reporter = None
        self.pli_handler = None
        self.nack_responder = None
        self.video_track = None

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
            except Exception as e:
                handle_error("closing video encoder", e)
            finally:
                self.video_encoder = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(
        description="blend2d + webcodecs WHIP クライアント"
    )
    parser.add_argument("--url", required=True, help="WHIP エンドポイント URL")
    parser.add_argument("--token", help="Bearer トークン（認証用）")
    parser.add_argument("--duration", type=int, help="配信時間（秒）")
    parser.add_argument(
        "--codec",
        choices=["h264", "h265", "av1"],
        default="h264",
        help="映像コーデック (デフォルト: h264)",
    )

    args = parser.parse_args()

    logger.info(f"Video codec: {args.codec}")
    logger.info(f"WHIP endpoint: {args.url}")

    client = WHIPBlend2DClient(args.url, args.token, args.codec)

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
