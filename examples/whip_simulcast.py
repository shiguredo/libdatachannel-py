"""
WHIP (WebRTC-HTTP Ingestion Protocol) クライアント (Simulcast)

webcodecs-py でエンコードして libdatachannel-py で WHIP 配信します。

使い方:
    # カメラとマイクを使用（デバイス番号 0 がデフォルト）
    uv run python examples/whip_simulcast.py --url https://example.com/whip/channel

    # デバイス番号を指定
    uv run python examples/whip_simulcast.py --url https://example.com/whip/channel --video-input-device 1 --audio-input-device 1

    # テストパターンで配信（blend2d + テスト音声）
    uv run python examples/whip_simulcast.py --url https://example.com/whip/channel --fake-capture-device

    # H.265 で 60fps 配信
    uv run python examples/whip_simulcast.py --url https://example.com/whip/channel --video-codec-type h265 --framerate 60

    # Simulcast 層数を指定 (デフォルト: 3, 最大: 4)
    uv run python examples/whip_simulcast.py --url https://example.com/whip/channel --whip-simulcast-total-layers 3
"""

import argparse
from dataclasses import dataclass
import logging
import queue
import random
import re
import threading
import time
from math import pi
from typing import List, Optional
from urllib.parse import urljoin

import httpx
import numpy as np
import structlog

# portaudio-py
import portaudio as pa

# uvc-py
import uvc

# blend2d-py
from blend2d import CompOp, Context, Image, Path

# webcodecs-py
from webcodecs import (
    AudioData,
    AudioDataInit,
    AudioEncoder,
    AudioEncoderConfig,
    AudioSampleFormat,
    AVCNalUnitType,
    EncodedAudioChunk,
    EncodedVideoChunk,
    HardwareAccelerationEngine,
    HEVCNalUnitType,
    LatencyMode,
    parse_avc_annexb,
    parse_hevc_annexb,
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
    IceServer,
    NalUnit,
    OpusRtpPacketizer,
    PeerConnection,
    PliHandler,
    RtcpNackResponder,
    RtcpSrReporter,
    RtpPacketizationConfig,
    Track,
)

logger = structlog.get_logger(__name__)

RTP_HEADER_EXT_URI_MID = "urn:ietf:params:rtp-hdrext:sdes:mid"
RTP_HEADER_EXT_URI_RID = "urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id"


# ============================================================================
# ユーティリティ関数（whep.py と共有）
# ============================================================================


@dataclass
class VideoLayerState:
    ssrc: int
    rid: str
    sequence_number: int
    timestamp: int
    last_dts_usec: int = 0
    first_dts_usec: Optional[int] = None
    last_keyframe_epoch: int = 0


def handle_error(context: str, error: Exception) -> None:
    """エラーハンドリング"""
    logger.error(f"Error {context}: {error}")
    if logger.isEnabledFor(logging.DEBUG):
        import traceback

        traceback.print_exc()


def get_nal_type_name(nal_type: int) -> str:
    """H.264 NAL タイプ名を取得 (webcodecs-py の AVCNalUnitType を使用)"""
    try:
        return AVCNalUnitType(nal_type).name
    except ValueError:
        return f"Reserved/Unknown ({nal_type})"


def parse_link_header(link_header: str) -> List[IceServer]:
    """Link ヘッダーから ICE サーバーを取得"""
    ice_servers: List[IceServer] = []
    if not link_header:
        return ice_servers

    # Parse Link header: <turn:turn.example.com>; rel="ice-server"; username="user"; credential="pass"
    # Split by comma to handle multiple servers
    entries = []
    current = ""
    in_quotes = False

    for char in link_header:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            entries.append(current.strip())
            current = ""
            continue
        current += char
    if current:
        entries.append(current.strip())

    for entry in entries:
        # Extract URL from <...>
        url_match = re.match(r"<([^>]+)>", entry)
        if not url_match:
            continue

        url = url_match.group(1)

        # Skip TURN TCP as it's not supported by libdatachannel
        if "transport=tcp" in url.lower() or "?tcp" in url.lower():
            logger.info(f"Skipping TURN TCP server (not supported): {url}")
            continue

        # Check if it's an ICE server
        if 'rel="ice-server"' not in entry:
            continue

        if url.startswith("stun:") or url.startswith("turn:"):
            ice_server = IceServer(url)

            # Extract username
            username_match = re.search(r'username="([^"]+)"', entry)
            if username_match:
                ice_server.username = username_match.group(1)

            # Extract credential
            credential_match = re.search(r'credential="([^"]+)"', entry)
            if credential_match:
                ice_server.password = credential_match.group(1)

            ice_servers.append(ice_server)
            logger.info(f"Added ICE server from Link header: {url}")
            if hasattr(ice_server, "username") and ice_server.username:
                logger.info(f"  with username: {ice_server.username}")

    return ice_servers


def simulcast_layers_in_answer(answer: str) -> int:
    """Answer SDP の simulcast 受理レイヤ数を取得"""
    layers_start = answer.find("a=simulcast")
    if layers_start == -1:
        return 0

    layers_end = answer.find("\r\n", layers_start)
    if layers_end == -1:
        return 0

    layers_accepted = 1
    for i in range(layers_start, layers_end):
        if answer[i] == ";":
            layers_accepted += 1

    return layers_accepted


def get_h265_nal_type_name(nal_type: int) -> str:
    """H.265 NAL タイプ名を取得 (webcodecs-py の HEVCNalUnitType を使用)"""
    try:
        return HEVCNalUnitType(nal_type).name
    except ValueError:
        return f"Unknown({nal_type})"


# ============================================================================
# Blend2D レンダラー（--fake-capture-device 用）
# ============================================================================


class MovingShape:
    """アニメーションする図形の基底クラス"""

    def __init__(
        self, x: float, y: float, vx: float, vy: float, r: int, g: int, b: int, alpha: int
    ):
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
                self.shapes.append(MovingRect(x, y, w, h, vx, vy, color[0], color[1], color[2], alpha))
            else:
                radius = random.randint(20, 50)
                self.shapes.append(MovingCircle(x, y, radius, vx, vy, color[0], color[1], color[2], alpha))

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
        framerate: int = 30,
        bitrate: int = 5_000_000,
        simulcast_total_layers: int = 3,
        disable_audio_processing: bool = False,
    ):
        self.whip_url = whip_url
        self.bearer_token = bearer_token
        self.codec = codec.lower()
        self.use_fake_capture = use_fake_capture
        self.video_input_device = video_input_device
        self.audio_input_device = audio_input_device
        self.video_bitrate = bitrate
        self.simulcast_total_layers = simulcast_total_layers
        self.disable_audio_processing = disable_audio_processing

        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # webcodecs Encoders
        self.video_encoders: List[VideoEncoder] = []
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
        self.video_layer_states: List[VideoLayerState] = []
        self.video_send_lock = threading.Lock()

        # Frame counters
        self.video_frame_number = 0
        self.audio_frame_number = 0
        self.encoded_video_count = 0
        self.encoded_audio_count = 0

        # Encoded frame queue for audio pacing
        self.encoded_audio_queue: queue.Queue = queue.Queue()

        # Video settings
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = framerate

        # Audio settings
        self.audio_sample_rate = 48000
        self.audio_channels = 1  # モノラル（多くのマイクは1ch）

        # Blend2D renderer (for fake capture)
        self.renderer: Optional[Blend2DRenderer] = None
        self.audio_frame_size = 960  # 20ms @ 48kHz

        self.last_audio_dts_usec: int = 0

        # Key frame interval（90秒ごと、ただし最初のフレームと PLI 応答時はキーフレーム）
        self.key_frame_interval_frames = self.video_fps * 90

        # PLI によるキーフレーム強制フラグ（全レイヤへ一斉伝播させるための epoch）
        self.force_keyframe_epoch = 0

        # Camera capture (uvc-py)
        self.uvc_device: Optional[uvc.Device] = None
        self.capture_active = False

        # Audio capture (portaudio-py)
        self.audio_stream: Optional[pa.Stream] = None

        # Test pattern state
        self.pattern_seed = 0

        # Running flag
        self._running = False

    def connect(self) -> None:
        """WHIP サーバーに接続（OBS の libdatachannel 実装を模倣）"""
        logger.info("Connecting to WHIP endpoint", url=self.whip_url)

        # PeerConnection を作成（WHIP では ICE は Link ヘッダー経由で取得）
        config = Configuration()
        config.ice_servers = []
        config.disable_auto_gathering = True
        config.force_media_transport = True
        self.pc = PeerConnection(config)

        # 状態変更コールバック（デバッグ用）
        def on_state_change(state: PeerConnection.State) -> None:
            logger.info("PeerConnection state changed", state=str(state))

        def on_ice_state_change(state: PeerConnection.IceState) -> None:
            logger.debug("ICE state changed", state=str(state))

        def on_gathering_state_change(state: PeerConnection.GatheringState) -> None:
            logger.debug("Gathering state changed", state=str(state))

        self.pc.on_state_change(on_state_change)
        self.pc.on_ice_state_change(on_ice_state_change)
        self.pc.on_gathering_state_change(on_gathering_state_change)

        # SSRC と cname を生成（OBS と同様）
        self.base_ssrc = random.randint(1, 0xFFFFFFFF)
        self.cname = "".join(
            random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", k=16)
        )
        media_stream_id = "".join(
            random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", k=16)
        )

        # オーディオトラックを追加（OBS と同様に SSRC を設定）
        audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        audio_desc.add_opus_codec(111)
        audio_desc.add_ssrc(self.base_ssrc, self.cname, media_stream_id, f"{media_stream_id}-audio")
        self.audio_track = self.pc.add_track(audio_desc)

        # ビデオトラックを追加（OBS と同様に SSRC を設定）
        video_desc = Description.Video("video", Description.Direction.SendOnly)
        if self.codec == "av1":
            video_desc.add_av1_codec(35)
        elif self.codec == "h265":
            video_desc.add_h265_codec(97)
        else:
            video_desc.add_h264_codec(96)
        # RID/MID を SDP に反映して simulcast を明示
        video_desc.add_ext_map(Description.Entry.ExtMap(1, RTP_HEADER_EXT_URI_MID))
        video_desc.add_ext_map(Description.Entry.ExtMap(2, RTP_HEADER_EXT_URI_RID))
        if self.simulcast_total_layers > 1:
            for layer_index in range(self.simulcast_total_layers):
                video_desc.add_rid(str(layer_index))
        video_desc.add_ssrc(
            self.base_ssrc + 1, self.cname, media_stream_id, f"{media_stream_id}-video"
        )
        self.video_track = self.pc.add_track(video_desc)

        # エンコーダーをセットアップ（simulcast 前提の複数エンコーダ）
        self._setup_video_encoder()
        self._setup_audio_encoder()

        # SDP オファーを生成
        self.pc.set_local_description()
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise RuntimeError("Failed to create offer")

        logger.debug("Offer SDP:\n" + str(local_sdp))

        # WHIP サーバーにオファーを送信
        logger.info("Sending offer to WHIP server...")
        headers = {"Content-Type": "application/sdp"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                self.whip_url,
                content=str(local_sdp),
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 201:
                raise RuntimeError(f"WHIP server returned {response.status_code}: {response.text}")

            logger.debug("Response headers", headers=dict(response.headers))

            # セッション URL を取得
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whip_url, self.session_url)
            logger.debug("Resource URL", url=self.session_url)

            # Link ヘッダーから ICE サーバーを取得
            link_header = response.headers.get("Link")
            ice_servers = []
            if link_header:
                ice_servers = parse_link_header(link_header)
                if ice_servers:
                    logger.info("Found ICE servers in Link header", count=len(ice_servers))

            # リモート SDP を設定
            logger.debug("Answer SDP:\n" + response.text)
            if self.simulcast_total_layers > 1:
                layers_accepted = simulcast_layers_in_answer(response.text)
                if layers_accepted != self.simulcast_total_layers:
                    raise RuntimeError(
                        f"WHIP server only accepted {layers_accepted} simulcast layers"
                    )
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)

            # ICE サーバーがある場合、gathering を実行（OBS と同様に待たない）
            if ice_servers:
                logger.info("Starting ICE gathering with TURN servers...")
                self.pc.gather_local_candidates(ice_servers)

        logger.info("WHIP signaling completed")

    def _setup_video_encoder(self) -> None:
        """webcodecs-py ビデオエンコーダーをセットアップ"""

        # simulcast レイヤ数を 1〜4 に丸める
        simulcast_layers = max(1, min(self.simulcast_total_layers, 4))

        def on_output(layer_index: int):
            def handler(chunk: EncodedVideoChunk) -> None:
                if not self.video_track or not self.video_track.is_open():
                    return

                try:
                    data = np.zeros(chunk.byte_length, dtype=np.uint8)
                    chunk.copy_to(data)

                    layer = self.video_layer_states[layer_index]
                    dts_usec = chunk.timestamp
                    if layer.first_dts_usec is None:
                        # レイヤごとの RTP タイムスタンプ基準を固定
                        layer.first_dts_usec = dts_usec
                        layer.last_dts_usec = dts_usec

                    duration = dts_usec - layer.last_dts_usec
                    if duration < 0:
                        duration = 0

                    elapsed_seconds = float(duration) / 1_000_000.0
                    elapsed_timestamp = int(elapsed_seconds * 90000)
                    layer.timestamp += elapsed_timestamp

                    # 単一 packetizer を共有するため送信は直列化
                    with self.video_send_lock:
                        if self.video_config is None:
                            return
                        # レイヤごとに SSRC/RID/シーケンス/タイムスタンプを差し替える
                        self.video_config.ssrc = layer.ssrc
                        self.video_config.rid = layer.rid
                        self.video_config.sequence_number = layer.sequence_number
                        self.video_config.timestamp = layer.timestamp
                        self.video_track.send(bytes(data))
                        layer.sequence_number = self.video_config.sequence_number
                        layer.timestamp = self.video_config.timestamp

                    layer.last_dts_usec = dts_usec

                    if layer_index == 0:
                        self.encoded_video_count += 1
                        if self.encoded_video_count % 30 == 0:
                            logger.debug(
                                f"Sent #{self.encoded_video_count} dts={dts_usec / 1000:.0f}ms "
                                f"duration={duration / 1000:.1f}ms rtp_ts={layer.timestamp}"
                            )
                except Exception as e:
                    handle_error(f"sending encoded video (layer {layer_index})", e)

            return handler

        def on_error(layer_index: int):
            def handler(error: str) -> None:
                logger.error(f"Video encoder error (layer {layer_index}): {error}")

            return handler

        if self.codec == "av1":
            codec_string = "av01.0.04M.08"
        elif self.codec == "h265":
            codec_string = "hev1.1.6.L120.B0"
        else:
            codec_string = "avc1.64002A"  # H.264 High Profile Level 4.2

        # OBS と同様に段階的にビットレート/解像度を下げる
        bitrate_step = max(1, self.video_bitrate // simulcast_layers)
        width_step = self.video_width // simulcast_layers
        height_step = self.video_height // simulcast_layers

        for layer_index in range(simulcast_layers):
            if layer_index == 0:
                width = self.video_width
                height = self.video_height
                bitrate = self.video_bitrate
            else:
                scale = simulcast_layers - layer_index
                width = width_step * scale
                height = height_step * scale
                width -= width % 2
                height -= height % 2
                bitrate = max(1, self.video_bitrate - (bitrate_step * layer_index))

            encoder_config: VideoEncoderConfig = {
                "codec": codec_string,
                "width": width,
                "height": height,
                "bitrate": bitrate,
                "latency_mode": LatencyMode.REALTIME,
            }
            # 元フレームはそのまま渡し、内部リサイズに任せる

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

            encoder = VideoEncoder(on_output(layer_index), on_error(layer_index))
            encoder.configure(encoder_config)
            self.video_encoders.append(encoder)
            logger.info(
                "Video encoder configured",
                layer=layer_index,
                codec=codec_string,
                width=width,
                height=height,
                bitrate=bitrate,
            )

        # RTP パケッタイザーをセットアップ
        if self.codec == "av1":
            payload_type = 35
        elif self.codec == "h265":
            payload_type = 97
        else:
            payload_type = 96

        # SDP の extmap と揃えるため MID/RID を明示
        self.video_config = RtpPacketizationConfig(
            ssrc=self.base_ssrc + 1,  # Description と同じ SSRC
            cname=self.cname,
            payload_type=payload_type,
            clock_rate=90000,
        )
        self.video_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        self.video_config.timestamp = self.video_config.start_timestamp
        self.video_config.sequence_number = random.randint(0, 0xFFFF)
        self.video_config.mid_id = 1
        self.video_config.rid_id = 2
        self.video_config.mid = "video"

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
            logger.info("PLI received - forcing keyframe")
            # 全レイヤに確実に反映するため epoch を進める
            self.force_keyframe_epoch += 1

        self.pli_handler = PliHandler(on_pli)
        self.video_packetizer.add_to_chain(self.pli_handler)

        # NACK responder
        self.nack_responder = RtcpNackResponder()
        self.video_packetizer.add_to_chain(self.nack_responder)

        if self.video_track:
            self.video_track.set_media_handler(self.video_packetizer)

        # レイヤ状態を初期化
        self.video_layer_states.clear()
        for layer_index in range(simulcast_layers):
            # レイヤごとの SSRC/RID/シーケンスを独立させる
            start_ts = random.randint(0, 0xFFFFFFFF)
            self.video_layer_states.append(
                VideoLayerState(
                    ssrc=self.base_ssrc + 1 + layer_index,
                    rid=str(layer_index),
                    sequence_number=random.randint(0, 0xFFFF),
                    timestamp=start_ts,
                )
            )

    def _setup_audio_encoder(self) -> None:
        """webcodecs-py オーディオエンコーダーをセットアップ"""

        def on_output(chunk: EncodedAudioChunk) -> None:
            # エンコード結果をキューに入れる（送信は別スレッドで一定間隔で行う）
            try:
                data = np.zeros(chunk.byte_length, dtype=np.uint8)
                chunk.copy_to(data)
                self.encoded_audio_queue.put((chunk.timestamp, bytes(data)))
            except Exception as e:
                handle_error("queueing encoded audio", e)

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
            ssrc=self.base_ssrc,  # Description と同じ SSRC
            cname=self.cname,
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
        """カメラキャプチャを開始 (uvc-py)"""
        # デバイスを開く
        try:
            if self.video_input_device is not None:
                self.uvc_device = uvc.open(self.video_input_device)
            else:
                # デフォルトデバイス (index 0)
                self.uvc_device = uvc.open(0)
        except RuntimeError as e:
            logger.error(f"Failed to open camera: {e}")
            return

        logger.info(f"Camera device: {self.uvc_device.info.name}")

        # サポートされているフォーマットを確認
        formats = self.uvc_device.get_supported_formats()
        if not formats:
            logger.error("No supported formats found")
            return

        # 要求解像度に近いフォーマットを探す (NV12 優先)
        selected = None
        for fmt in formats:
            if (
                fmt.width == self.video_width
                and fmt.height == self.video_height
                and fmt.fps == self.video_fps
            ):
                if fmt.format == uvc.Format.NV12:
                    selected = fmt
                    break
                elif selected is None:
                    selected = fmt

        # 見つからなければ 30fps の NV12 フォーマットを探す
        if selected is None:
            for fmt in formats:
                if fmt.fps == 30 and fmt.format == uvc.Format.NV12:
                    selected = fmt
                    break

        # それでも見つからなければ最初のフォーマット
        if selected is None:
            selected = formats[0]

        # 実際の解像度に合わせる
        self.video_width = selected.width
        self.video_height = selected.height
        self.video_fps = selected.fps

        logger.info(
            f"Camera opened: {self.video_width}x{self.video_height} @ {self.video_fps}fps "
            f"(format: {selected.format})"
        )

        # キャプチャ開始 (NV12 出力)
        self.uvc_device.start(
            self.video_width,
            self.video_height,
            self.video_fps,
            capture_format=selected.format,
            output_format=uvc.Format.NV12,
        )
        self.capture_active = True

    def _start_audio_capture(self) -> None:
        """マイクキャプチャを開始 (portaudio-py)"""
        # デバイス情報を取得してチャンネル数を決定
        # audio_input_device が None の場合はシステムデフォルトを使用
        device = self.audio_input_device
        if device is None:
            device = pa.get_default_input_device()
            if device == pa.NO_DEVICE:
                logger.error("No default input device found")
                return

        device_info = pa.get_device_info(device)
        if device_info is None:
            logger.error(f"Failed to get device info for device {device}")
            return

        max_channels = device_info.max_input_channels
        if max_channels < 1:
            logger.error(f"Audio device {device} has no input channels")
            return
        # デバイスの最大チャンネル数を使用（モノラルかステレオ）
        self.audio_channels = min(max_channels, 2)

        # 入力パラメータを作成
        input_params = pa.StreamParameters(
            device=device,
            channel_count=self.audio_channels,
            sample_format=pa.FLOAT32,
            suggested_latency=device_info.default_low_input_latency,
        )

        # ストリームを開く
        self.audio_stream = pa.Stream(
            input_parameters=input_params,
            sample_rate=self.audio_sample_rate,
            frames_per_buffer=self.audio_frame_size,
        )
        self.audio_stream.start()
        logger.info(
            f"Audio capture started: {self.audio_sample_rate}Hz, {self.audio_channels}ch (device: {device_info.name})"
        )

    def _generate_test_audio(self) -> np.ndarray:
        """テストオーディオを生成（サイン波）"""
        t = np.linspace(
            self.audio_frame_number * self.audio_frame_size / self.audio_sample_rate,
            (self.audio_frame_number + 1) * self.audio_frame_size / self.audio_sample_rate,
            self.audio_frame_size,
            dtype=np.float32,
        )
        # 無音
        mono = np.zeros_like(t)
        if self.audio_channels == 1:
            return mono.reshape(-1, 1)
        else:
            return np.column_stack([mono] * self.audio_channels)

    def send_frames(self, duration: Optional[int] = None) -> None:
        """フレームを送信"""
        if not self.pc:
            raise RuntimeError("PeerConnection not initialized")

        # WHIP シグナリング後の接続完了を待機
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
        else:
            self._start_camera_capture()

        # 音声は fake モードの有無に関わらず明示的に開始
        if not self.use_fake_capture:
            self._start_audio_capture()

        logger.info(f"Sending frames: {self.video_width}x{self.video_height} @ {self.video_fps}fps")

        # フレーム送信（エンコード完了時に on_output で直接送信）
        frame_interval = 1.0 / self.video_fps
        audio_interval = self.audio_frame_size / self.audio_sample_rate
        self._running = True

        def video_encode_loop():
            """ビデオエンコードループ（送信は on_output コールバックで行う）"""
            if self.renderer:
                # Blend2D モード: 固定間隔でエンコード
                next_time = time.perf_counter()
                while self._running:
                    now = time.perf_counter()
                    if now >= next_time:
                        self._encode_video_frame()
                        next_time += frame_interval
                    time.sleep(0.001)
            else:
                # カメラモード (uvc-py): フレーム到着を待つ
                while self._running and self.uvc_device:
                    uvc_frame = self.uvc_device.get_frame()
                    if uvc_frame is not None:
                        self._encode_camera_frame(uvc_frame)
                    else:
                        time.sleep(0.001)

        def audio_encode_loop():
            """オーディオエンコードループ"""
            next_audio_time = time.perf_counter()
            while self._running:
                now = time.perf_counter()
                if now >= next_audio_time:
                    self._encode_audio_frame()
                    next_audio_time += audio_interval
                time.sleep(0.001)

        def audio_send_loop():
            """オーディオ送信ループ（一定間隔）"""
            next_time = time.perf_counter() + audio_interval
            while self._running:
                now = time.perf_counter()
                if now >= next_time:
                    self._send_encoded_audio()
                    next_time += audio_interval
                time.sleep(0.001)

        video_encode_thread = threading.Thread(target=video_encode_loop, daemon=True)
        audio_encode_thread = None
        audio_send_thread = None
        if not self.disable_audio_processing:
            audio_encode_thread = threading.Thread(target=audio_encode_loop, daemon=True)
            audio_send_thread = threading.Thread(target=audio_send_loop, daemon=True)

        video_encode_thread.start()
        if audio_encode_thread:
            audio_encode_thread.start()
        if audio_send_thread:
            audio_send_thread.start()

        # メインスレッドは終了条件の監視のみ
        try:
            start_time = time.time()
            while True:
                if duration and time.time() - start_time >= duration:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._running = False
            video_encode_thread.join(timeout=1.0)
            if audio_encode_thread:
                audio_encode_thread.join(timeout=1.0)
            if audio_send_thread:
                audio_send_thread.join(timeout=1.0)

    def _encode_camera_frame(self, uvc_frame: uvc.Frame) -> None:
        """カメラフレームをエンコード (NV12)"""
        if not self.video_encoders:
            return

        t0 = time.perf_counter()

        # NV12 フォーマットで Y と UV プレーンを取得
        y_plane, uv_plane = uvc_frame.to_nv12()

        t1 = time.perf_counter()

        self._encode_nv12_frame(y_plane, uv_plane, t0, t1)

    def _encode_video_frame(self) -> None:
        """ビデオフレームをエンコード（Blend2D モード用）"""
        if not self.video_encoders or not self.renderer:
            return

        t0 = time.perf_counter()
        bgra_frame = self.renderer.render_frame()
        t1 = time.perf_counter()

        self._encode_bgra_frame(bgra_frame, t0, t1)

    def _encode_bgra_frame(self, bgra_frame: np.ndarray, t0: float, t1: float) -> None:
        """BGRA フレームをエンコード"""

        # タイムスタンプはフレーム番号から計算（マイクロ秒単位）
        # これは webcodecs の timestamp として使用され、on_output で RTP timestamp に変換される
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

        # エンコード（PLI による強制キーフレームも考慮）
        force_epoch = self.force_keyframe_epoch
        is_keyframe = self.video_frame_number % self.key_frame_interval_frames == 0
        try:
            for layer_index, encoder in enumerate(self.video_encoders):
                layer = self.video_layer_states[layer_index]
                force_by_pli = layer.last_keyframe_epoch < force_epoch
                send_keyframe = is_keyframe or force_by_pli
                if force_by_pli and layer_index == 0:
                    logger.info(
                        f"Sending keyframe in response to PLI (frame #{self.video_frame_number})"
                    )
                if send_keyframe:
                    encoder.encode(frame, {"key_frame": True})
                    if force_by_pli:
                        layer.last_keyframe_epoch = force_epoch
                else:
                    encoder.encode(frame)
        except Exception as e:
            handle_error("encoding video frame", e)

        frame.close()
        self.video_frame_number += 1

        t2 = time.perf_counter()

        # パフォーマンス計測（1秒ごとに出力）
        if self.video_frame_number % self.video_fps == 0:
            render_ms = (t1 - t0) * 1000
            encode_ms = (t2 - t1) * 1000
            logger.debug(
                f"Frame #{self.video_frame_number}: render={render_ms:.1f}ms, encode={encode_ms:.1f}ms"
            )

    def _encode_nv12_frame(
        self, y_plane: np.ndarray, uv_plane: np.ndarray, t0: float, t1: float
    ) -> None:
        """NV12 フレームをエンコード"""
        if not self.video_encoders:
            return

        # タイムスタンプはフレーム番号から計算（マイクロ秒単位）
        timestamp_us = int(self.video_frame_number * 1_000_000 / self.video_fps)

        # NV12 VideoFrame を作成
        nv12_init: VideoFrameBufferInit = {
            "format": VideoPixelFormat.NV12,
            "coded_width": self.video_width,
            "coded_height": self.video_height,
            "timestamp": timestamp_us,
        }
        frame = VideoFrame(y_plane, uv_plane, nv12_init)  # type: ignore[call-overload]

        # エンコード（PLI による強制キーフレームも考慮）
        force_epoch = self.force_keyframe_epoch
        is_keyframe = self.video_frame_number % self.key_frame_interval_frames == 0
        try:
            for layer_index, encoder in enumerate(self.video_encoders):
                layer = self.video_layer_states[layer_index]
                force_by_pli = layer.last_keyframe_epoch < force_epoch
                send_keyframe = is_keyframe or force_by_pli
                if force_by_pli and layer_index == 0:
                    logger.info(
                        f"Sending keyframe in response to PLI (frame #{self.video_frame_number})"
                    )
                if send_keyframe:
                    encoder.encode(frame, {"key_frame": True})
                    if force_by_pli:
                        layer.last_keyframe_epoch = force_epoch
                else:
                    encoder.encode(frame)
        except Exception as e:
            handle_error("encoding video frame", e)

        frame.close()
        self.video_frame_number += 1

        t2 = time.perf_counter()

        # パフォーマンス計測（1秒ごとに出力）
        if self.video_frame_number % self.video_fps == 0:
            capture_ms = (t1 - t0) * 1000
            encode_ms = (t2 - t1) * 1000
            logger.debug(
                f"Frame #{self.video_frame_number}: capture={capture_ms:.1f}ms, encode={encode_ms:.1f}ms"
            )

    def _encode_audio_frame(self) -> None:
        """オーディオフレームをエンコード (portaudio-py)"""
        if not self.audio_encoder:
            return

        # オーディオを取得
        if self.use_fake_capture:
            # テスト音声（無音）
            audio_samples = self._generate_test_audio()
        else:
            # マイクからオーディオを取得 (portaudio-py)
            if not self.audio_stream:
                return
            try:
                # float32 形式で読み込み (shape: [frames, channels])
                audio_samples = self.audio_stream.read_float32(self.audio_frame_size)
            except Exception as e:
                handle_error("reading audio", e)
                return

        # AudioData を作成（timestamp はサンプル数から計算）
        timestamp_us = int(
            self.audio_frame_number * self.audio_frame_size * 1_000_000 / self.audio_sample_rate
        )

        init: AudioDataInit = {
            "format": AudioSampleFormat.F32,
            "sample_rate": self.audio_sample_rate,
            "number_of_frames": len(audio_samples),
            "number_of_channels": self.audio_channels,
            "timestamp": timestamp_us,
            "data": audio_samples.astype(np.float32),
        }
        audio_data = AudioData(init)

        # エンコード（送信は別スレッドで pacing）
        try:
            self.audio_encoder.encode(audio_data)
        except Exception as e:
            handle_error("encoding audio frame", e)

        audio_data.close()
        self.audio_frame_number += 1

    def _send_encoded_audio(self) -> None:
        """エンコード済みオーディオを送信"""
        if self.encoded_audio_queue.empty():
            return
        if not self.audio_track or not self.audio_track.is_open():
            return

        try:
            timestamp_us, data = self.encoded_audio_queue.get_nowait()

            # duration を計算（前フレームとの差分）
            duration = timestamp_us - self.last_audio_dts_usec

            # duration を秒に変換
            elapsed_seconds = float(duration) / 1_000_000.0

            # クロックレートに変換してタイムスタンプをインクリメント
            elapsed_timestamp = int(elapsed_seconds * 48000)
            if self.audio_config is not None:
                self.audio_config.timestamp = self.audio_config.timestamp + elapsed_timestamp

            # パケッタイザが RTP header を更新する
            self.audio_track.send(data)

            # 状態を更新
            self.last_audio_dts_usec = timestamp_us
            self.encoded_audio_count += 1

            if self.encoded_audio_count % 100 == 0:
                logger.debug(f"Sent encoded audio frame #{self.encoded_audio_count}")
        except queue.Empty:
            pass
        except Exception as e:
            handle_error("sending encoded audio", e)

    def disconnect(self) -> None:
        """切断処理"""
        logger.info("Starting graceful shutdown...")

        # キャプチャを停止
        self.capture_active = False
        if self.uvc_device:
            self.uvc_device.stop()
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

        # エンコーダーをフラッシュ（残データを送信）
        if self.video_encoders:
            for encoder in self.video_encoders:
                try:
                    encoder.flush()
                except Exception:
                    pass
        if self.audio_encoder:
            try:
                self.audio_encoder.flush()
            except Exception:
                pass

        time.sleep(0.5)

        # MediaHandler チェーンを解除（循環参照を防ぐ）
        if self.video_track:
            self.video_track.set_media_handler(None)
        if self.audio_track:
            self.audio_track.set_media_handler(None)

        # リソースをクリーンアップ
        self.nack_responder = None
        self.pli_handler = None
        self.video_sr_reporter = None
        self.audio_sr_reporter = None
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_config = None
        self.audio_config = None
        self.video_layer_states = []
        self.video_track = None
        self.audio_track = None

        # Blend2D レンダラーをクリーンアップ
        if self.renderer:
            self.renderer.ctx = None
            self.renderer.img = None
            self.renderer = None

        if self.pc:
            try:
                self.pc.close()
            except Exception as e:
                handle_error("closing PeerConnection", e)
            finally:
                self.pc = None

        if self.video_encoders:
            for encoder in self.video_encoders:
                try:
                    encoder.close()
                except Exception:
                    pass
            self.video_encoders = []

        if self.audio_encoder:
            try:
                self.audio_encoder.close()
            except Exception:
                pass
            finally:
                self.audio_encoder = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(
        description="WHIP クライアント（webcodecs-py ベース, simulcast 対応）"
    )
    parser.add_argument("--url", required=True, help="WHIP エンドポイント URL")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグログを出力",
    )
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
        default=0,
        help="ビデオ入力デバイス番号 (デフォルト: 0)",
    )
    parser.add_argument(
        "--audio-input-device",
        type=int,
        help="オーディオ入力デバイス番号 (未指定時はシステムデフォルト)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=30,
        help="フレームレート (デフォルト: 30)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=5_000_000,
        help="ビットレート (デフォルト: 5000000)",
    )
    parser.add_argument(
        "--whip-simulcast-total-layers",
        type=int,
        default=3,
        help="Simulcast レイヤ数 (デフォルト: 3, 最大: 4)",
    )
    parser.add_argument(
        "--disable-audio-processing",
        action="store_true",
        help="オーディオのエンコード・送信を無効にする（デバッグ用）",
    )

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

    if args.whip_simulcast_total_layers < 1 or args.whip_simulcast_total_layers > 4:
        raise SystemExit("--whip-simulcast-total-layers は 1〜4 の範囲で指定してください")

    logger.info(
        "Video settings",
        codec=args.video_codec_type,
        framerate=args.framerate,
        bitrate=args.bitrate,
        simulcast_layers=args.whip_simulcast_total_layers,
    )
    logger.info(f"WHIP endpoint: {args.url}")
    if args.fake_capture_device:
        logger.info("Using fake capture device (blend2d video + test audio)")
    else:
        audio_dev_str = (
            str(args.audio_input_device) if args.audio_input_device is not None else "default"
        )
        logger.info(f"Video device: {args.video_input_device}, Audio device: {audio_dev_str}")

    client = WHIPClient(
        args.url,
        args.token,
        args.video_codec_type,
        args.fake_capture_device,
        args.video_input_device,
        args.audio_input_device,
        args.framerate,
        args.bitrate,
        args.whip_simulcast_total_layers,
        args.disable_audio_processing,
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
