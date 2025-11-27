"""
Ayame シグナリングクライアント

libdatachannel-py で Ayame シグナリングサーバーに接続します。

デフォルトで Ayame Labo (https://ayame-labo.shiguredo.app/) に接続します。

使い方:
    # 送信側（カメラとマイクを使用）
    uv run python examples/ayame.py --room-id test-room --sendrecv

    # 送信側（テストパターン）
    uv run python examples/ayame.py --room-id test-room --sendrecv --fake-capture-device

    # 受信側
    uv run python examples/ayame.py --room-id test-room --recvonly --display

    # シグナリングキーを指定
    uv run python examples/ayame.py --room-id test-room --signaling-key YOUR_KEY --sendrecv

    # 別の Ayame サーバーに接続
    uv run python examples/ayame.py --url wss://your-ayame.example.com/signaling --room-id test-room --sendrecv
"""

import argparse
import json
import logging
import queue
import random
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import sounddevice as sd

from misc import handle_error

# libdatachannel-py
from libdatachannel import (
    Candidate,
    Configuration,
    Description,
    H264RtpDepacketizer,
    H264RtpPacketizer,
    IceServer,
    NalUnit,
    OpusRtpDepacketizer,
    OpusRtpPacketizer,
    PeerConnection,
    PliHandler,
    RtcpNackResponder,
    RtcpSrReporter,
    RtpPacketizationConfig,
    Track,
    WebSocket,
    WebSocketConfiguration,
)

# webcodecs-py
from webcodecs import (
    AudioData,
    AudioDataInit,
    AudioEncoder,
    AudioEncoderConfig,
    AudioSampleFormat,
    EncodedAudioChunk,
    EncodedVideoChunk,
    EncodedVideoChunkInit,
    EncodedVideoChunkType,
    HardwareAccelerationEngine,
    LatencyMode,
    VideoDecoder,
    VideoDecoderConfig,
    VideoEncoder,
    VideoEncoderConfig,
    VideoFrame,
    VideoFrameBufferInit,
    VideoPixelFormat,
)

logger = logging.getLogger(__name__)


class AyameClient:
    """Ayame シグナリングクライアント"""

    def __init__(
        self,
        signaling_url: str,
        room_id: str,
        client_id: Optional[str] = None,
        signaling_key: Optional[str] = None,
        direction: str = "sendrecv",
        use_fake_capture: bool = False,
        video_input_device: Optional[int] = None,
        audio_input_device: Optional[int] = None,
        display_video: bool = False,
        framerate: int = 30,
        bitrate: int = 5_000_000,
    ):
        self.signaling_url = signaling_url
        self.room_id = room_id
        self.client_id = client_id or str(uuid4())
        self.signaling_key = signaling_key
        self.direction = direction
        self.use_fake_capture = use_fake_capture
        self.video_input_device = video_input_device if video_input_device is not None else 0
        self.audio_input_device = audio_input_device
        self.display_video = display_video
        self.framerate = framerate
        self.bitrate = bitrate

        # WebSocket
        self.ws: Optional[WebSocket] = None

        # PeerConnection
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None

        # ICE servers (from accept message)
        self.ice_servers: list[IceServer] = []

        # State
        self.is_offer = False
        self.is_exist_user = False
        self.authz_metadata: Any = None
        self.running = False
        self.connected = threading.Event()

        # Video settings
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = framerate

        # Audio settings
        self.audio_sample_rate = 48000
        self.audio_channels = 1
        self.audio_frame_size = 960  # 20ms @ 48kHz

        # Encoders (for sendrecv/sendonly)
        self.video_encoder: Optional[VideoEncoder] = None
        self.audio_encoder: Optional[AudioEncoder] = None
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_config: Optional[RtpPacketizationConfig] = None
        self.audio_config: Optional[RtpPacketizationConfig] = None

        # Decoders (for sendrecv/recvonly)
        self.video_decoder: Optional[VideoDecoder] = None
        self.decoder_configured = False

        # Frame counters
        self.video_frame_number = 0
        self.audio_frame_number = 0
        self.encoded_video_count = 0
        self.encoded_audio_count = 0
        self.received_video_count = 0
        self.received_audio_count = 0
        self.decoded_frame_count = 0

        # Timestamps
        self.last_video_dts_usec: int = 0
        self.last_audio_dts_usec: int = 0

        # Queues
        self.video_queue: queue.Queue = queue.Queue(maxsize=30)
        self.audio_queue: queue.Queue = queue.Queue(maxsize=50)
        self.encoded_audio_queue: queue.Queue = queue.Queue()
        self.display_queue: Optional[queue.Queue] = (
            queue.Queue(maxsize=10) if display_video else None
        )

        # Camera/Audio capture
        self.camera = None
        self.camera_thread: Optional[threading.Thread] = None
        self.audio_stream = None
        self.capture_active = False

        # Key frame interval
        self.key_frame_interval_frames = self.video_fps * 90  # 90 seconds

        # Callbacks
        self.on_open: Optional[Callable[[], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[str], None]] = None

    def connect(self) -> None:
        """Ayame シグナリングサーバーに接続"""
        logger.info(f"Connecting to Ayame: {self.signaling_url}")
        logger.info(f"Room ID: {self.room_id}, Client ID: {self.client_id}")

        self.running = True

        # WebSocket 接続
        ws_config = WebSocketConfiguration()
        # wss:// (TLS) 接続のため TLS 検証を無効化
        ws_config.disable_tls_verification = True
        self.ws = WebSocket(ws_config)

        self.ws.on_open(self._on_ws_open)
        self.ws.on_message(self._on_ws_message)
        self.ws.on_error(self._on_ws_error)
        self.ws.on_closed(self._on_ws_closed)

        self.ws.open(self.signaling_url)

        # 接続完了を待機
        if not self.connected.wait(timeout=30.0):
            raise RuntimeError("Connection timeout")

        logger.info("Connected to Ayame")

    def _on_ws_open(self) -> None:
        """WebSocket 接続時"""
        logger.info("WebSocket connected")

        # register メッセージを送信
        register_message = {
            "type": "register",
            "roomId": self.room_id,
            "clientId": self.client_id,
        }
        if self.signaling_key:
            register_message["key"] = self.signaling_key

        self._send_ws(register_message)
        logger.info("Sent register message")

    def _on_ws_message(self, message: str | bytes) -> None:
        """WebSocket メッセージ受信時"""
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "ping":
                self._send_ws({"type": "pong"})
            elif msg_type == "accept":
                self._handle_accept(data)
            elif msg_type == "reject":
                reason = data.get("reason", "REJECTED")
                logger.error(f"Connection rejected: {reason}")
                if self.on_disconnect:
                    self.on_disconnect(reason)
            elif msg_type == "offer":
                self._handle_offer(data)
            elif msg_type == "answer":
                self._handle_answer(data)
            elif msg_type == "candidate":
                self._handle_candidate(data)
            elif msg_type == "bye":
                logger.info("Received bye message")
                if self.on_disconnect:
                    self.on_disconnect("BYE")
        except Exception as e:
            handle_error("processing WebSocket message", e)

    def _on_ws_error(self, error: str) -> None:
        """WebSocket エラー時"""
        logger.error(f"WebSocket error: {error}")

    def _on_ws_closed(self) -> None:
        """WebSocket 切断時"""
        logger.info("WebSocket closed")
        if self.on_disconnect:
            self.on_disconnect("WS-CLOSED")

    def _send_ws(self, message: dict) -> None:
        """WebSocket でメッセージを送信"""
        if self.ws:
            self.ws.send(json.dumps(message))

    def _handle_accept(self, data: dict) -> None:
        """accept メッセージを処理"""
        logger.info("Received accept message")

        self.authz_metadata = data.get("authzMetadata")
        self.is_exist_user = data.get("isExistUser", False)

        # ICE servers を取得
        ice_servers_data = data.get("iceServers", [])
        for server_data in ice_servers_data:
            urls = server_data.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            for url in urls:
                ice_server = IceServer(url)
                if "username" in server_data:
                    ice_server.username = server_data["username"]
                if "credential" in server_data:
                    ice_server.password = server_data["credential"]
                self.ice_servers.append(ice_server)
                logger.info(f"Added ICE server: {url}")

        # 既存ユーザーがいる場合は offer を送信（この時点で PeerConnection を作成）
        if self.is_exist_user:
            logger.info("Existing user found, sending offer")
            self._create_peer_connection()
            self._send_offer()
        else:
            logger.info("No existing user, waiting for offer")

        if self.on_open:
            self.on_open()

    def _create_peer_connection(self) -> None:
        """PeerConnection を作成"""
        config = Configuration()
        config.ice_servers = self.ice_servers
        self.pc = PeerConnection(config)

        # 状態変化をログ出力
        def on_state_change(state: PeerConnection.State) -> None:
            logger.info(f"PeerConnection state: {state}")

        def on_gathering_state_change(state: PeerConnection.GatheringState) -> None:
            logger.info(f"ICE gathering state: {state}")

        def on_local_candidate(candidate) -> None:
            mid = candidate.mid()
            # mid から sdpMLineIndex を推測
            # audio は通常 index 0、video は通常 index 1
            sdp_m_line_index = 0
            if mid:
                if mid == "1" or "video" in mid.lower():
                    sdp_m_line_index = 1
            logger.debug(
                f"Local ICE candidate (mid={mid}, mLineIndex={sdp_m_line_index}): "
                f"{candidate.candidate()[:50]}..."
            )
            # Ayame に ICE candidate を送信
            # candidate.candidate() は "candidate:..." 形式（a= なし）
            candidate_msg = {
                "type": "candidate",
                "ice": {
                    "candidate": candidate.candidate(),
                    "sdpMid": mid if mid else "0",
                    "sdpMLineIndex": sdp_m_line_index,
                },
            }
            self._send_ws(candidate_msg)

        self.pc.on_state_change(on_state_change)
        self.pc.on_gathering_state_change(on_gathering_state_change)
        self.pc.on_local_candidate(on_local_candidate)

        # オーディオトラックを追加
        if self.direction == "sendrecv":
            audio_desc = Description.Audio("audio", Description.Direction.SendRecv)
        elif self.direction == "sendonly":
            audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        else:
            audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # ビデオトラックを追加
        if self.direction == "sendrecv":
            video_desc = Description.Video("video", Description.Direction.SendRecv)
        elif self.direction == "sendonly":
            video_desc = Description.Video("video", Description.Direction.SendOnly)
        else:
            video_desc = Description.Video("video", Description.Direction.RecvOnly)
        video_desc.add_h264_codec(96)
        self.video_track = self.pc.add_track(video_desc)

        # エンコーダー/デコーダーをセットアップ
        if self.direction in ["sendrecv", "sendonly"]:
            self._setup_video_encoder()
            self._setup_audio_encoder()

        if self.direction in ["sendrecv", "recvonly"]:
            self._setup_video_depacketizer()
            self._setup_audio_depacketizer()
            if self.display_video:
                self._setup_video_decoder()

        logger.info(f"PeerConnection created with direction: {self.direction}")

    def _send_offer(self) -> None:
        """offer を送信"""
        if not self.pc:
            return

        self.pc.set_local_description()
        local_sdp = self.pc.local_description()
        if local_sdp:
            sdp_str = str(local_sdp)
            logger.debug(f"Local offer SDP:\n{sdp_str}")
            offer_message = {
                "type": "offer",
                "sdp": sdp_str,
            }
            self._send_ws(offer_message)
            self.is_offer = True
            logger.info("Sent offer")

    def _handle_offer(self, data: dict) -> None:
        """offer を処理"""
        logger.info("Received offer")

        sdp = data.get("sdp", "")
        logger.debug(f"Remote SDP:\n{sdp}")

        # PeerConnection を作成（offer を受け取った側はトラックを追加しない）
        config = Configuration()
        config.ice_servers = self.ice_servers
        self.pc = PeerConnection(config)

        # 状態変化をログ出力
        def on_state_change(state: PeerConnection.State) -> None:
            logger.info(f"PeerConnection state: {state}")

        def on_gathering_state_change(state: PeerConnection.GatheringState) -> None:
            logger.info(f"ICE gathering state: {state}")

        def on_local_candidate(candidate) -> None:
            mid = candidate.mid()
            # mid から sdpMLineIndex を推測
            # audio は通常 index 0、video は通常 index 1
            sdp_m_line_index = 0
            if mid:
                if mid == "1" or "video" in mid.lower():
                    sdp_m_line_index = 1
            logger.debug(
                f"Local ICE candidate (mid={mid}, mLineIndex={sdp_m_line_index}): "
                f"{candidate.candidate()[:50]}..."
            )
            # Ayame に ICE candidate を送信
            # candidate.candidate() は "candidate:..." 形式（a= なし）
            candidate_msg = {
                "type": "candidate",
                "ice": {
                    "candidate": candidate.candidate(),
                    "sdpMid": mid if mid else "0",
                    "sdpMLineIndex": sdp_m_line_index,
                },
            }
            self._send_ws(candidate_msg)

        self.pc.on_state_change(on_state_change)
        self.pc.on_gathering_state_change(on_gathering_state_change)
        self.pc.on_local_candidate(on_local_candidate)

        # on_track コールバックを設定（トラックを受信するため）
        def on_track(track: Track) -> None:
            mid = track.mid()
            logger.info(f"Received track with mid: {mid}")

            if mid == "0" or "audio" in mid.lower():
                self.audio_track = track
                if self.direction in ["sendrecv", "sendonly"]:
                    self._setup_audio_encoder()
                    if self.audio_track and self.audio_packetizer:
                        self.audio_track.set_media_handler(self.audio_packetizer)
                if self.direction in ["sendrecv", "recvonly"]:
                    self._setup_audio_depacketizer()
            elif mid == "1" or "video" in mid.lower():
                self.video_track = track
                if self.direction in ["sendrecv", "sendonly"]:
                    self._setup_video_encoder()
                    if self.video_track and self.video_packetizer:
                        self.video_track.set_media_handler(self.video_packetizer)
                if self.direction in ["sendrecv", "recvonly"]:
                    if self.display_video:
                        self._setup_video_decoder()
                    self._setup_video_depacketizer()

        self.pc.on_track(on_track)

        # remote description を設定
        remote_desc = Description(sdp, Description.Type.Offer)
        self.pc.set_remote_description(remote_desc)

        # answer を生成
        self.pc.set_local_description()
        local_sdp = self.pc.local_description()
        if local_sdp:
            logger.debug(f"Local SDP:\n{local_sdp}")
            answer_message = {
                "type": "answer",
                "sdp": str(local_sdp),
            }
            self._send_ws(answer_message)
            logger.info("Sent answer")

        self.connected.set()

    def _handle_answer(self, data: dict) -> None:
        """answer を処理"""
        logger.info("Received answer")

        if not self.pc:
            return

        sdp = data.get("sdp", "")
        logger.debug(f"Remote answer SDP:\n{sdp}")
        remote_desc = Description(sdp, Description.Type.Answer)
        self.pc.set_remote_description(remote_desc)

        self.connected.set()

    def _handle_candidate(self, data: dict) -> None:
        """ICE candidate を処理"""
        ice_data = data.get("ice")
        if ice_data and self.pc:
            candidate_str = ice_data.get("candidate", "")
            sdp_mid = ice_data.get("sdpMid", "")
            # 空の候補は end-of-candidates を示す
            if not candidate_str:
                logger.debug("Received end-of-candidates signal")
                return
            candidate = Candidate(candidate_str, sdp_mid)
            self.pc.add_remote_candidate(candidate)
            logger.debug(f"Added remote ICE candidate (mid={sdp_mid}): {candidate_str[:60]}...")

    def _setup_video_encoder(self) -> None:
        """ビデオエンコーダーをセットアップ"""

        def on_output(chunk: EncodedVideoChunk) -> None:
            if not self.video_track or not self.video_track.is_open():
                return

            try:
                data = np.zeros(chunk.byte_length, dtype=np.uint8)
                chunk.copy_to(data)

                dts_usec = chunk.timestamp
                duration = dts_usec - self.last_video_dts_usec
                elapsed_seconds = float(duration) / 1_000_000.0
                elapsed_timestamp = int(elapsed_seconds * 90000)
                self.video_config.timestamp = self.video_config.timestamp + elapsed_timestamp

                self.video_track.send(bytes(data))
                self.last_video_dts_usec = dts_usec
                self.encoded_video_count += 1

                if self.encoded_video_count % 30 == 0:
                    logger.debug(f"Sent video frame #{self.encoded_video_count}")
            except Exception as e:
                handle_error("sending encoded video", e)

        def on_error(error: str) -> None:
            logger.error(f"Video encoder error: {error}")

        self.video_encoder = VideoEncoder(on_output, on_error)

        encoder_config: VideoEncoderConfig = {
            "codec": "avc1.64002A",
            "width": self.video_width,
            "height": self.video_height,
            "bitrate": self.bitrate,
            "latency_mode": LatencyMode.REALTIME,
            "avc": {"format": "annexb"},
            "hardware_acceleration_engine": HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX,
        }

        self.video_encoder.configure(encoder_config)
        logger.info("Video encoder configured: H.264")

        # RTP パケッタイザー
        self.video_config = RtpPacketizationConfig(
            ssrc=random.randint(1, 0xFFFFFFFF),
            cname="video-stream",
            payload_type=96,
            clock_rate=90000,
        )
        self.video_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        self.video_config.timestamp = self.video_config.start_timestamp
        self.video_config.sequence_number = random.randint(0, 0xFFFF)

        self.video_packetizer = H264RtpPacketizer(
            NalUnit.Separator.LongStartSequence,
            self.video_config,
            1200,
        )

        video_sr_reporter = RtcpSrReporter(self.video_config)
        self.video_packetizer.add_to_chain(video_sr_reporter)

        def on_pli():
            logger.info("PLI received - requesting keyframe")

        pli_handler = PliHandler(on_pli)
        self.video_packetizer.add_to_chain(pli_handler)

        nack_responder = RtcpNackResponder()
        self.video_packetizer.add_to_chain(nack_responder)

        if self.video_track:
            self.video_track.set_media_handler(self.video_packetizer)

    def _setup_audio_encoder(self) -> None:
        """オーディオエンコーダーをセットアップ"""

        def on_output(chunk: EncodedAudioChunk) -> None:
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

        # RTP パケッタイザー
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

        audio_sr_reporter = RtcpSrReporter(self.audio_config)
        self.audio_packetizer.add_to_chain(audio_sr_reporter)

        if self.audio_track:
            self.audio_track.set_media_handler(self.audio_packetizer)

    def _setup_video_decoder(self) -> None:
        """ビデオデコーダーをセットアップ"""

        def on_output(frame: VideoFrame) -> None:
            self.decoded_frame_count += 1
            if self.decoded_frame_count % 30 == 0:
                logger.info(
                    f"Decoded frame #{self.decoded_frame_count}: "
                    f"{frame.coded_width}x{frame.coded_height}"
                )

            if self.display_queue:
                try:
                    width = frame.coded_width
                    height = frame.coded_height
                    rgb_size = width * height * 3
                    rgb_buffer = np.zeros(rgb_size, dtype=np.uint8)
                    frame.copy_to(rgb_buffer, {"format": VideoPixelFormat.RGB})
                    rgb_frame = rgb_buffer.reshape((height, width, 3))
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    self.display_queue.put_nowait(bgr_frame)
                except queue.Full:
                    pass
                except Exception as e:
                    if self.decoded_frame_count <= 5:
                        logger.error(f"Error converting frame: {e}")

            frame.close()

        def on_error(error: str) -> None:
            logger.error(f"Video decoder error: {error}")

        self.video_decoder = VideoDecoder(on_output, on_error)
        logger.info("Video decoder created")

    def _configure_decoder(self, data: bytes) -> None:
        """デコーダーを設定"""
        if self.decoder_configured or not self.video_decoder:
            return

        config: VideoDecoderConfig = {
            "codec": "avc1.64001F",
            "coded_width": 1920,
            "coded_height": 1080,
        }

        try:
            self.video_decoder.configure(config)
            self.decoder_configured = True
            logger.info("Video decoder configured")
        except Exception as e:
            logger.error(f"Failed to configure decoder: {e}")

    def _setup_video_depacketizer(self) -> None:
        """ビデオデパケッタイザーをセットアップ"""
        if self.video_track:
            h264_depacketizer = H264RtpDepacketizer()
            self.video_track.on_frame(self._on_video_frame)
            self.video_track.set_media_handler(h264_depacketizer)
            logger.info("H.264 depacketizer set for video track")

    def _on_video_frame(self, data: bytes, frame_info) -> None:
        """ビデオフレームを受信"""
        self.received_video_count += 1

        # NAL ユニットを解析してキーフレームかどうか判定
        has_keyframe = False
        offset = 0
        while offset < len(data):
            if offset + 4 <= len(data) and data[offset : offset + 4] == b"\x00\x00\x00\x01":
                start_code_len = 4
            elif offset + 3 <= len(data) and data[offset : offset + 3] == b"\x00\x00\x01":
                start_code_len = 3
            else:
                offset += 1
                continue

            nal_start = offset + start_code_len
            if nal_start >= len(data):
                break

            nal_type = data[nal_start] & 0x1F
            if nal_type in [5, 7, 8]:  # IDR, SPS, PPS
                has_keyframe = True
                break

            next_offset = nal_start + 1
            while next_offset < len(data):
                if (
                    next_offset + 4 <= len(data)
                    and data[next_offset : next_offset + 4] == b"\x00\x00\x00\x01"
                ) or (
                    next_offset + 3 <= len(data)
                    and data[next_offset : next_offset + 3] == b"\x00\x00\x01"
                ):
                    break
                next_offset += 1
            offset = next_offset

        # デコーダーを設定
        if not self.decoder_configured:
            self._configure_decoder(data)

        # デコード
        if self.video_decoder and self.decoder_configured:
            try:
                chunk_type = (
                    EncodedVideoChunkType.KEY if has_keyframe else EncodedVideoChunkType.DELTA
                )
                init: EncodedVideoChunkInit = {
                    "type": chunk_type,
                    "timestamp": frame_info.timestamp,
                    "data": np.frombuffer(data, dtype=np.uint8),
                }
                chunk = EncodedVideoChunk(init)
                self.video_decoder.decode(chunk)
            except Exception as e:
                if self.received_video_count <= 5:
                    logger.error(f"Decode error: {e}")

        if self.received_video_count % 30 == 0:
            logger.info(f"Received video frame #{self.received_video_count}")

    def _setup_audio_depacketizer(self) -> None:
        """オーディオデパケッタイザーをセットアップ"""
        if self.audio_track:
            opus_depacketizer = OpusRtpDepacketizer()
            self.audio_track.on_frame(self._on_audio_frame)
            self.audio_track.set_media_handler(opus_depacketizer)
            logger.info("Opus depacketizer set for audio track")

    def _on_audio_frame(self, data: bytes, frame_info) -> None:
        """オーディオフレームを受信"""
        self.received_audio_count += 1
        if self.received_audio_count % 50 == 0:
            logger.debug(f"Received audio frame #{self.received_audio_count}")

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

        if actual_width != self.video_width or actual_height != self.video_height:
            self.video_width = actual_width
            self.video_height = actual_height

        self.capture_active = True

        def capture_thread():
            while self.capture_active:
                ret, frame = self.camera.read()
                if ret:
                    try:
                        self.video_queue.put_nowait(frame)
                    except queue.Full:
                        pass

        self.camera_thread = threading.Thread(target=capture_thread, daemon=True)
        self.camera_thread.start()

    def _start_audio_capture(self) -> None:
        """マイクキャプチャを開始"""
        device = self.audio_input_device
        if device is None:
            device = sd.default.device[0]
        device_info = sd.query_devices(device, "input")
        max_channels = device_info["max_input_channels"]
        if max_channels < 1:
            logger.error(f"Audio device {device} has no input channels")
            return
        self.audio_channels = min(max_channels, 2)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio capture status: {status}")
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

        self.audio_stream = sd.InputStream(
            device=device,
            samplerate=self.audio_sample_rate,
            channels=self.audio_channels,
            dtype=np.float32,
            blocksize=self.audio_frame_size,
            callback=audio_callback,
        )
        self.audio_stream.start()
        logger.info(
            f"Audio capture started: {self.audio_sample_rate}Hz, {self.audio_channels}ch"
        )

    def _generate_test_frame(self) -> np.ndarray:
        """テストパターンを生成"""
        frame = np.zeros((self.video_height, self.video_width, 4), dtype=np.uint8)

        # カラーバー
        bar_width = self.video_width // 8
        colors = [
            (255, 255, 255, 255),  # White
            (255, 255, 0, 255),  # Yellow
            (0, 255, 255, 255),  # Cyan
            (0, 255, 0, 255),  # Green
            (255, 0, 255, 255),  # Magenta
            (255, 0, 0, 255),  # Red
            (0, 0, 255, 255),  # Blue
            (0, 0, 0, 255),  # Black
        ]
        for i, color in enumerate(colors):
            x_start = i * bar_width
            x_end = (i + 1) * bar_width
            frame[:, x_start:x_end] = color

        return frame

    def _generate_test_audio(self) -> np.ndarray:
        """テストオーディオを生成（無音）"""
        mono = np.zeros(self.audio_frame_size, dtype=np.float32)
        if self.audio_channels == 1:
            return mono.reshape(-1, 1)
        else:
            return np.column_stack([mono] * self.audio_channels)

    def send_frames(self, duration: Optional[int] = None) -> None:
        """フレームを送受信"""
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

        if self.on_connect:
            self.on_connect()

        # 送信側の場合はキャプチャを開始
        if self.direction in ["sendrecv", "sendonly"]:
            if self.use_fake_capture:
                logger.info("Using fake capture device")
            else:
                self._start_camera_capture()
                self._start_audio_capture()

        frame_interval = 1.0 / self.video_fps
        audio_interval = self.audio_frame_size / self.audio_sample_rate
        self._loop_running = True

        def video_encode_loop():
            next_time = time.perf_counter()
            while self._loop_running:
                now = time.perf_counter()
                if now >= next_time:
                    self._encode_video_frame()
                    next_time += frame_interval
                time.sleep(0.001)

        def audio_encode_loop():
            next_time = time.perf_counter()
            while self._loop_running:
                now = time.perf_counter()
                if now >= next_time:
                    self._encode_audio_frame()
                    next_time += audio_interval
                time.sleep(0.001)

        def audio_send_loop():
            next_time = time.perf_counter() + audio_interval
            while self._loop_running:
                now = time.perf_counter()
                if now >= next_time:
                    self._send_encoded_audio()
                    next_time += audio_interval
                time.sleep(0.001)

        threads = []
        if self.direction in ["sendrecv", "sendonly"]:
            video_thread = threading.Thread(target=video_encode_loop, daemon=True)
            audio_thread = threading.Thread(target=audio_encode_loop, daemon=True)
            send_thread = threading.Thread(target=audio_send_loop, daemon=True)
            video_thread.start()
            audio_thread.start()
            send_thread.start()
            threads.extend([video_thread, audio_thread, send_thread])

        try:
            start_time = time.time()
            while self.running:
                if duration and time.time() - start_time >= duration:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._loop_running = False
            for t in threads:
                t.join(timeout=1.0)

    def _encode_video_frame(self) -> None:
        """ビデオフレームをエンコード"""
        if not self.video_encoder:
            return

        # フレームを取得
        if self.use_fake_capture:
            bgra_frame = self._generate_test_frame()
        else:
            try:
                bgr_frame = self.video_queue.get_nowait()
                bgra_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
            except queue.Empty:
                return

        timestamp_us = int(self.video_frame_number * 1_000_000 / self.video_fps)

        bgra_init: VideoFrameBufferInit = {
            "format": VideoPixelFormat.BGRA,
            "coded_width": self.video_width,
            "coded_height": self.video_height,
            "timestamp": timestamp_us,
        }
        bgra_video_frame = VideoFrame(bgra_frame, bgra_init)

        # BGRA -> I420
        i420_size = self.video_width * self.video_height * 3 // 2
        i420_buffer = np.zeros(i420_size, dtype=np.uint8)
        bgra_video_frame.copy_to(i420_buffer, {"format": VideoPixelFormat.I420})
        bgra_video_frame.close()

        i420_init: VideoFrameBufferInit = {
            "format": VideoPixelFormat.I420,
            "coded_width": self.video_width,
            "coded_height": self.video_height,
            "timestamp": timestamp_us,
        }
        frame = VideoFrame(i420_buffer, i420_init)

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

    def _encode_audio_frame(self) -> None:
        """オーディオフレームをエンコード"""
        if not self.audio_encoder:
            return

        if self.use_fake_capture:
            audio_samples = self._generate_test_audio()
        else:
            try:
                audio_samples = self.audio_queue.get_nowait()
            except queue.Empty:
                return

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

            duration = timestamp_us - self.last_audio_dts_usec
            elapsed_seconds = float(duration) / 1_000_000.0
            elapsed_timestamp = int(elapsed_seconds * 48000)
            self.audio_config.timestamp = self.audio_config.timestamp + elapsed_timestamp

            self.audio_track.send(data)
            self.last_audio_dts_usec = timestamp_us
            self.encoded_audio_count += 1
        except queue.Empty:
            pass
        except Exception as e:
            handle_error("sending encoded audio", e)

    def disconnect(self) -> None:
        """切断処理"""
        logger.info("Starting graceful shutdown...")

        self.running = False
        self.capture_active = False

        # キャプチャを停止
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
            self.camera_thread = None
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

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

        # デコーダーをクリーンアップ
        if self.video_decoder:
            try:
                self.video_decoder.close()
            except Exception:
                pass
            self.video_decoder = None

        # トラックをクリーンアップ（PeerConnection より先に参照を解放）
        self.video_track = None
        self.audio_track = None

        # MediaHandler チェーンをクリーンアップ
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_config = None
        self.audio_config = None

        # PeerConnection をクローズ
        if self.pc:
            try:
                self.pc.close()
            except Exception as e:
                handle_error("closing PeerConnection", e)
            self.pc = None

        # エンコーダーをクリーンアップ
        if self.video_encoder:
            try:
                self.video_encoder.close()
            except Exception:
                pass
            self.video_encoder = None
        if self.audio_encoder:
            try:
                self.audio_encoder.close()
            except Exception:
                pass
            self.audio_encoder = None

        # WebSocket を閉じる（PeerConnection の後に閉じる）
        if self.ws:
            self.ws.close()
            time.sleep(0.5)
            self.ws = None

        logger.info("Graceful shutdown completed")


def display_frames(client: AyameClient) -> bool:
    """フレームを表示（メインスレッドで実行）"""
    if not client.display_queue:
        return False

    logger.info("Starting video display...")

    window_name = "Ayame Video"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 100, 100)
    logger.info("Window created (Press 'q' or ESC to close)")

    frame_count = 0
    window_closed = False

    while client.running:
        try:
            frame = client.display_queue.get(timeout=0.1)
            frame_count += 1

            if frame_count == 1:
                logger.info(f"First frame: shape={frame.shape}")

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User pressed 'q' or ESC")
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
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
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    window_closed = True
                    break
            except cv2.error:
                window_closed = True
                break

    logger.info(f"Display finished. Frames displayed: {frame_count}")
    if not window_closed:
        cv2.destroyAllWindows()

    return window_closed


AYAME_LABO_URL = "wss://ayame-labo.shiguredo.app/signaling"


def main():
    parser = argparse.ArgumentParser(description="Ayame シグナリングクライアント")
    parser.add_argument(
        "--url",
        default=AYAME_LABO_URL,
        help=f"Ayame シグナリング URL (デフォルト: {AYAME_LABO_URL})",
    )
    parser.add_argument("--room-id", required=True, help="ルーム ID")
    parser.add_argument("--client-id", help="クライアント ID (省略時は自動生成)")
    parser.add_argument("--signaling-key", help="シグナリングキー")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグログを出力",
    )
    parser.add_argument("--duration", type=int, help="接続時間（秒）")

    # Direction
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--sendrecv",
        action="store_true",
        help="送受信モード（デフォルト）",
    )
    direction_group.add_argument(
        "--sendonly",
        action="store_true",
        help="送信のみモード",
    )
    direction_group.add_argument(
        "--recvonly",
        action="store_true",
        help="受信のみモード",
    )

    # Capture options
    parser.add_argument(
        "--fake-capture-device",
        action="store_true",
        help="テストパターンを使用",
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

    # Video options
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

    # Display
    parser.add_argument(
        "--display",
        action="store_true",
        help="受信した映像を表示",
    )

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Direction を決定
    if args.sendonly:
        direction = "sendonly"
    elif args.recvonly:
        direction = "recvonly"
    else:
        direction = "sendrecv"

    logger.info(f"Ayame signaling URL: {args.url}")
    logger.info(f"Room ID: {args.room_id}")
    logger.info(f"Direction: {direction}")
    if args.fake_capture_device:
        logger.info("Using fake capture device")
    elif direction in ["sendrecv", "sendonly"]:
        audio_dev_str = (
            str(args.audio_input_device) if args.audio_input_device is not None else "default"
        )
        logger.info(f"Video device: {args.video_input_device}, Audio device: {audio_dev_str}")

    client = AyameClient(
        args.url,
        args.room_id,
        client_id=args.client_id,
        signaling_key=args.signaling_key,
        direction=direction,
        use_fake_capture=args.fake_capture_device,
        video_input_device=args.video_input_device,
        audio_input_device=args.audio_input_device,
        display_video=args.display,
        framerate=args.framerate,
        bitrate=args.bitrate,
    )

    try:
        client.connect()

        if args.display and direction in ["sendrecv", "recvonly"]:
            # 受信スレッドを開始
            def receive_thread():
                try:
                    client.send_frames(args.duration)
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
            client.send_frames(args.duration)

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        handle_error("running Ayame client", e)
    finally:
        try:
            client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()
