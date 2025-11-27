"""
WHEP (WebRTC-HTTP Egress Protocol) クライアント

draft-ietf-wish-whep-03 準拠の WHEP クライアント実装。
webcodecs-py でデコードして OpenCV で表示します。

使い方:
    # H.264 で映像を受信（表示なし）
    uv run python examples/whep.py --url https://example.com/whep/channel

    # H.264 で映像を受信して表示
    uv run python examples/whep.py --url https://example.com/whep/channel --display

    # H.265 で映像を受信して表示
    uv run python examples/whep.py --url https://example.com/whep/channel --display --video-codec-type h265

    # デバッグログを有効にして受信
    uv run python examples/whep.py --url https://example.com/whep/channel --debug
"""

import argparse
import logging
import queue
import subprocess
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
from misc import get_nal_type_name, handle_error, parse_link_header

# webcodecs-py
from webcodecs import (
    EncodedVideoChunk,
    EncodedVideoChunkInit,
    EncodedVideoChunkType,
    HardwareAccelerationEngine,
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

logger = logging.getLogger(__name__)


def build_hvcc(vps_data: bytes, sps_data: bytes, pps_data: bytes) -> bytes:
    """HEVC Decoder Configuration Record (HVCC) を生成

    ISO/IEC 14496-15 Section 8.3.3.1 に準拠
    """
    # HVCC header
    hvcc = bytearray()

    # configurationVersion = 1
    hvcc.append(1)

    # general_profile_space (2 bits) + general_tier_flag (1 bit) + general_profile_idc (5 bits)
    # デフォルト: Main profile (1)
    hvcc.append(0x01)

    # general_profile_compatibility_flags (32 bits)
    hvcc.extend([0x60, 0x00, 0x00, 0x00])

    # general_constraint_indicator_flags (48 bits)
    hvcc.extend([0x90, 0x00, 0x00, 0x00, 0x00, 0x00])

    # general_level_idc = 120 (Level 4.0)
    hvcc.append(120)

    # min_spatial_segmentation_idc (12 bits) with reserved 4 bits = 0xF000
    hvcc.extend([0xF0, 0x00])

    # parallelismType (2 bits) with reserved 6 bits = 0xFC
    hvcc.append(0xFC)

    # chromaFormat (2 bits) with reserved 6 bits = 0xFC (4:2:0 = 1)
    hvcc.append(0xFD)

    # bitDepthLumaMinus8 (3 bits) with reserved 5 bits = 0xF8
    hvcc.append(0xF8)

    # bitDepthChromaMinus8 (3 bits) with reserved 5 bits = 0xF8
    hvcc.append(0xF8)

    # avgFrameRate (16 bits) = 0
    hvcc.extend([0x00, 0x00])

    # constantFrameRate (2 bits) + numTemporalLayers (3 bits) +
    # temporalIdNested (1 bit) + lengthSizeMinusOne (2 bits)
    # = 0x00 | 0x00 | 0x00 | 0x03 = 0x03
    hvcc.append(0x03)

    # numOfArrays = 3 (VPS, SPS, PPS)
    hvcc.append(3)

    # VPS array
    hvcc.append(0xA0)  # array_completeness (1) + reserved (1) + NAL_unit_type (6) = 32
    hvcc.extend([0x00, 0x01])  # numNalus = 1
    hvcc.extend(len(vps_data).to_bytes(2, "big"))  # nalUnitLength
    hvcc.extend(vps_data)

    # SPS array
    hvcc.append(0xA1)  # NAL_unit_type = 33
    hvcc.extend([0x00, 0x01])  # numNalus = 1
    hvcc.extend(len(sps_data).to_bytes(2, "big"))  # nalUnitLength
    hvcc.extend(sps_data)

    # PPS array
    hvcc.append(0xA2)  # NAL_unit_type = 34
    hvcc.extend([0x00, 0x01])  # numNalus = 1
    hvcc.extend(len(pps_data).to_bytes(2, "big"))  # nalUnitLength
    hvcc.extend(pps_data)

    return bytes(hvcc)


def get_h265_nal_type_name(nal_type: int) -> str:
    """H.265 NAL ユニットタイプ名を取得

    Args:
        nal_type: NAL ユニットタイプ値

    Returns:
        NAL ユニットタイプの名前
    """
    h265_nal_type_names = {
        0: "TRAIL_N",
        1: "TRAIL_R",
        2: "TSA_N",
        3: "TSA_R",
        4: "STSA_N",
        5: "STSA_R",
        6: "RADL_N",
        7: "RADL_R",
        8: "RASL_N",
        9: "RASL_R",
        16: "BLA_W_LP",
        17: "BLA_W_RADL",
        18: "BLA_N_LP",
        19: "IDR_W_RADL",
        20: "IDR_N_LP",
        21: "CRA_NUT",
        32: "VPS",
        33: "SPS",
        34: "PPS",
        35: "AUD",
        36: "EOS",
        37: "EOB",
        38: "FD",
        39: "PREFIX_SEI",
        40: "SUFFIX_SEI",
    }
    return h265_nal_type_names.get(nal_type, f"Reserved/Unknown ({nal_type})")


class WHEPClient:
    """WHEP クライアント（draft-ietf-wish-whep-03 準拠）"""

    def __init__(
        self,
        whep_url: str,
        bearer_token: Optional[str] = None,
        display_video: bool = False,
        video_codec: str = "h264",
    ):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.display_video = display_video
        self.video_codec = video_codec

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
        self.frame_queue: Optional[queue.Queue] = (
            queue.Queue(maxsize=30) if display_video else None
        )
        self.dropped_frame_count = 0

        # Running flag
        self.running = True

    def connect(self) -> None:
        """WHEP サーバーに接続"""
        logger.info(f"Connecting to WHEP endpoint: {self.whep_url}")

        # PeerConnection を作成（自動 gathering を無効化）
        config = Configuration()
        config.ice_servers = []
        config.disable_auto_gathering = True
        self.pc = PeerConnection(config)

        # オーディオトラックを追加（RecvOnly）
        audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # ビデオトラックを追加（RecvOnly）
        video_desc = Description.Video("video", Description.Direction.RecvOnly)
        if self.video_codec == "h265":
            video_desc.add_h265_codec(96)
        else:
            video_desc.add_h264_codec(96)
        self.video_track = self.pc.add_track(video_desc)

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
                raise RuntimeError(
                    f"WHEP server returned {response.status_code}: {response.text}"
                )

            # セッション URL を取得
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whep_url, self.session_url)

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
                    # デコーダーから返されたフレームのサイズを使用
                    width = frame.coded_width
                    height = frame.coded_height

                    if self.decoded_frame_count == 1:
                        logger.info(f"First decoded frame: {width}x{height}")

                    # NV12 バッファを作成（VideoToolbox は NV12 で出力）
                    nv12_size = width * height * 3 // 2
                    nv12_buffer = np.zeros(nv12_size, dtype=np.uint8)

                    # copy_to で NV12 に変換
                    frame.copy_to(nv12_buffer, {"format": VideoPixelFormat.NV12})

                    # NV12 を YUV 形式に変換（OpenCV 用）
                    yuv_frame = nv12_buffer.reshape((height * 3 // 2, width))

                    # NV12 → BGR 変換
                    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)

                    self.frame_queue.put_nowait(bgr_frame)
                except queue.Full:
                    self.dropped_frame_count += 1
                    if self.dropped_frame_count % 10 == 1:
                        logger.warning(f"Frame dropped (total: {self.dropped_frame_count})")
                except Exception as e:
                    if self.decoded_frame_count <= 5:
                        logger.error(f"Error converting frame: {e}")

            frame.close()

        def on_error(error: str) -> None:
            logger.error(f"Video decoder error: {error}")

        self.video_decoder = VideoDecoder(on_output, on_error)
        logger.info("Video decoder created (will configure on first frame)")

    def _configure_decoder(self, data: bytes) -> None:
        """デコーダーを設定（SPS/PPS または VPS/SPS/PPS から）"""
        if self.decoder_configured or not self.video_decoder:
            return

        if self.video_codec == "h265":
            # H.265: VPS/SPS/PPS を抽出
            vps_data = None
            sps_data = None
            pps_data = None

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

                # H.265 NAL タイプは (byte >> 1) & 0x3F
                nal_type = (data[nal_start] >> 1) & 0x3F

                # 次のスタートコードを探す
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

                # NAL ユニットデータを抽出（スタートコードなし）
                nal_data = data[nal_start:next_offset]

                if nal_type == 32:  # VPS
                    vps_data = nal_data
                    logger.info(f"Found VPS: {len(vps_data)} bytes")
                elif nal_type == 33:  # SPS
                    sps_data = nal_data
                    logger.info(f"Found SPS: {len(sps_data)} bytes")
                elif nal_type == 34:  # PPS
                    pps_data = nal_data
                    logger.info(f"Found PPS: {len(pps_data)} bytes")

                offset = next_offset

            if vps_data and sps_data and pps_data:
                # HVCC (HEVC Decoder Configuration Record) を生成
                hvcc = build_hvcc(vps_data, sps_data, pps_data)
                logger.info(f"Built HVCC: {len(hvcc)} bytes")

                # NOTE: webcodecs-py の現在の実装では coded_width/coded_height が
                # デルタフレームのデコード時に必要。最初のキーフレームで自動取得できないため
                # 暫定的にダミー値を設定（実際の解像度はフレームから取得される）
                config: VideoDecoderConfig = {
                    "codec": "hev1.1.6.L120.B0",  # H.265 Main Profile Level 4.0
                    "coded_width": 1920,  # ダミー値（webcodecs-py の制限回避用）
                    "coded_height": 1080,  # ダミー値（webcodecs-py の制限回避用）
                    "hardware_acceleration_engine": HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX,
                }

                try:
                    self.video_decoder.configure(config)
                    self.decoder_configured = True
                    logger.info("H.265 Video decoder configured")
                except Exception as e:
                    logger.error(f"Failed to configure H.265 decoder: {e}")
        else:
            # H.264: SPS/PPS を探す
            sps_data = None
            pps_data = None

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

                # NAL タイプを取得
                nal_type = data[nal_start] & 0x1F

                # 次のスタートコードまでを取得
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

                nal_data = data[nal_start:next_offset]

                if nal_type == 7:  # SPS
                    sps_data = nal_data
                    logger.info(f"Found SPS: {len(sps_data)} bytes")
                elif nal_type == 8:  # PPS
                    pps_data = nal_data
                    logger.info(f"Found PPS: {len(pps_data)} bytes")

                offset = next_offset

            if sps_data and pps_data:
                # NOTE: webcodecs-py の現在の実装では coded_width/coded_height が
                # デルタフレームのデコード時に必要。暫定的にダミー値を設定
                config: VideoDecoderConfig = {
                    "codec": "avc1.64001F",  # H.264 High Profile
                    "coded_width": 1920,  # ダミー値（webcodecs-py の制限回避用）
                    "coded_height": 1080,  # ダミー値（webcodecs-py の制限回避用）
                    "hardware_acceleration_engine": HardwareAccelerationEngine.APPLE_VIDEO_TOOLBOX,
                }

                try:
                    self.video_decoder.configure(config)
                    self.decoder_configured = True
                    logger.info("H.264 Video decoder configured")
                except Exception as e:
                    logger.error(f"Failed to configure H.264 decoder: {e}")

    def _on_video_frame(self, data: bytes, frame_info) -> None:
        """ビデオフレームを受信"""
        self.video_frame_count += 1

        # NAL ユニットを解析
        nal_units = []
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

            if self.video_codec == "h265":
                # H.265 NAL タイプは (byte >> 1) & 0x3F
                nal_header = data[nal_start]
                nal_type = (nal_header >> 1) & 0x3F

                # H.265 IDR: 19 (IDR_W_RADL), 20 (IDR_N_LP)
                # H.265 CRA: 21 (CRA_NUT)
                # H.265 VPS/SPS/PPS: 32, 33, 34
                if nal_type in [19, 20, 21, 32, 33, 34]:
                    has_keyframe = True

                nal_units.append(
                    {
                        "type": nal_type,
                        "type_name": get_h265_nal_type_name(nal_type),
                    }
                )
            else:
                # H.264 NAL タイプ
                nal_header = data[nal_start]
                nal_type = nal_header & 0x1F

                if nal_type == 5:  # IDR
                    has_keyframe = True
                elif nal_type in [7, 8]:  # SPS, PPS
                    has_keyframe = True

                nal_units.append(
                    {
                        "type": nal_type,
                        "type_name": get_nal_type_name(nal_type),
                    }
                )

            # 次のスタートコードへ
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
                    EncodedVideoChunkType.KEY
                    if has_keyframe
                    else EncodedVideoChunkType.DELTA
                )

                init: EncodedVideoChunkInit = {
                    "type": chunk_type,
                    "timestamp": frame_info.timestamp,
                    "data": data,
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

        # running フラグを先に False にして、デコード処理を停止
        self.running = False

        # デコーダーを先にクリーンアップ（エラー抑制のため）
        if self.video_decoder:
            try:
                self.video_decoder.close()
            except Exception:
                pass
            finally:
                self.video_decoder = None

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
                            f"DELETE request returned status {response.status_code}"
                        )
            except Exception as e:
                handle_error("terminating WHEP session", e)

        time.sleep(0.5)

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
    """フレームを表示（ffplay を使用）"""
    logger.info("Starting video display with ffplay...")

    frame_count = 0
    ffplay_process = None
    video_size = None

    try:
        while client.running:
            if client.frame_queue is None:
                break

            # キューからフレームを取得
            try:
                frame = client.frame_queue.get(timeout=0.1)
                frame_count += 1

                # 最初のフレームで ffplay を起動
                if ffplay_process is None:
                    height, width = frame.shape[:2]
                    video_size = f"{width}x{height}"
                    logger.info(f"First frame: shape={frame.shape}, starting ffplay...")

                    ffplay_process = subprocess.Popen(
                        [
                            "ffplay",
                            "-f", "rawvideo",
                            "-pixel_format", "bgr24",
                            "-video_size", video_size,
                            "-framerate", "30",
                            "-i", "-",
                            "-autoexit",
                            "-loglevel", "quiet",
                        ],
                        stdin=subprocess.PIPE,
                    )

                # フレームを ffplay に送信
                if ffplay_process and ffplay_process.stdin and ffplay_process.poll() is None:
                    try:
                        ffplay_process.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.info("ffplay closed")
                        break
                else:
                    # ffplay が終了した
                    break

            except queue.Empty:
                # ffplay が終了したかチェック
                if ffplay_process and ffplay_process.poll() is not None:
                    logger.info("ffplay exited")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        if ffplay_process:
            try:
                if ffplay_process.stdin:
                    ffplay_process.stdin.close()
            except Exception:
                pass
            ffplay_process.terminate()
            ffplay_process.wait(timeout=2.0)

    logger.info(f"Display finished. Frames displayed: {frame_count}")
    return ffplay_process is not None and ffplay_process.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="WHEP クライアント（draft-ietf-wish-whep-03 準拠）"
    )
    parser.add_argument("--url", required=True, help="WHEP エンドポイント URL")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグログを出力",
    )
    parser.add_argument("--token", help="Bearer トークン（認証用）")
    parser.add_argument("--duration", type=int, help="受信時間（秒）")
    parser.add_argument("--display", action="store_true", help="映像を表示")
    parser.add_argument(
        "--video-codec-type",
        choices=["h264", "h265"],
        default="h264",
        help="映像コーデック (デフォルト: h264)",
    )

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Video codec: {args.video_codec_type}")
    logger.info(f"WHEP endpoint: {args.url}")

    client = WHEPClient(
        args.url, args.token, args.display, video_codec=args.video_codec_type
    )

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
