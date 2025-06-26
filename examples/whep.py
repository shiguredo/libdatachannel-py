import argparse
import json
import logging
import re
import signal
import sys
import time
from typing import Optional, List
from urllib.parse import urljoin

import httpx

from libdatachannel import (
    PeerConnection,
    Configuration,
    IceServer,
    RtpDepacketizer,
    OpusRtpDepacketizer,
    H264RtpDepacketizer,
    NalUnit,
    MediaHandler,
    PyMediaHandler,
    make_message,
    Description,
    TransportPolicy,
    RtcpReceivingSession,
)
from libdatachannel.codec import (
    VideoCodecType,
    VideoDecoder,
    EncodedImage,
    VideoFrame,
    create_openh264_video_decoder,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MediaStats:
    """メディア統計情報を管理するクラス"""
    
    def __init__(self):
        self.start_time = time.time()
        self.audio_frames = 0
        self.video_frames = 0
        self.audio_bytes = 0
        self.video_bytes = 0
        self.last_stats_time = time.time()
        self.last_audio_frames = 0
        self.last_video_frames = 0
    
    def update_audio(self, size: int):
        self.audio_frames += 1
        self.audio_bytes += size
    
    def update_video(self, size: int):
        self.video_frames += 1
        self.video_bytes += size
    
    def log_stats(self):
        now = time.time()
        elapsed = now - self.last_stats_time
        if elapsed >= 1.0:  # 1秒ごとに統計を出力
            audio_fps = (self.audio_frames - self.last_audio_frames) / elapsed
            video_fps = (self.video_frames - self.last_video_frames) / elapsed
            
            total_elapsed = now - self.start_time
            logger.info(
                f"Stats - Audio: {audio_fps:.1f} fps, "
                f"Video: {video_fps:.1f} fps, "
                f"Total: {self.audio_bytes + self.video_bytes:,} bytes, "
                f"Duration: {total_elapsed:.1f}s"
            )
            
            self.last_stats_time = now
            self.last_audio_frames = self.audio_frames
            self.last_video_frames = self.video_frames


class H264DepacketizerHandler(PyMediaHandler):
    """H.264 RTPデパケタイザーのPythonハンドラー"""
    def __init__(self, stats: MediaStats, openh264_path: Optional[str] = None):
        super().__init__()
        self.stats = stats
        self.depacketizer = H264RtpDepacketizer()
        self.frame_count = 0
        self.decoder = None
        self.openh264_path = openh264_path
        self.decoded_frame_count = 0
        
        # OpenH264デコーダーを初期化
        if self.openh264_path:
            self._init_decoder()
    
    def incoming(self, messages, send):
        """RTPパケットを受信してデパケタイズ"""
        # デパケタイザーのincomingを明示的に呼ぶ
        depacketized = self.depacketizer.incoming(messages, send)
        
        # デパケタイズされたメッセージを処理して統計を更新
        for msg in depacketized:
            if msg:
                size = len(msg)
                self.frame_count += 1
                self.stats.update_video(size)
                
                # 最初の数フレームを詳細にログ
                if self.frame_count <= 5:
                    self._log_frame_details(msg, size)
        
        # デパケタイズされたメッセージを返す（次のハンドラーに渡される）
        return depacketized
    
    def _init_decoder(self):
        """OpenH264デコーダーを初期化"""
        try:
            self.decoder = create_openh264_video_decoder(self.openh264_path)
            
            # デコーダー設定
            settings = VideoDecoder.Settings()
            settings.codec_type = VideoCodecType.H264
            
            if not self.decoder.init(settings):
                logger.error("Failed to initialize OpenH264 decoder")
                self.decoder = None
                return
            
            # デコードコールバックを設定
            self.decoder.set_on_decode(self._on_decoded_frame)
            logger.info(f"OpenH264 decoder initialized successfully with library: {self.openh264_path}")
            
        except Exception as e:
            logger.error(f"Failed to create OpenH264 decoder: {e}")
            self.decoder = None
    
    def _on_decoded_frame(self, frame: VideoFrame):
        """デコードされたフレームを処理"""
        self.decoded_frame_count += 1
        
        if self.decoded_frame_count <= 5:
            logger.info(f"Decoded frame #{self.decoded_frame_count}:")
            logger.info(f"  Format: {frame.format}")
            logger.info(f"  Size: {frame.width()}x{frame.height()}")
            logger.info(f"  Timestamp: {frame.timestamp}")
            
            # フレームタイプに応じてバッファを取得
            if frame.format == ImageFormat.I420:
                buffer = frame.i420_buffer
                if buffer:
                    logger.info(f"  I420 buffer - Y stride: {buffer.stride_y()}, U stride: {buffer.stride_u()}, V stride: {buffer.stride_v()}")
            elif frame.format == ImageFormat.NV12:
                buffer = frame.nv12_buffer
                if buffer:
                    logger.info(f"  NV12 buffer - Y stride: {buffer.stride_y()}, UV stride: {buffer.stride_uv()}")
    
    def _log_frame_details(self, msg, size):
        """フレームの詳細をログ出力"""
        logger.info(f"video depacketized frame #{self.frame_count}: size={size} bytes")
        
        # frame_infoがあるか確認
        frame_info = getattr(msg, 'frame_info', None)
        if frame_info:
            logger.info(f"  Frame info found: PT={frame_info.payload_type}, TS={frame_info.timestamp}")
        
        # データの内容を確認
        data = bytes(msg)
        logger.info(f"  First 20 bytes: {data[:20].hex() if len(data) >= 20 else data.hex()}")
        
        # H.264 NALユニットのスタートコードを確認
        if len(data) >= 4:
            start_code_len = 0
            if data[:4] == b'\x00\x00\x00\x01':
                start_code_len = 4
                logger.info(f"  *** H.264 NAL unit with long start code detected! ***")
            elif data[:3] == b'\x00\x00\x01':
                start_code_len = 3
                logger.info(f"  *** H.264 NAL unit with short start code detected! ***")
            
            if start_code_len > 0 and len(data) > start_code_len:
                try:
                    # スタートコードを除いたNALユニットデータ
                    nal_data = data[start_code_len:]
                    nal_unit = NalUnit(nal_data)
                    
                    # NalUnitクラスで全ての情報を取得
                    forbidden_bit = nal_unit.forbidden_bit()
                    nri = nal_unit.nri()
                    nal_type = nal_unit.unit_type()
                    
                    logger.info(f"  NAL unit type: {nal_type}")
                    logger.info(f"  NAL header: forbidden_bit={forbidden_bit}, NRI={nri}")
                    
                    # ペイロードを取得
                    payload = nal_unit.payload()
                    if payload:
                        logger.info(f"  NAL payload size: {len(payload)} bytes")
                        
                        # NALタイプ別の処理
                        if nal_type == 7:  # SPS
                            logger.info(f"  *** SPS (Sequence Parameter Set) detected ***")
                            if len(payload) > 4:
                                logger.info(f"  SPS payload (first 4 bytes): {bytes(payload[:4]).hex()}")
                        elif nal_type == 8:  # PPS
                            logger.info(f"  *** PPS (Picture Parameter Set) detected ***")
                        elif nal_type == 5:  # IDR
                            logger.info(f"  *** IDR (Instantaneous Decoder Refresh) slice detected ***")
                        elif nal_type == 1:  # Non-IDR slice
                            logger.info(f"  *** Non-IDR slice detected ***")
                        elif nal_type == 6:  # SEI
                            logger.info(f"  *** SEI (Supplemental Enhancement Information) detected ***")
                        elif nal_type == 9:  # AUD
                            logger.info(f"  *** AUD (Access Unit Delimiter) detected ***")
                        
                except Exception as e:
                    logger.error(f"  Failed to parse NAL unit: {e}")
        
        # デコーダーが有効な場合はデコード
        if self.decoder and size > 0:
            try:
                import numpy as np
                
                # EncodedImageを作成
                encoded_image = EncodedImage()
                # データはすでにスタートコード付きでデパケタイズされているので、そのままnumpy配列に変換
                encoded_image.data = np.frombuffer(data, dtype=np.uint8)
                
                # frame_infoからタイムスタンプを設定
                if hasattr(msg, 'frame_info') and msg.frame_info:
                    # タイムスタンプをtimedeltaに変換（RTPタイムスタンプは90kHzの場合が多い）
                    timestamp_ms = msg.frame_info.timestamp / 90.0  # 90kHz -> ms
                    encoded_image.timestamp = timestamp_ms / 1000.0  # ms -> seconds
                
                # デバッグ用：引数の型を確認
                logger.debug(f"Decoding with encoded_image type: {type(encoded_image)}, data type: {type(encoded_image.data)}")
                
                # デコード実行（numpy配列を直接渡す）
                self.decoder.decode(encoded_image.data)
                
            except Exception as e:
                logger.error(f"Failed to decode frame: {e}")



class WHEPClient:
    """WHEP クライアントの実装"""
    
    def __init__(self, endpoint_url: str, bearer_token: Optional[str] = None, openh264_path: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.bearer_token = bearer_token
        self.openh264_path = openh264_path
        self.session_url: Optional[str] = None
        self.pc: Optional[PeerConnection] = None
        self.session: Optional[httpx.Client] = None
        self.stats = MediaStats()
        self._running = False
        self.video_handler = None
    
    def connect(self):
        """WHEP エンドポイントに接続"""
        logger.info(f"Connecting to WHEP endpoint: {self.endpoint_url}")
        
        # Create peer connection
        config = Configuration()
        # No default ICE servers - will use TURN from Link header
        config.ice_servers = []
        config.ice_transport_policy = TransportPolicy.Relay  # TURNを強制
        
        # Try to disable auto gathering if available (for later adding ICE servers from Link header)
        if hasattr(config, "disable_auto_gathering"):
            config.disable_auto_gathering = True
        
        self.pc = PeerConnection(config)
        
        # コネクション状態の変更を監視
        self.pc.on_state_change(self._on_state_change)
        self.pc.on_ice_state_change(self._on_ice_state_change)
        
        # Add audio track FIRST (before video)
        audio_desc = Description.Audio("audio", Description.Direction.SendRecv)
        audio_desc.add_opus_codec(111)
        logger.info("Audio description created with Opus codec")
        self.audio_track = self.pc.add_track(audio_desc)
        
        # Add video track SECOND (after audio)
        video_desc = Description.Video("video", Description.Direction.SendRecv)
        video_desc.add_h264_codec(96)  # H.264のみ、payload type 96
        self.video_track = self.pc.add_track(video_desc)
        logger.info("Audio track added with Opus codec (PT=111)")
        logger.info("Video track added with H.264 codec (PT=96)")
        
        # Set up media handlers
        self._setup_media_handlers()
        
        # Create offer
        self.pc.set_local_description()
        
        # Get local SDP
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise Exception("Failed to create offer")
        
        # Convert to recvonly
        offer_str = str(local_sdp)
        sdp_lines = offer_str.split('\n')
        modified_sdp = []
        for line in sdp_lines:
            if line.startswith('a=sendrecv'):
                modified_sdp.append('a=recvonly')
            elif line.startswith('a=sendonly'):
                modified_sdp.append('a=recvonly')
            else:
                modified_sdp.append(line)
        
        offer_sdp = '\n'.join(modified_sdp)
        logger.debug(f"SDP Offer:\n{offer_sdp}")
        
        # Send offer to WHEP server
        logger.info("Sending offer to WHEP server...")
        
        # Create HTTP session
        headers = {"Content-Type": "application/sdp"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        
        self.session = httpx.Client(timeout=10.0)
        
        try:
            response = self.session.post(
                self.endpoint_url,
                content=offer_sdp,
                headers=headers,
                follow_redirects=True,
            )
            
            if response.status_code == 201:
                # 成功：SDP アンサーとセッション URL を取得
                answer_sdp = response.text
                self.session_url = response.headers.get("Location")
                
                # Location ヘッダーが相対パスの場合の処理
                if self.session_url and not self.session_url.startswith(('http://', 'https://')):
                    self.session_url = urljoin(self.endpoint_url, self.session_url)
                
                logger.info(f"Session created: {self.session_url}")
                logger.debug(f"SDP Answer:\n{answer_sdp}")
                
                # Parse Link header for ICE servers
                link_header = response.headers.get("Link")
                if link_header:
                    ice_servers = self._parse_link_header(link_header)
                    if ice_servers:
                        logger.info(f"Found {len(ice_servers)} ICE server(s) in Link header")
                        # Try to add ICE servers if method is available
                        if hasattr(self.pc, "gather_local_candidates"):
                            self.pc.gather_local_candidates(ice_servers)
                        else:
                            logger.warning("Cannot add ICE servers after PeerConnection creation")
                
                # Set remote SDP
                answer_desc = Description(answer_sdp, Description.Type.Answer)
                self.pc.set_remote_description(answer_desc)
                
                # 接続が確立されるまで待機
                self._wait_for_connection()
                
                logger.info("WHEP connection established successfully")
                self._running = True
                
            elif response.status_code == 409:
                # まだパブリッシャーがいない
                retry_after = int(response.headers.get("Retry-After", "5"))
                raise Exception(f"No active publisher, retry after {retry_after} seconds")
            else:
                error_text = response.text
                raise Exception(f"Failed to create session: {response.status_code} - {error_text}")
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.disconnect()
            raise
    
    def _setup_media_handlers(self):
        """メディアハンドラーを設定"""
        # 音声用のハンドラーを設定
        # RTCPセッションを最初に設定
        audio_rtcp = RtcpReceivingSession()
        self.audio_track.set_media_handler(audio_rtcp)
        
        # OpusRtpDepacketizerを直接チェーンに追加
        audio_depacketizer = OpusRtpDepacketizer()
        audio_rtcp.add_to_chain(audio_depacketizer)
        
        # 音声用の簡易ハンドラーを追加（統計のみ）
        class AudioStatsHandler(PyMediaHandler):
            def __init__(self, stats):
                super().__init__()
                self.stats = stats
                self.frame_count = 0
            
            def incoming(self, messages, send):
                for msg in messages:
                    if msg:
                        self.stats.update_audio(len(msg))
                        self.frame_count += 1
                        
                        # 最初の数フレームだけログ
                        if self.frame_count <= 5:
                            frame_info = getattr(msg, 'frame_info', None)
                            if frame_info:
                                logger.info(f"Audio frame #{self.frame_count}: PT={frame_info.payload_type}, size={len(msg)} bytes")
                
                return messages
        
        audio_handler = AudioStatsHandler(self.stats)
        audio_depacketizer.add_to_chain(audio_handler)
        
        def on_audio_open():
            logger.info(f"Audio track opened! is_open: {self.audio_track.is_open()}")
        
        self.audio_track.on_open(on_audio_open)
        logger.info(f"Audio media handler configured, track is_open: {self.audio_track.is_open()}")
        
        # ビデオ用のハンドラーを設定
        # RTCPセッションを最初に設定
        video_rtcp = RtcpReceivingSession()
        self.video_track.set_media_handler(video_rtcp)
        
        # H264デパケタイザーハンドラーをチェーンに追加（カスタム実装）
        self.video_handler = H264DepacketizerHandler(self.stats, self.openh264_path)
        video_rtcp.add_to_chain(self.video_handler)
        
        # on_frameコールバックは設定しない（MediaHandlerチェーンが処理）
        
        def on_video_open():
            logger.info(f"Video track opened! is_open: {self.video_track.is_open()}")
        
        self.video_track.on_open(on_video_open)
        logger.info(f"Video media handler configured, track is_open: {self.video_track.is_open()}")
    
    # 削除 - connect メソッドに統合
    
    def _parse_link_header(self, link_header: str) -> List[IceServer]:
        """Link ヘッダーから ICE サーバー情報を解析 (whip.py の実装を参考)"""
        ice_servers = []
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
    
    def _parse_ice_servers(self, headers) -> List[IceServer]:
        """Link ヘッダーから ICE サーバー情報を解析"""
        link_header = headers.get("Link")
        return self._parse_link_header(link_header)
    
    def _wait_for_connection(self):
        """接続が確立されるまで待機"""
        timeout = 60  # 60秒のタイムアウト（TURNの場合時間がかかる）
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            state = self.pc.state()
            ice_state = self.pc.ice_state()
            
            if state == PeerConnection.State.Connected:
                return
            elif state == PeerConnection.State.Failed or state == PeerConnection.State.Closed:
                raise Exception(f"Connection failed: {state}")
            elif ice_state == PeerConnection.IceState.Failed:
                raise Exception("ICE connection failed")
            
            # 定期的に状態をログ出力
            if int(time.time() - start_time) % 5 == 0:
                logger.debug(f"Waiting for connection... State: {state}, ICE: {ice_state}")
            
            time.sleep(0.1)
        
        raise Exception("Connection timeout")
    
    def _on_state_change(self, state):
        """PeerConnection の状態変更ハンドラー"""
        logger.info(f"Connection state changed: {state}")
    
    def _on_ice_state_change(self, state):
        """ICE 接続状態の変更ハンドラー"""
        logger.info(f"ICE state changed: {state}")
        if state == PeerConnection.IceState.Connected:
            logger.info("ICE connection established!")
    
    def run(self, duration: Optional[int] = None):
        """メディアを受信"""
        logger.info("Starting media reception...")
        start_time = time.time()
        
        try:
            while self._running:
                # 統計情報を出力
                self.stats.log_stats()
                
                # 指定された期間が経過したら終了
                if duration and time.time() - start_time >= duration:
                    logger.info(f"Specified duration ({duration}s) reached")
                    break
                
                # 接続状態を確認
                if self.pc.state() != PeerConnection.State.Connected:
                    logger.warning("Connection lost")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """切断処理"""
        self._running = False
        
        # セッションを削除
        if self.session and self.session_url:
            try:
                response = self.session.delete(self.session_url)
                if response.status_code == 200:
                    logger.info("Session terminated successfully")
                else:
                    logger.warning(f"Failed to terminate session: {response.status_code}")
            except Exception as e:
                logger.error(f"Error terminating session: {e}")
        
        # トラックをクリア
        if hasattr(self, 'audio_track') and self.audio_track:
            self.audio_track.on_open(lambda: None)  # コールバックをクリア
            self.audio_track.on_frame(lambda x: None)
            self.audio_track = None
            
        if hasattr(self, 'video_track') and self.video_track:
            self.video_track.on_open(lambda: None)  # コールバックをクリア
            self.video_track.on_frame(lambda x: None)
            self.video_track = None
        
        # PeerConnection をクローズ
        if self.pc:
            # コールバックをクリア
            self.pc.on_state_change(lambda x: None)
            self.pc.on_ice_state_change(lambda x: None)
            self.pc.close()
            self.pc = None
        
        # HTTP セッションをクローズ
        if self.session:
            self.session.close()
            self.session = None
        
        # 最終統計を出力
        logger.info(
            f"Final stats - Audio frames: {self.stats.audio_frames}, "
            f"Video frames: {self.stats.video_frames}, "
            f"Total bytes: {self.stats.audio_bytes + self.stats.video_bytes:,}"
        )
        
        # デコーダーの統計も出力
        if self.video_handler and self.video_handler.decoder:
            logger.info(f"Decoded frames: {self.video_handler.decoded_frame_count}")
        
        # デコーダーをクリーンアップ
        if self.video_handler and self.video_handler.decoder:
            try:
                self.video_handler.decoder.release()
            except Exception as e:
                logger.error(f"Error releasing decoder: {e}")


def main():
    parser = argparse.ArgumentParser(description="WHEP client example")
    parser.add_argument(
        "--url",
        required=True,
        help="WHEP endpoint URL"
    )
    parser.add_argument(
        "--token",
        help="Bearer token for authentication"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Duration in seconds to receive media (default: unlimited)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--openh264",
        help="Path to OpenH264 library for H.264 decoding"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # WHEP クライアントを作成
    client = WHEPClient(args.url, args.token, args.openh264)
    
    # シグナルハンドラーを設定
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        if client:
            client._running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 接続
        client.connect()
        
        # メディアを受信
        client.run(args.duration)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # 必ず切断処理を実行
        client.disconnect()


if __name__ == "__main__":
    main()