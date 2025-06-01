import argparse
import logging
import queue
import re
import threading
import time
from typing import List, Optional
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
import sounddevice as sd

from libdatachannel import (
    AV1RtpPacketizer,
    Configuration,
    Description,
    IceServer,
    OpusRtpPacketizer,
    PeerConnection,
    PliHandler,
    RtcpNackResponder,
    RtcpSrReporter,
    RtpPacketizationConfig,
    Track,
)
from libdatachannel.codec import (
    AudioCodecType,
    AudioEncoder,
    AudioFrame,
    ImageFormat,
    VideoCodecType,
    VideoEncoder,
    VideoFrame,
    VideoFrameBufferI420,
    create_aom_video_encoder,
    create_opus_audio_encoder,
)
from libdatachannel.libyuv import (
    FourCC,
    RotationMode,
    convert_to_i420,
    rgb24_to_i420,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WHIPClient:
    """Minimal WHIP client for sending test video and audio"""

    def __init__(self, whip_url: str, bearer_token: Optional[str] = None):
        self.whip_url = whip_url
        self.bearer_token = bearer_token
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Encoders
        self.video_encoder: Optional[VideoEncoder] = None
        self.audio_encoder: Optional[AudioEncoder] = None

        # RTP components
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_sr_reporter = None
        self.audio_sr_reporter = None
        self.pli_handler = None
        self.nack_responder = None

        # Frame counters
        self.video_frame_number = 0
        self.audio_timestamp_ms = 0

        # Video settings
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = 30

        # Audio settings
        self.audio_sample_rate = 48000
        self.audio_channels = 2

        # Key frame interval settings
        self.key_frame_interval_seconds = 2.0
        self.last_key_frame_time = None

        # Camera and audio capture
        self.camera = None
        self.camera_thread = None
        self.audio_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.video_queue = queue.Queue(maxsize=30)
        self.capture_active = False

    def _parse_link_header(self, link_header: str) -> List[IceServer]:
        """Parse Link header for ICE servers"""
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

    def connect(self):
        """Connect to WHIP server"""
        logger.info(f"Connecting to WHIP endpoint: {self.whip_url}")

        # Create peer connection
        config = Configuration()
        # No default ICE servers - will use TURN from Link header
        config.ice_servers = []

        # Try to disable auto gathering if available (for later adding ICE servers from Link header)
        if hasattr(config, "disable_auto_gathering"):
            config.disable_auto_gathering = True

        self.pc = PeerConnection(config)

        # Add video track
        video_desc = Description.Video("video", Description.Direction.SendOnly)
        video_desc.add_av1_codec(35)
        self.video_track = self.pc.add_track(video_desc)

        # Add audio track
        audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # Set up encoders
        self._setup_video_encoder()
        self._setup_audio_encoder()

        # Create offer
        self.pc.set_local_description()

        # Get local SDP
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise Exception("Failed to create offer")

        # Send offer to WHIP server
        logger.info("Sending offer to WHIP server...")
        with httpx.Client(timeout=10.0) as client:
            headers = {
                "Content-Type": "application/sdp",
            }
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"

            response = client.post(
                self.whip_url,
                content=str(local_sdp),
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 201:
                raise Exception(f"WHIP server returned {response.status_code}: {response.text}")

            # Get session URL
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whip_url, self.session_url)

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
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)

        logger.info("Connected to WHIP server")

    def _setup_video_encoder(self):
        """Set up AV1 video encoder"""
        self.video_encoder = create_aom_video_encoder()

        settings = VideoEncoder.Settings()
        settings.codec_type = VideoCodecType.AV1
        settings.width = self.video_width
        settings.height = self.video_height
        settings.bitrate = 2000000  # 2 Mbps
        settings.fps = self.video_fps

        if not self.video_encoder.init(settings):
            raise Exception("Failed to initialize video encoder")

        # Set up RTP packetizer
        video_config = RtpPacketizationConfig(
            ssrc=1234567, cname="video-stream", payload_type=35, clock_rate=90000
        )

        self.video_packetizer = AV1RtpPacketizer(
            AV1RtpPacketizer.Packetization.TemporalUnit, video_config
        )

        # Add RTCP SR reporter
        self.video_sr_reporter = RtcpSrReporter(video_config)
        self.video_packetizer.add_to_chain(self.video_sr_reporter)

        # Add PLI handler
        def on_pli():
            logger.info("PLI received - Picture Loss Indication")
        
        self.pli_handler = PliHandler(on_pli)
        self.video_packetizer.add_to_chain(self.pli_handler)

        # Add NACK responder for retransmission
        self.nack_responder = RtcpNackResponder()
        self.video_packetizer.add_to_chain(self.nack_responder)
        logger.info("NACK responder added for video track")

        # Set packetizer on track
        if not self.video_track:
            raise RuntimeError("Video track not initialized")
        self.video_track.set_media_handler(self.video_packetizer)

        # Set encoder callback
        def on_encoded(encoded_image):
            if self.video_track and self.video_track.is_open():
                try:
                    data = encoded_image.data.tobytes()
                    self.video_track.send(data)
                except Exception as e:
                    logger.error(f"Error sending encoded video: {e}")

        self.video_encoder.set_on_encode(on_encoded)

    def _setup_audio_encoder(self):
        """Set up Opus audio encoder"""
        self.audio_encoder = create_opus_audio_encoder()

        settings = AudioEncoder.Settings()
        settings.codec_type = AudioCodecType.OPUS
        settings.sample_rate = self.audio_sample_rate
        settings.channels = self.audio_channels
        settings.bitrate = 64000  # 64 kbps
        settings.frame_duration_ms = 20

        if not self.audio_encoder.init(settings):
            raise Exception("Failed to initialize audio encoder")

        # Set up RTP packetizer
        audio_config = RtpPacketizationConfig(
            ssrc=7654321, cname="audio-stream", payload_type=111, clock_rate=48000
        )

        self.audio_packetizer = OpusRtpPacketizer(audio_config)

        # Add RTCP SR reporter
        self.audio_sr_reporter = RtcpSrReporter(audio_config)
        self.audio_packetizer.add_to_chain(self.audio_sr_reporter)

        # Set packetizer on track
        if not self.audio_track:
            raise RuntimeError("Audio track not initialized")
        self.audio_track.set_media_handler(self.audio_packetizer)

        # Set encoder callback
        def on_encoded(encoded_audio):
            if self.audio_track and self.audio_track.is_open():
                try:
                    data = encoded_audio.data.tobytes()
                    self.audio_track.send(data)
                except Exception as e:
                    logger.error(f"Error sending encoded audio: {e}")

        self.audio_encoder.set_on_encode(on_encoded)

    def _capture_camera(self):
        """Capture video from camera"""
        logger.info("Starting camera capture...")
        
        # Open camera (0 for default camera)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            return
            
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.video_fps)
        
        # Get actual camera properties
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} fps")
        
        while self.capture_active:
            ret, frame = self.camera.read()
            if ret:
                try:
                    self.video_queue.put_nowait(frame)
                except queue.Full:
                    # Drop frame if queue is full
                    pass
            else:
                logger.error("Failed to read frame from camera")
                time.sleep(0.1)
        
        self.camera.release()
        logger.info("Camera capture stopped")

    def _capture_audio(self):
        """Capture audio from microphone"""
        logger.info("Starting audio capture...")
        
        # Audio callback
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert to mono if needed
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            # Convert to stereo for Opus encoder
            stereo_data = np.column_stack((audio_data, audio_data))
            
            try:
                self.audio_queue.put_nowait(stereo_data.astype(np.float32))
            except queue.Full:
                # Drop audio if queue is full
                pass
        
        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=1,  # Capture mono
                callback=audio_callback,
                blocksize=int(self.audio_sample_rate * 0.02),  # 20ms blocks
                dtype='float32'
            ):
                logger.info(f"Audio capture started at {self.audio_sample_rate} Hz")
                while self.capture_active:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
        
        logger.info("Audio capture stopped")

    def send_frames(self, duration: Optional[int] = None, use_camera: bool = False, use_mic: bool = False):
        """Send video and audio frames from camera/mic or dummy data"""
        if not self.pc:
            raise RuntimeError("PeerConnection not initialized. Call connect() first.")

        # Wait for connection
        timeout = 10.0
        start_time = time.time()
        while self.pc.state() != PeerConnection.State.Connected:
            if time.time() - start_time > timeout:
                raise Exception("Connection timeout")
            time.sleep(0.1)

        logger.info("Connection established")

        # Start capture threads if needed
        if use_camera or use_mic:
            self.capture_active = True
            
            if use_camera:
                self.camera_thread = threading.Thread(target=self._capture_camera)
                self.camera_thread.start()
                logger.info("Started camera capture thread")
            
            if use_mic:
                self.audio_thread = threading.Thread(target=self._capture_audio)
                self.audio_thread.start()
                logger.info("Started audio capture thread")
            
            # Give capture threads time to start
            time.sleep(1.0)

        # Frame intervals
        video_interval = 1.0 / self.video_fps
        audio_interval = 0.02  # 20ms

        start_time = time.time()
        next_video_time = start_time
        next_audio_time = start_time

        try:
            while True:
                current_time = time.time()

                # Check duration
                if duration and current_time - start_time >= duration:
                    break

                # Send video frame
                if current_time >= next_video_time:
                    if use_camera:
                        self._send_video_frame()
                    else:
                        self._send_dummy_video_frame()
                    next_video_time += video_interval

                # Send audio frame
                if current_time >= next_audio_time:
                    if use_mic:
                        self._send_audio_frame()
                    else:
                        self._send_dummy_audio_frame()
                    next_audio_time += audio_interval

                # Sleep until next frame
                next_time = min(next_video_time, next_audio_time)
                sleep_time = max(0, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if use_camera or use_mic:
                # Stop capture threads
                self.capture_active = False
                if use_camera and self.camera_thread:
                    self.camera_thread.join(timeout=2.0)
                if use_mic and self.audio_thread:
                    self.audio_thread.join(timeout=2.0)
                logger.info("Stopped capture threads")

    def _send_video_frame(self):
        """Send a video frame from camera"""
        if not self.video_encoder:
            return

        try:
            # Get frame from queue (non-blocking)
            bgr_frame = self.video_queue.get_nowait()
        except queue.Empty:
            # No frame available, skip this iteration
            return

        # Keep OpenCV's BGR format and pass it directly
        # rgb24_to_i420 might actually expect BGR despite its name
        frame = bgr_frame
        
        # Ensure frame is the expected size
        if frame.shape[:2] != (self.video_height, self.video_width):
            frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        # Ensure the frame is contiguous in memory
        frame = np.ascontiguousarray(frame)
        
        # Create I420 buffer
        buffer = VideoFrameBufferI420.create(self.video_width, self.video_height)
        
        # Convert BGR to I420 using libyuv (despite the name rgb24_to_i420)
        rgb24_to_i420(
            frame,
            self.video_width * 3,  # RGB stride
            buffer.y,
            buffer.u,
            buffer.v,
            buffer.stride_y(),
            buffer.stride_u(),
            buffer.stride_v(),
            self.video_width,
            self.video_height
        )
        
        # Create video frame
        frame = VideoFrame()
        frame.format = ImageFormat.I420
        frame.i420_buffer = buffer
        frame.timestamp = self.video_frame_number / self.video_fps
        frame.frame_number = self.video_frame_number

        # Encode frame
        try:
            self.video_encoder.encode(frame)
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")

        self.video_frame_number += 1

        # Check if it's time to request a key frame (AFTER encoding the frame)
        current_time = time.time()

        # Initialize last_key_frame_time if this is the first frame
        if self.last_key_frame_time is None:
            self.last_key_frame_time = current_time
        time_since_last_key = current_time - self.last_key_frame_time

        # Request key frame every 2 seconds
        if time_since_last_key >= self.key_frame_interval_seconds:
            logger.info(f"Requesting key frame (time since last: {time_since_last_key:.2f}s)")
            try:
                # Force the encoder to generate a key frame (intra frame)
                self.video_encoder.force_intra_next_frame()
            except Exception as e:
                logger.error(f"Error requesting keyframe: {e}")
            self.last_key_frame_time = current_time

        # Debug log every 30 frames (1 second)
        if self.video_frame_number % 30 == 0:
            logger.info(f"Frame {self.video_frame_number}: total frames sent")

    def _send_audio_frame(self):
        """Send an audio frame from microphone"""
        if not self.audio_encoder:
            return

        try:
            # Get audio data from queue (non-blocking)
            audio_data = self.audio_queue.get_nowait()
        except queue.Empty:
            # No audio available, send silence
            samples = int(self.audio_sample_rate * 0.02)  # 960 samples
            audio_data = np.zeros((samples, self.audio_channels), dtype=np.float32)

        # Create audio frame
        frame = AudioFrame()
        frame.sample_rate = self.audio_sample_rate
        frame.pcm = audio_data
        frame.timestamp = self.audio_timestamp_ms / 1000.0

        # Encode frame
        self.audio_encoder.encode(frame)
        self.audio_timestamp_ms += 20

    def _send_dummy_video_frame(self):
        """Send a black dummy video frame"""
        if not self.video_encoder:
            return

        # Create I420 frame
        frame = VideoFrame()
        frame.format = ImageFormat.I420

        # Create I420 buffer
        buffer = VideoFrameBufferI420.create(self.video_width, self.video_height)

        # Fill with black (Y=16, U=128, V=128)
        # Y plane
        y_data = np.full((self.video_height, buffer.stride_y()), 16, dtype=np.uint8)
        buffer.y = y_data

        # U plane
        u_height = self.video_height // 2
        u_data = np.full((u_height, buffer.stride_u()), 128, dtype=np.uint8)
        buffer.u = u_data

        # V plane
        v_height = self.video_height // 2
        v_data = np.full((v_height, buffer.stride_v()), 128, dtype=np.uint8)
        buffer.v = v_data

        frame.i420_buffer = buffer
        frame.timestamp = self.video_frame_number / self.video_fps
        frame.frame_number = self.video_frame_number

        # Encode frame
        try:
            self.video_encoder.encode(frame)
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")

        self.video_frame_number += 1

        # Check if it's time to request a key frame
        current_time = time.time()
        if self.last_key_frame_time is None:
            self.last_key_frame_time = current_time
        time_since_last_key = current_time - self.last_key_frame_time

        if time_since_last_key >= self.key_frame_interval_seconds:
            logger.info(f"Requesting key frame (time since last: {time_since_last_key:.2f}s)")
            try:
                self.video_encoder.force_intra_next_frame()
            except Exception as e:
                logger.error(f"Error requesting keyframe: {e}")
            self.last_key_frame_time = current_time

    def _send_dummy_audio_frame(self):
        """Send a silent dummy audio frame"""
        if not self.audio_encoder:
            return

        # Create audio frame
        frame = AudioFrame()
        frame.sample_rate = self.audio_sample_rate

        # Generate 20ms of silence
        samples = int(self.audio_sample_rate * 0.02)  # 960 samples
        silence = np.zeros((samples, self.audio_channels), dtype=np.float32)

        frame.pcm = silence
        frame.timestamp = self.audio_timestamp_ms / 1000.0

        # Encode frame
        self.audio_encoder.encode(frame)
        self.audio_timestamp_ms += 20

    def disconnect(self):
        """Disconnect from WHIP server with graceful shutdown"""
        logger.info("Starting graceful shutdown...")

        # First, send DELETE request to WHIP server
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
            except httpx.TimeoutException:
                logger.error("DELETE request timed out")
            except httpx.RequestError as e:
                logger.error(f"DELETE request failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during DELETE: {e}")

        # Wait a bit for graceful shutdown
        logger.info("Waiting for graceful shutdown...")
        time.sleep(0.5)

        # Stop capture if active
        self.capture_active = False
        
        # Release camera if open
        if self.camera and self.camera.isOpened():
            self.camera.release()
            logger.info("Camera released")
        
        # Clean up resources in proper order
        logger.info("Cleaning up resources...")

        # Clear RTP components first
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_sr_reporter = None
        self.audio_sr_reporter = None
        self.pli_handler = None
        self.nack_responder = None

        # Close tracks before closing PeerConnection
        self.video_track = None
        self.audio_track = None

        # Close PeerConnection
        if self.pc:
            try:
                self.pc.close()
            except Exception as e:
                logger.error(f"Error closing PeerConnection: {e}")
            finally:
                self.pc = None

        # Finally release encoders
        if self.video_encoder:
            try:
                self.video_encoder.release()
            except Exception as e:
                logger.error(f"Error releasing video encoder: {e}")
            finally:
                self.video_encoder = None

        if self.audio_encoder:
            try:
                self.audio_encoder.release()
            except Exception as e:
                logger.error(f"Error releasing audio encoder: {e}")
            finally:
                self.audio_encoder = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="WHIP client with camera/microphone support")
    parser.add_argument("--url", required=True, help="WHIP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--camera", action="store_true", help="Use camera for video capture")
    parser.add_argument("--mic", action="store_true", help="Use microphone for audio capture")

    args = parser.parse_args()

    # Log what sources are being used
    video_source = "camera" if args.camera else "dummy (black video)"
    audio_source = "microphone" if args.mic else "dummy (silence)"
    logger.info(f"Video source: {video_source}")
    logger.info(f"Audio source: {audio_source}")
    
    if args.camera:
        logger.info("Make sure you have a camera connected")
    if args.mic:
        logger.info("Make sure you have microphone permissions enabled")

    client = WHIPClient(args.url, args.token)

    try:
        client.connect()
        client.send_frames(args.duration, use_camera=args.camera, use_mic=args.mic)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always disconnect gracefully
        try:
            client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()
