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
    rgb24_to_i420,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        # Debug counters
        self._audio_callback_count = 0
        self._mic_callback_count = 0
        self._audio_send_count = 0
        self._last_audio_send_time = None

    def _handle_error(self, context: str, error: Exception):
        """Unified error handling"""
        logger.error(f"Error {context}: {error}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback

            traceback.print_exc()

    def _create_black_i420_buffer(self) -> VideoFrameBufferI420:
        """Create a black I420 video buffer"""
        buffer = VideoFrameBufferI420.create(self.video_width, self.video_height)

        # Fill with black (Y=16, U=128, V=128)
        buffer.y = np.full((self.video_height, buffer.stride_y()), 16, dtype=np.uint8)
        buffer.u = np.full((self.video_height // 2, buffer.stride_u()), 128, dtype=np.uint8)
        buffer.v = np.full((self.video_height // 2, buffer.stride_v()), 128, dtype=np.uint8)

        return buffer

    def _check_and_request_keyframe(self):
        """Check if it's time to request a key frame"""
        current_time = time.time()
        if self.last_key_frame_time is None:
            self.last_key_frame_time = current_time

        time_since_last_key = current_time - self.last_key_frame_time
        if time_since_last_key >= self.key_frame_interval_seconds:
            logger.info(f"Requesting key frame (time since last: {time_since_last_key:.2f}s)")
            try:
                self.video_encoder.force_intra_next_frame()
            except Exception as e:
                self._handle_error("requesting keyframe", e)
            self.last_key_frame_time = current_time

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

        # Add audio track FIRST (before video)
        audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        # Opus codec with proper parameters
        # Default: maxplaybackrate=48000;stereo=1;sprop-stereo=1;minptime=10;ptime=20;useinbandfec=1;usedtx=0
        audio_desc.add_opus_codec(111)
        # Log the actual audio description
        logger.info("Audio description created with Opus codec")
        self.audio_track = self.pc.add_track(audio_desc)

        # Add video track SECOND (after audio)
        video_desc = Description.Video("video", Description.Direction.SendOnly)
        video_desc.add_av1_codec(35)
        self.video_track = self.pc.add_track(video_desc)
        logger.info("Audio track added with Opus codec (PT=111)")
        logger.info(f"Audio track state after adding: is_open={self.audio_track.is_open()}")

        # Set up encoders
        self._setup_video_encoder()
        self._setup_audio_encoder()

        # Create offer
        self.pc.set_local_description()

        # Get local SDP
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise Exception("Failed to create offer")

        # Log SDP for debugging
        # Log SDP media sections order
        sdp_lines = str(local_sdp).split("\n")
        media_lines = [line.strip() for line in sdp_lines if line.startswith("m=")]
        logger.info(f"SDP media sections order: {media_lines}")
        audio_lines = [
            line.strip() for line in sdp_lines if "opus" in line.lower() or "a=rtpmap:111" in line
        ]
        logger.info(f"SDP audio codec lines: {audio_lines}")

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
        settings.bitrate = 2500000  # 2 Mbps
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
                    self._handle_error("sending encoded video", e)

        self.video_encoder.set_on_encode(on_encoded)

    def _setup_audio_encoder(self):
        """Set up Opus audio encoder"""
        logger.info("Setting up audio encoder...")
        self.audio_encoder = create_opus_audio_encoder()

        settings = AudioEncoder.Settings()
        settings.codec_type = AudioCodecType.OPUS
        settings.sample_rate = self.audio_sample_rate
        settings.channels = self.audio_channels
        settings.bitrate = 96000  # 96 kbps
        settings.frame_duration_ms = 20

        # Note: Opus encoder is configured with these settings

        logger.info(
            f"Audio encoder settings: sample_rate={settings.sample_rate}, "
            f"channels={settings.channels}, bitrate={settings.bitrate}, "
            f"frame_duration_ms={settings.frame_duration_ms}"
        )

        if not self.audio_encoder.init(settings):
            raise Exception("Failed to initialize audio encoder")
        logger.info("Audio encoder initialized successfully")

        # Set up RTP packetizer
        audio_config = RtpPacketizationConfig(
            ssrc=7654321, cname="audio-stream", payload_type=111, clock_rate=48000
        )

        # Initialize RTP timestamps
        import random

        audio_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        audio_config.timestamp = audio_config.start_timestamp
        audio_config.sequence_number = random.randint(0, 0xFFFF)

        # Store config for later use
        self.audio_config = audio_config

        # Track last audio timestamp for duration calculation
        self.last_audio_timestamp_us = 0

        logger.info(
            f"Audio RTP config: SSRC={audio_config.ssrc}, "
            f"cname={audio_config.cname}, payload_type={audio_config.payload_type}, "
            f"clock_rate={audio_config.clock_rate}, "
            f"initial_timestamp={audio_config.timestamp}, initial_seq={audio_config.sequence_number}"
        )

        self.audio_packetizer = OpusRtpPacketizer(audio_config)
        logger.info("OpusRtpPacketizer created")

        # Add RTCP SR reporter
        self.audio_sr_reporter = RtcpSrReporter(audio_config)
        self.audio_packetizer.add_to_chain(self.audio_sr_reporter)
        logger.info("RTCP SR reporter added to audio chain")

        # Set packetizer on track
        if not self.audio_track:
            raise RuntimeError("Audio track not initialized")

        # Log track state before setting handler
        logger.info(
            f"Audio track state before set_media_handler: is_open={self.audio_track.is_open()}"
        )

        self.audio_track.set_media_handler(self.audio_packetizer)
        logger.info("Audio packetizer set on track")

        # Log track state after setting handler
        logger.info(
            f"Audio track state after set_media_handler: is_open={self.audio_track.is_open()}"
        )

        # Set encoder callback
        def on_encoded(encoded_audio):
            # Use info level for first few callbacks to ensure visibility
            if not hasattr(self, "_audio_callback_count"):
                self._audio_callback_count = 0
            self._audio_callback_count += 1

            log_level = logging.INFO if self._audio_callback_count <= 10 else logging.DEBUG

            # Calculate timestamp in microseconds
            timestamp_us = int(encoded_audio.timestamp.total_seconds() * 1000000)

            logger.log(
                log_level,
                f"Audio on_encoded callback #{self._audio_callback_count}: "
                f"data size={len(encoded_audio.data)}, timestamp={timestamp_us}us",
            )

            # Log first few bytes of encoded data for debugging
            if self._audio_callback_count <= 5:
                data_bytes = encoded_audio.data.tobytes()
                first_bytes = " ".join(f"{b:02x}" for b in data_bytes[: min(20, len(data_bytes))])
                logger.info(f"First bytes of encoded Opus data: {first_bytes}")
                logger.info(
                    f"Encoded data size: {len(data_bytes)} bytes (expected ~120-400 bytes for 20ms @ 64kbps)"
                )
                # Check if this looks like valid Opus data
                if len(data_bytes) > 0:
                    toc_byte = data_bytes[0]
                    logger.info(
                        f"Opus TOC byte: 0x{toc_byte:02x} (config={toc_byte >> 3}, s={toc_byte >> 2 & 1}, c={toc_byte & 3})"
                    )

                    # Decode TOC byte according to RFC 6716
                    config = (toc_byte >> 3) & 0x1F
                    s = (toc_byte >> 2) & 0x01
                    c = toc_byte & 0x03

                    # Configuration mode
                    if config < 12:
                        mode = "SILK-only"
                        bandwidth = ["NB", "MB", "WB", "SWB"][config // 3]
                        frame_size_ms = [10, 20, 40, 60][config % 3]
                    elif config < 16:
                        mode = "Hybrid"
                        bandwidth = ["SWB", "FB"][config - 12] if config < 14 else "FB"
                        frame_size_ms = [10, 20][config % 2]
                    else:
                        mode = "CELT-only"
                        bandwidth = ["NB", "WB", "SWB", "FB"][config - 16] if config < 20 else "FB"
                        frame_size_ms = [2.5, 5, 10, 20][config - 16] if config < 20 else 20

                    logger.info(
                        f"Opus packet: mode={mode}, bandwidth={bandwidth}, "
                        f"frame_size={frame_size_ms}ms, stereo={s}, frames_per_packet={c + 1}"
                    )

            # Check track state
            track_open = self.audio_track and self.audio_track.is_open()
            if self._audio_callback_count <= 5:
                logger.info(
                    f"Audio track state in callback: exists={self.audio_track is not None}, "
                    f"is_open={track_open}"
                )

            if track_open:
                try:
                    data = encoded_audio.data.tobytes()

                    # Calculate duration since last packet
                    if self.last_audio_timestamp_us > 0:
                        duration_us = timestamp_us - self.last_audio_timestamp_us
                    else:
                        # First packet, assume 20ms duration
                        duration_us = 20000  # 20ms in microseconds

                    self.last_audio_timestamp_us = timestamp_us

                    # Convert duration to seconds
                    elapsed_seconds = duration_us / 1000000.0

                    # Use the built-in method to convert seconds to timestamp increment
                    elapsed_timestamp = audio_config.seconds_to_timestamp(elapsed_seconds)

                    # Update RTP timestamp
                    audio_config.timestamp = audio_config.timestamp + elapsed_timestamp

                    if self._audio_callback_count <= 10:
                        logger.info(
                            f"Duration: {duration_us}us, elapsed_timestamp: {elapsed_timestamp}, "
                            f"new RTP timestamp: {audio_config.timestamp}, "
                            f"sequence_number: {audio_config.sequence_number}"
                        )

                    # Check if we need to send RTCP SR (similar to reference implementation)
                    if self.audio_sr_reporter:
                        # Get elapsed time in clock rate from last RTCP sender report
                        report_elapsed_timestamp = (
                            audio_config.timestamp
                            - self.audio_sr_reporter.last_reported_timestamp()
                        )

                        # Check if last report was at least 1 second ago
                        if audio_config.timestamp_to_seconds(report_elapsed_timestamp) > 1:
                            self.audio_sr_reporter.set_needs_to_report()
                            if self._audio_callback_count <= 10:
                                logger.info(
                                    f"Setting RTCP SR needs to report flag (elapsed: {audio_config.timestamp_to_seconds(report_elapsed_timestamp):.2f}s)"
                                )

                    result = self.audio_track.send(data)
                    if self._audio_callback_count <= 10:
                        logger.info(f"track.send() returned: {result} (sent {len(data)} bytes)")
                    else:
                        logger.log(
                            log_level,
                            f"Sent audio data to track: {len(data)} bytes, result={result}",
                        )
                except Exception as e:
                    self._handle_error("sending encoded audio", e)
            else:
                logger.warning("Audio track not open, cannot send data")

        self.audio_encoder.set_on_encode(on_encoded)
        logger.info("Audio encoder callback set")

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
            _ = time_info  # Unused
            if status:
                logger.warning(f"Audio callback status: {status}")

            # Make a copy of the input data to avoid memory issues
            indata_copy = indata.copy()

            # Debug: log raw input data
            if not hasattr(self, "_mic_callback_count"):
                self._mic_callback_count = 0
            self._mic_callback_count += 1

            # Check if all zeros
            is_silence = np.allclose(indata_copy, 0.0)

            if self._mic_callback_count <= 10 or (self._mic_callback_count % 50 == 0):
                logger.info(
                    f"[MIC] Callback #{self._mic_callback_count}: indata shape={indata_copy.shape}, "
                    f"dtype={indata_copy.dtype}, frames={frames}"
                )
                logger.info(
                    f"[MIC] Raw input range: min={indata_copy.min():.6f}, max={indata_copy.max():.6f}, "
                    f"RMS={np.sqrt(np.mean(indata_copy**2)):.6f}, all_zeros={is_silence}"
                )
                # Log some actual samples
                if indata_copy.shape[0] > 0:
                    logger.info(f"[MIC] First 10 samples: {indata_copy[:10, 0]}")
                # Check if input is silence
                if is_silence:
                    logger.warning(
                        "[MIC] Input is all zeros! Check microphone permissions or device selection."
                    )

            # Keep original stereo data if available
            if indata_copy.shape[1] >= 2:
                # Use first two channels if more than 2 channels
                stereo_data = indata_copy[:, :2]
            elif indata_copy.shape[1] == 1:
                # Convert mono to stereo by duplicating the channel
                stereo_data = np.column_stack((indata_copy[:, 0], indata_copy[:, 0]))
            else:
                logger.error(f"Unexpected audio shape: {indata_copy.shape}")
                return

            # Ensure the data is float32 and contiguous
            stereo_data = np.ascontiguousarray(stereo_data, dtype=np.float32)

            try:
                self.audio_queue.put_nowait(stereo_data)
                # Debug: log first few frames
                if hasattr(self, "_audio_callback_count"):
                    self._audio_callback_count += 1
                else:
                    self._audio_callback_count = 1

                if self._audio_callback_count <= 5:
                    logger.info(
                        f"Audio capture callback #{self._audio_callback_count}: "
                        f"stereo_data shape={stereo_data.shape}, "
                        f"min={stereo_data.min():.6f}, max={stereo_data.max():.6f}"
                    )
            except queue.Full:
                # Drop audio if queue is full
                pass

        # Start audio stream
        try:
            # Log available audio devices
            logger.info("Available audio devices:")
            devices = sd.query_devices()
            logger.info(devices)

            # Find default input device
            default_input = sd.default.device[0]
            logger.info(f"Default input device index: {default_input}")
            if default_input is not None and default_input < len(devices):
                default_device = devices[default_input]
                logger.info(
                    f"Default input device: {default_device['name']}, channels: {default_device['max_input_channels']}"
                )

            # Check for macOS permission hint
            import platform

            if platform.system() == "Darwin":
                logger.info(
                    "Running on macOS - ensure Terminal/Python has microphone access in System Preferences > Security & Privacy > Privacy > Microphone"
                )

            # Try with different channel configurations
            input_channels = 1  # Try mono first
            if default_input is not None and default_input < len(devices):
                max_channels = devices[default_input]["max_input_channels"]
                if max_channels >= 2:
                    input_channels = 2  # Use stereo if available
                logger.info(
                    f"Using {input_channels} input channel(s) (device supports max {max_channels})"
                )

            with sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=input_channels,  # Use detected channel count
                callback=audio_callback,
                blocksize=int(self.audio_sample_rate * 0.02),  # 20ms blocks
                dtype="float32",
                device=default_input,  # Explicitly use default input device
            ):
                logger.info(
                    f"Audio capture started at {self.audio_sample_rate} Hz with device {default_input}"
                )
                while self.capture_active:
                    time.sleep(0.1)
        except Exception as e:
            self._handle_error("audio capture", e)

        logger.info("Audio capture stopped")

    def send_frames(
        self, duration: Optional[int] = None, use_camera: bool = False, use_mic: bool = False
    ):
        """Send video and audio frames from camera/mic or fake data"""
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

        # Log track states after connection
        logger.info("Track states after connection:")
        logger.info(
            f"  Video track: exists={self.video_track is not None}, "
            f"is_open={self.video_track.is_open() if self.video_track else 'N/A'}"
        )
        logger.info(
            f"  Audio track: exists={self.audio_track is not None}, "
            f"is_open={self.audio_track.is_open() if self.audio_track else 'N/A'}"
        )

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
                        self._send_fake_video_frame()
                    next_video_time += video_interval

                # Send audio frame
                if current_time >= next_audio_time:
                    # Debug: log timing
                    if hasattr(self, "_audio_send_count"):
                        self._audio_send_count += 1
                    else:
                        self._audio_send_count = 1

                    if self._audio_send_count <= 10 or self._audio_send_count % 50 == 0:
                        logger.debug(
                            f"Sending audio frame #{self._audio_send_count} at time={current_time:.3f}, next_audio_time={next_audio_time:.3f}"
                        )

                    if use_mic:
                        self._send_audio_frame()
                    else:
                        self._send_fake_audio_frame()
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

        # OpenCV's BGR format matches libyuv's RGB24 memory layout (B,G,R)
        # So we can pass BGR directly to rgb24_to_i420
        frame = bgr_frame

        # Ensure frame is the expected size
        if frame.shape[:2] != (self.video_height, self.video_width):
            frame = cv2.resize(frame, (self.video_width, self.video_height))

        # Ensure the frame is contiguous in memory
        frame = np.ascontiguousarray(frame)

        # Create I420 buffer
        buffer = VideoFrameBufferI420.create(self.video_width, self.video_height)

        # Convert BGR to I420 using libyuv
        # Note: libyuv's RGB24 = B,G,R in memory = OpenCV's BGR
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
            self.video_height,
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
            self._handle_error("encoding frame", e)

        self.video_frame_number += 1
        self._check_and_request_keyframe()

        # Debug log every 30 frames (1 second)
        if self.video_frame_number % 30 == 0:
            logger.info(f"Frame {self.video_frame_number}: total frames sent")

    def _send_audio_frame(self):
        """Send an audio frame from microphone"""
        if not self.audio_encoder:
            logger.warning("No audio encoder available")
            return

        # Track time between calls
        current_time = time.time()
        if self._last_audio_send_time is not None:
            time_diff = (current_time - self._last_audio_send_time) * 1000  # ms
            if time_diff > 25 or time_diff < 15:  # Should be ~20ms
                logger.warning(
                    f"Audio frame timing issue: {time_diff:.1f}ms between frames (expected 20ms)"
                )
        self._last_audio_send_time = current_time

        try:
            # Get audio data from queue (non-blocking)
            audio_data = self.audio_queue.get_nowait()
            queue_was_empty = False
        except queue.Empty:
            # No audio available, send silence
            queue_was_empty = True
            samples = int(self.audio_sample_rate * 0.02)  # 960 samples
            audio_data = np.zeros((samples, self.audio_channels), dtype=np.float32)
            if self.audio_timestamp_ms % 1000 <= 20:
                logger.warning(f"Audio queue empty at {self.audio_timestamp_ms}ms, sending silence")

        # Ensure audio data has the correct shape and type
        # The shape must be (samples, channels)
        if audio_data.ndim == 1:
            # If mono, reshape to (samples, 1)
            audio_data = audio_data.reshape(-1, 1)

        # Verify shape
        if audio_data.ndim != 2:
            logger.error(f"Invalid audio data shape: {audio_data.shape}, expected 2D array")
            return

        # Ensure correct dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure audio data is contiguous in memory
        audio_data = np.ascontiguousarray(audio_data)

        # For mic audio, amplify if it's too quiet
        max_val = np.abs(audio_data).max()
        rms_val = np.sqrt(np.mean(audio_data**2))

        # Log audio levels before processing
        if self.audio_timestamp_ms % 1000 <= 20:  # Log every second
            logger.info(f"[Audio Level] Before processing: max={max_val:.6f}, RMS={rms_val:.6f}")

        # Only amplify if RMS is very low (not just peak)
        if rms_val > 0 and rms_val < 0.01:  # Very quiet based on RMS
            # Gentle amplification based on RMS, not peak
            target_rms = 0.05  # Target RMS level
            gain = min(target_rms / rms_val, 5.0)  # Limit gain to 5x
            audio_data = audio_data * gain
            if self.audio_timestamp_ms <= 100 or self.audio_timestamp_ms % 1000 <= 20:
                logger.info(f"Amplifying quiet audio: RMS was {rms_val:.6f}, gain={gain:.2f}x")
        elif max_val > 0.8:  # Check for potential clipping
            logger.warning(f"Audio may be clipping: max_val={max_val:.6f}")

        # Ensure audio data is within valid range
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Debug: Log detailed info for first few frames
        if self.audio_timestamp_ms <= 100:
            logger.info(f"[Audio Debug] Frame at {self.audio_timestamp_ms}ms:")
            logger.info(f"  - Shape: {audio_data.shape}")
            logger.info(f"  - dtype: {audio_data.dtype}")
            logger.info(f"  - is C-contiguous: {audio_data.flags['C_CONTIGUOUS']}")
            logger.info(f"  - min/max: {audio_data.min():.6f} / {audio_data.max():.6f}")
            logger.info(f"  - First 10 samples (L): {audio_data[:10, 0]}")
            logger.info(f"  - First 10 samples (R): {audio_data[:10, 1]}")

        # Create audio frame
        frame = AudioFrame()
        frame.sample_rate = self.audio_sample_rate
        frame.pcm = audio_data
        frame.timestamp = self.audio_timestamp_ms / 1000.0  # seconds as float

        # Log frame details before encoding
        if (
            self.audio_timestamp_ms <= 100 or self.audio_timestamp_ms % 1000 <= 20
        ):  # Log first few frames and periodically
            timestamp_sec = (
                frame.timestamp.total_seconds()
                if hasattr(frame.timestamp, "total_seconds")
                else frame.timestamp
            )
            logger.info("[AudioFrame Debug] Created frame:")
            logger.info(f"  - sample_rate: {frame.sample_rate}")
            logger.info(f"  - timestamp: {timestamp_sec:.3f}s (ms: {self.audio_timestamp_ms})")
            logger.info(f"  - samples(): {frame.samples()}")
            logger.info(f"  - channels(): {frame.channels()}")
            logger.info(
                f"  - Expected RTP timestamp increment: {int(frame.samples())} (for {frame.sample_rate}Hz)"
            )

            # Check if PCM data is accessible from frame
            if hasattr(frame, "pcm"):
                logger.info(f"  - frame.pcm shape: {frame.pcm.shape}")
                logger.info(f"  - frame.pcm dtype: {frame.pcm.dtype}")

        # Debug log every second
        if self.audio_timestamp_ms % 1000 == 0:
            timestamp_sec = (
                frame.timestamp.total_seconds()
                if hasattr(frame.timestamp, "total_seconds")
                else frame.timestamp
            )
            logger.info(
                f"Audio progress: timestamp={timestamp_sec:.3f}s, "
                f"shape={audio_data.shape}, "
                f"min={audio_data.min():.6f}, max={audio_data.max():.6f}, "
                f"mean={audio_data.mean():.6f}, "
                f"queue_empty={queue_was_empty}, queue_size={self.audio_queue.qsize()}"
            )

            # Check RMS (Root Mean Square) for better volume indication
            rms = np.sqrt(np.mean(audio_data**2))
            source = "silence" if queue_was_empty else "mic"
            logger.info(f"Audio RMS: {rms:.6f} ({source} audio)")

            # Also log track state periodically
            if self.audio_track:
                logger.info(f"Audio track state: is_open={self.audio_track.is_open()}")

        # Encode frame
        log_level = logging.INFO if self.audio_timestamp_ms <= 200 else logging.DEBUG
        logger.log(
            log_level,
            f"[Encode] Calling audio_encoder.encode() with timestamp: {self.audio_timestamp_ms / 1000.0:.3f}s",
        )

        # Debug: Check if encoder callback gets called
        if self.audio_timestamp_ms <= 200:
            logger.info(
                f"[Pre-encode] Audio data summary: shape={audio_data.shape}, "
                f"dtype={audio_data.dtype}, min={audio_data.min():.6f}, "
                f"max={audio_data.max():.6f}, RMS={np.sqrt(np.mean(audio_data**2)):.6f}"
            )
            # Log actual PCM data samples to verify it's not silence
            logger.info(
                f"[Pre-encode] PCM samples (first 5): L={audio_data[:5, 0]}, R={audio_data[:5, 1]}"
            )

        try:
            self.audio_encoder.encode(frame)
            logger.log(log_level, "[Encode] audio_encoder.encode() completed successfully")
        except Exception as e:
            self._handle_error("in audio_encoder.encode()", e)
        self.audio_timestamp_ms += 20

    def _send_fake_video_frame(self):
        """Send a black fake video frame"""
        if not self.video_encoder:
            return

        # Create I420 frame with black buffer
        frame = VideoFrame()
        frame.format = ImageFormat.I420
        frame.i420_buffer = self._create_black_i420_buffer()
        frame.timestamp = self.video_frame_number / self.video_fps
        frame.frame_number = self.video_frame_number

        # Encode frame
        try:
            self.video_encoder.encode(frame)
        except Exception as e:
            self._handle_error("encoding frame", e)

        self.video_frame_number += 1
        self._check_and_request_keyframe()

    def _send_fake_audio_frame(self):
        """Send a silent fake audio frame"""
        if not self.audio_encoder:
            logger.warning("No audio encoder available for fake frame")
            return

        # Create audio frame
        frame = AudioFrame()
        frame.sample_rate = self.audio_sample_rate

        # Generate 20ms of audio (sine wave)
        samples = int(self.audio_sample_rate * 0.02)  # 960 samples

        # Create a gentle 440Hz sine wave (A4 note, musical and not harsh)
        t = np.arange(samples) / self.audio_sample_rate + (self.audio_timestamp_ms / 1000.0)
        frequency = 440.0  # A4 note
        audio_signal = np.sin(2 * np.pi * frequency * t) * 0.1  # 10% volume (gentle)

        # Create mono or stereo based on settings
        if self.audio_channels == 1:
            audio_data = audio_signal.reshape(-1, 1).astype(np.float32)
        else:
            audio_data = np.column_stack((audio_signal, audio_signal)).astype(np.float32)

        # Ensure audio data is contiguous in memory
        audio_data = np.ascontiguousarray(audio_data)

        # Ensure audio data is within valid range
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Debug: Log detailed info for first few frames
        if self.audio_timestamp_ms <= 100:
            logger.info(f"[Fake Audio Debug] Frame at {self.audio_timestamp_ms}ms:")
            logger.info(f"  - Shape: {audio_data.shape}")
            logger.info(f"  - dtype: {audio_data.dtype}")
            logger.info(f"  - is C-contiguous: {audio_data.flags['C_CONTIGUOUS']}")
            logger.info(f"  - min/max: {audio_data.min():.6f} / {audio_data.max():.6f}")
            logger.info(f"  - First 10 samples (L): {audio_data[:10, 0]}")
            logger.info(f"  - First 10 samples (R): {audio_data[:10, 1]}")

        frame.pcm = audio_data
        frame.timestamp = self.audio_timestamp_ms / 1000.0

        # Log frame details before encoding
        if self.audio_timestamp_ms <= 100:  # Only log first few frames
            timestamp_sec = (
                frame.timestamp.total_seconds()
                if hasattr(frame.timestamp, "total_seconds")
                else frame.timestamp
            )
            logger.info("[Fake AudioFrame Debug] Created frame:")
            logger.info(f"  - sample_rate: {frame.sample_rate}")
            logger.info(f"  - timestamp: {timestamp_sec:.3f}s")
            logger.info(f"  - samples(): {frame.samples()}")
            logger.info(f"  - channels(): {frame.channels()}")

            # Check if PCM data is accessible from frame
            if hasattr(frame, "pcm"):
                logger.info(f"  - frame.pcm shape: {frame.pcm.shape}")
                logger.info(f"  - frame.pcm dtype: {frame.pcm.dtype}")

        # Debug log every second
        if self.audio_timestamp_ms % 1000 == 0:
            timestamp_sec = (
                frame.timestamp.total_seconds()
                if hasattr(frame.timestamp, "total_seconds")
                else frame.timestamp
            )
            logger.info(
                f"Fake audio progress: timestamp={timestamp_sec:.3f}s, "
                f"shape={audio_data.shape}, "
                f"min={audio_data.min():.6f}, max={audio_data.max():.6f}, "
                f"mean={audio_data.mean():.6f}"
            )

            # Check RMS (Root Mean Square) for better volume indication
            rms = np.sqrt(np.mean(audio_data**2))
            logger.info(f"Fake audio RMS: {rms:.6f} (expected ~0.212 for 30% sine wave)")

            # Also log track state periodically
            if self.audio_track:
                logger.info(f"Audio track state (fake): is_open={self.audio_track.is_open()}")

        # Encode frame
        log_level = logging.INFO if self.audio_timestamp_ms <= 200 else logging.DEBUG
        logger.log(
            log_level,
            f"[Fake Encode] Calling audio_encoder.encode() with timestamp: {self.audio_timestamp_ms / 1000.0:.3f}s",
        )
        try:
            self.audio_encoder.encode(frame)
            logger.log(log_level, "[Fake Encode] audio_encoder.encode() completed successfully")
        except Exception as e:
            logger.error(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            self._handle_error("encoding fake audio frame", e)
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
                self._handle_error("closing PeerConnection", e)
            finally:
                self.pc = None

        # Finally release encoders
        if self.video_encoder:
            try:
                self.video_encoder.release()
            except Exception as e:
                self._handle_error("releasing video encoder", e)
            finally:
                self.video_encoder = None

        if self.audio_encoder:
            try:
                self.audio_encoder.release()
            except Exception as e:
                self._handle_error("releasing audio encoder", e)
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
    video_source = "camera" if args.camera else "fake (black video)"
    audio_source = "microphone" if args.mic else "fake (440Hz tone)"
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
        client._handle_error("", e)
    finally:
        # Always disconnect gracefully
        try:
            client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()
