import argparse
import logging
import queue
import threading
import time
from typing import Optional, List
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
import sounddevice as sd

# Error handling helper function
def handle_error(context: str, error: Exception):
    """Helper function to handle errors consistently"""
    logger.error(f"Error {context}: {error}", exc_info=True)

# Parse link header for ICE servers
def parse_link_header(link_header: str) -> list:
    """Parse Link header for ICE server information"""
    # Simple implementation - in production you'd want a more robust parser
    ice_servers = []
    # This is a placeholder - implement based on actual Link header format
    return ice_servers

from libdatachannel import (
    AV1RtpPacketizer,
    Configuration,
    Description,
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


class SimulcastLayer:
    """Represents a single simulcast layer with its own encoder and RTP configuration"""
    
    def __init__(self, rid: str, width: int, height: int, bitrate: int, fps: int, ssrc: int):
        self.rid = rid
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.fps = fps
        self.ssrc = ssrc
        
        # Encoder and RTP components
        self.encoder: Optional[VideoEncoder] = None
        self.packetizer = None
        self.sr_reporter = None
        self.pli_handler = None
        self.nack_responder = None
        self.rtp_config = None
        
        # Frame counters
        self.frame_number = 0
        self.last_key_frame_time = None
        self.key_frame_count = 0


class WHIPSimulcastClient:
    """WHIP client with simulcast support"""

    def __init__(self, whip_url: str, bearer_token: Optional[str] = None, simulcast_layers: int = 3):
        self.whip_url = whip_url
        self.bearer_token = bearer_token
        self.simulcast_layers = simulcast_layers
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Simulcast layers
        self.layers: List[SimulcastLayer] = []
        
        # Audio encoder
        self.audio_encoder: Optional[AudioEncoder] = None
        self.audio_packetizer = None
        self.audio_sr_reporter = None

        # Frame counters
        self.audio_timestamp_ms = 0

        # Video settings (base layer)
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = 30

        # Audio settings
        self.audio_sample_rate = 48000
        self.audio_channels = 2

        # Key frame interval settings
        self.key_frame_interval_seconds = 2.0

        # Camera and audio capture
        self.camera = None
        self.camera_thread = None
        self.audio_thread = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.video_queue = queue.Queue(maxsize=30)
        self.capture_active = False

        # Audio accumulation buffer for 10ms blocks
        self.audio_accumulator = []

    def _create_simulcast_layers(self):
        """Create simulcast layers with different resolutions and bitrates"""
        base_ssrc = 1234567
        
        # Define layer configurations (based on OBS implementation)
        # OBS uses progressive scaling where each layer is a fraction of the original
        layer_configs = [
            {"rid": "3", "scale": 1.0, "bitrate_ratio": 1.0},      # Highest quality (only for 4 layers)
            {"rid": "2", "scale": 0.75, "bitrate_ratio": 0.6},     # High quality
            {"rid": "1", "scale": 0.5, "bitrate_ratio": 0.3},      # Medium quality
            {"rid": "0", "scale": 0.25, "bitrate_ratio": 0.1},     # Low quality
        ]
        
        # Create layers based on the number requested (OBS style)
        # For N layers, we create layers 0 to N-1
        start_idx = len(layer_configs) - self.simulcast_layers
        for i, config in enumerate(layer_configs[start_idx:]):
            width = int(self.video_width * config["scale"])
            height = int(self.video_height * config["scale"])
            bitrate = int(2500000 * config["bitrate_ratio"])  # Base bitrate: 2.5 Mbps
            
            layer = SimulcastLayer(
                rid=config["rid"],
                width=width,
                height=height,
                bitrate=bitrate,
                fps=self.video_fps,
                ssrc=base_ssrc + i
            )
            self.layers.append(layer)
            
            logger.info(f"Created simulcast layer: rid={layer.rid}, resolution={width}x{height}, bitrate={bitrate}")

    def _setup_video_encoder_for_layer(self, layer: SimulcastLayer, track: Track):
        """Set up AV1 video encoder for a simulcast layer"""
        layer.encoder = create_aom_video_encoder()

        settings = VideoEncoder.Settings()
        settings.codec_type = VideoCodecType.AV1
        settings.width = layer.width
        settings.height = layer.height
        settings.bitrate = layer.bitrate
        settings.fps = layer.fps

        if not layer.encoder.init(settings):
            raise Exception(f"Failed to initialize video encoder for layer {layer.rid}")

        # Set up RTP packetizer with RID
        layer.rtp_config = RtpPacketizationConfig(
            ssrc=layer.ssrc, 
            cname=f"video-stream-{layer.rid}", 
            payload_type=35, 
            clock_rate=90000
        )
        
        # Set RID for this layer
        layer.rtp_config.rid = layer.rid

        # Initialize RTP timestamps
        import random
        layer.rtp_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        layer.rtp_config.timestamp = layer.rtp_config.start_timestamp
        layer.rtp_config.sequence_number = random.randint(0, 0xFFFF)

        logger.info(
            f"Video RTP config for {layer.rid}: SSRC={layer.rtp_config.ssrc}, "
            f"cname={layer.rtp_config.cname}, rid={layer.rtp_config.rid}"
        )

        layer.packetizer = AV1RtpPacketizer(
            AV1RtpPacketizer.Packetization.TemporalUnit, layer.rtp_config
        )

        # Add RTCP SR reporter
        layer.sr_reporter = RtcpSrReporter(layer.rtp_config)
        layer.packetizer.add_to_chain(layer.sr_reporter)

        # Add PLI handler
        def on_pli():
            logger.info(f"PLI received for layer {layer.rid}")

        layer.pli_handler = PliHandler(on_pli)
        layer.packetizer.add_to_chain(layer.pli_handler)

        # Add NACK responder for retransmission
        layer.nack_responder = RtcpNackResponder()
        layer.packetizer.add_to_chain(layer.nack_responder)
        
        # Set encoder callback
        def on_encoded(encoded_image):
            if track and track.is_open():
                try:
                    data = encoded_image.data.tobytes()
                    track.send(data)
                except Exception as e:
                    handle_error(f"sending encoded video for {layer.rid}", e)

        layer.encoder.set_on_encode(on_encoded)

    def connect(self):
        """Connect to WHIP server with simulcast support"""
        logger.info(f"Connecting to WHIP endpoint with {self.simulcast_layers} simulcast layers")

        # Create simulcast layers
        self._create_simulcast_layers()

        # Create peer connection
        config = Configuration()
        config.ice_servers = []

        if hasattr(config, "disable_auto_gathering"):
            config.disable_auto_gathering = True

        self.pc = PeerConnection(config)

        # Add audio track
        audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)

        # Add video track with simulcast
        video_desc = Description.Video("video", Description.Direction.SendOnly)
        video_desc.add_av1_codec(35)
        
        # Add RID entries for each simulcast layer
        for layer in self.layers:
            video_desc.add_rid(layer.rid)
            
        self.video_track = self.pc.add_track(video_desc)
        
        logger.info(f"Added video track with {len(self.layers)} simulcast layers")

        # Set up encoders
        self._setup_audio_encoder()
        
        # For simulcast, we need to handle multiple encoders
        # In this simplified version, we'll use the first layer's packetizer on the track
        if self.layers:
            self._setup_video_encoder_for_layer(self.layers[0], self.video_track)
            self.video_track.set_media_handler(self.layers[0].packetizer)
            
            # Set up other layers
            for layer in self.layers[1:]:
                self._setup_video_encoder_for_layer(layer, self.video_track)

        # Create offer
        self.pc.set_local_description()

        # Get local SDP
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise Exception("Failed to create offer")

        # Modify SDP to add simulcast
        modified_sdp = self._add_simulcast_to_sdp(str(local_sdp))
        
        logger.info("Modified SDP with simulcast:")
        logger.info(modified_sdp)

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
                content=modified_sdp,
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
                ice_servers = parse_link_header(link_header)
                if ice_servers:
                    logger.info(f"Found {len(ice_servers)} ICE server(s) in Link header")
                    if hasattr(self.pc, "gather_local_candidates"):
                        self.pc.gather_local_candidates(ice_servers)

            # Set remote SDP
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)

        logger.info("Connected to WHIP server with simulcast")

    def _add_simulcast_to_sdp(self, sdp: str) -> str:
        """Add simulcast attributes to SDP"""
        lines = sdp.split('\n')
        modified_lines = []
        in_video_section = False
        
        for line in lines:
            modified_lines.append(line)
            
            # Check if we're entering video section
            if line.startswith('m=video'):
                in_video_section = True
            elif line.startswith('m='):
                in_video_section = False
                
            # Add simulcast attributes after video section
            if in_video_section and line.startswith('a=ssrc:'):
                # Check if this is the last SSRC line
                next_line_idx = lines.index(line) + 1
                if next_line_idx >= len(lines) or not lines[next_line_idx].startswith('a=ssrc:'):
                    # Add simulcast attribute
                    rid_list = ";".join([layer.rid for layer in self.layers])
                    modified_lines.append(f"a=simulcast:send {rid_list}")
                    
                    # Add RID attributes
                    for layer in self.layers:
                        modified_lines.append(f"a=rid:{layer.rid} send")
                    
                    # Add SSRC for each layer
                    for layer in self.layers:
                        modified_lines.append(f"a=ssrc:{layer.ssrc} cname:{layer.rtp_config.cname}")
                        modified_lines.append(f"a=ssrc:{layer.ssrc} rid:{layer.rid}")
        
        return '\n'.join(modified_lines)

    def _setup_audio_encoder(self):
        """Set up Opus audio encoder"""
        logger.info("Setting up audio encoder...")
        self.audio_encoder = create_opus_audio_encoder()

        settings = AudioEncoder.Settings()
        settings.codec_type = AudioCodecType.OPUS
        settings.sample_rate = self.audio_sample_rate
        settings.channels = self.audio_channels
        settings.bitrate = 96000
        settings.frame_duration_ms = 20

        if not self.audio_encoder.init(settings):
            raise Exception("Failed to initialize audio encoder")

        # Set up RTP packetizer
        audio_config = RtpPacketizationConfig(
            ssrc=7654321, cname="audio-stream", payload_type=111, clock_rate=48000
        )

        import random
        audio_config.start_timestamp = random.randint(0, 0xFFFFFFFF)
        audio_config.timestamp = audio_config.start_timestamp
        audio_config.sequence_number = random.randint(0, 0xFFFF)

        self.audio_config = audio_config
        self.last_audio_timestamp_us = 0

        self.audio_packetizer = OpusRtpPacketizer(audio_config)

        # Add RTCP SR reporter
        self.audio_sr_reporter = RtcpSrReporter(audio_config)
        self.audio_packetizer.add_to_chain(self.audio_sr_reporter)

        if not self.audio_track:
            raise RuntimeError("Audio track not initialized")

        self.audio_track.set_media_handler(self.audio_packetizer)

        # Set encoder callback
        def on_encoded(encoded_audio):
            timestamp_us = int(encoded_audio.timestamp.total_seconds() * 1000000)
            
            if self.audio_track and self.audio_track.is_open():
                try:
                    data = encoded_audio.data.tobytes()
                    
                    if self.last_audio_timestamp_us > 0:
                        duration_us = timestamp_us - self.last_audio_timestamp_us
                    else:
                        duration_us = 20000
                    
                    self.last_audio_timestamp_us = timestamp_us
                    
                    elapsed_seconds = duration_us / 1000000.0
                    elapsed_timestamp = audio_config.seconds_to_timestamp(elapsed_seconds)
                    audio_config.timestamp = audio_config.timestamp + elapsed_timestamp
                    
                    # Check if we need to send RTCP SR
                    if self.audio_sr_reporter:
                        report_elapsed_timestamp = (
                            audio_config.timestamp
                            - self.audio_sr_reporter.last_reported_timestamp()
                        )
                        if audio_config.timestamp_to_seconds(report_elapsed_timestamp) > 1:
                            self.audio_sr_reporter.set_needs_to_report()
                    
                    self.audio_track.send(data)
                except Exception as e:
                    handle_error("sending encoded audio", e)

        self.audio_encoder.set_on_encode(on_encoded)

    def _capture_camera(self):
        """Capture video from camera"""
        logger.info("Starting camera capture...")

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
        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} fps")

        while self.capture_active:
            ret, frame = self.camera.read()
            if ret:
                try:
                    self.video_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                logger.error("Failed to read frame from camera")
                time.sleep(0.1)

        self.camera.release()
        logger.info("Camera capture stopped")

    def _capture_audio(self):
        """Capture audio from microphone"""
        logger.info("Starting audio capture...")

        def audio_callback(indata, frames, time_info, status):
            _ = time_info
            if status:
                logger.warning(f"Audio callback status: {status}")

            indata_copy = indata.copy()

            if indata_copy.shape[1] >= 2:
                stereo_data = indata_copy[:, :2]
            elif indata_copy.shape[1] == 1:
                stereo_data = np.column_stack((indata_copy[:, 0], indata_copy[:, 0]))
            else:
                logger.error(f"Unexpected audio shape: {indata_copy.shape}")
                return

            stereo_data = np.ascontiguousarray(stereo_data, dtype=np.float32)

            self.audio_accumulator.append(stereo_data)

            if len(self.audio_accumulator) >= 2:
                combined_data = np.concatenate(self.audio_accumulator[:2], axis=0)
                self.audio_accumulator = self.audio_accumulator[2:]

                try:
                    if self.audio_queue.qsize() >= 2:
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.audio_queue.put_nowait(combined_data)
                except queue.Full:
                    pass

        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            
            input_channels = 1
            if default_input is not None and default_input < len(devices):
                max_channels = devices[default_input]["max_input_channels"]
                if max_channels >= 2:
                    input_channels = 2

            with sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=input_channels,
                callback=audio_callback,
                blocksize=int(self.audio_sample_rate * 0.01),
                dtype="float32",
                device=default_input,
                latency="low",
            ):
                logger.info(f"Audio capture started at {self.audio_sample_rate} Hz")
                while self.capture_active:
                    time.sleep(0.1)
        except Exception as e:
            handle_error("audio capture", e)

        logger.info("Audio capture stopped")

    def send_frames(self, duration: Optional[int] = None, use_camera: bool = False, use_mic: bool = False):
        """Send video and audio frames"""
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

            if use_mic:
                self.audio_thread = threading.Thread(target=self._capture_audio)
                self.audio_thread.start()

            time.sleep(1.0)

        # Frame intervals
        video_interval = 1.0 / self.video_fps
        audio_interval = 0.02

        start_time = time.time()
        next_video_time = start_time
        next_audio_time = start_time

        try:
            while True:
                current_time = time.time()

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
                    if use_mic:
                        self._send_audio_frame()
                    else:
                        self._send_fake_audio_frame()
                    next_audio_time += audio_interval

                # Sleep until next frame
                next_time = min(next_video_time, next_audio_time)
                sleep_time = max(0, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.010))
        finally:
            if use_camera or use_mic:
                self.capture_active = False
                if use_camera and self.camera_thread:
                    self.camera_thread.join(timeout=2.0)
                if use_mic and self.audio_thread:
                    self.audio_thread.join(timeout=2.0)

    def _send_video_frame(self):
        """Send a video frame from camera to all simulcast layers"""
        try:
            bgr_frame = self.video_queue.get_nowait()
        except queue.Empty:
            return

        # Process for each simulcast layer
        for layer in self.layers:
            if not layer.encoder:
                continue
                
            # Resize frame for this layer if needed
            if layer.width != self.video_width or layer.height != self.video_height:
                resized_frame = cv2.resize(bgr_frame, (layer.width, layer.height))
            else:
                resized_frame = bgr_frame
                
            # Convert to I420
            frame = np.ascontiguousarray(resized_frame)
            buffer = VideoFrameBufferI420.create(layer.width, layer.height)
            
            rgb24_to_i420(
                frame,
                layer.width * 3,
                buffer.y,
                buffer.u,
                buffer.v,
                buffer.stride_y(),
                buffer.stride_u(),
                buffer.stride_v(),
                layer.width,
                layer.height,
            )
            
            # Create video frame
            video_frame = VideoFrame()
            video_frame.format = ImageFormat.I420
            video_frame.i420_buffer = buffer
            video_frame.timestamp = layer.frame_number / layer.fps
            video_frame.frame_number = layer.frame_number
            
            # Check for keyframe
            self._check_and_request_keyframe_for_layer(layer)
            
            # Encode frame
            try:
                layer.encoder.encode(video_frame)
            except Exception as e:
                handle_error(f"encoding frame for {layer.rid}", e)
            
            layer.frame_number += 1

    def _send_fake_video_frame(self):
        """Send fake video frames to all simulcast layers"""
        for layer in self.layers:
            if not layer.encoder:
                continue
                
            # Create pattern buffer for this layer
            buffer = self._create_pattern_i420_buffer_for_layer(layer)
            
            # Create video frame
            video_frame = VideoFrame()
            video_frame.format = ImageFormat.I420
            video_frame.i420_buffer = buffer
            video_frame.timestamp = layer.frame_number / layer.fps
            video_frame.frame_number = layer.frame_number
            
            # Check for keyframe
            self._check_and_request_keyframe_for_layer(layer)
            
            # Encode frame
            try:
                layer.encoder.encode(video_frame)
            except Exception as e:
                handle_error(f"encoding fake frame for {layer.rid}", e)
            
            layer.frame_number += 1

    def _create_pattern_i420_buffer_for_layer(self, layer: SimulcastLayer) -> VideoFrameBufferI420:
        """Create a patterned I420 video buffer for a specific layer"""
        buffer = VideoFrameBufferI420.create(layer.width, layer.height)
        
        import random
        random.seed(layer.key_frame_count)
        
        # Different patterns for different layers
        if layer.rid == "h":
            # High quality - full gradient
            direction = random.choice(["horizontal", "vertical", "diagonal"])
        elif layer.rid == "m":
            # Medium quality - simpler gradient
            direction = random.choice(["horizontal", "vertical"])
        else:
            # Low quality - solid color
            direction = "solid"
        
        # Random colors
        r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        r2, g2, b2 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        
        # Convert to YUV
        y1 = int(0.299 * r1 + 0.587 * g1 + 0.114 * b1)
        u1 = int(-0.169 * r1 - 0.331 * g1 + 0.5 * b1 + 128)
        v1 = int(0.5 * r1 - 0.419 * g1 - 0.081 * b1 + 128)
        
        y2 = int(0.299 * r2 + 0.587 * g2 + 0.114 * b2)
        u2 = int(-0.169 * r2 - 0.331 * g2 + 0.5 * b2 + 128)
        v2 = int(0.5 * r2 - 0.419 * g2 - 0.081 * b2 + 128)
        
        # Clamp values
        y1, y2 = np.clip([y1, y2], 16, 235)
        u1, u2 = np.clip([u1, u2], 16, 240)
        v1, v2 = np.clip([v1, v2], 16, 240)
        
        if direction == "horizontal":
            y_gradient = np.linspace(y1, y2, layer.width, dtype=np.uint8)
            buffer.y[:layer.height, :layer.width] = y_gradient
            
            u_gradient = np.linspace(u1, u2, layer.width // 2, dtype=np.uint8)
            v_gradient = np.linspace(v1, v2, layer.width // 2, dtype=np.uint8)
            buffer.u[:layer.height // 2, :layer.width // 2] = u_gradient
            buffer.v[:layer.height // 2, :layer.width // 2] = v_gradient
            
        elif direction == "vertical":
            y_gradient = np.linspace(y1, y2, layer.height, dtype=np.uint8).reshape(-1, 1)
            buffer.y[:layer.height, :layer.width] = y_gradient
            
            u_gradient = np.linspace(u1, u2, layer.height // 2, dtype=np.uint8).reshape(-1, 1)
            v_gradient = np.linspace(v1, v2, layer.height // 2, dtype=np.uint8).reshape(-1, 1)
            buffer.u[:layer.height // 2, :layer.width // 2] = u_gradient
            buffer.v[:layer.height // 2, :layer.width // 2] = v_gradient
            
        elif direction == "diagonal":
            x_grad = np.linspace(0, 1, layer.width)
            y_grad = np.linspace(0, 1, layer.height).reshape(-1, 1)
            diagonal = (x_grad + y_grad) / 2
            buffer.y[:layer.height, :layer.width] = (y1 + (y2 - y1) * diagonal).astype(np.uint8)
            
            x_grad_half = np.linspace(0, 1, layer.width // 2)
            y_grad_half = np.linspace(0, 1, layer.height // 2).reshape(-1, 1)
            diagonal_half = (x_grad_half + y_grad_half) / 2
            buffer.u[:layer.height // 2, :layer.width // 2] = (u1 + (u2 - u1) * diagonal_half).astype(np.uint8)
            buffer.v[:layer.height // 2, :layer.width // 2] = (v1 + (v2 - v1) * diagonal_half).astype(np.uint8)
        else:  # solid
            buffer.y[:layer.height, :layer.width] = y1
            buffer.u[:layer.height // 2, :layer.width // 2] = u1
            buffer.v[:layer.height // 2, :layer.width // 2] = v1
        
        # Fill padding if needed
        if buffer.stride_y() > layer.width:
            buffer.y[:, layer.width:buffer.stride_y()] = 16
        if buffer.stride_u() > layer.width // 2:
            buffer.u[:, layer.width // 2:buffer.stride_u()] = 128
        if buffer.stride_v() > layer.width // 2:
            buffer.v[:, layer.width // 2:buffer.stride_v()] = 128
            
        return buffer

    def _check_and_request_keyframe_for_layer(self, layer: SimulcastLayer):
        """Check if it's time to request a key frame for a layer"""
        current_time = time.time()
        if layer.last_key_frame_time is None:
            layer.last_key_frame_time = current_time
            
        time_since_last_key = current_time - layer.last_key_frame_time
        if time_since_last_key >= self.key_frame_interval_seconds:
            logger.debug(f"Requesting key frame for {layer.rid}")
            try:
                if layer.encoder:
                    layer.encoder.force_intra_next_frame()
                    layer.key_frame_count += 1
            except Exception as e:
                handle_error(f"requesting keyframe for {layer.rid}", e)
            layer.last_key_frame_time = current_time

    def _send_audio_frame(self):
        """Send an audio frame from microphone"""
        if not self.audio_encoder:
            return

        try:
            audio_data = self.audio_queue.get_nowait()
        except queue.Empty:
            # Send silence
            samples = int(self.audio_sample_rate * 0.02)
            audio_data = np.zeros((samples, self.audio_channels), dtype=np.float32)

        # Process audio data
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        audio_data = np.ascontiguousarray(audio_data)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Create audio frame
        frame = AudioFrame()
        frame.sample_rate = self.audio_sample_rate
        frame.pcm = audio_data
        frame.timestamp = self.audio_timestamp_ms / 1000.0

        # Encode frame
        try:
            self.audio_encoder.encode(frame)
        except Exception as e:
            handle_error("encoding audio frame", e)
        
        self.audio_timestamp_ms += 20

    def _send_fake_audio_frame(self):
        """Send a fake audio frame"""
        if not self.audio_encoder:
            return

        # Create audio frame
        frame = AudioFrame()
        frame.sample_rate = self.audio_sample_rate

        # Generate 20ms of 440Hz sine wave
        samples = int(self.audio_sample_rate * 0.02)
        t = np.arange(samples) / self.audio_sample_rate + (self.audio_timestamp_ms / 1000.0)
        frequency = 440.0
        audio_signal = np.sin(2 * np.pi * frequency * t) * 0.1

        if self.audio_channels == 1:
            audio_data = audio_signal.reshape(-1, 1).astype(np.float32)
        else:
            audio_data = np.column_stack((audio_signal, audio_signal)).astype(np.float32)

        audio_data = np.ascontiguousarray(audio_data)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        frame.pcm = audio_data
        frame.timestamp = self.audio_timestamp_ms / 1000.0

        # Encode frame
        try:
            self.audio_encoder.encode(frame)
        except Exception as e:
            handle_error("encoding fake audio frame", e)
        
        self.audio_timestamp_ms += 20

    def disconnect(self):
        """Disconnect from WHIP server"""
        logger.info("Starting graceful shutdown...")

        # Send DELETE request
        if self.session_url:
            try:
                with httpx.Client(timeout=5.0) as client:
                    headers = {}
                    if self.bearer_token:
                        headers["Authorization"] = f"Bearer {self.bearer_token}"

                    response = client.delete(self.session_url, headers=headers)
                    if response.status_code in [200, 204]:
                        logger.info("WHIP session terminated successfully")
            except Exception as e:
                logger.error(f"Error during DELETE: {e}")

        time.sleep(0.5)

        # Stop capture
        self.capture_active = False

        # Release camera
        if self.camera and self.camera.isOpened():
            self.camera.release()

        # Clean up resources
        for layer in self.layers:
            layer.packetizer = None
            layer.sr_reporter = None
            layer.pli_handler = None
            layer.nack_responder = None
            if layer.encoder:
                try:
                    layer.encoder.release()
                except Exception as e:
                    handle_error(f"releasing encoder for {layer.rid}", e)
                layer.encoder = None

        self.audio_packetizer = None
        self.audio_sr_reporter = None

        self.video_track = None
        self.audio_track = None

        if self.pc:
            try:
                self.pc.close()
            except Exception as e:
                handle_error("closing PeerConnection", e)
            finally:
                self.pc = None

        if self.audio_encoder:
            try:
                self.audio_encoder.release()
            except Exception as e:
                handle_error("releasing audio encoder", e)
            finally:
                self.audio_encoder = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="WHIP client with simulcast support")
    parser.add_argument("--url", required=True, help="WHIP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--camera", action="store_true", help="Use camera for video capture")
    parser.add_argument("--mic", action="store_true", help="Use microphone for audio capture")
    parser.add_argument("--layers", type=int, default=3, choices=[1, 2, 3, 4], help="Number of simulcast layers (1-4)")

    args = parser.parse_args()

    video_source = "camera" if args.camera else "fake (pattern video)"
    audio_source = "microphone" if args.mic else "fake (440Hz tone)"
    logger.info(f"Video source: {video_source}")
    logger.info(f"Audio source: {audio_source}")
    logger.info(f"Simulcast layers: {args.layers}")

    client = WHIPSimulcastClient(args.url, args.token, simulcast_layers=args.layers)

    try:
        client.connect()
        client.send_frames(args.duration, use_camera=args.camera, use_mic=args.mic)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    main()