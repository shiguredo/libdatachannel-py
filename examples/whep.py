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
    Configuration,
    Description,
    IceServer,
    OpusRtpDepacketizer,
    PeerConnection,
    RtcpReceivingSession,
    RtpDepacketizer,
    Track,
)
from libdatachannel.codec import (
    AudioCodecType,
    AudioDecoder,
    AudioFrame,
    EncodedAudio,
    EncodedImage,
    ImageFormat,
    VideoCodecType,
    VideoDecoder,
    VideoFrame,
    create_openh264_video_decoder,
    create_opus_audio_decoder,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WHEPClient:
    """Minimal WHEP client for receiving and playing back video and audio"""

    def __init__(self, whep_url: str, bearer_token: Optional[str] = None, no_video: bool = False, timeout: Optional[int] = None):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.no_video = no_video
        self.timeout = timeout
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Decoders
        self.video_decoder: Optional[VideoDecoder] = None
        self.audio_decoder: Optional[AudioDecoder] = None

        # Playback queues
        self.video_queue = queue.Queue(maxsize=60)  # Increased for better buffering
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Playback threads
        self.video_thread = None
        self.audio_thread = None
        self.playback_active = False

        # Audio playback settings
        self.audio_stream = None
        self.audio_sample_rate = 48000
        self.audio_channels = 2
        self._audio_message_count = 0

        # Window name for OpenCV
        self.window_name = "WHEP Player"

        # Statistics
        self.video_frames_received = 0
        self.audio_frames_received = 0
        self.last_stats_time = time.time()
        
        # H264 NAL unit buffer
        self._h264_nal_buffer = []

    def _handle_error(self, context: str, error: Exception):
        """Unified error handling"""
        logger.error(f"Error {context}: {error}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()

    def _parse_link_header(self, link_header: str) -> List[IceServer]:
        """Parse Link header for ICE servers"""
        ice_servers = []
        if not link_header:
            return ice_servers

        # Parse Link header similar to WHIP client
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
        """Connect to WHEP server"""
        logger.info(f"Connecting to WHEP endpoint: {self.whep_url}")

        # Create peer connection
        config = Configuration()
        config.ice_servers = []
        
        # Set max message size if needed
        if hasattr(config, "max_message_size"):
            config.max_message_size = 16384

        # Try to disable auto gathering if available
        if hasattr(config, "disable_auto_gathering"):
            config.disable_auto_gathering = True

        self.pc = PeerConnection(config)

        # Add receive-only tracks
        # Add audio track FIRST (before video)
        audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)
        logger.info("Added audio track with Opus codec (PT=111)")

        # Add video track SECOND (after audio)
        video_desc = Description.Video("video", Description.Direction.RecvOnly)
        video_desc.add_h264_codec(96)
        self.video_track = self.pc.add_track(video_desc)
        logger.info("Added video track with H264 codec (PT=96)")

        # Create offer
        self.pc.set_local_description()

        # Get local SDP
        local_sdp = self.pc.local_description()
        if not local_sdp:
            raise Exception("Failed to create offer")

        # Log SDP for debugging
        sdp_lines = str(local_sdp).split("\n")
        media_lines = [line.strip() for line in sdp_lines if line.startswith("m=")]
        logger.info(f"SDP media sections order: {media_lines}")
        
        # Log full SDP offer for debugging
        logger.debug("SDP Offer:")
        logger.debug(str(local_sdp))

        # Send offer to WHEP server
        logger.info("Sending offer to WHEP server...")
        with httpx.Client(timeout=10.0) as client:
            headers = {
                "Content-Type": "application/sdp",
            }
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"

            response = client.post(
                self.whep_url,
                content=str(local_sdp),
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 201:
                logger.error(f"WHEP server error response headers: {dict(response.headers)}")
                error_body = response.text if response.text else "(empty response body)"
                logger.error(f"WHEP server error response body: {error_body}")
                
                # Check if this might be a WHIP endpoint instead of WHEP
                if "whip" in self.whep_url.lower() and "whep" not in self.whep_url.lower():
                    logger.warning("URL contains 'whip' but not 'whep' - are you using the correct endpoint?")
                
                raise Exception(f"WHEP server returned {response.status_code}: {error_body}")

            # Get session URL
            self.session_url = response.headers.get("Location")
            if self.session_url and not self.session_url.startswith("http"):
                self.session_url = urljoin(self.whep_url, self.session_url)

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
            
            # Log answer SDP for debugging
            logger.debug("SDP Answer:")
            logger.debug(response.text)

        logger.info("Connected to WHEP server")

    def _setup_video_decoder(self):
        """Set up H264 video decoder"""
        if not self.video_track:
            logger.warning("Video track is None in _setup_video_decoder")
            return
            
        # Create H264 decoder
        import os
        openh264_path = os.environ.get("OPENH264_PATH")
        if not openh264_path:
            raise Exception("OPENH264_PATH environment variable not set")
        self.video_decoder = create_openh264_video_decoder(openh264_path)
        
        # Initialize decoder with settings
        settings = VideoDecoder.Settings()
        settings.codec_type = VideoCodecType.H264
        # Note: width and height are optional for H264 decoder
        # They will be determined from SPS

        if not self.video_decoder.init(settings):
            raise Exception("Failed to initialize video decoder")
            
        logger.info("H264 video decoder initialized successfully")
        
        # Counter for decoded frames
        self._decoded_frame_count = 0
        
        # Set decoder callback AFTER initialization
        def on_decoded(decoded_frame):
            try:
                self._decoded_frame_count += 1
                logger.info(f"🎥 Video frame decoded #{self._decoded_frame_count}: {decoded_frame.width()}x{decoded_frame.height()}, format: {decoded_frame.format}")
                
                # Put frame in queue
                try:
                    self.video_queue.put_nowait(decoded_frame)
                except queue.Full:
                    logger.warning(f"Video queue full, dropping frame #{self.video_frames_received}")
            except Exception as e:
                logger.error(f"Error in on_decoded callback: {e}")
                import traceback
                traceback.print_exc()

        def on_error(error_msg):
            logger.error(f"Video decoder error: {error_msg}")

        self.video_decoder.set_on_decode(on_decoded)
        if hasattr(self.video_decoder, 'set_on_error'):
            self.video_decoder.set_on_error(on_error)
        logger.info("Video decoder callback set")

        # Initialize counter
        self._video_message_count = 0
        self._h264_timestamp = 0
        
        # Set track message handler for H264
        def on_video_message(data):
            try:
                self._video_message_count += 1
                
                # Always check if this is RTP data
                if len(data) >= 12 and (data[0] >> 6) == 2:  # RTP version 2
                    # Extract RTP header info
                    pt = data[1] & 0x7F
                    seq = (data[2] << 8) | data[3]
                    timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                    
                    # RTP header is typically 12 bytes (without extensions)
                    header_size = 12
                    
                    # Check for header extension
                    if data[0] & 0x10:
                        if len(data) >= header_size + 4:
                            ext_length = ((data[header_size + 2] << 8) | data[header_size + 3]) * 4
                            header_size += 4 + ext_length
                    
                    # Extract payload
                    if header_size < len(data):
                        # Log only first 20 messages for debugging
                        if self._video_message_count <= 20:
                            logger.info(f"Video message #{self._video_message_count}: {len(data)} bytes")
                            preview = ' '.join(f'{b:02x}' for b in data[:min(30, len(data))])
                            logger.info(f"Data preview: {preview}")
                            logger.info("Detected RTP packet, manually extracting payload")
                            logger.info(f"Extracted payload: {len(data) - header_size} bytes from RTP packet (PT={pt}, seq={seq})")
                        
                        # Always extract the payload
                        data = data[header_size:]
                    else:
                        logger.warning("Invalid RTP packet - header size exceeds packet size")
                        return
                else:
                    # Not RTP data - log for debugging if within first 20 messages
                    if self._video_message_count <= 20:
                        logger.info(f"Video message #{self._video_message_count}: {len(data)} bytes (not RTP)")
                        preview = ' '.join(f'{b:02x}' for b in data[:min(30, len(data))])
                        logger.info(f"Data preview: {preview}")
                
                # Ensure we have valid data after RTP extraction
                if len(data) == 0:
                    logger.warning(f"Empty payload after RTP extraction in message #{self._video_message_count}")
                    return
                
                # The data should be H264 NAL units after RTP extraction
                if self.video_decoder and len(data) > 0:
                    try:
                        # Process H264 RTP payload according to RFC 6184
                        nal_type = None
                        if len(data) > 0:
                            nal_type = data[0] & 0x1F
                            
                        if self._video_message_count <= 20:
                            logger.info(f"NAL type: {nal_type} ({self._get_nal_type_name(nal_type) if nal_type else 'Unknown'})")
                        
                        # Process based on NAL type
                        if nal_type >= 1 and nal_type <= 23:  # Single NAL unit
                            # Add start code prefix to NAL unit
                            nal_with_start_code = bytearray([0x00, 0x00, 0x00, 0x01])
                            nal_with_start_code.extend(data)
                            self._h264_nal_buffer.append(nal_with_start_code)
                            
                            if self._video_message_count <= 20 and nal_type > 15:
                                logger.warning(f"Unusual NAL type {nal_type} in single NAL unit range")
                            
                        elif nal_type == 24:  # STAP-A (Single Time Aggregation Packet)
                            # Skip STAP-A header
                            i = 1
                            while i < len(data) - 2:
                                # Read NAL size (2 bytes)
                                nal_size = (data[i] << 8) | data[i + 1]
                                i += 2
                                
                                if i + nal_size <= len(data):
                                    # Extract NAL unit
                                    nal_unit = data[i:i + nal_size]
                                    if len(nal_unit) > 0:
                                        # Add start code and NAL to buffer
                                        nal_with_start_code = bytearray([0x00, 0x00, 0x00, 0x01])
                                        nal_with_start_code.extend(nal_unit)
                                        self._h264_nal_buffer.append(nal_with_start_code)
                                        
                                        if self._video_message_count <= 20:
                                            sub_nal_type = nal_unit[0] & 0x1F if len(nal_unit) > 0 else None
                                            logger.info(f"  STAP-A contains NAL type {sub_nal_type}: {self._get_nal_type_name(sub_nal_type)}")
                                    i += nal_size
                                else:
                                    break
                                    
                        elif nal_type == 28:  # FU-A (Fragmentation Unit)
                            if len(data) < 2:
                                return
                                
                            # FU header
                            fu_header = data[1]
                            start_bit = (fu_header >> 7) & 0x01
                            end_bit = (fu_header >> 6) & 0x01
                            nal_unit_type = fu_header & 0x1F
                            
                            if start_bit:
                                # First fragment - create NAL header
                                nal_header = (data[0] & 0xE0) | nal_unit_type
                                self._fu_buffer = bytearray([0x00, 0x00, 0x00, 0x01, nal_header])
                                self._fu_buffer.extend(data[2:])  # Skip FU indicator and header
                                if self._video_message_count <= 20:
                                    logger.info(f"  FU-A start for NAL type {nal_unit_type}: {self._get_nal_type_name(nal_unit_type)}")
                            elif hasattr(self, '_fu_buffer'):
                                # Continuation fragment
                                self._fu_buffer.extend(data[2:])  # Skip FU indicator and header
                                
                                if end_bit:
                                    # Last fragment - add complete NAL to buffer
                                    self._h264_nal_buffer.append(self._fu_buffer)
                                    if self._video_message_count <= 20:
                                        logger.info(f"  FU-A complete, total size: {len(self._fu_buffer)} bytes")
                                    del self._fu_buffer
                        else:
                            # Other NAL types (including reserved types) - treat as single NAL unit
                            if self._video_message_count <= 20:
                                logger.info(f"NAL type: {nal_type} (Reserved or other type)")
                            # Add start code prefix to NAL unit
                            nal_with_start_code = bytearray([0x00, 0x00, 0x00, 0x01])
                            nal_with_start_code.extend(data)
                            self._h264_nal_buffer.append(nal_with_start_code)
                        
                        # Check if we should decode
                        should_decode = False
                        
                        # Decode on IDR frames (NAL type 5) or access unit delimiter (type 9)
                        if nal_type == 5:  # IDR frame
                            should_decode = True
                            logger.info(f"IDR frame detected at message #{self._video_message_count}, decoding immediately")
                        elif nal_type == 9:  # Access unit delimiter - marks new frame
                            # Decode previous frame if exists
                            if len(self._h264_nal_buffer) > 1:  # More than just the delimiter
                                should_decode = True
                                if self._video_message_count <= 20:
                                    logger.info("Access unit delimiter detected, decoding previous frame")
                        elif nal_type in [7, 8]:  # SPS/PPS
                            if self._video_message_count <= 20:
                                logger.info(f"Received {self._get_nal_type_name(nal_type)}")
                        elif nal_type == 1:  # Non-IDR slice - also decode after accumulating
                            if len(self._h264_nal_buffer) >= 3:  # At least a few NALs
                                should_decode = True
                                if self._video_message_count <= 20:
                                    logger.info(f"Non-IDR slice with {len(self._h264_nal_buffer)} NALs, decoding")
                        elif len(self._h264_nal_buffer) >= 5:  # Multiple NALs accumulated
                            should_decode = True
                            if self._video_message_count <= 20:
                                logger.info(f"Accumulated {len(self._h264_nal_buffer)} NALs, decoding")
                        
                        if should_decode and len(self._h264_nal_buffer) > 0:
                            # For access unit delimiter, exclude it from current decode
                            nals_to_decode = self._h264_nal_buffer[:-1] if nal_type == 9 else self._h264_nal_buffer
                            
                            if len(nals_to_decode) > 0:
                                # Combine all NAL units
                                combined_data = bytearray()
                                for nal in nals_to_decode:
                                    combined_data.extend(nal)
                                
                                # Create EncodedImage
                                encoded_image = EncodedImage()
                                np_data = np.array(combined_data, dtype=np.uint8)
                                encoded_image.data = np_data
                                
                                # Set timestamp
                                from datetime import timedelta
                                encoded_image.timestamp = timedelta(milliseconds=self._h264_timestamp)
                                
                                # Increment frame counter before logging
                                self.video_frames_received += 1
                                
                                if self.video_frames_received <= 10 or self.video_frames_received % 100 == 0:
                                    logger.info(f"Decoding H264 frame #{self.video_frames_received}: {len(combined_data)} bytes, {len(nals_to_decode)} NALs, timestamp={self._h264_timestamp}ms")
                                    # Log NAL types in this frame
                                    nal_types_in_frame = []
                                    for nal in nals_to_decode:
                                        if len(nal) > 4:
                                            # Check if this looks like a NAL with start code
                                            if nal[0] == 0x00 and nal[1] == 0x00 and nal[2] == 0x00 and nal[3] == 0x01:
                                                nal_type = nal[4] & 0x1F  # After start code
                                                nal_types_in_frame.append(nal_type)
                                            else:
                                                # This might be raw RTP data - log warning
                                                logger.warning(f"NAL buffer contains non-NAL data: {' '.join(f'{b:02x}' for b in nal[:10])}")
                                    logger.info(f"  NAL types in frame: {[self._get_nal_type_name(t) for t in nal_types_in_frame]}")
                                    
                                    # Debug first bytes
                                    if len(combined_data) > 10 and self.video_frames_received <= 10:
                                        preview = ' '.join(f'{b:02x}' for b in combined_data[:20])
                                        logger.info(f"  Frame data preview: {preview}")
                                
                                # Decode
                                try:
                                    result = self.video_decoder.decode(encoded_image)
                                    if self.video_frames_received <= 5 or self.video_frames_received % 100 == 0:
                                        logger.info(f"Decode called for frame #{self.video_frames_received}, result: {result}")
                                    
                                    # Flush decoder periodically, not every frame
                                    if hasattr(self.video_decoder, 'flush') and self.video_frames_received % 30 == 0:
                                        self.video_decoder.flush()
                                        logger.info(f"Flushed decoder at frame {self.video_frames_received}")
                                        
                                    # Check if decoder is still valid
                                    if self.video_frames_received % 100 == 0:
                                        logger.info(f"Decoder status check: decoder={self.video_decoder is not None}, callback set={hasattr(self.video_decoder, '_on_decoded')}")
                                except Exception as e:
                                    logger.error(f"Failed to decode frame #{self.video_frames_received}: {e}")
                                    import traceback
                                    traceback.print_exc()
                                
                                # Update timestamp
                                self._h264_timestamp += 33  # ~30fps
                                
                                # Clear buffer after decoding
                                self._h264_nal_buffer = []
                                
                                if self._video_message_count <= 20:
                                    logger.info(f"Buffer cleared after frame #{self.video_frames_received}, remaining NALs: {len(self._h264_nal_buffer)}")
                            
                    except Exception as e:
                        logger.error(f"Error processing H264 data: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                self._handle_error("processing video message", e)
        
        def _get_nal_type_name(nal_type):
            """Get human-readable name for NAL unit type"""
            nal_names = {
                1: "Non-IDR slice",
                2: "Slice data partition A",
                3: "Slice data partition B",
                4: "Slice data partition C",
                5: "IDR slice",
                6: "SEI",
                7: "SPS",
                8: "PPS",
                9: "Access unit delimiter",
                10: "End of sequence",
                11: "End of stream",
                12: "Filler data",
                13: "SPS extension",
                14: "Prefix NAL unit",
                15: "Subset SPS",
                16: "Reserved",
                17: "Reserved",
                18: "Reserved",
                19: "Auxiliary slice",
                20: "Slice extension",
                21: "Slice extension for depth view",
                22: "Reserved",
                23: "Reserved",
                24: "STAP-A",
                25: "STAP-B",
                26: "MTAP16",
                27: "MTAP24",
                28: "FU-A",
                29: "FU-B",
                30: "Unspecified",
                31: "Unspecified",
            }
            return nal_names.get(nal_type, f"Type {nal_type}")
        
        self._get_nal_type_name = _get_nal_type_name

        if self.video_track:
            # Set up RTCP receiving session
            rtcp_session = RtcpReceivingSession()
            self.video_track.set_media_handler(rtcp_session)
            
            # NOTE: Manual RTP depacketization since H264RtpDepacketizer doesn't seem to work as expected
            # H264RtpDepacketizer returns raw RTP packets instead of depacketized H264 NAL units
            
            # Set message handler on track
            self.video_track.on_message(on_video_message)
            
            logger.info(f"Video track setup: is_open={self.video_track.is_open()}, with RTCP session (H264 depacketizer disabled)")
        else:
            logger.warning("Video track is None!")

        logger.info("Video decoder setup complete")

    def _setup_audio_decoder(self):
        """Set up Opus audio decoder"""
        self.audio_decoder = create_opus_audio_decoder()

        settings = AudioDecoder.Settings()
        settings.codec_type = AudioCodecType.OPUS
        settings.sample_rate = self.audio_sample_rate
        settings.channels = self.audio_channels

        if not self.audio_decoder.init(settings):
            raise Exception("Failed to initialize audio decoder")

        # Set decoder callback
        def on_decoded(decoded_frame):
            try:
                # Normalize audio to -1.0 to 1.0 range
                pcm_data = decoded_frame.pcm
                pcm_max = np.abs(pcm_data).max()
                if pcm_max > 1.0:
                    # Normalize to -1.0 to 1.0 range
                    decoded_frame.pcm = pcm_data / 32768.0  # Assuming 16-bit range
                
                self.audio_queue.put_nowait(decoded_frame)
                self.audio_frames_received += 1
                
                if self.audio_frames_received <= 5 or self.audio_frames_received % 100 == 0:
                    normalized_data = decoded_frame.pcm
                    pcm_stats = f"min={normalized_data.min():.6f}, max={normalized_data.max():.6f}, mean={normalized_data.mean():.6f}"
                    logger.info(f"Audio frame decoded #{self.audio_frames_received}: {decoded_frame.samples()} samples, {decoded_frame.channels()} channels, {pcm_stats}")
            except queue.Full:
                # Drop frame if queue is full
                pass

        self.audio_decoder.set_on_decode(on_decoded)

        # Set track message handler
        def on_audio_message(data):
            try:
                self._audio_message_count += 1
                if self._audio_message_count <= 10 or self._audio_message_count % 100 == 0:
                    logger.info(f"Audio message #{self._audio_message_count}: {len(data)} bytes")
                    
                # Create EncodedAudio from data
                if self.audio_decoder:
                    encoded_audio = EncodedAudio()
                    # Convert bytes to numpy array and copy
                    np_data = np.frombuffer(data, dtype=np.uint8).copy()
                    encoded_audio.data = np_data
                    # Set timestamp
                    from datetime import timedelta
                    # Audio timestamp (20ms per frame for Opus)
                    timestamp_ms = self._audio_message_count * 20
                    encoded_audio.timestamp = timedelta(milliseconds=timestamp_ms)
                    self.audio_decoder.decode(encoded_audio)
            except Exception as e:
                self._handle_error("decoding audio", e)

        if self.audio_track:
            # Set up RTCP receiving session first
            rtcp_session = RtcpReceivingSession()
            self.audio_track.set_media_handler(rtcp_session)
            
            # Then chain RTP depacketizer for Opus
            opus_depacketizer = OpusRtpDepacketizer()
            self.audio_track.chain_media_handler(opus_depacketizer)
            
            # Now set message handler on track
            self.audio_track.on_message(on_audio_message)
            
            logger.info("Audio track setup with RTCP session and RTP depacketizer")

        logger.info("Audio decoder setup complete")

    def _play_video(self):
        """Play video frames using OpenCV"""
        logger.info("Starting video playback thread...")
        
        # Wait for main thread to be ready (macOS requirement)
        time.sleep(0.5)
        
        # Try different window creation methods for macOS compatibility
        window_created = False
        try:
            # Try simple window first (more compatible with macOS)
            cv2.namedWindow(self.window_name)
            window_created = True
            logger.info("OpenCV window created with default settings")
            
            # Show a black frame initially
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Waiting for video...", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(self.window_name, black_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            logger.error(f"Failed to create OpenCV window: {e}")
            logger.info("Continuing without display - frames are being decoded successfully")
            # Continue without display to show decoding is working
            while self.playback_active:
                try:
                    frame = self.video_queue.get(timeout=0.1)
                    logger.info(f"Decoded frame received but cannot display: {frame.width()}x{frame.height()}")
                except queue.Empty:
                    pass
            return
        
        frame_count = 0

        while self.playback_active:
            try:
                # Get frame from queue with timeout
                frame = self.video_queue.get(timeout=0.1)
                frame_count += 1
                
                # Resize window on first frame
                if frame_count == 1:
                    logger.info(f"First video frame received! Frame size: {frame.width()}x{frame.height()}")
                    try:
                        cv2.resizeWindow(self.window_name, frame.width(), frame.height())
                    except cv2.error:
                        pass  # Ignore resize errors

                # Convert frame to BGR for OpenCV display
                if frame.format == ImageFormat.I420:
                    # Get I420 buffer
                    i420_buffer = frame.i420_buffer
                    
                    # Extract Y, U, V planes
                    height = i420_buffer.height()
                    width = i420_buffer.width()
                    
                    y_plane = np.array(i420_buffer.y[:height, :width], copy=True)
                    u_plane = np.array(i420_buffer.u[:height//2, :width//2], copy=True)
                    v_plane = np.array(i420_buffer.v[:height//2, :width//2], copy=True)
                    
                    # Upsample U and V planes
                    u_upsampled = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
                    v_upsampled = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    # Stack to create YUV image
                    yuv = np.stack([y_plane, u_upsampled, v_upsampled], axis=-1).astype(np.uint8)
                    
                    # Convert YUV to BGR
                    bgr_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    
                    # Display frame
                    cv2.imshow(self.window_name, bgr_frame)
                    
                    if frame_count <= 5:
                        logger.info(f"Displayed video frame #{frame_count}")
                    
                    # Handle window events
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User pressed 'q', stopping playback")
                        self.playback_active = False

            except queue.Empty:
                # No frame available
                if window_created:
                    # Check if window is closed
                    try:
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Window closed, stopping playback")
                            self.playback_active = False
                    except cv2.error:
                        pass  # Window might not exist yet
            except Exception as e:
                self._handle_error("playing video", e)
        
        # Decode any remaining H264 NAL units
        if hasattr(self, '_h264_nal_buffer') and len(self._h264_nal_buffer) > 0:
            logger.info(f"Decoding remaining H264 NAL units: {len(self._h264_nal_buffer)} NALs")
            # Process remaining NALs
            combined_data = bytearray()
            for nal in self._h264_nal_buffer:
                combined_data.extend(nal)
            if len(combined_data) > 0:
                encoded_image = EncodedImage()
                np_data = np.array(combined_data, dtype=np.uint8)
                encoded_image.data = np_data
                from datetime import timedelta
                encoded_image.timestamp = timedelta(milliseconds=self._h264_timestamp)
                try:
                    if self.video_decoder:
                        self.video_decoder.decode(encoded_image)
                        # Try to flush decoder
                        if hasattr(self.video_decoder, 'flush'):
                            self.video_decoder.flush()
                            logger.info("Flushed decoder for remaining frames")
                except Exception as e:
                    logger.error(f"Error decoding remaining NALs: {e}")

        if window_created:
            cv2.destroyWindow(self.window_name)
        logger.info(f"Video playback stopped. Total frames displayed: {frame_count}")

    def _play_audio(self):
        """Play audio frames using sounddevice"""
        logger.info("Starting audio playback...")

        # Buffer for accumulating audio samples
        self.audio_buffer = []
        
        def audio_callback(outdata, frames, time_info, status):
            _ = time_info  # Unused
            if status:
                logger.warning(f"Audio playback status: {status}")

            # Initialize with silence
            outdata.fill(0)
            
            samples_needed = frames
            samples_written = 0

            # First, use any buffered samples
            if self.audio_buffer:
                buffered_samples = len(self.audio_buffer)
                samples_to_use = min(buffered_samples, samples_needed)
                buffered_data = np.vstack(self.audio_buffer[:samples_to_use])
                outdata[:samples_to_use] = buffered_data
                self.audio_buffer = self.audio_buffer[samples_to_use:]
                samples_written += samples_to_use
                samples_needed -= samples_to_use

            # Then get more frames from queue if needed
            while samples_needed > 0:
                try:
                    audio_frame = self.audio_queue.get_nowait()
                    pcm_data = audio_frame.pcm
                    
                    if pcm_data.shape[0] <= samples_needed:
                        # Use entire frame
                        outdata[samples_written:samples_written + pcm_data.shape[0]] = pcm_data
                        samples_written += pcm_data.shape[0]
                        samples_needed -= pcm_data.shape[0]
                    else:
                        # Use part of frame, buffer the rest
                        outdata[samples_written:] = pcm_data[:samples_needed]
                        # Buffer remaining samples
                        for i in range(samples_needed, pcm_data.shape[0]):
                            self.audio_buffer.append(pcm_data[i:i+1])
                        samples_needed = 0
                        
                except queue.Empty:
                    # No more audio available
                    break

        try:
            with sd.OutputStream(
                samplerate=self.audio_sample_rate,
                channels=self.audio_channels,
                callback=audio_callback,
                blocksize=int(self.audio_sample_rate * 0.02),  # 20ms blocks
                dtype="float32",
            ):
                logger.info(f"Audio playback started at {self.audio_sample_rate} Hz")
                while self.playback_active:
                    time.sleep(0.1)
        except Exception as e:
            self._handle_error("audio playback", e)

        logger.info("Audio playback stopped")

    def start_playback(self):
        """Start receiving and playing media"""
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
        
        # Check track states
        logger.info("Track states after connection:")
        logger.info(f"  Video track: exists={self.video_track is not None}, is_open={self.video_track.is_open() if self.video_track else 'N/A'}")
        logger.info(f"  Audio track: exists={self.audio_track is not None}, is_open={self.audio_track.is_open() if self.audio_track else 'N/A'}")
        
        # Set up decoders after connection is established
        self._setup_video_decoder()
        self._setup_audio_decoder()
        
        # Debug: check message count after a delay
        def check_messages():
            time.sleep(2)
            logger.info(f"Message counts after 2s: video={getattr(self, '_video_message_count', 0)}, audio={getattr(self, '_audio_message_count', 0)}")
        
        threading.Thread(target=check_messages).start()

        # Start playback threads
        self.playback_active = True

        if not self.no_video:
            self.video_thread = threading.Thread(target=self._play_video)
            self.video_thread.daemon = True  # Make thread daemon to ensure clean shutdown
            self.video_thread.start()
            logger.info("Started video playback thread")
        else:
            logger.info("Video playback disabled")

        self.audio_thread = threading.Thread(target=self._play_audio)
        self.audio_thread.start()
        logger.info("Started audio playback thread")

        # Monitor statistics
        start_time = time.time()
        timeout = getattr(self, 'timeout', None)
        
        try:
            while self.playback_active:
                current_time = time.time()
                
                # Check timeout
                if timeout and (current_time - start_time) >= timeout:
                    logger.info(f"Timeout reached ({timeout} seconds)")
                    break
                
                if current_time - self.last_stats_time >= 5.0:  # Log stats every 5 seconds
                    video_messages = getattr(self, '_video_message_count', 0)
                    audio_messages = getattr(self, '_audio_message_count', 0)
                    elapsed = int(current_time - start_time)
                    video_fps = self.video_frames_received / elapsed if elapsed > 0 else 0
                    audio_fps = self.audio_frames_received / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"[{elapsed}s] Video: {video_messages} msgs, {self.video_frames_received} sent to decoder, {self._decoded_frame_count} decoded ({video_fps:.1f} fps), "
                        f"Audio: {audio_messages} msgs, {self.audio_frames_received} decoded ({audio_fps:.1f} fps), "
                        f"Queues: V={self.video_queue.qsize()} A={self.audio_queue.qsize()}"
                    )
                    self.last_stats_time = current_time
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Stop playback
            self.playback_active = False
            if self.video_thread:
                self.video_thread.join(timeout=2.0)
            if self.audio_thread:
                self.audio_thread.join(timeout=2.0)

    def disconnect(self):
        """Disconnect from WHEP server"""
        logger.info("Starting graceful shutdown...")

        # First, send DELETE request to WHEP server
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
                        logger.warning(f"DELETE request returned status {response.status_code}")
            except Exception as e:
                logger.error(f"DELETE request failed: {e}")

        # Stop playback
        self.playback_active = False

        # Clean up resources
        logger.info("Cleaning up resources...")

        # Close tracks
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

        # Release decoders
        if self.video_decoder:
            try:
                if hasattr(self.video_decoder, 'release'):
                    self.video_decoder.release()
                    logger.info("Released video decoder")
            except Exception as e:
                logger.error(f"Error releasing video decoder: {e}")
        if self.audio_decoder:
            try:
                if hasattr(self.audio_decoder, 'release'):
                    self.audio_decoder.release()
                    logger.info("Released audio decoder")
            except Exception as e:
                logger.error(f"Error releasing audio decoder: {e}")
        
        self.video_decoder = None
        self.audio_decoder = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="WHEP client for WebRTC playback")
    parser.add_argument("--url", required=True, help="WHEP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--no-video", action="store_true", help="Disable video display (audio only)")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")

    args = parser.parse_args()

    logger.info("Starting WHEP client...")
    logger.info("Press 'q' in the video window or Ctrl+C to stop")

    client = WHEPClient(args.url, args.token, args.no_video, args.timeout)

    try:
        client.connect()
        client.start_playback()
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