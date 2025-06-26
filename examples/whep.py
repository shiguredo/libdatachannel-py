import argparse
import logging
import queue
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import httpx
import numpy as np
import cv2

from libdatachannel import (
    Configuration,
    Description,
    H264RtpDepacketizer,
    IceServer,
    NalUnit,
    OpusRtpDepacketizer,
    PeerConnection,
    Track,
)
from libdatachannel.codec import (
    VideoCodecType,
    VideoDecoder,
    create_openh264_video_decoder,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WHEPClient:
    """Minimal WHEP client for receiving test video and audio"""

    def __init__(self, whep_url: str, bearer_token: Optional[str] = None, openh264_path: Optional[str] = None, display_video: bool = False):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.openh264_path = openh264_path
        self.display_video = display_video
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Track counters
        self.video_frame_count = 0
        self.audio_frame_count = 0
        self.decoded_frame_count = 0
        
        # Decoder
        self.video_decoder = None
        
        # OpenCV display
        self.window_name = "WHEP Video"
        self.frame_queue = queue.Queue(maxsize=10) if display_video else None

    def _handle_error(self, context: str, error: Exception):
        """Unified error handling"""
        logger.error(f"Error {context}: {error}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback

            traceback.print_exc()

    def _parse_link_header(self, link_header: str) -> list[IceServer]:
        """Parse Link header for ICE servers"""
        ice_servers: list[IceServer] = []
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
            import re

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
        # No default ICE servers - will use TURN from Link header
        config.ice_servers = []

        # Try to disable auto gathering if available (for later adding ICE servers from Link header)
        if hasattr(config, "disable_auto_gathering"):
            config.disable_auto_gathering = True

        self.pc = PeerConnection(config)

        # Add audio track FIRST (before video) - RecvOnly
        audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        # Opus codec with proper parameters
        audio_desc.add_opus_codec(111)
        logger.info("Audio description created with Opus codec")
        self.audio_track = self.pc.add_track(audio_desc)

        # Add video track SECOND (after audio) - RecvOnly
        video_desc = Description.Video("video", Description.Direction.RecvOnly)
        video_desc.add_h264_codec(96)  # H.264 codec
        self.video_track = self.pc.add_track(video_desc)
        logger.info("Audio track added with Opus codec (PT=111)")
        logger.info("Video track added with H.264 codec (PT=96)")

        # Set up depacketizers and handlers
        self._setup_video_depacketizer()
        self._setup_audio_depacketizer()
        
        # Initialize video decoder if OpenH264 path is provided
        if self.openh264_path:
            self._setup_video_decoder()

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
                raise Exception(f"WHEP server returned {response.status_code}: {response.text}")

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

        logger.info("Connected to WHEP server")

    def _get_nal_type_name(self, nal_type: int) -> str:
        """Get human-readable name for NAL unit type"""
        nal_type_names = {
            0: "Unspecified",
            1: "Coded slice (non-IDR)",
            2: "Coded slice data partition A",
            3: "Coded slice data partition B",
            4: "Coded slice data partition C",
            5: "Coded slice (IDR)",
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
            19: "Coded slice of auxiliary picture",
            20: "Coded slice extension",
        }
        return nal_type_names.get(nal_type, f"Reserved/Unknown ({nal_type})")

    def _setup_video_depacketizer(self):
        """Set up H.264 RTP depacketizer for video track"""
        if self.video_track:
            # H264RtpDepacketizer takes a NalUnit.Separator type
            # Default is LongStartSequence (0x00000001)
            h264_depacketizer = H264RtpDepacketizer()
            
            # Set up frame handler for H.264 frames
            def on_video_frame(data: bytes, frame_info):
                self.video_frame_count += 1
                
                # Parse NAL units using manual parsing
                # Note: libdatachannel's NalUnit class is primarily for creating NAL units,
                # not for parsing existing H.264 stream data
                nal_units = []
                try:
                    # Parse the H.264 frame data for NAL units
                    offset = 0
                    while offset < len(data):
                        # Look for start code (0x00000001 or 0x000001)
                        if offset + 4 <= len(data) and data[offset:offset+4] == b'\x00\x00\x00\x01':
                            start_code_len = 4
                        elif offset + 3 <= len(data) and data[offset:offset+3] == b'\x00\x00\x01':
                            start_code_len = 3
                        else:
                            offset += 1
                            continue
                        
                        # Find the start of NAL unit data
                        nal_start = offset + start_code_len
                        if nal_start >= len(data):
                            break
                            
                        # Find the next start code or end of data
                        next_offset = nal_start
                        while next_offset < len(data):
                            if (next_offset + 4 <= len(data) and data[next_offset:next_offset+4] == b'\x00\x00\x00\x01') or \
                               (next_offset + 3 <= len(data) and data[next_offset:next_offset+3] == b'\x00\x00\x01'):
                                break
                            next_offset += 1
                        
                        # Extract NAL unit
                        nal_data = data[nal_start:next_offset]
                        if nal_data:
                            # Create NalUnit object from bytes
                            try:
                                nal_unit = NalUnit(nal_data)
                                
                                # Get NAL unit properties from the NalUnit object
                                nal_type = nal_unit.unit_type()
                                nal_ref_idc = nal_unit.nri()
                                forbidden_bit = 1 if nal_unit.forbidden_bit() else 0
                                
                                nal_units.append({
                                    'type': nal_type,
                                    'ref_idc': nal_ref_idc,
                                    'forbidden': forbidden_bit,
                                    'size': len(nal_data),
                                    'type_name': self._get_nal_type_name(nal_type),
                                    'nal_unit_obj': nal_unit  # Store the NalUnit object
                                })
                            except Exception:
                                # Fallback to manual parsing if NalUnit construction fails
                                nal_header = nal_data[0]
                                nal_type = nal_header & 0x1F
                                nal_ref_idc = (nal_header >> 5) & 0x03
                                forbidden_bit = (nal_header >> 7) & 0x01
                                
                                nal_units.append({
                                    'type': nal_type,
                                    'ref_idc': nal_ref_idc,
                                    'forbidden': forbidden_bit,
                                    'size': len(nal_data),
                                    'type_name': self._get_nal_type_name(nal_type)
                                })
                        
                        offset = next_offset
                
                except Exception as e:
                    logger.error(f"Error parsing NAL units: {e}")
                
                # Decode the frame with OpenH264 if decoder is available
                if self.video_decoder and len(data) > 0:
                    try:
                        from libdatachannel.codec import EncodedImage
                        import numpy as np
                        
                        # Create EncodedImage with the H.264 frame data
                        encoded_image = EncodedImage()
                        # Convert bytes to numpy array - make a copy to ensure correct format
                        np_data = np.frombuffer(data, dtype=np.uint8).copy()
                        encoded_image.data = np_data
                        # Convert timestamp to timedelta (microseconds)
                        from datetime import timedelta
                        encoded_image.timestamp = timedelta(microseconds=frame_info.timestamp)
                        
                        # Decode the frame
                        self.video_decoder.decode(encoded_image)
                        
                        if self.video_frame_count % 30 == 0:
                            logger.debug(f"Decoded video frame #{self.video_frame_count}")
                    except Exception as e:
                        if self.video_frame_count <= 2:
                            logger.error(f"Error decoding video frame: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                
                if self.video_frame_count % 30 == 0 or any(unit['type'] in [5, 7, 8] for unit in nal_units):  # Log every 30 frames or key frames
                    logger.info(
                        f"Video frame #{self.video_frame_count}: "
                        f"size={len(data)} bytes, timestamp={frame_info.timestamp}, "
                        f"NAL units: {len(nal_units)}"
                    )
                    for i, unit in enumerate(nal_units):
                        logger.info(
                            f"  NAL[{i}]: type={unit['type']} ({unit['type_name']}), "
                            f"ref_idc={unit['ref_idc']}, size={unit['size']} bytes, "
                            f"forbidden={unit['forbidden']}"
                        )
            
            self.video_track.on_frame(on_video_frame)
            self.video_track.set_media_handler(h264_depacketizer)
            logger.info("H.264 depacketizer and handlers set for video track")

    def _setup_audio_depacketizer(self):
        """Set up Opus RTP depacketizer for audio track"""
        if self.audio_track:
            # OpusRtpDepacketizer for Opus packets
            opus_depacketizer = OpusRtpDepacketizer()
            
            # Set up frame handler for Opus frames
            def on_audio_frame(data: bytes, frame_info):
                self.audio_frame_count += 1
                if self.audio_frame_count % 50 == 0:  # Log every 50 frames
                    logger.info(
                        f"Audio frame #{self.audio_frame_count}: "
                        f"size={len(data)} bytes, timestamp={frame_info.timestamp}"
                    )
            
            self.audio_track.on_frame(on_audio_frame)
            self.audio_track.set_media_handler(opus_depacketizer)
            logger.info("Opus depacketizer and handlers set for audio track")

    def _setup_video_decoder(self):
        """Set up OpenH264 video decoder"""
        try:
            # Load OpenH264 library
            import os
            if not os.path.exists(self.openh264_path):
                logger.error(f"OpenH264 library not found at: {self.openh264_path}")
                return
                
            # Create OpenH264 decoder
            self.video_decoder = create_openh264_video_decoder(self.openh264_path)
            
            # Initialize decoder settings
            settings = VideoDecoder.Settings()
            settings.codec_type = VideoCodecType.H264
            
            if self.video_decoder.init(settings):
                logger.info(f"OpenH264 decoder initialized successfully from: {self.openh264_path}")
                
                # Set up decoder callback
                def on_decoded_frame(frame):
                    self.decoded_frame_count += 1
                    if self.decoded_frame_count % 30 == 0:  # Log every 30 decoded frames
                        logger.info(
                            f"Decoded frame #{self.decoded_frame_count}: "
                            f"{frame.width()}x{frame.height()}, format={frame.format}"
                        )
                    
                    # Put frame in queue for display if enabled
                    if self.display_video and self.frame_queue:
                        try:
                            # Convert I420 to RGB for OpenCV display
                            import numpy as np
                            from libdatachannel import libyuv
                            
                            width = frame.width()
                            height = frame.height()
                            
                            # Create RGB buffer
                            rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
                            
                            # Convert I420 to RGB using libyuv
                            if frame.format.name == "I420" and frame.i420_buffer:
                                i420_buffer = frame.i420_buffer
                                libyuv.i420_to_rgb24(
                                    i420_buffer.y, i420_buffer.u, i420_buffer.v,
                                    i420_buffer.stride_y(), i420_buffer.stride_u(), i420_buffer.stride_v(),
                                    rgb_buffer, width * 3,
                                    width, height
                                )
                                
                                # Put frame in queue (non-blocking)
                                try:
                                    self.frame_queue.put_nowait(rgb_buffer)
                                    if self.decoded_frame_count == 1:
                                        logger.info("First frame added to display queue")
                                except queue.Full:
                                    # Drop frame if queue is full
                                    if self.decoded_frame_count % 30 == 0:
                                        logger.warning("Display queue is full, dropping frame")
                        except Exception as e:
                            logger.error(f"Error converting frame for display: {e}")
                
                self.video_decoder.set_on_decode(on_decoded_frame)
            else:
                logger.error("Failed to initialize OpenH264 decoder")
                self.video_decoder = None
                
        except Exception as e:
            logger.error(f"Error setting up OpenH264 decoder: {e}")
            self.video_decoder = None

    def receive_frames(self, duration: Optional[int] = None):
        """Receive video and audio frames"""
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

        start_time = time.time()
        last_video_count = 0
        last_audio_count = 0

        try:
            while True:
                current_time = time.time()

                # Check duration
                if duration and current_time - start_time >= duration:
                    break

                # Log statistics every second
                if int(current_time - start_time) > int(current_time - start_time - 1):
                    video_fps = self.video_frame_count - last_video_count
                    audio_fps = self.audio_frame_count - last_audio_count
                    last_video_count = self.video_frame_count
                    last_audio_count = self.audio_frame_count

                    logger.info(
                        f"Stats: Video {video_fps} fps (total: {self.video_frame_count} frames), "
                        f"Audio {audio_fps} fps (total: {self.audio_frame_count} frames)"
                    )

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        logger.info(
            f"Receive completed. Total frames - Video: {self.video_frame_count}, Audio: {self.audio_frame_count}"
        )
        if self.video_decoder:
            logger.info(f"Total decoded frames: {self.decoded_frame_count}")

    def disconnect(self):
        """Disconnect from WHEP server with graceful shutdown"""
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
            except httpx.TimeoutException:
                logger.error("DELETE request timed out")
            except httpx.RequestError as e:
                logger.error(f"DELETE request failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during DELETE: {e}")

        # Wait a bit for graceful shutdown
        logger.info("Waiting for graceful shutdown...")
        time.sleep(0.5)

        # Clean up resources in proper order
        logger.info("Cleaning up resources...")
        
        # Clean up video decoder if it exists
        if self.video_decoder:
            try:
                self.video_decoder.release()
                logger.info("Video decoder released")
            except Exception as e:
                self._handle_error("releasing video decoder", e)
            finally:
                self.video_decoder = None
        
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

        logger.info("Graceful shutdown completed")


def display_frames(client: WHEPClient, stop_event: threading.Event):
    """Display frames from queue using OpenCV (must run on main thread for macOS)"""
    logger.info("Starting display_frames function")
    
    logger.info(f"Creating window: {client.window_name}")
    cv2.namedWindow(client.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(client.window_name, 640, 480)  # Set initial window size
    cv2.moveWindow(client.window_name, 100, 100)   # Position window
    logger.info("Window created")
    
    frame_count = 0
    while not stop_event.is_set():
        try:
            # Get frame from queue with timeout
            frame = client.frame_queue.get(timeout=0.1)
            frame_count += 1
            
            if frame_count == 1:
                logger.info(f"Got first frame from queue: shape={frame.shape}")
            
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Display frame
            cv2.imshow(client.window_name, bgr_frame)
            
            if frame_count % 30 == 0:
                logger.info(f"Displayed {frame_count} frames")
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q', stopping display")
                stop_event.set()
                break
                
        except queue.Empty:
            # No frame available, continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q' while waiting, stopping display")
                stop_event.set()
                break
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Exiting display loop. Total frames displayed: {frame_count}")
    cv2.destroyAllWindows()
    logger.info("Windows destroyed")


def main():
    parser = argparse.ArgumentParser(description="WHEP client for receiving media")
    parser.add_argument("--url", required=True, help="WHEP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--openh264", help="Path to OpenH264 library for H.264 decoding")
    parser.add_argument("--display", action="store_true", help="Display video using OpenCV")

    args = parser.parse_args()

    logger.info("Starting WHEP client...")
    logger.info(f"WHEP endpoint: {args.url}")
    if args.openh264:
        logger.info(f"OpenH264 library path: {args.openh264}")
    if args.display:
        logger.info("Video display enabled")

    client = WHEPClient(args.url, args.token, args.openh264, args.display)
    
    # Event to signal stop
    stop_event = threading.Event()
    
    # Start connection in a separate thread
    def run_client():
        try:
            client.connect()
            client.receive_frames(args.duration)
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
        except Exception as e:
            client._handle_error("", e)
        finally:
            stop_event.set()
            # Always disconnect gracefully
            try:
                client.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    if args.display:
        if not args.openh264:
            logger.error("--openh264 is required when --display is specified")
            return
            
        logger.info("Display mode enabled, starting client in thread")
        # Start client in a thread
        client_thread = threading.Thread(target=run_client)
        client_thread.start()
        
        # Display frames on main thread (required for macOS)
        logger.info("Starting display on main thread")
        display_frames(client, stop_event)
        
        # Wait for client thread to finish
        logger.info("Waiting for client thread to finish")
        client_thread.join()
        logger.info("Client thread finished")
    else:
        # Run client directly if no display
        logger.info("Display mode disabled, running client directly")
        run_client()


if __name__ == "__main__":
    main()