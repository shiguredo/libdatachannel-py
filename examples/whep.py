import argparse
import logging
import time
from typing import Optional
from urllib.parse import urljoin

import httpx

from libdatachannel import (
    Configuration,
    Description,
    H264RtpDepacketizer,
    IceServer,
    OpusRtpDepacketizer,
    PeerConnection,
    Track,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WHEPClient:
    """Minimal WHEP client for receiving test video and audio"""

    def __init__(self, whep_url: str, bearer_token: Optional[str] = None):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None

        # Track counters
        self.video_frame_count = 0
        self.audio_frame_count = 0
        
        # RTP packet statistics
        self.video_packets_received = 0
        self.video_bytes_received = 0
        self.audio_packets_received = 0
        self.audio_bytes_received = 0

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
        self._setup_depacketizers()
        self._setup_track_handlers()

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

    def _setup_depacketizers(self):
        """Set up RTP depacketizers for audio and video"""
        # Video H.264 depacketizer
        if self.video_track:
            # H264RtpDepacketizer takes a NalUnit.Separator type
            # Default is LongStartSequence (0x00000001)
            h264_depacketizer = H264RtpDepacketizer()
            
            # Set up message handler for raw RTP packets
            def on_video_message(data: bytes):
                self.video_packets_received += 1
                self.video_bytes_received += len(data)
                if self.video_packets_received % 100 == 0:  # Log every 100 packets
                    logger.info(f"Video RTP stats: {self.video_packets_received} packets, {self.video_bytes_received} bytes")
            
            # Set up frame handler for H.264 frames
            def on_video_frame(data: bytes, frame_info):
                self.video_frame_count += 1
                if self.video_frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(
                        f"Video frame #{self.video_frame_count}: "
                        f"size={len(data)} bytes, timestamp={frame_info.timestamp}"
                    )
            
            self.video_track.on_message(on_video_message)
            self.video_track.on_frame(on_video_frame)
            self.video_track.set_media_handler(h264_depacketizer)
            logger.info("H.264 depacketizer and handlers set for video track")
        
        # Audio Opus depacketizer
        if self.audio_track:
            # OpusRtpDepacketizer for Opus packets
            opus_depacketizer = OpusRtpDepacketizer()
            
            # Set up message handler for raw RTP packets
            def on_audio_message(data: bytes):
                self.audio_packets_received += 1
                self.audio_bytes_received += len(data)
                if self.audio_packets_received % 100 == 0:  # Log every 100 packets
                    logger.info(f"Audio RTP stats: {self.audio_packets_received} packets, {self.audio_bytes_received} bytes")
            
            # Set up frame handler for Opus frames
            def on_audio_frame(data: bytes, frame_info):
                self.audio_frame_count += 1
                if self.audio_frame_count % 50 == 0:  # Log every 50 frames
                    logger.info(
                        f"Audio frame #{self.audio_frame_count}: "
                        f"size={len(data)} bytes, timestamp={frame_info.timestamp}"
                    )
            
            self.audio_track.on_message(on_audio_message)
            self.audio_track.on_frame(on_audio_frame)
            self.audio_track.set_media_handler(opus_depacketizer)
            logger.info("Opus depacketizer and handlers set for audio track")

    def _setup_track_handlers(self):
        """Set up message handlers for receiving raw RTP packets (if depacketizers are not used)"""
        # Note: When using depacketizers, these handlers won't be called
        # as the depacketizer intercepts the RTP packets
        pass

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
                    logger.info(
                        f"  RTP: Video {self.video_packets_received} packets ({self.video_bytes_received} bytes), "
                        f"Audio {self.audio_packets_received} packets ({self.audio_bytes_received} bytes)"
                    )

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        logger.info(
            f"Receive completed. Total frames - Video: {self.video_frame_count}, Audio: {self.audio_frame_count}"
        )
        logger.info(
            f"Total RTP packets - Video: {self.video_packets_received} ({self.video_bytes_received} bytes), "
            f"Audio: {self.audio_packets_received} ({self.audio_bytes_received} bytes)"
        )

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


def main():
    parser = argparse.ArgumentParser(description="WHEP client for receiving media")
    parser.add_argument("--url", required=True, help="WHEP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--duration", type=int, help="Duration in seconds")

    args = parser.parse_args()

    logger.info("Starting WHEP client...")
    logger.info(f"WHEP endpoint: {args.url}")

    client = WHEPClient(args.url, args.token)

    try:
        client.connect()
        client.receive_frames(args.duration)
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