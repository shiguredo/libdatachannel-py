import argparse
import logging
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import httpx

from libdatachannel import (
    Configuration,
    Description,
    IceServer,
    PeerConnection,
    RtcpReceivingSession,
    Track,
)
from wish import parse_link_header

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleWHEPClient:
    """Minimal WHEP client for receiving RTP packets without decoding"""

    def __init__(self, whep_url: str, bearer_token: Optional[str] = None, timeout: Optional[int] = None):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None
        
        # Statistics
        self.video_packets_received = 0
        self.audio_packets_received = 0
        self.last_stats_time = time.time()
        self.running = False

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
                ice_servers = parse_link_header(link_header)
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

    def _setup_tracks(self):
        """Set up track message handlers"""
        # Video track handler
        def on_video_message(data):
            self.video_packets_received += 1
            
            # Extract RTP header info if this is an RTP packet
            if len(data) >= 12 and (data[0] >> 6) == 2:  # RTP version 2
                pt = data[1] & 0x7F
                seq = (data[2] << 8) | data[3]
                timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                
                if self.video_packets_received <= 10 or self.video_packets_received % 1000 == 0:
                    logger.info(f"Video RTP packet #{self.video_packets_received}: PT={pt}, seq={seq}, timestamp={timestamp}, size={len(data)} bytes")
            else:
                if self.video_packets_received <= 10:
                    logger.info(f"Video message #{self.video_packets_received}: {len(data)} bytes (not RTP)")

        # Audio track handler  
        def on_audio_message(data):
            self.audio_packets_received += 1
            
            # Extract RTP header info if this is an RTP packet
            if len(data) >= 12 and (data[0] >> 6) == 2:  # RTP version 2
                pt = data[1] & 0x7F
                seq = (data[2] << 8) | data[3]
                timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                
                if self.audio_packets_received <= 10 or self.audio_packets_received % 1000 == 0:
                    logger.info(f"Audio RTP packet #{self.audio_packets_received}: PT={pt}, seq={seq}, timestamp={timestamp}, size={len(data)} bytes")
            else:
                if self.audio_packets_received <= 10:
                    logger.info(f"Audio message #{self.audio_packets_received}: {len(data)} bytes (not RTP)")

        if self.video_track:
            # Set up RTCP receiving session
            rtcp_session = RtcpReceivingSession()
            self.video_track.set_media_handler(rtcp_session)
            
            # Set message handler on track
            self.video_track.on_message(on_video_message)
            
            logger.info(f"Video track setup: is_open={self.video_track.is_open()}")

        if self.audio_track:
            # Set up RTCP receiving session
            rtcp_session = RtcpReceivingSession()
            self.audio_track.set_media_handler(rtcp_session)
            
            # Set message handler on track
            self.audio_track.on_message(on_audio_message)
            
            logger.info(f"Audio track setup: is_open={self.audio_track.is_open()}")

    def start_receiving(self):
        """Start receiving RTP packets"""
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
        
        # Set up track handlers
        self._setup_tracks()
        
        # Monitor statistics
        self.running = True
        start_time = time.time()
        timeout_seconds = getattr(self, 'timeout', None)
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check timeout
                if timeout_seconds and (current_time - start_time) >= timeout_seconds:
                    logger.info(f"Timeout reached ({timeout_seconds} seconds)")
                    break
                
                # Log statistics every 5 seconds
                if current_time - self.last_stats_time >= 5.0:
                    elapsed = int(current_time - start_time)
                    video_pps = self.video_packets_received / elapsed if elapsed > 0 else 0
                    audio_pps = self.audio_packets_received / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"[{elapsed}s] Video: {self.video_packets_received} packets ({video_pps:.1f} pps), "
                        f"Audio: {self.audio_packets_received} packets ({audio_pps:.1f} pps)"
                    )
                    self.last_stats_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.running = False

    def disconnect(self):
        """Disconnect from WHEP server"""
        logger.info("Starting graceful shutdown...")

        # Stop receiving
        self.running = False

        # Send DELETE request to WHEP server
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
                logger.error(f"Error closing PeerConnection: {e}")
            finally:
                self.pc = None

        logger.info("Graceful shutdown completed")


def main():
    parser = argparse.ArgumentParser(description="Simple WHEP client for receiving RTP packets")
    parser.add_argument("--url", required=True, help="WHEP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")

    args = parser.parse_args()

    logger.info("Starting simple WHEP client...")
    logger.info("Press Ctrl+C to stop")

    client = SimpleWHEPClient(args.url, args.token, args.timeout)

    try:
        client.connect()
        client.start_receiving()
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