"""
WHEP client with proper RTP depacketization using custom MediaHandler
"""
import argparse
import logging
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
    RtpDepacketizer,
    H264RtpDepacketizer,
    NalUnit,
)
from wish import parse_link_header
from custom_media_handler import RtpDepacketizingHandler

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleWHEPClient:
    """WHEP client with proper RTP depacketization"""

    def __init__(self, whep_url: str, bearer_token: Optional[str] = None, timeout: Optional[int] = None):
        self.whep_url = whep_url
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.pc: Optional[PeerConnection] = None
        self.video_track: Optional[Track] = None
        self.audio_track: Optional[Track] = None
        self.session_url: Optional[str] = None
        
        # Statistics
        self.video_payloads_received = 0
        self.audio_payloads_received = 0
        self.last_stats_time = time.time()
        self.running = False

    def connect(self):
        """Connect to WHEP server"""
        logger.info(f"Connecting to WHEP endpoint: {self.whep_url}")

        # Create peer connection
        config = Configuration()
        config.ice_servers = []
        
        if hasattr(config, "max_message_size"):
            config.max_message_size = 16384

        if hasattr(config, "disable_auto_gathering"):
            config.disable_auto_gathering = True

        self.pc = PeerConnection(config)

        # Add receive-only tracks
        audio_desc = Description.Audio("audio", Description.Direction.RecvOnly)
        audio_desc.add_opus_codec(111)
        self.audio_track = self.pc.add_track(audio_desc)
        logger.info("Added audio track with Opus codec (PT=111)")

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

        sdp_lines = str(local_sdp).split("\n")
        media_lines = [line.strip() for line in sdp_lines if line.startswith("m=")]
        logger.info(f"SDP media sections order: {media_lines}")
        
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

            # Set remote SDP
            answer = Description(response.text, Description.Type.Answer)
            self.pc.set_remote_description(answer)
            
            logger.debug("SDP Answer:")
            logger.debug(response.text)

        logger.info("Connected to WHEP server")

    def _setup_tracks(self):
        """Set up track message handlers with custom RTP depacketizer"""
        
        # Video payload handler
        def on_video_payload(payload: bytes, payload_type: int, timestamp: int):
            self.video_payloads_received += 1
            
            if self.video_payloads_received <= 10 or self.video_payloads_received % 1000 == 0:
                logger.info(f"Video payload #{self.video_payloads_received}: PT={payload_type}, size={len(payload)} bytes")
                
                # For H264, check NAL unit type
                if payload_type == 96 and len(payload) > 0:
                    nal_type = payload[0] & 0x1F
                    logger.info(f"  H264 NAL type: {nal_type} ({self._get_nal_type_name(nal_type)})")
                    
                    if len(payload) >= 5:
                        preview = ' '.join(f'{b:02x}' for b in payload[:5])
                        logger.info(f"  Payload preview: {preview}...")

        # Audio payload handler
        def on_audio_payload(payload: bytes, payload_type: int, timestamp: int):
            self.audio_payloads_received += 1
            
            if self.audio_payloads_received <= 10 or self.audio_payloads_received % 1000 == 0:
                logger.info(f"Audio payload #{self.audio_payloads_received}: PT={payload_type}, size={len(payload)} bytes")

        if self.video_track:
            # Set up RTCP receiving session
            rtcp_session = RtcpReceivingSession()
            self.video_track.set_media_handler(rtcp_session)
            
            # Use custom RTP depacketizer
            video_depacketizer = RtpDepacketizingHandler(on_video_payload)
            self.video_track.chain_media_handler(video_depacketizer)
            
            # The track's on_message will still receive raw RTP packets
            # But our custom handler will extract the payloads
            self.video_track.on_message(lambda data: None)  # Ignore raw messages
            
            logger.info(f"Video track setup with custom RTP depacketizer")

        if self.audio_track:
            # Set up RTCP receiving session
            rtcp_session = RtcpReceivingSession()
            self.audio_track.set_media_handler(rtcp_session)
            
            # Use custom RTP depacketizer for audio
            audio_depacketizer = RtpDepacketizingHandler(on_audio_payload)
            self.audio_track.chain_media_handler(audio_depacketizer)
            
            # Ignore raw messages
            self.audio_track.on_message(lambda data: None)
            
            logger.info(f"Audio track setup with custom RTP depacketizer")
    
    def _get_nal_type_name(self, nal_type):
        """Get human-readable name for NAL unit type"""
        nal_names = {
            1: "Non-IDR slice",
            5: "IDR slice", 
            6: "SEI",
            7: "SPS",
            8: "PPS",
            9: "Access unit delimiter",
            24: "STAP-A",
            28: "FU-A",
        }
        return nal_names.get(nal_type, f"Type {nal_type}")

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
                    video_pps = self.video_payloads_received / elapsed if elapsed > 0 else 0
                    audio_pps = self.audio_payloads_received / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"[{elapsed}s] Video: {self.video_payloads_received} RTP payloads ({video_pps:.1f}/s), "
                        f"Audio: {self.audio_payloads_received} RTP payloads ({audio_pps:.1f}/s)"
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
    parser = argparse.ArgumentParser(description="WHEP client with custom RTP depacketizer")
    parser.add_argument("--url", required=True, help="WHEP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")

    args = parser.parse_args()

    logger.info("Starting WHEP client with custom MediaHandler...")
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