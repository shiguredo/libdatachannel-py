import argparse
import asyncio
import logging
import re
from typing import List, Optional
from urllib.parse import urljoin

import httpx
import numpy as np

from libdatachannel import (
    AV1RtpPacketizer,
    Configuration,
    Description,
    IceServer,
    OpusRtpPacketizer,
    PeerConnection,
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

        # Track if we've already cleaned up
        self._cleaned_up = False

    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        if not self._cleaned_up:
            logger.warning("WHIPClient being destroyed without proper cleanup!")
            # Synchronous cleanup in destructor
            # Clear tracks
            self.video_track = None
            self.audio_track = None

            # Close PeerConnection
            if self.pc:
                try:
                    self.pc.close()
                except Exception:
                    pass
            self.pc = None

            # Release encoders
            if self.video_encoder:
                try:
                    self.video_encoder.release()
                except Exception:
                    pass
            self.video_encoder = None

            if self.audio_encoder:
                try:
                    self.audio_encoder.release()
                except Exception:
                    pass
            self.audio_encoder = None

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
                    logger.debug(f"  with username: {ice_server.username}")

        return ice_servers

    async def connect(self):
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "Content-Type": "application/sdp",
            }
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"

            response = await client.post(
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

        # Set packetizer on track
        if not self.video_track:
            raise RuntimeError("Video track not initialized")
        self.video_track.set_media_handler(self.video_packetizer)

        # Set encoder callback
        self.video_encoder.set_on_encode(self._on_video_encoded)

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
        self.audio_encoder.set_on_encode(self._on_audio_encoded)

    def _on_video_encoded(self, encoded_image):
        """Handle encoded video frame"""
        if self.video_track and self.video_track.is_open():
            try:
                data = encoded_image.data.tobytes()
                self.video_track.send(data)
                logger.debug(f"Sent video frame: {len(data)} bytes")
            except Exception as e:
                logger.error(f"Error sending video: {e}")

    def _on_audio_encoded(self, encoded_audio):
        """Handle encoded audio frame"""
        if self.audio_track and self.audio_track.is_open():
            try:
                data = encoded_audio.data.tobytes()
                self.audio_track.send(data)
                logger.debug(f"Sent audio frame: {len(data)} bytes")
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    async def send_frames(self, duration: Optional[int] = None):
        """Send test video and audio frames"""
        logger.info("Starting to send frames...")

        # Check if PeerConnection exists
        if not self.pc:
            raise RuntimeError("PeerConnection not initialized. Call connect() first.")

        # Wait for connection
        timeout = 10.0
        start_time = asyncio.get_event_loop().time()
        while self.pc.state() != PeerConnection.State.Connected:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise Exception("Connection timeout")
            await asyncio.sleep(0.1)

        logger.info("Connection established, sending frames")

        # Frame intervals
        video_interval = 1.0 / self.video_fps
        audio_interval = 0.02  # 20ms

        start_time = asyncio.get_event_loop().time()
        next_video_time = start_time
        next_audio_time = start_time

        try:
            while True:
                current_time = asyncio.get_event_loop().time()

                # Check duration
                if duration and current_time - start_time >= duration:
                    break

                # Send video frame
                if current_time >= next_video_time:
                    self._send_video_frame()
                    next_video_time += video_interval

                # Send audio frame
                if current_time >= next_audio_time:
                    self._send_audio_frame()
                    next_audio_time += audio_interval

                # Sleep until next frame
                next_time = min(next_video_time, next_audio_time)
                sleep_time = max(0, next_time - asyncio.get_event_loop().time())
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("Frame sending cancelled")
            raise

    def _send_video_frame(self):
        """Send a black video frame"""
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
        self.video_encoder.encode(frame)
        self.video_frame_number += 1

    def _send_audio_frame(self):
        """Send a silent audio frame"""
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

    async def disconnect(self):
        """Disconnect from WHIP server with graceful shutdown"""
        logger.info("Starting graceful shutdown...")

        # First, send DELETE request to WHIP server
        if self.session_url:
            logger.info("Sending DELETE request to WHIP server...")
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    headers = {}
                    if self.bearer_token:
                        headers["Authorization"] = f"Bearer {self.bearer_token}"

                    response = await client.delete(self.session_url, headers=headers)
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
        await asyncio.sleep(0.5)

        # Clean up resources in proper order
        logger.info("Cleaning up resources...")

        # Clear RTP components first
        self.video_packetizer = None
        self.audio_packetizer = None
        self.video_sr_reporter = None
        self.audio_sr_reporter = None
        logger.debug("RTP components cleared")

        # Close tracks before closing PeerConnection
        self.video_track = None
        self.audio_track = None
        logger.debug("Tracks cleared")

        # Close PeerConnection
        if self.pc:
            try:
                self.pc.close()
                logger.debug("PeerConnection closed")
            except Exception as e:
                logger.error(f"Error closing PeerConnection: {e}")
            finally:
                self.pc = None

        # Finally release encoders
        if self.video_encoder:
            try:
                self.video_encoder.release()
                logger.debug("Video encoder released")
            except Exception as e:
                logger.error(f"Error releasing video encoder: {e}")
            finally:
                self.video_encoder = None

        if self.audio_encoder:
            try:
                self.audio_encoder.release()
                logger.debug("Audio encoder released")
            except Exception as e:
                logger.error(f"Error releasing audio encoder: {e}")
            finally:
                self.audio_encoder = None

        logger.info("Graceful shutdown completed")
        self._cleaned_up = True


async def main():
    parser = argparse.ArgumentParser(description="Minimal WHIP client")
    parser.add_argument("--url", required=True, help="WHIP endpoint URL")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--duration", type=int, help="Duration in seconds")

    args = parser.parse_args()

    client = WHIPClient(args.url, args.token)

    # Flag to track if we're shutting down
    shutting_down = False

    # Set up signal handler for graceful shutdown
    loop = asyncio.get_event_loop()
    send_task = None

    def signal_handler(sig):
        nonlocal shutting_down
        if not shutting_down:
            shutting_down = True
            logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
            if send_task and not send_task.done():
                send_task.cancel()

    # Register signal handlers
    import signal

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    try:
        await client.connect()
        send_task = asyncio.create_task(client.send_frames(args.duration))
        await send_task
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except asyncio.CancelledError:
        logger.info("Task cancelled")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always disconnect gracefully
        logger.info("Ensuring disconnect is called...")
        try:
            await client.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

        # Remove signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main())
