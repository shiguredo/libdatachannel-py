import logging
import re
from typing import List

from libdatachannel import IceServer

logger = logging.getLogger(__name__)


def handle_error(context: str, error: Exception):
    """Unified error handling

    Args:
        context: Description of where the error occurred
        error: The exception that was raised
    """
    logger.error(f"Error {context}: {error}")
    if logger.isEnabledFor(logging.DEBUG):
        import traceback

        traceback.print_exc()


def get_nal_type_name(nal_type: int) -> str:
    """Get human-readable name for NAL unit type

    Args:
        nal_type: NAL unit type value

    Returns:
        Human-readable name of the NAL unit type
    """
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


def parse_link_header(link_header: str) -> List[IceServer]:
    """Parse Link header for ICE servers

    Args:
        link_header: Link header string from HTTP response

    Returns:
        List of IceServer objects parsed from the header
    """
    ice_servers: List[IceServer] = []
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
