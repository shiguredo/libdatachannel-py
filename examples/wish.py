"""Common utilities for WHIP/WHEP clients."""

import re
from typing import List

from libdatachannel import IceServer


def parse_link_header(link_header: str) -> List[IceServer]:
    """Parse Link header for ICE servers
    
    Args:
        link_header: Link header string from HTTP response
        
    Returns:
        List of IceServer objects parsed from the header
    """
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

    return ice_servers