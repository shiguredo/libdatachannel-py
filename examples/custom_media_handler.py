"""
Custom MediaHandler implementation for proper RTP depacketization in Python
"""
import logging
from typing import Callable, List
from libdatachannel import MediaHandler, Message

logger = logging.getLogger(__name__)


class CallbackMediaHandler(MediaHandler):
    """
    A MediaHandler that forwards processed messages to a Python callback.
    This allows us to capture the output of RTP depacketizers.
    """
    
    def __init__(self, callback: Callable[[bytes], None]):
        super().__init__()
        self.callback = callback
        self._message_count = 0
    
    def incoming(self, messages: List[Message], send: Callable) -> None:
        """Process incoming messages and forward to callback"""
        # Process each message
        for message in messages:
            self._message_count += 1
            
            # Get message data as bytes
            if hasattr(message, 'data'):
                data = bytes(message.data())
            else:
                # If message is already bytes-like
                data = bytes(message)
            
            # Call the Python callback with the processed data
            self.callback(data)
            
            if self._message_count <= 5:
                logger.debug(f"CallbackMediaHandler: Forwarded message #{self._message_count}, size={len(data)} bytes")
        
        # Don't forward to next handler (we're the end of the chain)
        # If you want to continue the chain, call: super().incoming(messages, send)
    
    def outgoing(self, messages: List[Message], send: Callable) -> None:
        """Pass through outgoing messages"""
        # For outgoing, just pass through
        super().outgoing(messages, send)


class RtpDepacketizingHandler(MediaHandler):
    """
    A MediaHandler that manually depacketizes RTP packets.
    This is a workaround for the current binding limitations.
    """
    
    def __init__(self, payload_callback: Callable[[bytes, int, int], None]):
        super().__init__()
        self.payload_callback = payload_callback
        self._packet_count = 0
    
    def incoming(self, messages: List[Message], send: Callable) -> None:
        """Extract RTP payload and forward to callback"""
        result_messages = []
        
        for message in messages:
            self._packet_count += 1
            
            # Get message data
            if hasattr(message, 'data'):
                data = bytes(message.data())
            else:
                data = bytes(message)
            
            # Check if this is an RTP packet
            if len(data) >= 12 and (data[0] >> 6) == 2:  # RTP version 2
                # Extract RTP header info
                pt = data[1] & 0x7F
                seq = (data[2] << 8) | data[3]
                timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                
                # Calculate header size
                header_size = 12
                cc = data[0] & 0x0F  # CSRC count
                header_size += cc * 4
                
                # Check for header extension
                if data[0] & 0x10:  # Extension bit
                    if len(data) >= header_size + 4:
                        ext_length = ((data[header_size + 2] << 8) | data[header_size + 3]) * 4
                        header_size += 4 + ext_length
                
                # Extract payload
                if header_size < len(data):
                    payload = data[header_size:]
                    
                    if self._packet_count <= 5:
                        logger.debug(f"RtpDepacketizingHandler: Packet #{self._packet_count} - PT={pt}, payload size={len(payload)} bytes")
                    
                    # Call the payload callback
                    self.payload_callback(payload, pt, timestamp)
                    
                    # Create a new message with just the payload
                    # This would be passed to the next handler in the chain
                    # For now, we're just using the callback
            else:
                # Not RTP, pass through
                if self._packet_count <= 5:
                    logger.debug(f"RtpDepacketizingHandler: Non-RTP message #{self._packet_count}")
        
        # Continue the chain with depacketized messages
        # super().incoming(result_messages, send)