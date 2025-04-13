#!/usr/bin/env python3
import hashlib
import base64

#########################
# Varint encoding (unsigned)
#########################
def encode_varint(i: int) -> bytes:
    """Encode an integer as a varint."""
    output = bytearray()
    while True:
        byte = i & 0x7F
        i >>= 7
        if i:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            break
    return bytes(output)

#########################
# Minimal CBOR encoding helpers
#########################
def cbor_encode_int(n: int) -> bytes:
    """Encode a small positive integer in CBOR (only supports n < 256)."""
    if n < 24:
        return bytes([n])
    elif n < 256:
        return bytes([0x18, n])
    else:
        raise NotImplementedError("Integer too large")

def cbor_encode_text(s: str) -> bytes:
    """Encode a text string in CBOR (only supports strings with length < 256)."""
    b = s.encode('utf-8')
    length = len(b)
    if length < 24:
        return bytes([0x60 + length]) + b
    elif length < 256:
        return bytes([0x78, length]) + b
    else:
        raise NotImplementedError("Text string too long")

def cbor_encode_bytes(b: bytes) -> bytes:
    """Encode a byte string in CBOR (only supports lengths < 256)."""
    length = len(b)
    if length < 24:
        return bytes([0x40 + length]) + b
    elif length < 256:
        return bytes([0x58, length]) + b
    else:
        raise NotImplementedError("Byte string too long")

def cbor_encode_array(items: list) -> bytes:
    """Encode a CBOR array (only supports arrays with less than 24 items)."""
    length = len(items)
    if length < 24:
        header = bytes([0x80 + length])
    elif length < 256:
        header = bytes([0x98, length])
    else:
        raise NotImplementedError("Array too long")
    return header + b''.join(items)

def cbor_encode_map(d: dict) -> bytes:
    """Encode a CBOR map (only supports maps with less than 24 pairs).
       The keys here are assumed to be text strings (already encoded via cbor_encode_text).
       The values must be pre‐encoded as bytes.
    """
    n = len(d)
    if n < 24:
        header = bytes([0xA0 + n])
    elif n < 256:
        header = bytes([0xB8, n])
    else:
        raise NotImplementedError("Map too large")
    encoded = header
    # Iterating in insertion order (keys are text strings)
    for key, value in d.items():
        encoded += cbor_encode_text(key) + value
    return encoded

def cbor_encode_tag(tag: int, content: bytes) -> bytes:
    """Encode a tagged value in CBOR.
       For tag 42 (used for CID in IPLD dag-cbor), the encoding is:
         - if tag < 24: one byte; else if tag < 256: 0xD8 plus one byte tag.
    """
    if tag < 24:
        tag_prefix = bytes([0xC0 + tag])
    elif tag < 256:
        tag_prefix = bytes([0xD8, tag])
    else:
        raise NotImplementedError("Tag too large")
    return tag_prefix + content

#########################
# CID generation functions
#########################
def generate_cid(data: bytes) -> bytes:
    """
    Create a CIDv1 (raw block) from the given data.
    Uses:
      - CID version: 1 (0x01)
      - Codec: raw (multicodec 0x55)
      - Multihash: sha256 (0x12) with 32-byte digest (0x20)
    The binary structure is:
      cid = <cid_version><codec><multihash>
    """
    digest = hashlib.sha256(data).digest()
    multihash = bytes([0x12, 0x20]) + digest  # sha256 code and length followed by digest
    cid = bytes([0x01, 0x55]) + multihash
    return cid

def cid_to_string(cid: bytes) -> str:
    """
    Convert the CID binary to its multibase base32-encoded string representation.
    The expected format for CIDv1 raw blocks in IPFS is a lower-case base32
    string prefixed with "b".
    """
    # Base32-encode, then remove any '=' padding and convert to lowercase.
    b32 = base64.b32encode(cid).decode('ascii').lower().rstrip('=')
    return 'b' + b32

#########################
# CAR file generation function
#########################
def generate_car(text: str) -> bytes:
    """
    Generate a CAR (Content Addressable aRchive) file as bytes from a text string.
    The resulting CAR file includes:
      - A header block encoded in CBOR (with version 1 and roots set to our only CID)
      - A single block (the CID followed by the raw data)
    No directory wrapping is done.
    """
    # Convert text into bytes (the file data)
    data = text.encode('utf-8')
    # Compute the CID (a CIDv1 for raw data)
    cid = generate_cid(data)

    # Build the header.
    # According to the CAR v1 spec the header is a CBOR map:
    #   {
    #     "version": 1,
    #     "roots": [ <CID> ]
    #   }
    # When encoding a CID in dag-cbor, it should be wrapped as a tagged bytes value with tag 42.
    cid_cbor = cbor_encode_tag(42, cbor_encode_bytes(cid))
    header_obj = {
        "version": cbor_encode_int(1),
        "roots": cbor_encode_array([cid_cbor])
    }
    header_cbor = cbor_encode_map(header_obj)
    # Prepend the header with its varint-encoded length
    header_length = encode_varint(len(header_cbor))

    # Build the block: block bytes consist of the CID followed by the raw file data.
    block = cid + data
    block_length = encode_varint(len(block))

    # Final CAR file is: <varint(len(header_cbor))><header_cbor><varint(len(block))><block>
    return header_length + header_cbor + block_length + block

#########################
# Test the implementation
#########################
if __name__ == "__main__":
    # The text to embed (with a newline)
    text = "My name is Matt."
    car_bytes = generate_car(text)
    
    # Compute the CID string for the data
    cid = generate_cid(text.encode('utf-8'))
    cid_str = cid_to_string(cid)
    
    # Expected CID as given in the test
    expected_cid = "bafkreiehm2wufzc2krgdkox6fvpixxw2ncvmvqwilvqbeqqy2x4g5wkewm"
    
    print("Computed CID:", cid_str)
    print("Matches expected:", cid_str == expected_cid)
    
    # Optional: Write the CAR file to disk if needed:
    # with open("output.car", "wb") as f:
    #     f.write(car_bytes)
