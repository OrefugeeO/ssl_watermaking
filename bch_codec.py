"""
BCH codec utilities for watermarking error correction.

Provides BCH encoding/decoding for bit strings, with automatic scheme
selection based on message length, error rate tolerance, and encoded
bit budget.

Uses the **galois** library for BCH operations at bit-level.  Since
the upstream pipeline stores byte-aligned codewords, each *k*-bit
segment is rounded down to ``(k // 8) * 8`` effective data bits per
codeword and the ECC part is byte-aligned to ``ceil(m * t / 8)``
bytes (matching the convention previously supplied by *bchlib*).
"""

import math

import galois


# ---------------------------------------------------------------------------
# BCH parameter table: (n, k, t_design)
#   n = codeword length (bits)
#   k = data length per segment (bits)
#   t_design = designed error-correction capability (bits/codeword)
#
# The *actual* t is determined by galois at construction time and may
# differ slightly from t_design.
# ---------------------------------------------------------------------------
BCH_PARAMS = [
    # n=63 (primitive BCH codes with m=6)
    (63, 57, 1),
    (63, 51, 2),
    (63, 45, 3),
    (63, 39, 4),
    (63, 36, 5),
    (63, 30, 6),
    (63, 24, 7),
    (63, 18, 10),
    (63, 16, 11),
    (63, 10, 13),
    (63, 7, 15),
    # n=127 (primitive BCH codes with m=7)
    (127, 120, 1),
    (127, 113, 2),
    (127, 106, 3),
    (127, 99, 4),
    (127, 92, 5),
    (127, 85, 6),
    (127, 78, 7),
    (127, 71, 9),
    (127, 64, 10),
    (127, 57, 11),
    (127, 50, 13),
    (127, 43, 14),
    (127, 36, 15),
    (127, 29, 21),
    (127, 22, 23),
    (127, 15, 27),
    (127, 8, 31),
    # n=255 (primitive BCH codes with m=8)
    (255, 247, 1),
    (255, 239, 2),
    (255, 231, 3),
    (255, 223, 4),
    (255, 215, 5),
    (255, 207, 6),
    (255, 199, 7),
    (255, 191, 8),
    (255, 187, 9),
    (255, 179, 10),
    (255, 171, 11),
    (255, 163, 12),
    (255, 155, 13),
    (255, 147, 14),
    (255, 139, 15),
    (255, 131, 18),
    (255, 123, 19),
    (255, 115, 21),
    (255, 107, 22),
    (255, 99, 23),
    (255, 91, 25),
    (255, 87, 26),
    (255, 79, 27),
    (255, 71, 29),
    (255, 63, 30),
    (255, 55, 31),
    (255, 47, 42),
    (255, 45, 43),
    (255, 37, 45),
    (255, 29, 47),
    (255, 21, 55),
    (255, 13, 59),
    (255, 9, 63),
]
"""
(n, k, t_actual) tuples for BCH schemes, verified against galois.
n   — 码字长度（总比特数）
k   — 信息比特长度（原始数据）
t   — galois 验证的实际可纠正最大错误比特数

All entries are primitive BCH codes over GF(2^m):
  63  → m=6,  127 → m=7,  255 → m=8
"""


# ---------------------------------------------------------------------------
# Utility: bits <-> bytes
# ---------------------------------------------------------------------------
def bits_to_bytes(bitstring):
    """Convert a binary string (e.g. "01010101...") to bytes.

    The bitstring is zero-padded to a multiple of 8 bits if necessary.

    Args:
        bitstring: str of '0' and '1' characters.

    Returns:
        bytes object.
    """
    if len(bitstring) % 8 != 0:
        bitstring = bitstring + "0" * (8 - len(bitstring) % 8)
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i + 8], 2))
    return bytes(out)


def bytes_to_bits(data):
    """Convert bytes to a binary string.

    Args:
        data: bytes or bytearray.

    Returns:
        str of '0' and '1' characters.
    """
    return "".join(format(b, "08b") for b in data)


# ---------------------------------------------------------------------------
# Helpers: construct BCH object & compute byte-aligned dimensions
# ---------------------------------------------------------------------------
def _build_bch(n, k):
    """Construct a galois BCH object for given (n, k)."""
    return galois.BCH(n, k)


def _bch_dimensions(n, k, t_actual):
    """Return (data_byte_len, ecc_byte_len, effective_k_bits, codeword_bits).

    The byte-alignment convention mirrors the original bchlib behaviour:
    * data is rounded down to ``(k // 8) * 8`` bits,
    * ecc bytes = ``ceil(m * t / 8)`` where ``m = ceil(log2(n+1))``.
    """
    m = int(math.log2(n + 1))
    data_byte_len = k // 8
    ecc_byte_len = (m * t_actual + 7) // 8
    effective_k_bits = data_byte_len * 8
    codeword_bits = (data_byte_len + ecc_byte_len) * 8
    return data_byte_len, ecc_byte_len, effective_k_bits, codeword_bits


# ---------------------------------------------------------------------------
# Scheme selection
# ---------------------------------------------------------------------------
def select_bch_scheme(msg_len_bits, max_error_rate, max_encoded_bits):
    """Select the optimal BCH scheme from the built-in parameter table.

    Iterates through ``BCH_PARAMS`` looking for schemes that satisfy:

        - ``t_actual >= max_error_rate * actual_codeword_bits``
        - ``num_segments = ceil(msg_len_bits / effective_data_bits)``
        - ``total_encoded_bits = num_segments * actual_codeword_bits <= max_encoded_bits``

    where *effective_data_bits* = ``(k // 8) * 8`` and
    *actual_codeword_bits* = ``(k // 8 + ecc_bytes) * 8``.

    Among satisfying schemes the one with the smallest *total_encoded_bits*
    is returned.

    Since *galois* directly supports all (n, k) combinations in the table,
    no ``try/except`` fallback is needed (unlike *bchlib*).

    Args:
        msg_len_bits: Length of the raw message in bits.
        max_error_rate: Maximum expected bit-error rate (0 < rate < 1).
        max_encoded_bits: Maximum allowed encoded bit budget.

    Returns:
        tuple: ``(n, k, t_actual, num_segments, total_encoded_bits, bch_obj)``

    Raises:
        ValueError: If no scheme in the table meets the constraints.
    """
    candidates = []

    for n, k, t_design in BCH_PARAMS:
        bch_obj = _build_bch(n, k)
        t_actual = bch_obj.t

        data_byte_len, ecc_byte_len, effective_k_bits, actual_codeword_bits = (
            _bch_dimensions(n, k, t_actual)
        )

        # Skip schemes that are too small to carry any data
        if effective_k_bits == 0:
            continue

        # Constraint 1: error correction capability

        if t_actual < max_error_rate * actual_codeword_bits:
            continue
        # print(f"Evaluating BCH(n={n}, k={k}, t_design={t_design}, t_actual={t_actual}): \n"
        #     "       "
        #     f"effective_k={effective_k_bits} bits | "
        #     f"data_bytes={data_byte_len}, ecc_bytes={ecc_byte_len} | "
        #     f"actual_codeword={actual_codeword_bits} bits\n"
        #     "       "
        #     f"required_capability={max_error_rate * actual_codeword_bits} | "
        #     f"error_rate={t_actual / actual_codeword_bits:.4f}")

        # Constraint 2: number of segments
        num_segments = math.ceil(msg_len_bits / effective_k_bits)
        total_encoded_bits = num_segments * actual_codeword_bits

        # Constraint 3: encoded budget
        if total_encoded_bits > max_encoded_bits:
            continue
        # print(f"  Total encoded bits: {total_encoded_bits} (max: {max_encoded_bits})")
        
        candidates.append((n, k, t_actual, num_segments, total_encoded_bits, bch_obj))

    if not candidates:
        raise ValueError(
            "No BCH scheme satisfies the constraints "
            f"(msg_len={msg_len_bits}, max_err_rate={max_error_rate}, "
            f"max_encoded={max_encoded_bits})"
        )

    # Choose the scheme with smallest total_encoded_bits
    candidates.sort(key=lambda x: x[4])
    return candidates[0]


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def bch_encode(msg_bitstring, bch_scheme):
    """Encode a bit string using the given BCH scheme.

    The message is split into segments of ``(k // 8) * 8`` effective data
    bits.  If the last segment is incomplete it is zero-padded.  Each
    segment is BCH-encoded and the resulting codewords are concatenated.

    Internally, the function:
        1. pads each segment to *k* bits (zero-fill),
        2. calls ``bch_obj.encode()`` to obtain an *n*-bit codeword,
        3. stores the data bytes and the byte-aligned ECC bytes
           (using the same ``ceil(m*t/8)`` convention as the original
           *bchlib* pipeline).

    Args:
        msg_bitstring: Raw message as a string of '0'/'1' characters.
        bch_scheme: Tuple ``(n, k, t, num_segments, total_encoded_bits, bch_obj)``
            as returned by :func:`select_bch_scheme`.

    Returns:
        str: Encoded bit string (concatenated codewords).
    """
    n, k, t, num_segments, total_encoded_bits, bch_obj = bch_scheme

    data_byte_len, ecc_byte_len, effective_k_bits, codeword_bits_len = (
        _bch_dimensions(n, k, t)
    )

    # Pad the message to an exact multiple of effective_k_bits
    padded_len = num_segments * effective_k_bits
    if len(msg_bitstring) < padded_len:
        msg_bitstring = msg_bitstring + "0" * (padded_len - len(msg_bitstring))

    encoded_bits_list = []

    for seg_idx in range(num_segments):
        seg_bits = msg_bitstring[seg_idx * effective_k_bits : (seg_idx + 1) * effective_k_bits]

        # Build a GF(2) message vector of exactly k bits (zero-padded)
        bit_ints = [int(c) for c in seg_bits] + [0] * (k - effective_k_bits)
        msg_gf2 = galois.GF2(bit_ints)

        # Galois encodes k bits → n-bit systematic codeword
        codeword_gf2 = bch_obj.encode(msg_gf2)

        # --- data part: first effective_k_bits → data_byte_len bytes ----
        data_bits_str = ''.join(str(b) for b in codeword_gf2[:effective_k_bits].tolist())
        data_bytes = bits_to_bytes(data_bits_str)
        # Ensure exact length
        if len(data_bytes) < data_byte_len:
            data_bytes = data_bytes + b'\x00' * (data_byte_len - len(data_bytes))
        else:
            data_bytes = data_bytes[:data_byte_len]

        # --- ECC part: bits [k : n] → ecc_byte_len bytes ---------------
        ecc_bits_list = codeword_gf2[k:].tolist()
        ecc_bits_str = ''.join(str(b) for b in ecc_bits_list)
        ecc_bytes = bits_to_bytes(ecc_bits_str)
        # Pad / truncate to exact ecc_byte_len
        if len(ecc_bytes) < ecc_byte_len:
            ecc_bytes = ecc_bytes + b'\x00' * (ecc_byte_len - len(ecc_bytes))
        else:
            ecc_bytes = ecc_bytes[:ecc_byte_len]

        codeword_bits = bytes_to_bits(bytes(data_bytes)) + bytes_to_bits(ecc_bytes)
        encoded_bits_list.append(codeword_bits)

    return "".join(encoded_bits_list)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def bch_decode(encoded_bitstring, bch_scheme, original_msg_len):
    """Decode and error-correct a BCH-encoded bit string.

    The encoded bit string is split into byte-aligned codewords.
    Each codeword is decoded using :meth:`galois.BCH.decode` (which
    returns the corrected *k*-bit message in a single call).

    The recovered data segments are concatenated and truncated to the
    original message length.

    Args:
        encoded_bitstring: Bit string of concatenated BCH codewords.
        bch_scheme: Tuple ``(n, k, t, num_segments, total_encoded_bits, bch_obj)``
            as returned by :func:`select_bch_scheme`.
        original_msg_len: Length of the original raw message in bits.

    Returns:
        str: Recovered original bit string.
    """
    n, k, t, num_segments, total_encoded_bits, bch_obj = bch_scheme

    data_byte_len, ecc_byte_len, effective_k_bits, codeword_bits_len = (
        _bch_dimensions(n, k, t)
    )

    recovered_bits_list = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * codeword_bits_len
        seg_end = seg_start + codeword_bits_len
        seg_bits = encoded_bitstring[seg_start:seg_end]

        # Pad if the encoded string was truncated
        if len(seg_bits) < codeword_bits_len:
            seg_bits = seg_bits + "0" * (codeword_bits_len - len(seg_bits))

        codeword_bytes = bytearray(bits_to_bytes(seg_bits))

        if len(codeword_bytes) > data_byte_len + ecc_byte_len:
            codeword_bytes = codeword_bytes[:data_byte_len + ecc_byte_len]
        elif len(codeword_bytes) < data_byte_len + ecc_byte_len:
            codeword_bytes += b"\x00" * ((data_byte_len + ecc_byte_len) - len(codeword_bytes))

        data_part = codeword_bytes[:data_byte_len]
        ecc_part = codeword_bytes[data_byte_len:data_byte_len + ecc_byte_len]

        # Reconstruct the n-bit codeword:
        #   data bits (effective_k_bits) + zero-pad to k + ecc bits (first n-k of ecc_part bits)
        data_bits_str = bytes_to_bits(bytes(data_part))[:effective_k_bits]
        # Pad data to k bits
        data_bits_padded = data_bits_str + "0" * (k - effective_k_bits)

        ecc_bits_str = bytes_to_bits(bytes(ecc_part))
        n_minus_k = n - k
        ecc_bits_truncated = ecc_bits_str[:n_minus_k]
        if len(ecc_bits_truncated) < n_minus_k:
            ecc_bits_truncated = ecc_bits_truncated + "0" * (n_minus_k - len(ecc_bits_truncated))

        codeword_bits_str = data_bits_padded + ecc_bits_truncated
        # Ensure exactly n bits
        if len(codeword_bits_str) < n:
            codeword_bits_str = codeword_bits_str + "0" * (n - len(codeword_bits_str))
        codeword_bits_str = codeword_bits_str[:n]

        try:
            codeword_gf2 = galois.GF2([int(c) for c in codeword_bits_str])
            decoded_gf2 = bch_obj.decode(codeword_gf2)
            recovered_bits = ''.join(str(b) for b in decoded_gf2[:effective_k_bits].tolist())
        except Exception:
            # Decode raised an exception; fall back to raw uncorrected data
            recovered_bits = data_bits_str[:effective_k_bits]

        recovered_bits_list.append(recovered_bits)

    full_recovered = "".join(recovered_bits_list)
    return full_recovered[:original_msg_len]


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: BCH(63, 45, 3) with 90-bit message
    print("=== Test 1: BCH(63,45,3), 90-bit message ===")
    scheme = select_bch_scheme(
        msg_len_bits=90, max_error_rate=0.01, max_encoded_bits=512
    )
    n, k, t, num_seg, total_bits, bch_obj = scheme
    data_bytes, ecc_bytes, _, _ = _bch_dimensions(n, k, t)
    print(f"Selected: n={n}, k={k}, t={t}, "
          f"data_bytes={data_bytes}, ecc_bytes={ecc_bytes}, "
          f"segments={num_seg}, total_bits={total_bits}")

    msg = "1010101011" * 9  # 90 bits
    encoded = bch_encode(msg, scheme)
    print(f"Encoded length: {len(encoded)} bits")

    # Flip 1 bit in first codeword (t=3 can correct 3)
    corrupted = list(encoded)
    corrupted[3] = "0" if corrupted[3] == "1" else "1"
    corrupted = "".join(corrupted)

    decoded = bch_decode(corrupted, scheme, len(msg))
    bit_errors = sum(1 for a, b in zip(msg, decoded) if a != b)
    print(f"Bit errors after decoding (1 flipped bit): {bit_errors}")
    print(f"Match: {msg == decoded}")

    # Test 2: Flip 2 bits, should still be corrected (t=3)
    print("\n=== Test 2: 2 flipped bits ===")
    corrupted2 = list(encoded)
    corrupted2[3] = "0" if corrupted2[3] == "1" else "1"
    corrupted2[10] = "0" if corrupted2[10] == "1" else "1"
    corrupted2 = "".join(corrupted2)
    decoded2 = bch_decode(corrupted2, scheme, len(msg))
    bit_errors2 = sum(1 for a, b in zip(msg, decoded2) if a != b)
    print(f"Bit errors after decoding (2 flipped bits): {bit_errors2}")
    print(f"Match: {msg == decoded2}")

    # Test 3: Flip 4 bits (exceeds t=3), may have errors
    print("\n=== Test 3: 4 flipped bits (exceeds t=3) ===")
    corrupted3 = list(encoded)
    for pos in [3, 10, 20, 50]:
        corrupted3[pos] = "0" if corrupted3[pos] == "1" else "1"
    corrupted3 = "".join(corrupted3)
    decoded3 = bch_decode(corrupted3, scheme, len(msg))
    bit_errors3 = sum(1 for a, b in zip(msg, decoded3) if a != b)
    print(f"Bit errors after decoding (4 flipped bits): {bit_errors3}")
    print(f"Match: {msg == decoded3}")

    # Test 4: BCH(255, 131, 20) — galois reports t_actual=18
    print("\n=== Test 4: BCH(255,131,20), 256-bit message ===")
    scheme4 = select_bch_scheme(
        msg_len_bits=256, max_error_rate=0.05, max_encoded_bits=1024
    )
    n4, k4, t4, num_seg4, total_bits4, bch_obj4 = scheme4
    data_bytes4, ecc_bytes4, _, _ = _bch_dimensions(n4, k4, t4)
    print(f"Selected: n={n4}, k={k4}, t={t4}, "
          f"data_bytes={data_bytes4}, ecc_bytes={ecc_bytes4}, "
          f"segments={num_seg4}, total_bits={total_bits4}")

    msg4 = "01" * 128  # 256 bits
    encoded4 = bch_encode(msg4, scheme4)
    print(f"Encoded length: {len(encoded4)} bits")

    corrupted4 = list(encoded4)
    corrupted4[5] = "0" if corrupted4[5] == "1" else "1"
    corrupted4[100] = "0" if corrupted4[100] == "1" else "1"
    corrupted4[200] = "0" if corrupted4[200] == "1" else "1"
    corrupted4 = "".join(corrupted4)

    decoded4 = bch_decode(corrupted4, scheme4, len(msg4))
    bit_errors4 = sum(1 for a, b in zip(msg4, decoded4) if a != b)
    print(f"Bit errors after decoding: {bit_errors4}")
    print(f"Match: {msg4 == decoded4}")

    # Test 5: ValueError for impossible constraints
    print("\n=== Test 5: Impossible constraints ===")
    try:
        select_bch_scheme(
            msg_len_bits=10000, max_error_rate=0.9, max_encoded_bits=100
        )
    except ValueError as e:
        print(f"Correctly raised: {e}")