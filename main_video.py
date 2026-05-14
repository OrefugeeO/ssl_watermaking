"""
main_video.py — Video-level watermark encoding and tracing entry script.

Supports two modes:
    --mode encode : Read a video, apply a frame-level protocol
                    (n 0bit + m multibit frames per cycle),
                    embed watermark carriers, and output an encoded video.
    --mode trace  : Read an encoded video, auto-infer the key from 0bit frames,
                    then decode all subsequent frames as multibit and match
                    against the message list.

Dependencies: OpenCV (cv2), PIL, torch, numpy, pandas, and the local modules
              (utils, utils_img, encode, decode, evaluate, bch_codec,
               data_augmentation, video_utils).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import bch_codec
import decode
import galois
import utils
import video_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
#  ENCODE VIDEO
# ===========================================================================
def encode_video(params):
    """
    Main encoding pipeline:
      1. Read video frames
      2. Load models & carriers
      3. Read message list from msg_path, prepare slices per message
      4. Encode frame-by-frame according to (n_0bit, m_multibit) cycle protocol,
         cycling through the message list
      5. Write encoded video
    """
    # -------------------------------------------------------------------
    # Step 1: Read video
    # -------------------------------------------------------------------
    input_path = params.input_video_path
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    print(">>> Reading video from %s..." % input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError("Cannot open video: %s" % input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_pil = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))
    cap.release()

    T = len(frames_pil)
    print("  Video: %d frames, %dx%d, %.2f fps" % (T, width, height, fps))
    if T == 0:
        raise ValueError("No frames read from video.")

    # -------------------------------------------------------------------
    # Step 2: Read message list from msg_path (one message per line)
    # -------------------------------------------------------------------
    msg_path = params.msg_path
    if not os.path.exists(msg_path):
        raise FileNotFoundError("Message file not found: %s" % msg_path)
    with open(msg_path, "r", encoding="utf-8") as f:
        msg_list = [line.strip() for line in f if line.strip()]
    if not msg_list:
        raise ValueError("No messages found in %s" % msg_path)
    print(">>> Loaded %d messages from %s" % (len(msg_list), msg_path))

    # -------------------------------------------------------------------
    # Step 3: Load models
    # -------------------------------------------------------------------
    model_0bit = video_utils.load_model(
        params.model_path_0bit, params.model_name_0bit, params.normlayer_path_0bit,
        task_name="0bit",
    )
    model_multibit = video_utils.load_model(
        params.model_path_multibit, params.model_name_multibit,
        params.normlayer_path_multibit,
        task_name="multibit",
    )

    D_0bit = model_0bit(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)
    D_multibit = model_multibit(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)

    # -------------------------------------------------------------------
    # Step 4: Load / generate carriers and angle
    # -------------------------------------------------------------------
    key = params.key if params.key is not None else "test"

    carrier_0bit = video_utils.load_or_generate_carrier(
        key, 1, D_0bit, params.carrier_dir, model_0bit,
    )
    angle_0bit = utils.pvalue_angle(dim=D_0bit, k=1, proba=params.target_fpr_0bit)

    # Determine K_multibit from the first message
    first_msg = msg_list[0]
    K_multibit = video_utils.infer_num_bits(
        key, first_msg, params.carrier_dir,
        params.use_bch, params.max_error_rate, params.max_encoded_bits,
    )
    print(">>> [multibit] Inferred K_multibit = %d" % K_multibit)
    # Show original message lengths and the effect of layer-0 zero-padding
    msg_lengths = [len(utils.string_to_binary(m)) for m in msg_list]
    max_msg_len = max(msg_lengths)
    print(">>> [multibit] Message original bit lengths: %s (max=%d)"
          % (msg_lengths, max_msg_len))
    if params.use_bch:
        # First zero-padding: align all messages to max length (layer 0)
        # This is implied by the fact that all messages go through the same BCH scheme
        print(">>> [multibit] Layer-0 zero-pad: all messages aligned to %d bits" % max_msg_len)

    carrier_multibit = video_utils.load_or_generate_carrier(
        key, K_multibit, D_multibit, params.carrier_dir, model_multibit,
    )

    # -------------------------------------------------------------------
    # Step 5: Prepare messages — each frame embeds the FULL BCH encoded string
    # -------------------------------------------------------------------
    print(">>> [multibit] Preparing messages...")
    msg_encoded_bitstrings = []  # per-message full encoded bitstring (K_multibit bits)

    cached_bch_scheme = None  # reuse BCH scheme across messages (avoid repeated selection)
    for msg_idx, msg_text in enumerate(msg_list):
        _msg_slices, bch_scheme, raw_bitstring, encoded_bitstring = (
            video_utils.prepare_message(
                msg_text, key, K_multibit,
                params.use_bch, params.max_error_rate, params.max_encoded_bits,
                bch_scheme_override=cached_bch_scheme,
            )
        )
        if bch_scheme is not None and cached_bch_scheme is None:
            cached_bch_scheme = bch_scheme  # cache for subsequent messages

        msg_encoded_bitstrings.append(encoded_bitstring)

        # Show expansion chain for the first message
        if msg_idx == 0 and params.use_bch:
            n, k, t, num_seg, total_enc, _bch_obj = cached_bch_scheme
            effective_k = (k // 8) * 8
            print(
                ">>> [multibit] Expansion chain (first message): "
                "original=%d → layer0_pad=%d → layer1_effective_k=%d → "
                "layer2_k=%d → BCH_encoded=%d"
                % (len(raw_bitstring), max_msg_len, effective_k, k, len(encoded_bitstring))
            )
            if len(encoded_bitstring) < K_multibit:
                print(">>> [multibit] Layer-3 zero-pad: %d → %d bits"
                      % (len(encoded_bitstring), K_multibit))

    num_msgs = len(msg_encoded_bitstrings)
    print("  Prepared %d messages for frame-level round-robin embedding (K=%d)"
          % (num_msgs, K_multibit))

    # Write message records for all unique messages
    for msg_idx, msg_text in enumerate(msg_list):
        video_utils.write_message_records(
            key, msg_text, msg_encoded_bitstrings[msg_idx]
        )

    # -------------------------------------------------------------------
    # Step 6: Build per-frame encode parameters
    # -------------------------------------------------------------------
    class _Params0bit:
        pass

    p0 = _Params0bit()
    p0.optimizer = params.optimizer_0bit
    p0.scheduler = params.scheduler_0bit
    p0.epochs = params.epochs_0bit
    p0.target_psnr = params.target_psnr_0bit
    p0.lambda_w = params.lambda_w_0bit
    p0.lambda_i = params.lambda_i_0bit
    p0.verbose = params.verbose
    p0.batch_size = params.batch_size_0bit
    p0.data_augmentation = params.data_augmentation_0bit

    class _ParamsMultibit:
        pass

    p1 = _ParamsMultibit()
    p1.optimizer = params.optimizer_multibit
    p1.scheduler = params.scheduler_multibit
    p1.epochs = params.epochs_multibit
    p1.target_psnr = params.target_psnr_multibit
    p1.lambda_w = params.lambda_w_multibit
    p1.lambda_i = params.lambda_i_multibit
    p1.verbose = params.verbose
    p1.batch_size = params.batch_size_multibit
    p1.data_augmentation = params.data_augmentation_multibit

    # -------------------------------------------------------------------
    # Step 7: Encode frames (batch processing)
    # -------------------------------------------------------------------
    n_0bit = params.n_0bit
    m_multibit = params.m_multibit
    cycle_len = n_0bit + m_multibit

    print(
        ">>> Encoding %d frames (n_0bit=%d, m_multibit=%d, cycle_len=%d)..."
        % (T, n_0bit, m_multibit, cycle_len)
    )

    # Step 7a: Classify all frames — frame-level message round-robin
    from collections import defaultdict

    multibit_counter = 0
    frame_plan = []  # [(orig_idx, pil, "0bit"|"multibit", bit_str_or_None)]

    for i in range(T):
        cycle_index = i % cycle_len
        if cycle_index < n_0bit:
            frame_plan.append((i, frames_pil[i], "0bit", None))
        else:
            msg_idx = multibit_counter % num_msgs
            bit_str = msg_encoded_bitstrings[msg_idx]  # FULL BCH encoded string
            multibit_counter += 1
            frame_plan.append((i, frames_pil[i], "multibit", bit_str))

    # Step 7b: Gather and batch-encode (one model at a time to save GPU memory)
    encoded_map = {}  # orig_idx → encoded PIL

    # --- 0bit frames (release model_multibit first, not needed here) ---
    del model_multibit
    torch.cuda.empty_cache()

    bit0_items = [(idx, pil) for idx, pil, t, _ in frame_plan if t == "0bit"]
    if bit0_items:
        bit0_indices, bit0_pils = zip(*bit0_items)
        print(
            ">>> Batch-encoding %d 0bit frames (batch_size=%d)..."
            % (len(bit0_pils), p0.batch_size)
        )
        encoded_0bit = video_utils.encode_frame_batch(
            list(bit0_pils),
            carrier_0bit,
            model_0bit,
            p0,
            angle=angle_0bit,
            is_multibit=False,
        )
        for k, idx in enumerate(bit0_indices):
            encoded_map[idx] = encoded_0bit[k]

    # --- 0bit done: release model_0bit, reload model_multibit ---
    del model_0bit
    torch.cuda.empty_cache()

    model_multibit = video_utils.load_model(
        params.model_path_multibit, params.model_name_multibit,
        params.normlayer_path_multibit, task_name="multibit",
    )

    # --- multibit frames, grouped by bit_str ---
    multibit_groups = defaultdict(list)
    for idx, pil, _, bit_str in frame_plan:
        if bit_str is not None:
            multibit_groups[bit_str].append((idx, pil))

    for bit_str, group in multibit_groups.items():
        indices, pils = zip(*group)
        msg_tensors = [
            torch.tensor([[c == "1" for c in bit_str]], dtype=torch.bool).to(
                device, non_blocking=True
            )
            for _ in indices
        ]
        print(
            ">>> Batch-encoding %d multibit frames (msg=%s, batch_size=%d)..."
            % (
                len(pils),
                bit_str[:20] + ("..." if len(bit_str) > 20 else ""),
                p1.batch_size,
            )
        )
        encoded_batch = video_utils.encode_frame_batch(
            list(pils),
            carrier_multibit,
            model_multibit,
            p1,
            msg_tensors=msg_tensors,
            is_multibit=True,
        )
        for k, idx in enumerate(indices):
            encoded_map[idx] = encoded_batch[k]

    # Step 7c: Reassemble in original order
    encoded_frames = [encoded_map[i] for i in range(T)]

    # -------------------------------------------------------------------
    # Step 8: Write output video (with audio via ffmpeg)
    # -------------------------------------------------------------------
    import shutil
    import subprocess

    output_dir = os.path.join(params.output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "%s_encoded.mp4" % video_name)
    video_only_path = os.path.join(output_dir, "%s_video_only.mp4" % video_name)

    # 8a: Write pure video frames via OpenCV
    print(">>> Writing pure video to %s..." % video_only_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_only_path, fourcc, fps, (width, height))

    for pil_img in encoded_frames:
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    # 8b: Merge audio from source video using ffmpeg
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        print(">>> Merging audio with ffmpeg...")
        cmd = [
            ffmpeg_bin,
            "-i", video_only_path,
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            "-y",
            output_video_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(">>> Audio merged successfully. Output: %s" % output_video_path)
            # Remove the video-only temp file
            os.remove(video_only_path)
        except subprocess.CalledProcessError as e:
            print(
                "WARNING: ffmpeg failed, keeping video-only output.\n"
                "  stderr: %s" % (e.stderr.decode() if e.stderr else str(e))
            )
            # Fallback: rename video-only to final path
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(video_only_path, output_video_path)
    else:
        print("WARNING: ffmpeg not found on PATH. Output video will have no audio.")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        os.rename(video_only_path, output_video_path)
        print(">>> Output (no audio): %s" % output_video_path)

    print(">>> Encoding complete. Output: %s" % output_video_path)


# ===========================================================================
#  TRACE VIDEO
# ===========================================================================
def trace_video(params):
    """
    Main tracing pipeline — two-pass scan with BCH/hamming hybrid matching.

    Pass 1 (0bit voting):
      1. For every frame, decode_0bit against all available carriers.
      2. Accumulate key_votes[matched_key] for confidence > threshold matches.
      3. After scanning all frames:
         - best_key = key with most votes
         - if best_key votes / total_frames < trace_0bit_ratio_threshold:
           → no copyright detected, output empty CSV, return
         - else: locked_key = best_key

    Pass 2 (multibit tracing):
      1. Load carrier_multibit_<locked_key>.pth
      2. Load workspace/database/message/<locked_key>/records.csv
      3. For every frame:
         a. decode_multibit → decoded_bits
         b. Try BCH decode → recovered_bits (if bch_meta.json exists)
         c. Check exact match between recovered_bits and each encoded_bits in records
         d. If exact match → hamming_distance=0, is_exact_match=True
         e. If no exact match → hamming distance matching against all records
         f. Write one row per matched record (or at minimum, the best match)
    """
    # -------------------------------------------------------------------
    # Step 1: Read video
    # -------------------------------------------------------------------
    input_path = params.input_video_path
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    print(">>> [Pass 1] Reading video from %s..." % input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError("Cannot open video: %s" % input_path)

    frames_pil = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))
    cap.release()

    T = len(frames_pil)
    print("  Video: %d frames" % T)

    # -------------------------------------------------------------------
    # Step 2: Load models
    # -------------------------------------------------------------------
    model_0bit = video_utils.load_model(
        params.model_path_0bit, params.model_name_0bit, params.normlayer_path_0bit,
        task_name="0bit",
    )
    model_multibit = video_utils.load_model(
        params.model_path_multibit, params.model_name_multibit,
        params.normlayer_path_multibit,
        task_name="multibit",
    )

    D_0bit = model_0bit(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)
    angle_0bit = utils.pvalue_angle(dim=D_0bit, k=1, proba=params.target_fpr_0bit)

    # -------------------------------------------------------------------
    # Step 3: Parameters
    # -------------------------------------------------------------------
    confidence_threshold = params.trace_confidence_threshold
    ratio_threshold = params.trace_0bit_ratio_threshold
    similarity_threshold = params.trace_multibit_similarity_threshold
    carrier_dir = params.carrier_dir

    carrier_files_0bit = sorted(
        f for f in os.listdir(carrier_dir)
        if f.startswith("carrier_0bit_") and f.endswith(".pth")
    )

    # ===================================================================
    # PASS 1: 0bit voting — iterate ALL frames, accumulate key votes
    # ===================================================================
    print(">>> [Pass 1] 0bit voting across all %d frames (threshold=%.0f%%)..."
          % (T, ratio_threshold * 100))

    key_votes = {}

    for i in tqdm(range(T), desc="Pass 1: 0bit voting", unit="frame"):
        pil_img = frames_pil[i]
        best_confidence = 0.0
        best_carrier_name = None

        for cf in carrier_files_0bit:
            carrier_path = os.path.join(carrier_dir, cf)
            carrier = torch.load(carrier_path, map_location=device)
            carrier = carrier.to(device, non_blocking=True)

            try:
                decoded = decode.decode_0bit(
                    [pil_img], carrier, angle_0bit, model_0bit
                )
                confidence = 1 - 10 ** decoded[0]["log10_pvalue"]
            except Exception:
                confidence = 0.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_carrier_name = os.path.splitext(cf)[0]

        if best_confidence > confidence_threshold and best_carrier_name is not None:
            matched_key = best_carrier_name.replace("carrier_0bit_", "")
            key_votes[matched_key] = key_votes.get(matched_key, 0) + 1

    # --- Determine best key ---
    if not key_votes:
        print(">>> [Pass 1] No 0bit matches found. No copyright watermark detected.")
        _write_empty_trace_csv(params, video_name)
        return

    best_key = max(key_votes, key=key_votes.get)
    best_votes = key_votes[best_key]
    vote_ratio = best_votes / T

    print(">>> [Pass 1] 0bit voting results:")
    for k, v in sorted(key_votes.items(), key=lambda x: -x[1]):
        print("  %s: %d votes (%.1f%%)" % (k, v, v / T * 100))
    print("  Best key: %s (%d/%d = %.1f%%, threshold=%.1f%%)"
          % (best_key, best_votes, T, vote_ratio * 100, ratio_threshold * 100))

    if vote_ratio < ratio_threshold:
        print(">>> [Pass 1] Best key vote ratio (%.1f%%) below threshold (%.1f%%). "
              "No copyright watermark detected."
              % (vote_ratio * 100, ratio_threshold * 100))
        _write_empty_trace_csv(params, video_name)
        return

    locked_key = best_key
    print(">>> [Pass 1] Copyright confirmed: key = %s" % locked_key)

    # ===================================================================
    # PASS 2: multibit tracing — iterate ALL frames, hybrid matching
    # ===================================================================
    print(">>> [Pass 2] Multibit tracing with key=%s..." % locked_key)

    # --- Load records.csv ---
    records_path = os.path.join(
        "workspace", "database", "message", locked_key, "records.csv"
    )
    records_df = None
    if os.path.exists(records_path):
        records_df = pd.read_csv(records_path)
        print("  Loaded %d message records from %s" % (len(records_df), records_path))
    else:
        print("  WARNING: No records.csv found at %s; multibit matching disabled."
              % records_path)

    # --- Load BCH metadata if available ---
    bch_scheme = None
    original_msg_len = None
    bch_meta_path = os.path.join(
        "workspace", "database", "message", locked_key, "bch_meta.json"
    )
    if os.path.exists(bch_meta_path):
        try:
            with open(bch_meta_path, "r") as f:
                bch_meta = json.load(f)
            orig_bits = bch_meta["original_num_bits"]
            n, k, t = bch_meta["n"], bch_meta["k"], bch_meta["t"]
            num_seg = bch_meta["num_segments"]
            total_enc = bch_meta["total_encoded_bits"]
            bch_obj = galois.BCH(n, k)
            bch_scheme = (n, k, t, num_seg, total_enc, bch_obj)
            original_msg_len = orig_bits
            print("  Loaded BCH scheme from %s (original=%d bits, n=%d, k=%d, t=%d)"
                  % (bch_meta_path, orig_bits, n, k, t))
        except Exception as e:
            print("  WARNING: Failed to load BCH meta: %s" % e)
    else:
        print("  No BCH metadata found (bch_meta.json); using raw hamming matching only.")

    # --- Load multibit carrier ---
    carrier_multibit_path = os.path.join(
        carrier_dir, "carrier_multibit_%s.pth" % locked_key
    )
    if not os.path.exists(carrier_multibit_path):
        print("  WARNING: No multibit carrier at %s; multibit matching disabled."
              % carrier_multibit_path)
    else:
        carrier_multibit = torch.load(carrier_multibit_path, map_location=device)
        carrier_multibit = carrier_multibit.to(device, non_blocking=True)

    rows = []

    for i in tqdm(range(T), desc="Pass 2: multibit trace", unit="frame"):
        pil_img = frames_pil[i]

        decoded_bits_str = ""
        recovered_bits_str = ""
        matched_bits_str = ""
        raw_msg = ""
        hamming_dist = ""
        similarity = ""
        is_exact_match = False

        if not os.path.exists(carrier_multibit_path) or records_df is None:
            # No carrier or no records → write empty row
            pass
        else:
            try:
                decoded = decode.decode_multibit(
                    [pil_img], carrier_multibit, model_multibit
                )
                msg_tensor = decoded[0]["msg"]
                decoded_bits_str = "".join(
                    map(str, msg_tensor.type(torch.int).tolist())
                )

                # --- Step A: Attempt BCH decode ---
                if bch_scheme is not None and original_msg_len is not None:
                    try:
                        recovered_bits_str = bch_codec.bch_decode(
                            decoded_bits_str, bch_scheme, original_msg_len
                        )
                    except Exception:
                        recovered_bits_str = ""
                else:
                    recovered_bits_str = ""

                # --- Step B: Exact match check (BCH recovered vs encoded_bits) ---
                exact_match_record = None
                if recovered_bits_str:
                    for _, rec in records_df.iterrows():
                        candidate_bits = str(rec["encoded_bits"])
                        if recovered_bits_str == candidate_bits:
                            exact_match_record = rec
                            break

                if exact_match_record is not None:
                    # Exact match — BCH decode succeeded
                    hamming_dist = 0
                    min_len = min(len(decoded_bits_str),
                                  len(str(exact_match_record["encoded_bits"])))
                    similarity = 1.0
                    matched_bits_str = str(exact_match_record["encoded_bits"])
                    raw_msg = str(exact_match_record["raw_msg"])
                    is_exact_match = True
                else:
                    # --- Step C: Hamming distance matching (fallback) ---
                    K_decoded = len(decoded_bits_str)
                    best_dist = K_decoded + 1
                    best_rec = None

                    for _, rec in records_df.iterrows():
                        candidate_bits = str(rec["encoded_bits"])
                        min_len = min(K_decoded, len(candidate_bits))
                        dist = utils.compute_hamming_distance(
                            decoded_bits_str[:min_len], candidate_bits[:min_len]
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_rec = rec

                    if best_rec is not None:
                        total_bits = min(K_decoded, len(str(best_rec["encoded_bits"])))
                        hamming_dist = best_dist
                        _sim = 1.0 - best_dist / total_bits if total_bits > 0 else 0.0
                        if _sim >= similarity_threshold:
                            similarity = _sim
                            matched_bits_str = str(best_rec["encoded_bits"])
                            raw_msg = str(best_rec["raw_msg"])
                            is_exact_match = False

            except Exception as e:
                tqdm.write("  Frame %d: multibit decode failed: %s" % (i, e))

        rows.append({
            "video_name": video_name,
            "frame_index": i,
            "encode_type": "multibit",
            "key_used": locked_key,
            "decoded_bits": decoded_bits_str,
            "recovered_bits": recovered_bits_str,
            "matched_bits": matched_bits_str,
            "hamming_distance": hamming_dist,
            "similarity": similarity,
            "raw_msg": raw_msg,
            "is_exact_match": is_exact_match,
        })

    # ===================================================================
    # Step 4: Write trace results CSV
    # ===================================================================
    output_dir = os.path.join(params.output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "trace_results.csv")

    df = pd.DataFrame(rows)
    columns_order = [
        "video_name", "frame_index", "encode_type",
        "key_used",
        "decoded_bits", "recovered_bits", "matched_bits",
        "hamming_distance", "similarity",
        "raw_msg", "is_exact_match",
    ]
    # Ensure all expected columns exist, fill missing with empty string
    for col in columns_order:
        if col not in df.columns:
            df[col] = ""
    df = df[columns_order]
    df.to_csv(output_csv, index=False)
    print(">>> Trace results saved to %s" % output_csv)
    print(
        "  Total frames: %d, exact matches: %d, hamming matches: %d"
        % (
            T,
            sum(1 for r in rows if r.get("is_exact_match") is True),
            sum(1 for r in rows
                if r.get("is_exact_match") is False
                and r.get("hamming_distance") != ""),
        )
    )


def _write_empty_trace_csv(params, video_name):
    """Write an empty trace_results.csv when no copyright is detected."""
    output_dir = os.path.join(params.output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "trace_results.csv")

    columns = [
        "video_name", "frame_index", "encode_type",
        "key_used",
        "decoded_bits", "recovered_bits", "matched_bits",
        "hamming_distance", "similarity",
        "raw_msg", "is_exact_match",
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(output_csv, index=False)
    print(">>> Empty trace results saved to %s (no copyright detected)" % output_csv)


# ===========================================================================
#  ARGPARSE
# ===========================================================================
def get_parser():
    parser = argparse.ArgumentParser(
        description="Video-level watermark encoding and tracing"
    )

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    # ---- 基础 / 模式参数 ----
    group = parser.add_argument_group("基础参数 (Basic parameters)")
    aa("--mode", type=str, choices=["encode", "trace"], required=True,
       help="[encode+trace] 运行模式: encode(编码) 或 trace(溯源)")
    aa("--input_video_path", type=str, required=True,
       help="[encode+trace] 输入视频文件路径")
    aa("--output_dir", type=str, default="workspace/output/video_output/",
       help="[encode+trace] 输出根目录 (默认: workspace/output/video_output/)")
    aa("--verbose", type=int, default=1,
       help="[encode+trace] 日志详细级别 (0: 静默, 1: 进度, 2/3: 详细优化日志)")

    # ---- 帧序协议参数 (encode only) ----
    group = parser.add_argument_group("帧序协议参数 (Frame protocol parameters)")
    aa("--n_0bit", type=int, default=2,
       help="[encode only] 每个周期中 0bit 帧的数量")
    aa("--m_multibit", type=int, default=3,
       help="[encode only] 每个周期中 multibit 帧的数量")

    # ---- Key / carrier / 溯源 ----
    group = parser.add_argument_group("Key 与载体参数 (Key and carrier parameters)")
    aa("--key", type=str, default=None,
       help="[encode only] 用于确定性载波生成和消息数据库索引的 key 字符串（若未指定则使用 'test'）")
    aa("--carrier_dir", type=str, default="workspace/database/carriers/",
       help="[encode+trace] 载体文件目录 (默认: workspace/database/carriers/)")
    aa("--trace_confidence_threshold", type=float, default=0.95,
       help="[trace only] 0bit 检测置信度阈值，高于此值认为匹配成功 (默认: 0.95)")
    aa("--trace_0bit_ratio_threshold", type=float, default=0.30,
       help="[trace only] 第一遍扫描后，最高匹配 key 的投票数占总帧数比例阈值，低于此值认为无版权水印 (默认: 0.30)")
    aa("--trace_multibit_similarity_threshold", type=float, default=0.60,
       help="[trace only] Pass 2 multibit 匹配时相似度的最低门限，低于此值的匹配结果将被丢弃 (默认: 0.60)")

    # ---- 消息参数 (encode only) ----
    group = parser.add_argument_group("消息参数 (Message parameters)")
    aa("--msg_path", type=str, default="workspace/input/input_messages/msgs.txt",
       help="[encode only] 消息文本文件路径 (每行一条消息) (默认: workspace/input/input_messages/msgs.txt)")
    aa("--msg_type", type=str, default="text", choices=["text", "bit"],
       help="[encode only] 输入消息类型: text(文本字符串) 或 bit(二进制字符串) (默认: text)")
    aa("--output_msg_type", type=str, default="bit", choices=["text", "bit"],
       help="[encode only] 输出消息类型: text(文本字符串) 或 bit(二进制字符串) (默认: bit)")
    aa("--use_bch", type=utils.bool_inst, default=False,
       help="[encode only] 是否使用 BCH 纠错编码 (默认: False)")
    aa("--max_error_rate", type=float, default=0.05,
       help="[encode only] BCH 编码容忍的最大比特错误率 (默认: 0.05)")
    aa("--max_encoded_bits", type=int, default=256,
       help="[encode only] BCH 编码后允许的最大比特数预算 (默认: 256)")

    # ---- 0bit 编码参数 ----
    group = parser.add_argument_group("0bit 编码参数 (0bit encoding parameters)")
    aa("--target_psnr_0bit", type=float, default=42.0,
       help="[encode only] 0bit 编码目标 PSNR 值，控制水印不可见性 (默认: 42.0)")
    aa("--target_fpr_0bit", type=float, default=1e-6,
       help="[encode+trace] 0bit 目标误报率 (False Positive Rate)，决定超锥角度 (默认: 1e-6)")
    aa("--model_name_0bit", type=str, default="resnet50",
       help="[encode+trace] 0bit backbone 模型名称 (默认: resnet50)")
    aa("--model_path_0bit", type=str, default="models/dino_r50_plus.pth",
       help="[encode+trace] 0bit backbone 模型权重路径 (默认: models/dino_r50_plus.pth)")
    aa("--normlayer_path_0bit", type=str, default="normlayers/out2048_coco_resized.pth",
       help="[encode+trace] 0bit 归一化层权重路径 (默认: normlayers/out2048_coco_resized.pth)")
    aa("--epochs_0bit", type=int, default=100,
       help="[encode only] 0bit 编码每帧优化迭代次数 (默认: 100)")
    aa("--data_augmentation_0bit", type=str, default="all", choices=["none", "all"],
       help="[encode only] 0bit 编码数据增强模式: none(仅基本变换) 或 all(含高级增强) (默认: all)")
    aa("--optimizer_0bit", type=str, default="Adam,lr=0.01",
       help="[encode only] 0bit 编码优化器，格式: 名称,key=value,... (默认: Adam,lr=0.01)")
    aa("--scheduler_0bit", type=str, default=None,
       help="[encode only] 0bit 编码学习率调度器，格式: 名称,key=value,... (默认: None)")
    aa("--batch_size_0bit", type=int, default=1,
       help="[encode only] 0bit 编码批次大小 (默认: 1)")
    aa("--lambda_w_0bit", type=float, default=1.0,
       help="[encode only] 0bit 水印损失权重 (默认: 1.0)")
    aa("--lambda_i_0bit", type=float, default=1.0,
       help="[encode only] 0bit 图像重建损失权重 (默认: 1.0)")

    # ---- multibit 编码参数 ----
    group = parser.add_argument_group("Multibit 编码参数 (Multibit encoding parameters)")
    aa("--target_psnr_multibit", type=float, default=42.0,
       help="[encode only] Multibit 编码目标 PSNR 值 (默认: 42.0)")
    aa("--target_fpr_multibit", type=float, default=1e-6,
       help="[encode only] Multibit 目标误报率（用于记录，multibit 使用 message_loss） (默认: 1e-6)")
    aa("--model_name_multibit", type=str, default="resnet50",
       help="[encode+trace] Multibit backbone 模型名称 (默认: resnet50)")
    aa("--model_path_multibit", type=str, default="models/dino_r50_plus.pth",
       help="[encode+trace] Multibit backbone 模型权重路径 (默认: models/dino_r50_plus.pth)")
    aa("--normlayer_path_multibit", type=str, default="normlayers/out2048_yfcc_resized.pth",
       help="[encode+trace] Multibit 归一化层权重路径 (默认: normlayers/out2048_yfcc_resized.pth)")
    aa("--epochs_multibit", type=int, default=200,
       help="[encode only] Multibit 编码每帧优化迭代次数 (默认: 200)")
    aa("--data_augmentation_multibit", type=str, default="all",
       choices=["none", "all"],
       help="[encode only] Multibit 编码数据增强模式: none(仅基本变换) 或 all(含高级增强) (默认: all)")
    aa("--optimizer_multibit", type=str, default="Adam,lr=0.01",
       help="[encode only] Multibit 编码优化器，格式: 名称,key=value,... (默认: Adam,lr=0.01)")
    aa("--scheduler_multibit", type=str, default=None,
       help="[encode only] Multibit 编码学习率调度器，格式: 名称,key=value,... (默认: None)")
    aa("--batch_size_multibit", type=int, default=1,
       help="[encode only] Multibit 编码批次大小 (默认: 1)")
    aa("--lambda_w_multibit", type=float, default=5e4,
       help="[encode only] Multibit 水印损失权重 (默认: 5e4)")
    aa("--lambda_i_multibit", type=float, default=1.0,
       help="[encode only] Multibit 图像重建损失权重 (默认: 1.0)")

    return parser


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    parser = get_parser()
    params = parser.parse_args()

    print(">>> Using device: %s" % ("GPU (%s)" % torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    torch.manual_seed(0)
    np.random.seed(0)

    if params.mode == "encode":
        if params.n_0bit is None or params.m_multibit is None:
            print(
                "ERROR: --n_0bit and --m_multibit are required for encode mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        encode_video(params)
    elif params.mode == "trace":
        trace_video(params)
    else:
        print("ERROR: Unknown mode '%s'. Use 'encode' or 'trace'." % params.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()