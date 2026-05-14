"""
video_utils.py — Shared helpers for video-level watermark encoding and tracing.

Contains:
    - MultiImageDataset: wraps multiple PIL images for batched encoding
    - infer_num_bits: determine K for multibit carrier
    - load_or_generate_carrier: load / generate a carrier file
    - prepare_message: text → bits → (optional BCH) → slices
    - load_model: backbone + normlayer → NormLayerWrapper
    - encode_frame_batch: watermark multiple PIL images in one forward pass
"""

import json
import math
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

import bch_codec
import data_augmentation
import decode
import encode
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiImageDataset(Dataset):
    """Wraps multiple PIL images for batched encoding."""

    def __init__(self, pil_imgs):
        self.pil_imgs = pil_imgs
        self.transform = utils_img.default_transform

    def __len__(self):
        return len(self.pil_imgs)

    def __getitem__(self, idx):
        return self.transform(self.pil_imgs[idx]), 0


# ---------------------------------------------------------------------------
# K / num_bits inference
# ---------------------------------------------------------------------------
def infer_num_bits(key, msg_text, carrier_dir, use_bch, max_error_rate, max_encoded_bits):
    """
    Determine K (number of bits per multibit frame).

    Priority order (updated: BCH width first, so each frame embeds the FULL message):
        1. BCH meta (bch_meta.json)   → total_encoded_bits
        2. BCH scheme selection       → total_encoded_bits
        3. Existing multibit carrier  → carrier.shape[0]
        4. records.csv                → len(first encoded_bits)
        5. Message text length        → len(string_to_binary(msg_text))
    """
    # 1. Try bch_meta.json
    bch_meta_path = os.path.join("workspace", "database", "message", key, "bch_meta.json")
    if os.path.exists(bch_meta_path) and use_bch:
        with open(bch_meta_path, "r") as f:
            bch_meta = json.load(f)
        return bch_meta["total_encoded_bits"]

    # 2. Compute via BCH scheme selection
    if use_bch:
        original_bits = len(utils.string_to_binary(msg_text))
        max_enc = max_encoded_bits if max_encoded_bits is not None else original_bits * 10
        scheme = bch_codec.select_bch_scheme(original_bits, max_error_rate, max_enc)
        return scheme[4]  # total_encoded_bits

    # 3. Try existing carrier
    carrier_path = os.path.join(carrier_dir, "carrier_multibit_%s.pth" % key)
    if os.path.exists(carrier_path):
        carrier = torch.load(carrier_path, map_location="cpu")
        return carrier.shape[0]

    # 4. Try records.csv
    records_path = os.path.join("workspace", "database", "message", key, "records.csv")
    if os.path.exists(records_path):
        records_df = pd.read_csv(records_path)
        if not records_df.empty:
            encoded = str(records_df.iloc[0]["encoded_bits"])
            return len(encoded)

    # 5. Fallback: message text length
    return len(utils.string_to_binary(msg_text))


# ---------------------------------------------------------------------------
# Carrier loading / generation
# ---------------------------------------------------------------------------
def load_or_generate_carrier(key, K, D, carrier_dir, model):
    """
    Load existing carrier or generate a new one deterministically from *key*.

    Args:
        key: Key string.
        K: Number of carrier vectors (1 for 0bit, N for multibit).
        D: Feature dimension (computed from model if None).
        carrier_dir: Directory for carrier files.
        model: Wrapped backbone (used to infer D when needed).

    Returns:
        Carrier tensor (K x D), on *device*.
    """
    os.makedirs(carrier_dir, exist_ok=True)

    fname = "carrier_0bit_%s.pth" % key if K == 1 else "carrier_multibit_%s.pth" % key
    carrier_path = os.path.join(carrier_dir, fname)

    if D is None:
        D = model(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)

    if os.path.exists(carrier_path):
        print(">>> Loading carrier from %s" % carrier_path)
        carrier = torch.load(carrier_path, map_location=device)
        if D is not None:
            assert D == carrier.shape[1], (
                "Carrier dimension mismatch: expected %d, got %d"
                % (D, carrier.shape[1])
            )
    else:
        print(">>> Generating carrier into %s..." % carrier_path)
        if key is not None:
            carrier = utils.generate_carriers_with_key(
                K, D, key, output_fpath=carrier_path
            )
        else:
            carrier = utils.generate_carriers(K, D, output_fpath=carrier_path)

    return carrier.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Message preparation (text → bits → BCH → slices)
# ---------------------------------------------------------------------------
def prepare_message(msg_text, key, K_multibit, use_bch, max_error_rate, max_encoded_bits, bch_scheme_override=None):
    """
    Prepare the message for embedding across multiple multibit frames.

    Args:
        msg_text: The message text string.
        key: Key string for BCH meta persistence.
        K_multibit: Bits per multibit frame (carrier dimension).
        use_bch: Whether to use BCH encoding.
        max_error_rate: Maximum acceptable bit error rate for BCH selection.
        max_encoded_bits: Maximum encoded bits budget.
        bch_scheme_override: If provided, reuse this BCH scheme tuple instead of
                             loading / selecting again.  (n, k, t, num_seg, total_enc, bch_obj)

    Returns:
        slices: list of bit strings, each of length K_multibit.
        bch_scheme: BCH scheme tuple, or None.
        raw_bitstring: original bit string (pre-BCH).
        encoded_bitstring: final bit string (post-BCH, or same as raw).
    """
    raw_bitstring = utils.string_to_binary(msg_text)
    original_num_bits = len(raw_bitstring)

    bch_scheme = None
    encoded_bitstring = raw_bitstring

    if use_bch:
        if bch_scheme_override is not None:
            # Reuse previously selected BCH scheme — skip BCH selection & meta save
            bch_scheme = bch_scheme_override
            n, k, t, num_seg, total_enc, bch_obj = bch_scheme
            encoded_bitstring = bch_codec.bch_encode(raw_bitstring, bch_scheme)
        else:
            max_enc = (
                max_encoded_bits
                if max_encoded_bits is not None
                else original_num_bits * 10
            )

            # Try loading existing BCH meta
            bch_meta_path = os.path.join(
                "workspace", "database", "message", key, "bch_meta.json"
            )
            if os.path.exists(bch_meta_path):
                with open(bch_meta_path, "r") as f:
                    bch_meta = json.load(f)
                bch_obj = bch_codec._build_bch(bch_meta["n"], bch_meta["k"])
                bch_scheme = (
                    bch_meta["n"],
                    bch_meta["k"],
                    bch_meta["t"],
                    bch_meta["num_segments"],
                    bch_meta["total_encoded_bits"],
                    bch_obj,
                )
                print(">>> [multibit] Loaded BCH scheme from %s" % bch_meta_path)
            else:
                print(
                    ">>> [multibit] Selecting BCH scheme (original=%d bits, max_err_rate=%.3f, max_enc=%d)..."
                    % (original_num_bits, max_error_rate, max_enc)
                )
                bch_scheme = bch_codec.select_bch_scheme(
                    original_num_bits, max_error_rate, max_enc
                )

            n, k, t, num_seg, total_enc, bch_obj = bch_scheme
            print(
                "  Selected BCH(%d,%d,%d): %d segments, %d encoded bits"
                % (n, k, t, num_seg, total_enc)
            )

            encoded_bitstring = bch_codec.bch_encode(raw_bitstring, bch_scheme)

            # Persist BCH metadata
            bch_meta_dir = os.path.join("workspace", "database", "message", key)
            os.makedirs(bch_meta_dir, exist_ok=True)
            bch_meta_path = os.path.join(bch_meta_dir, "bch_meta.json")
            with open(bch_meta_path, "w") as f:
                json.dump(
                    {
                        "original_num_bits": original_num_bits,
                        "n": n,
                        "k": k,
                        "t": t,
                        "num_segments": num_seg,
                        "total_encoded_bits": total_enc,
                    },
                    f,
                )
            print("  BCH metadata saved to %s" % bch_meta_path)

    # Slice encoded_bitstring into chunks of K_multibit bits
    total_len = len(encoded_bitstring)
    num_slices = math.ceil(total_len / K_multibit)
    slices = []
    for i in range(num_slices):
        start = i * K_multibit
        end = min(start + K_multibit, total_len)
        chunk = encoded_bitstring[start:end]
        if len(chunk) < K_multibit:
            chunk = chunk + "0" * (K_multibit - len(chunk))
        slices.append(chunk)

    return slices, bch_scheme, raw_bitstring, encoded_bitstring


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(model_path, model_name, normlayer_path, task_name="unknown"):
    """Load backbone and normalization layer, return NormLayerWrapper (eval mode)."""
    print(">>> [%s] Building backbone and normalization layer..." % task_name)
    backbone = utils.build_backbone(path=model_path, name=model_name)
    normlayer = utils.load_normalization_layer(path=normlayer_path)
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


# Encode multiple frames in one batch
# ---------------------------------------------------------------------------
def encode_frame_batch(
    pil_imgs, carrier, model, params, msg_tensors=None, angle=None, is_multibit=False
):
    """
    Encode multiple PIL images in a single forward pass.

    For 0bit: all images share the same *carrier* and *angle*.
    For multibit: *msg_tensors* must be a list of 1×K bool tensors, one per image.

    Args:
        pil_imgs: List of PIL images.
        carrier: carrier tensor.
        model: NormLayerWrapper.
        params: optimization parameters (same structure as encode_single_frame).
        msg_tensors: List of (1×K) bool tensors for multibit mode.
        angle: hypercone angle (0bit only).
        is_multibit: True for multibit encoding.

    Returns:
        List of PIL images (watermarked), in the same order as input.
    """
    if len(pil_imgs) == 0:
        return []

    N = len(pil_imgs)

    # Build data augmentation
    if params.data_augmentation == "all":
        data_aug = data_augmentation.All()
    elif params.data_augmentation == "none":
        data_aug = data_augmentation.DifferentiableDataAugmentation()
    else:
        data_aug = data_augmentation.All()

    # Multi-image dataloader (batch_size from params controls GPU memory)
    dataset = MultiImageDataset(pil_imgs)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=0)

    if is_multibit:
        # Stack msg_tensors into N×K
        if msg_tensors is None:
            raise ValueError("msg_tensors must be provided for multibit encoding")
        msgs_stacked = torch.cat(msg_tensors, dim=0)  # N×K
        pt_imgs_out = encode.watermark_multibit(
            dataloader, msgs_stacked, carrier, model, data_aug, params
        )
    else:
        pt_imgs_out = encode.watermark_0bit(
            dataloader, carrier, angle, model, data_aug, params
        )

    # Convert all output tensors to PIL images
    pil_outs = []
    for i in range(N):
        pil_out = ToPILImage()(
            utils_img.unnormalize_img(pt_imgs_out[i]).squeeze(0).detach().cpu()
        )
        pil_outs.append(pil_out)

    return pil_outs


# ---------------------------------------------------------------------------
# Message record persistence
# ---------------------------------------------------------------------------
def write_message_records(key, msg_text, encoded_bitstring):
    """Write a single message record to the database."""
    utils.append_message_record(key, 0, "video_frame", msg_text, encoded_bitstring)