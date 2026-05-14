import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.transforms import functional
from augly.image import functional as aug_functional

import bch_codec
import decode

import utils_img
import utils

pd.options.display.float_format = "{:,.3f}".format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attacks_dict = {
    "none": lambda x : x,
    "rotation": functional.rotate,
    "grayscale": functional.rgb_to_grayscale,
    "contrast": functional.adjust_contrast,
    "brightness": functional.adjust_brightness,
    "hue": functional.adjust_hue,
    "hflip": functional.hflip,
    "vflip": functional.vflip,
    "blur": functional.gaussian_blur, # sigma = ksize*0.15 + 0.35  - ksize = (sigma-0.35)/0.15
    "jpeg": aug_functional.encoding_quality,
    "resize": utils_img.resize,
    "center_crop": utils_img.center_crop,
    "meme_format": aug_functional.meme_format,
    "overlay_emoji": aug_functional.overlay_emoji,
    "overlay_onto_screenshot": aug_functional.overlay_onto_screenshot,
}

attacks = [{'attack': 'none'}] \
    + [{'attack': 'meme_format'}] \
    + [{'attack': 'overlay_onto_screenshot'}] \
    + [{'attack': 'rotation', 'angle': jj} for jj in range(0,45,5)] \
    + [{'attack': 'center_crop', 'scale': 0.1*jj} for jj in range(1,11)] \
    + [{'attack': 'resize', 'scale': 0.1*jj} for jj in range(1,11)] \
    + [{'attack': 'blur', 'kernel_size': 1+2*jj} for jj in range(1,10)] \
    + [{'attack': 'jpeg', 'quality': 10*jj} for jj in range(1,11)] \
    + [{'attack': 'contrast', 'contrast_factor': 0.5*jj} for jj in range(0,5)] \
    + [{'attack': 'brightness', 'brightness_factor': 0.5*jj} for jj in range(1,5)] \
    + [{'attack': 'hue', 'hue_factor': -0.5 + 0.25*jj} for jj in range(0,5)] \
    + [{'attack': 'hue', 'hue_factor': 0.2}] 
    
def generate_attacks(img, attacks):
    """ Generate a list of attacked images from a PIL image. """
    attacked_imgs = []
    for attack in attacks:
        attack = attack.copy()
        attack_name = attack.pop('attack')
        attacked_imgs.append(attacks_dict[attack_name](img, **attack))
    return attacked_imgs


def evaluate_trace_carriers(imgs, filenames, model, angle, carriers_dir, threshold):
    """
    Trace watermarking evaluation: compare each image against every carrier in carriers_dir.
    
    Args:
        imgs: List of PIL images
        filenames: List of filenames corresponding to the images
        model: Neural net model to extract the features
        angle: Angle of the hypercone
        carriers_dir: Directory containing .pth carrier files
        threshold: Confidence threshold for matching (e.g. 0.95)
    
    Returns:
        df: DataFrame with columns: index, filename, carrier_name, R, log10_pvalue, confidence, matched
    """
    decoded_data = decode.decode_trace_carriers(imgs, model, angle, carriers_dir, threshold)
    df = pd.DataFrame(decoded_data)
    df['filename'] = df['index'].apply(lambda i: filenames[i])
    df = df[['index', 'filename', 'carrier_name', 'R', 'log10_pvalue', 'confidence', 'matched']]
    return df


def decode_0bit_from_folder(img_dir, carrier, angle, model):
    """
    Args:
        img_dir: Folder containing the images to decode
        carrier: Hypercone direction 1xD
        angle: Angle of the hypercone        
        model: Neural net model to extract the features

    Returns:
        df: Dataframe with the decoded message for each image
    """
    imgs, filenames = utils_img.pil_imgs_from_folder(img_dir)
    decoded_data = decode.decode_0bit(imgs, carrier, angle, model)
    df = pd.DataFrame(decoded_data)
    df['filename'] = filenames
    df['marked'] = df['R'] > 0
    df.drop(columns=['R', 'log10_pvalue'], inplace=True)
    return df


def evaluate_0bit_on_attacks(imgs, carrier, angle, model, params, attacks=attacks, save=True):
    """
    Args:
        imgs: Watermarked images, list of PIL Images
        carrier: Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features
        params: Must contain verbose, output_dir 
        attacks: List of attacks to apply
        save: Whether to save instances of attacked images for the first image

    Returns:
        df: Dataframe with the detection scores for each transformation
    """

    logs = []
    for ii, img in enumerate(tqdm(imgs)):
        
        attacked_imgs = generate_attacks(img, attacks)
        if ii==0 and save:
            imgs_dir = os.path.join(params.output_dir, 'imgs')
            for jj in range(len(attacks)):
                filename = "%i_%s.png" % (ii, str(attacks[jj]).replace(':', '_').replace('{', '').replace('}', ''))
                attacked_imgs[jj].save(os.path.join(imgs_dir, filename))

        decoded_data = decode.decode_0bit(attacked_imgs, carrier, angle, model)
        for jj in range(len(attacks)):
            attack = attacks[jj].copy()
            # change params name before logging to harmonize df between attacks
            attack_name = attack.pop('attack')
            param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values()))) 
            decoded_datum = decoded_data[jj]
            log = {
                "keyword": "evaluation",
                "img": ii,
                "attack": attack_name,
                **attack_params,
                "log10_pvalue": decoded_datum['log10_pvalue'],
                "R": decoded_datum['R'],
                "marked": decoded_datum['R']>0,
            }
            logs.append(log)
            if params.verbose>1:
                print("__log__:%s" % json.dumps(log))

    df = pd.DataFrame(logs).drop(columns='keyword')

    if params.verbose>0:
        print('\n%s'%df)
    return df


def decode_multibit_from_folder(img_dir, carrier, model, output_msg_type,
                                bch_scheme=None, original_msg_len=None):
    """
    Args:
        img_dir: Folder containing the images to decode
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features
        msg_type: Type of message to decode ('bit' or 'text')
        bch_scheme: Optional tuple (n,k,t,num_seg,total_enc,bch_obj) for BCH error correction
        original_msg_len: Original message length in bits (required if bch_scheme is provided)

    Returns:
        df: Dataframe with the decoded message for each image
    """
    imgs, filenames = utils_img.pil_imgs_from_folder(img_dir)
    decoded_data = decode.decode_multibit(imgs, carrier, model)
    df = pd.DataFrame(decoded_data)
    df['filename'] = filenames
    df['msg'] = df['msg'].apply(
        lambda x: ''.join(map(str,x.type(torch.int).tolist()))
    )

    # BCH decode if enabled
    if bch_scheme is not None:
        df['decoded_msg'] = df['msg'].apply(
            lambda x: bch_codec.bch_decode(x, bch_scheme, original_msg_len)
        )

    if output_msg_type == 'text':
        df['msg'] = df['msg'].apply(
            lambda x: utils.binary_to_string(x)
        )
    return df


def decode_multibit_trace_from_folder(img_dir, carrier, model, key,
                                       output_msg_type='bit'):
    """
    Decode multibit watermarks and trace each decoded message against
    the message database for the given key using Hamming distance.

    For each image, the decoded bits are compared against every record
    in workspace/database/message/<key>/records.csv.

    Args:
        img_dir: Folder containing the images to decode
        carrier (tensor of size KxD): K carriers of dimension D
        model: Neural net model to extract the features
        key: Key string identifying which message DB to query
        output_msg_type: 'bit' or 'text'

    Returns:
        df: DataFrame with columns:
            image_index, image_filename, decoded_bits, matched_bits,
            hamming_distance, similarity, raw_msg, key, is_best_match
    """
    imgs, filenames = utils_img.pil_imgs_from_folder(img_dir)
    decoded_data = decode.decode_multibit(imgs, carrier, model)

    # Load the message database
    records_df = utils.load_message_records(key)

    rows = []
    total_bits = carrier.shape[0]  # K

    for ii, decoded_datum in enumerate(decoded_data):
        decoded_bits = ''.join(
            map(str, decoded_datum['msg'].type(torch.int).tolist())
        )

        if records_df.empty:
            # No records to match against
            rows.append({
                'image_index': ii,
                'image_filename': filenames[ii],
                'decoded_bits': decoded_bits,
                'matched_bits': '',
                'hamming_distance': -1,
                'similarity': -1.0,
                'raw_msg': '',
                'key': key,
                'is_best_match': False,
            })
            continue

        # Compute Hamming distance against each record
        best_dist = total_bits + 1
        best_idx = -1

        for rec_idx, rec_row in records_df.iterrows():
            matched_bits = str(rec_row['encoded_bits'])
            dist = utils.compute_hamming_distance(decoded_bits, matched_bits)
            similarity = 1.0 - dist / total_bits if total_bits > 0 else 1.0
            is_best = False  # will update after we find best for this image

            rows.append({
                'image_index': ii,
                'image_filename': filenames[ii],
                'decoded_bits': decoded_bits,
                'matched_bits': matched_bits,
                'hamming_distance': dist,
                'similarity': similarity,
                'raw_msg': rec_row['raw_msg'],
                'key': key,
                'is_best_match': False,
            })

            if dist < best_dist:
                best_dist = dist
                best_idx = len(rows) - 1

        # Mark the best match for this image
        if best_idx >= 0:
            rows[best_idx]['is_best_match'] = True

    df = pd.DataFrame(rows)

    if output_msg_type == 'text':
        # Keep decoded_bits as-is; they are bit strings
        pass

    return df


def evaluate_multibit_on_attacks(imgs, carrier, model, msgs_orig, params, attacks=attacks, save=True,
                                 bch_scheme=None, msgs_orig_raw=None):
    """
    Args:
        imgs: Watermarked images, list of PIL Images
        carrier (tensor of size KxD): K carriers of dimension D, each one determines a bit
        model: Neural net model to extract the features
        msgs_orig (boolean tensor of size NxK): original messages (BCH-encoded if bch_scheme is provided)
        params: Must contain verbose, output_dir 
        attacks: List of attacks to apply
        save: Whether to save instances of attacked images for the first image
        bch_scheme: Optional tuple (n,k,t,num_seg,total_enc,bch_obj) for BCH error correction
        msgs_orig_raw: Optional boolean tensor (N x original_msg_len) — the raw messages before BCH encoding

    Returns:
        df: Dataframe with the decoding scores for each transformation
    """

    logs = []
    for ii, img in enumerate(tqdm(imgs)):
        
        attacked_imgs = generate_attacks(img, attacks)
        if ii==0 and save:
            imgs_dir = os.path.join(params.output_dir, 'imgs')
            for jj in range(len(attacks)):
                filename = "%i_%s.png" % (ii, str(attacks[jj]).replace(':', '_').replace('{', '').replace('}', ''))
                attacked_imgs[jj].save(os.path.join(imgs_dir, filename))

        decoded_data = decode.decode_multibit(attacked_imgs, carrier, model)
        for jj in range(len(attacks)):
            attack = attacks[jj].copy()
            # change params name before logging to harmonize df between attacks
            attack_name = attack.pop('attack')
            param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values()))) 
            decoded_datum = decoded_data[jj]
            diff = (~torch.logical_xor(msgs_orig[ii], decoded_datum['msg'])).tolist() # useful for bit accuracy metric
            log = {
                "keyword": "evaluation",
                "img": ii,
                "attack": attack_name,
                **attack_params,
                "msg_orig": ''.join(['1' if b else '0' for b in msgs_orig[ii].tolist()]),
                "msg_decoded": ''.join(['1' if b else '0' for b in decoded_datum['msg'].tolist()]),
                "bit_acc": np.sum(diff)/len(diff),
                "word_acc": int(np.sum(diff)==len(diff)),
            }

            # BCH decode: recover raw message and compute accuracy against original raw message
            if bch_scheme is not None and msgs_orig_raw is not None:
                msg_decoded_str = log["msg_decoded"]
                recovered_codeword = bch_codec.bch_decode(
                    msg_decoded_str, bch_scheme,
                    msgs_orig_raw.shape[1]
                )
                orig_raw_str = ''.join(
                    ['1' if b else '0' for b in msgs_orig_raw[ii].tolist()]
                )
                match_bits = sum(
                    1 for a, b in zip(recovered_codeword, orig_raw_str) if a == b
                )
                log["recovered_codeword"] = recovered_codeword
                log["bit_acc_bch"] = match_bits / len(orig_raw_str) if len(orig_raw_str) > 0 else 1.0

            logs.append(log)
            if params.verbose>1:
                print("__log__:%s" % json.dumps(log))

    df = pd.DataFrame(logs).drop(columns='keyword')

    if params.verbose>0:
        print('\n%s'%df)
    return df


def aggregate_df(df, params):
    """
    Reads the dataframe output by the previous function and returns average detection scores for each transformation
    """
    df['param0'] = df['param0'].fillna(-1)
    df_mean = df.groupby(['attack','param0'], as_index=False).mean(numeric_only=True).drop(columns='img')
    df_min = df.groupby(['attack','param0'], as_index=False).min(numeric_only=True).drop(columns='img')
    df_max = df.groupby(['attack','param0'], as_index=False).max(numeric_only=True).drop(columns='img')
    df_std = df.groupby(['attack','param0'], as_index=False).std(numeric_only=True).drop(columns='img')
    # df_agg = df.groupby(['attack','param0'], as_index=False).agg(['mean','min','max','std']).drop(columns='img')
    # ��������

    df_mean.columns = [f'{col}_mean' for col in df_mean.columns]
    df_min.columns = [f'{col}_min' for col in df_min.columns]
    df_max.columns = [f'{col}_max' for col in df_max.columns]
    df_std.columns = [f'{col}_std' for col in df_std.columns]

    # �ϲ�DataFrame
    df_agg = pd.concat([df_mean, df_min, df_max, df_std], axis=1)

    if params.verbose>0:
        print('\n%s'%df_mean)
        print('\n%s'%df_agg)
    return df_agg

