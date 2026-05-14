import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
from torchvision.transforms import ToPILImage

import galois

import bch_codec
import data_augmentation
import encode
import evaluate
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()
    # 创建一个argparse.ArgumentParser对象，赋值给变量parser。这个对象用于处理命令行参数。

    def aa(*args, **kwargs):
        # 定义一个内部函数aa，这个函数接受任意数量的位置参数和关键字参数。
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("Experiments parameters")
    # 创建一个新的参数组，组名为’Experiments parameters’。
    aa(  # 指定数据的目录data_dir
        "--data_dir",  # 参数的名称为--data_dir
        type=str,  # 类型为str
        default="workspace/input/input_images/",  # 默认值为"workspace/input/input_images/"
        help="Folder directory (Default: /workspace/input/input_images)",  # 命令行中使用-h或--help选项时显示
    )
    aa(  # 指定载体空间的方向，即嵌入水印的空间carrier_dir
        "--carrier_dir",
        type=str,
        default="workspace/database/carriers/",
        help="Directions of the latent space in which the watermark is embedded (Default: /database/carriers)",
    )
    aa(  # 指定是否保存有水印的图像save_images
        "--save_images",
        type=utils.bool_inst,
        default=True,
        help="Whether to save watermarked images (Default: False)",
    )
    aa(  # 指定是否评估检测器evaluate
        "--evaluate",
        type=utils.bool_inst,
        default=True,
        help="Whether to evaluate the detector (Default: True)",
    )
    aa(# 运行模式: 1=加密, 2=解密, 3=溯源
        "--mode",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Operation mode: 1=encode, 2=decode, 3=trace (Default: 1)",
    )
    aa(  # 指定是否在图像处理前将其resize以适应模型输入尺寸resize_to_fit
        "--resize_to_fit",
        type=utils.bool_inst,
        default=False,
        help="Whether to resize images to fit model input size (224x224) before processing (Default: False)",
    )
    aa(# 控制输出的详细程度verbose
        "--verbose", type=int, default=1
    )  

    group = parser.add_argument_group("Messages parameters")
    aa(  # 指定消息的类型msg_type
        "--msg_type",
        type=str,
        default="text",
        choices=["text", "bit"],
        help="Type of message (Default: bit)",
    )
    aa(  # 指定输出消息的类型output_msg_type
        "--output_msg_type",
        type=str,
        default="bit",
        choices=["text", "bit"],
        help="Type of message (Default: bit)",
    )
    aa(  # 指定消息文本文件的路径msg_path
        "--msg_path",
        type=str,
        default="workspace/input/input_messages/msgs.txt",
        help="Path to the messages text file (Default: workspace/input/input_messages/msgs.txt)",
    )
    aa(  # 指定消息的位数num_bits
        "--num_bits",
        type=int,
        default=30,
        help="Number of bits of the message. (Default: None)",
    )
    aa(  # 是否使用BCH纠错编码use_bch
        "--use_bch",
        type=utils.bool_inst,
        default=False,
        help="Whether to use BCH error-correcting code (Default: False)",
    )
    aa(  # BCH最大预期误码率max_error_rate
        "--max_error_rate",
        type=float,
        default=0.05,
        help="Maximum expected bit error rate for BCH scheme selection (Default: 0.10)",
    )
    aa(  # BCH最大编码比特预算max_encoded_bits
        "--max_encoded_bits",
        type=int,
        default=256,
        help="Maximum allowed encoded bits budget for BCH (Default: None)",
    )

    group = parser.add_argument_group("Marking parameters")
    aa(  # 指定目标PSNR值（以dB为单位）target_psnr
        "--target_psnr",
        type=float,
        default=42.0,
        help="Target PSNR value in dB. (Default: 42 dB)",
    )
    aa(  # 指定检测器的目标FPR值target_fpr
        "--target_fpr",
        type=float,
        default=1e-6,
        help="Target FPR of the dectector. (Default: 1e-6)",
    )
    aa(  # 指定密钥，用于确定性生成载体key
        "--key",
        type=str,
        default=None,
        help="Key string for deterministic carrier generation. If provided, carriers are generated from this key via SHA-256. (Default: None)",
    )

    group = parser.add_argument_group("Neural-Network parameters")
    aa(  # 指定神经网络的架构model_name
        "--model_name",
        type=str,
        default="resnet50",
        help="Marking network architecture. See https://pytorch.org/vision/stable/models.html and https://rwightman.github.io/pytorch-image-models/models/ (Default: resnet50)",
    )
    aa(  # 指定模型的路径model_path
        "--model_path",
        type=str,
        default="models/dino_r50_plus.pth",
        help="Path to the model (Default: /models/dino_r50_plus.pth)",
    )
    aa(  # 指定规范化层的路径normlayer_path
        "--normlayer_path",
        type=str,
        default="normlayers/out2048_yfcc_orig.pth",
        help="Path to the normalization layer (Default: /normlayers/out2048.pth)",
    )

    group = parser.add_argument_group("Optimization parameters")
    aa(  # 指定图像优化的周期数epochs
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs for image optimization. (Default: 300)",
    )
    aa(  # 指定在标记时间使用的数据增强类型data_augmentation
        "--data_augmentation",
        type=str,
        default="all",
        choices=["none", "all"],
        help="Type of data augmentation to use at marking time. (Default: All)",
    )
    aa(  # 指定要使用的优化器optimizer
        "--optimizer",
        type=str,
        default="Adam,lr=0.01",
        help="Optimizer to use. (Default: Adam,lr=0.01)",
    )
    aa(  # 指定要使用的调度器scheduler
        "--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)"
    )
    aa(  # 指定标记的批处理大小batch_size
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for marking. (Default: 128)",
    )
    aa(  # 指定水印损失的权重lambda_w
        "--lambda_w",
        type=float,
        default=5e4,
        help="Weight of the watermark loss. (Default: 1.0)",
    )
    aa(  # 指定图像损失的权重lambda_i
        "--lambda_i",
        type=float,
        default=1.0,
        help="Weight of the image loss. (Default: 1.0)",
    )

    return parser


def _bits_to_bool_list(bitstring):
    """Convert a bitstring ('0101...') to a list of bools."""
    return [c == "1" for c in bitstring]


def _bool_list_to_bits(bools):
    """Convert a list/tensor of bools to a bitstring ('0101...')."""
    return "".join("1" if b else "0" for b in bools)


def _bch_encode_msgs(msgs, bch_scheme):
    """BCH-encode each row of a boolean tensor, return encoded boolean tensor."""
    encoded_rows = []
    for row in msgs:
        bits = _bool_list_to_bits(row.tolist())
        encoded_bits = bch_codec.bch_encode(bits, bch_scheme)
        encoded_rows.append(_bits_to_bool_list(encoded_bits))
    return torch.tensor(encoded_rows)


def main(params):
    # Set seeds for reproductibility
    torch.manual_seed(0)
    np.random.seed(0)

    # =========================================================================
    # Step 1: Determine original message length
    # =========================================================================
    if params.msg_path is not None:
        original_num_bits = utils.get_num_bits(params.msg_path, params.msg_type)
        if params.num_bits != original_num_bits:
            warning_msg = (
                "WARNING: Number of bits in the loaded message ({a}) does not "
                "match the number of bits indicated in the num_bit argument ({b}). "
                "Setting num_bits to {a} "
                'Try with "--num_bit {a}" to remove the warning'.format(
                    a=original_num_bits, b=params.num_bits
                )
            )
            print(warning_msg)
        params.num_bits = original_num_bits
    else:
        original_num_bits = params.num_bits

    # =========================================================================
    # Step 2: BCH scheme selection (if enabled)
    # =========================================================================
    bch_scheme = None
    if params.use_bch:
        if params.max_encoded_bits is None:
            # Default: allow up to 10x expansion
            max_enc = original_num_bits * 10
        else:
            max_enc = params.max_encoded_bits

        if params.verbose > 0:
            print(
                ">>> Selecting BCH scheme (original=%d bits, "
                "max_err_rate=%.3f, max_encoded=%d)..."
                % (original_num_bits, params.max_error_rate, max_enc)
            )
        try:
            bch_scheme = bch_codec.select_bch_scheme(
                original_num_bits, params.max_error_rate, max_enc
            )
        except ValueError as e:
            print("ERROR: BCH scheme selection failed: %s" % e, file=sys.stderr)
            sys.exit(1)

        n, k, t, num_seg, total_enc_bits, _ = bch_scheme
        if params.verbose > 0:
            print(
                "  Selected BCH(%d,%d,%d): %d segments, %d encoded bits"
                % (n, k, t, num_seg, total_enc_bits)
            )
        params.num_bits = total_enc_bits

    # =========================================================================
    # Step 3: Output directory
    # =========================================================================
    key_name = params.key if params.key is not None else "test"
    if params.mode == 2:
        run_output_dir = os.path.join("workspace", "output", "decoded_output", "multibit", key_name)
    elif params.mode == 3:
        run_output_dir = os.path.join("workspace", "output", "traced_output", "multibit", key_name)
    else:
        run_output_dir = os.path.join("workspace", "output", "encoded_output", "multibit", key_name)
    params.output_dir = run_output_dir

    # =========================================================================
    # Step 4: Loads backbone and normalization layer
    # =========================================================================
    if params.verbose > 0:
        print(">>> Building backbone and normalization layer...")
    backbone = utils.build_backbone(path=params.model_path, name=params.model_name)
    normlayer = utils.load_normalization_layer(path=params.normlayer_path)
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # =========================================================================
    # Step 5: Load or generate carrier (with possibly-updated num_bits)
    # =========================================================================
    if not os.path.exists(params.carrier_dir):
        os.makedirs(params.carrier_dir, exist_ok=True)
    D = model(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)  # 特征维度
    K = params.num_bits  # 信息最大bit位数

    carrier_path = os.path.join(params.carrier_dir, "carrier_multibit_%s.pth" % key_name)
    if os.path.exists(carrier_path):
        if params.verbose > 0:
            print(">>> Loading carrier from %s" % carrier_path)
        carrier = torch.load(carrier_path)
        assert D == carrier.shape[1]
    else:
        if params.verbose > 0:
            print(
                ">>> Generating carrier into %s... (can take up to a minute)"
                % carrier_path
            )
        if params.key is not None:
            carrier = utils.generate_carriers_with_key(K, D, params.key, output_fpath=carrier_path)
        else:
            carrier = utils.generate_carriers(K, D, output_fpath=carrier_path)

    carrier = carrier.to(
        device, non_blocking=True
    )  # direction vectors of the hyperspace

    # =================================================================================================
    # Mode 3: Trace (hamming distance matching against message DB)
    # =================================================================================================
    if params.mode == 3:
        if params.verbose > 0:
            print(">>> Tracing watermarks via hamming distance (key=%s)..." % key_name)
        
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir, exist_ok=True)
        
        df = evaluate.decode_multibit_trace_from_folder(
            params.data_dir, carrier, model, key_name,
            output_msg_type=params.output_msg_type
        )
        
        trace_path = os.path.join(params.output_dir, "records.csv")
        df.to_csv(trace_path, index=False)
        if params.verbose > 0:
            print("Trace results saved in %s" % trace_path)
    
    # =================================================================================================
    # Mode 2: Decode only
    # =================================================================================================
    elif params.mode == 2:
        if params.verbose > 0:
            print(">>> Decoding watermarks...")
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir, exist_ok=True)

        # BCH decode if enabled — read metadata from encode output directory
        decode_bch_scheme = None
        decode_orig_bits = original_num_bits
        if params.use_bch:
            if bch_scheme is not None:
                # BCH scheme already available from scheme selection step
                decode_bch_scheme = bch_scheme
                decode_orig_bits = original_num_bits
            else:
                # Load BCH metadata from message database (same dir as records.csv)
                bch_meta_msg_dir = os.path.join("workspace", "database", "message", key_name)
                bch_meta_path = os.path.join(bch_meta_msg_dir, "bch_meta.json")
                if os.path.exists(bch_meta_path):
                    with open(bch_meta_path, "r") as f:
                        bch_meta = json.load(f)
                    orig_bits = bch_meta["original_num_bits"]
                    n, k, t = bch_meta["n"], bch_meta["k"], bch_meta["t"]
                    num_seg = bch_meta["num_segments"]
                    total_enc = bch_meta["total_encoded_bits"]
                    bch_obj = galois.BCH(n, k)
                    decode_bch_scheme = (n, k, t, num_seg, total_enc, bch_obj)
                    decode_orig_bits = orig_bits
                    if params.verbose > 0:
                        print(
                            ">>> Loaded BCH scheme from %s (original=%d bits)"
                            % (bch_meta_path, orig_bits)
                        )
                else:
                    print("ERROR: --use_bch set but bch_meta.json not found in %s"
                          % bch_meta_msg_dir, file=sys.stderr)
                    sys.exit(1)

        df = evaluate.decode_multibit_from_folder(
            params.data_dir, carrier, model, params.output_msg_type,
            bch_scheme=decode_bch_scheme, original_msg_len=decode_orig_bits
        )

        # For decode-only mode, replace df['msg'] with BCH-decoded result
        if params.use_bch and decode_bch_scheme is not None and 'decoded_msg' in df.columns:
            df['msg'] = df['decoded_msg']
            if params.verbose > 0:
                print(">>> BCH-decoded %d messages" % len(df))

        df_path = os.path.join(params.output_dir, "decodings.csv")
        df.to_csv(df_path, index=False)
        if params.verbose > 0:
            print("Results saved in %s" % df_path)
    # =================================================================================================
    # Mode 1: Encode
    # =================================================================================================
    elif params.mode == 1:
        # Load images
        if params.verbose > 0:
            print(">>> Loading images from %s..." % params.data_dir)
        dataloader = utils_img.get_dataloader(
            params.data_dir, batch_size=params.batch_size, resize_to_fit=params.resize_to_fit
        )

        # Generate messages (original length before BCH encoding)
        if params.verbose > 0:
            print(">>> Loading messages...")
        if params.msg_path is None:
            msgs = utils.generate_messages(len(dataloader.dataset), original_num_bits)
        else:
            if not os.path.exists(params.msg_path):
                if params.verbose > 0:
                    print("Generating random messages into %s..." % params.msg_path)
                os.makedirs(os.path.dirname(params.msg_path), exist_ok=True)
                msgs = utils.generate_messages(
                    len(dataloader.dataset), original_num_bits
                )
                utils.save_messages(msgs, params.msg_path)
            else:
                if params.verbose > 0:
                    print(
                        "Loading %s messages from %s..."
                        % (params.msg_type, params.msg_path)
                    )
                msgs = utils.load_messages(
                    params.msg_path, params.msg_type, len(dataloader.dataset)
                )

        # BCH-encode messages if enabled
        msgs_orig_raw = None
        if params.use_bch and bch_scheme is not None:
            msgs_orig_raw = msgs  # keep raw messages before BCH encoding
            if params.verbose > 0:
                print(">>> BCH-encoding messages...")
            msgs = _bch_encode_msgs(msgs, bch_scheme)

            # Save BCH metadata for decode phase (same dir as records.csv)
            bch_meta_msg_dir = os.path.join("workspace", "database", "message", key_name)
            if not os.path.exists(bch_meta_msg_dir):
                os.makedirs(bch_meta_msg_dir, exist_ok=True)
            n_sch, k_sch, t_sch, num_seg_sch, total_enc_sch, bch_obj_sch = bch_scheme
            bch_meta = {
                "original_num_bits": original_num_bits,
                "n": n_sch,
                "k": k_sch,
                "t": t_sch,
                "num_segments": num_seg_sch,
                "total_encoded_bits": total_enc_sch,
            }
            bch_meta_path = os.path.join(bch_meta_msg_dir, "bch_meta.json")
            with open(bch_meta_path, "w") as f:
                json.dump(bch_meta, f)
            if params.verbose > 0:
                print("  BCH metadata saved to %s" % bch_meta_path)

        # Construct data augmentation
        if params.data_augmentation == "all":
            data_aug = data_augmentation.All()
        elif params.data_augmentation == "none":
            data_aug = data_augmentation.DifferentiableDataAugmentation()

        # Marking
        if params.verbose > 0:
            print(">>> Marking images...")
        pt_imgs_out = encode.watermark_multibit(
            dataloader, msgs, carrier, model, data_aug, params
        )
        imgs_out = [
            ToPILImage()(utils_img.unnormalize_img(pt_img).squeeze(0))
            for pt_img in pt_imgs_out
        ]

        # Evaluate
        if params.evaluate:
            if params.verbose > 0:
                print(">>> Evaluating watermarks...")
            if not os.path.exists(params.output_dir):
                os.makedirs(params.output_dir)
            imgs_dir = os.path.join(params.output_dir, "imgs")
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
            df = evaluate.evaluate_multibit_on_attacks(
                imgs_out, carrier, model, msgs, params,
                bch_scheme=bch_scheme, msgs_orig_raw=msgs_orig_raw
            )
            df_agg = evaluate.aggregate_df(df, params)
            df_path = os.path.join(params.output_dir, "df.csv")
            df_agg_path = os.path.join(params.output_dir, "df_agg.csv")
            df.to_csv(df_path, index=False)
            df_agg.to_csv(df_agg_path)
            if params.verbose > 0:
                print("Results saved in %s" % df_path)

        # Write message records to message database
        if params.verbose > 0:
            print(">>> Writing message records to message database...")
        img_paths = [sample[0] for sample in dataloader.dataset.samples]
        for ii in range(len(imgs_out)):
            encoded_bits = _bool_list_to_bits(msgs[ii].tolist())
            # Get raw message text for the record
            if msgs_orig_raw is not None:
                raw_bits_str = _bool_list_to_bits(msgs_orig_raw[ii].tolist())
                raw_msg = utils.binary_to_string(raw_bits_str)
            else:
                raw_msg = utils.binary_to_string(encoded_bits)
            filename = os.path.basename(img_paths[ii])
            utils.append_message_record(key_name, ii, filename, raw_msg, encoded_bits)
        
        # Save
        if params.save_images:
            if not os.path.exists(params.output_dir):
                os.makedirs(params.output_dir, exist_ok=True)
            imgs_dir = os.path.join(params.output_dir, "imgs")
            if params.verbose > 0:
                print(">>> Saving images into %s..." % imgs_dir)
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
            for ii, img_out in enumerate(imgs_out):
                img_out.save(os.path.join(imgs_dir, "%i_out.png" % ii))


if __name__ == "__main__":

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
