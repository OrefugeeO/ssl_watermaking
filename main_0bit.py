import argparse
import datetime
import os

import numpy as np
import torch
from torchvision.transforms import ToPILImage

import data_augmentation
import encode
import evaluate
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    # 基本参数
    group = parser.add_argument_group("Experiments parameters")
    aa(# 输入图像文件夹的目录data_dir
        "--data_dir",
        type=str,
        default="workspace/input/input_images/",
        help="Folder directory (Default: /workspace/input/input_images)",
    )
    aa(# 水印嵌入的潜在空间方向的目录carrier_dir
        "--carrier_dir",
        type=str,
        default="workspace/database/carriers/",
        help="Directions of the latent space in which the watermark is embedded (Default: /database/carriers)",
    )
    aa(# 是否保存带水印的图像save_images
        "--save_images",
        type=utils.bool_inst,
        default=True,
        help="Whether to save watermarked images (Default: True)",
    )
    aa(# 是否评估检测器evaluate
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
    aa(# 溯源模式的置信度阈值，输入95表示阈值=0.95
        "--trace_confidence_threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for trace mode. Input 95 means 0.95. (Default: 0.95)",
    )
    aa(# 控制输出信息的详细程度verbose
        "--verbose", 
        type=int, 
        default=1
    )

    # 水印参数
    group = parser.add_argument_group("Marking parameters")
    aa(# 目标峰值信噪比（PSNR）值，以dB为单位target_psnr
        "--target_psnr",
        type=float,
        default=42.0,
        help="Target PSNR value in dB. (Default: 42 dB)",
    )
    aa(# 检测器的目标假阳性率（FPR）target_fpr
        "--target_fpr",
        type=float,
        default=1e-6,
        help="Target FPR of the dectector. (Default: 1e-6)",
    )
    aa(# 指定密钥，用于确定性生成载体key
        "--key",
        type=str,
        default=None,
        help="Key string for deterministic carrier generation. If provided, carriers are generated from this key via SHA-256. (Default: None)",
    )

    # 神经网络参数
    group = parser.add_argument_group("Neural-Network parameters")
    aa(# 用于水印嵌入的网络架构model_name
        "--model_name",
        type=str,
        default="resnet50",
        help="Marking network architecture. See https://pytorch.org/vision/stable/models.html and https://rwightman.github.io/pytorch-image-models/models/ (Default: resnet50)",
    )
    aa(# 模型的路径model_path
        "--model_path",
        type=str,
        default="models/dino_r50_plus.pth",
        help="Path to the model (Default: /models/dino_r50_plus.pth)",
    )
    aa(# 归一化层的路径normlayer_path
        "--normlayer_path",
        type=str,
        default="normlayers/out2048_yfcc_orig.pth",
        help="Path to the normalization layer (Default: /normlayers/out2048.pth)",
    )

    # 优化参数
    group = parser.add_argument_group("Optimization parameters")
    aa(# 图像优化的训练轮数epochs
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs for image optimization. (Default: 300)",
    )
    aa(# 在水印嵌入时使用的数据增强类型data_augmentation
        "--data_augmentation",
        type=str,
        default="all",
        choices=["none", "all"],
        help="Type of data augmentation to use at marking time. (Default: All)",
    )
    aa(# 使用的优化器optimizer
        "--optimizer",
        type=str,
        default="Adam,lr=0.01",
        help="Optimizer to use. (Default: Adam,lr=0.01)",
    )
    aa(# 使用的调度器scheduler
        "--scheduler", 
        type=str, 
        default=None, 
        help="Scheduler to use. (Default: None)"
    )
    aa(# 水印嵌入时的批处理大小batch_size
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for marking. (Default: 128)",
    )
    aa(# 水印损失的权重lambda_w
        "--lambda_w",
        type=float,
        default=1.0,
        help="Weight of the watermark loss. (Default: 1.0)",
    )
    aa(# 图像损失的权重lambda_i
        "--lambda_i",
        type=float,
        default=1.0,
        help="Weight of the image loss. (Default: 1.0)",
    )

    return parser


def check(print_data):
    print("=========================")
    for i in range(len(print_data)):
        print(print_data[i])
    print("=========================")


def main(params):
    # 设置随机种子以确保结果可重复
    torch.manual_seed(0)
    np.random.seed(0)

    # 动态生成输出目录
    key_name = params.key if params.key is not None else "test"
    if params.mode == 2:
        run_output_dir = os.path.join("workspace", "output", "decoded_output", "0bit", key_name)
    elif params.mode == 3:
        run_output_dir = os.path.join("workspace", "output", "traced_output", "0bit", key_name)
    else:
        run_output_dir = os.path.join("workspace", "output", "encoded_output", "0bit", key_name)
    params.output_dir = run_output_dir

    # 加载骨干网络和归一化层
    if params.verbose > 0:
        print(">>> Building backbone and normalization layer...")
    backbone = utils.build_backbone(
        path=params.model_path, name=params.model_name
    )  # 加载模型权重和架构
    normlayer = utils.load_normalization_layer(
        path=params.normlayer_path
    )  # 加载归一化层
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():  # 返回模型中所有可学习的参数（即权重和偏置）
        p.requires_grad = False  # 反向传播过程中不计算该参数的梯度
    model.eval()  # 设置为评估模式

    # 加载或生成载体和角度
    if not os.path.exists(params.carrier_dir):
        os.makedirs(params.carrier_dir, exist_ok=True)
    D = model(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)
    K = 1  # 0-bit always uses a single carrier

    carrier_path = os.path.join(params.carrier_dir, "carrier_0bit_%s.pth" % key_name)
    if os.path.exists(carrier_path):
        if params.verbose > 0:
            print(">>> Loading carrier from %s" % carrier_path)
        carrier = torch.load(carrier_path)
        assert D == carrier.shape[1]
    else:
        if params.verbose > 0:
            print(">>> Generating carrier into %s..." % carrier_path)
        if params.key is not None:
            carrier = utils.generate_carriers_with_key(K, D, params.key, output_fpath=carrier_path)
        else:
            carrier = utils.generate_carriers(K, D, output_fpath=carrier_path)

    carrier = carrier.to(device, non_blocking=True)
    angle = utils.pvalue_angle(dim=D, k=1, proba=params.target_fpr)

    # =================================================================================================
    # 模式 2: 解密
    if params.mode == 2:
        if params.verbose > 0:
            print(">>> Decoding watermarks...")
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir, exist_ok=True)
        df = evaluate.decode_0bit_from_folder(params.data_dir, carrier, angle, model)
        df_path = os.path.join(params.output_dir, "decodings.csv")
        df.to_csv(df_path, index=False)
        if params.verbose > 0:
            print("Results saved in %s" % df_path)
    # =================================================================================================
    # 模式 3: 溯源
    elif params.mode == 3:
        if params.verbose > 0:
            print(">>> Tracing watermarks against all carriers in %s..." % params.carrier_dir)
        imgs, filenames = utils_img.pil_imgs_from_folder(params.data_dir)
        threshold = params.trace_confidence_threshold
        df = evaluate.evaluate_trace_carriers(imgs, filenames, model, angle, params.carrier_dir, threshold)
        df = df.sort_values(
            by=["matched", "confidence"],
            ascending=[False, False],
            na_position="last",
            ignore_index=True,
        )
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir, exist_ok=True)
        df_path = os.path.join(params.output_dir, "trace_results.csv")
        with open(df_path, "w", encoding="utf-8") as f:
            f.write("# confidence = 1 - 10^(log10_pvalue)\n")
            f.write("# matched = confidence > threshold\n")
            f.write("# threshold = %s\n" % str(threshold))
            df.to_csv(f, index=False)
        if params.verbose > 0:
            print("Results saved in %s" % df_path)
    # =================================================================================================
    # 模式 1: 加密
    else:
        # Construct data augmentation
        if params.data_augmentation == "all":  # 写入时使用数据增强
            data_aug = data_augmentation.All()
        elif params.data_augmentation == "none":  # 不使用数据增强
            data_aug = data_augmentation.DifferentiableDataAugmentation()

        # Load images
        if params.verbose > 0:
            print(">>> Loading images from %s..." % params.data_dir)
        dataloader = utils_img.get_dataloader(
            params.data_dir, batch_size=params.batch_size
        )

        # Marking
        if params.verbose > 0:
            print(">>> Marking images...")
        """
            dataloader:数据加载器
            carrier:超空间方向向量
            angle:超空间角度
            model:模型
            data_aug:数据增强器
            params:参数
        """
        pt_imgs_out = encode.watermark_0bit(
            dataloader, carrier, angle, model, data_aug, params
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
                os.makedirs(params.output_dir, exist_ok=True)
            imgs_dir = os.path.join(params.output_dir, "imgs")
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
            df = evaluate.evaluate_0bit_on_attacks(
                imgs_out, carrier, angle, model, params
            )
            df_agg = evaluate.aggregate_df(df, params)
            df_path = os.path.join(params.output_dir, "df.csv")
            df_agg_path = os.path.join(params.output_dir, "df_agg.csv")
            df.to_csv(df_path, index=False)
            df_agg.to_csv(df_agg_path)
            if params.verbose > 0:
                print("Results saved in %s" % df_path)

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
