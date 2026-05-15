# 基于加密与可嵌入水印技术的数字藏品溯源算法研究与实现

基于 DINO 自监督特征学习的深度学习图像与视频数字水印系统。通过在神经网络特征空间中嵌入不可见的水印信息，实现对数字藏品（NFT 图像/视频）的版权保护与身份溯源。

---

## ✨ 功能特性

本项目支持三种水印工作模式，每种模式均提供完整的编码、解码与溯源流程：

| 模式 | 说明 |
|------|------|
| **0-bit 水印** | 二值检测 —— 判断图像是否嵌入了指定密钥的水印 |
| **多比特水印** | 多比特信息嵌入 —— 支持 BCH 纠错编码，可嵌入自定义消息 |
| **视频水印** | 帧级视频水印 —— 逐帧嵌入与解码，支持视频溯源 |

### 算法执行流程

#### 0-bit 水印流程

![0-bit 水印算法流程](extra/0bit.png?v=3cc8ed4)

#### 多比特水印流程

![多比特水印算法流程](extra/multibit.png?v=3cc8ed4)

#### 视频水印流程

![视频水印算法流程](extra/video.png?v=3cc8ed4)

---

## 📦 环境安装

```bash
# 克隆项目
git clone https://github.com/OrefugeeO/ssl_watermaking.git
cd ssl_watermarking

# 安装依赖
pip install -r requirements.txt
```

> 建议使用 Python 3.8+，CUDA 11.x 以上以获得 GPU 加速。

---

## 🧠 模型权重准备

水印的嵌入与提取需要两类预训练模型：

1. **特征提取网络**：将图像映射到高维特征空间
2. **归一化层**：对特征进行 PCA 白化，使其在隐空间中分布更均匀

项目实验中所用的权重如下：

### 骨干网络

| 模型 | 说明 | 下载链接 |
|------|------|----------|
| `dino_r50_plus.pth` | ResNet-50，使用 DINO 自监督方法训练 | [下载](https://dl.fbaipublicfiles.com/ssl_watermarking/dino_r50_plus.pth) |

### 归一化层（四选一）

| 权重文件 | 类型 | 适用场景 | 下载链接 |
|----------|------|----------|----------|
| `out2048_yfcc_orig.pth` | whitening | **推荐**，通用场景，基于 YFCC 原始尺寸图像计算 | [下载](https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_yfcc_orig.pth) |
| `out2048_yfcc_resized.pth` | whitening (resized) | **低分辨率推荐**，输入图像尺寸约 128×128 时使用 | [下载](https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_yfcc_resized.pth) |
| `out2048_coco_orig.pth` | whitening_v1 | 备选方案，基于 COCO 原始尺寸图像计算 | [下载](https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_coco_orig.pth) |
| `out2048_coco_resized.pth` | whitening_v1 (resized) | 备选低分辨率方案 | [下载](https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_coco_resized.pth) |

### 放置方式

下载后将权重文件放入项目对应目录：

```
ssl_watermarking/
├── models/
│   └── dino_r50_plus.pth          # 骨干网络权重
├── normlayers/
│   ├── out2048_yfcc_orig.pth      # 归一化层（推荐）
│   ├── out2048_yfcc_resized.pth   # 归一化层（低分辨率）
│   ├── out2048_coco_orig.pth      # 归一化层（备选）
│   └── out2048_coco_resized.pth   # 归一化层（备选低分辨率）
└── ...
```

> **说明**：骨干网络使用 [DINO](https://github.com/pierrefdz/dino/) 框架训练（在原版基础上额外加入了旋转变换增强）。归一化层采用 PCA 白化方法，分别在 YFCC 数据集（whitening 系列）和 COCO 数据集（whitening_v1 系列）的 10 万张图像上统计得到。对于分辨率较低的输入图像，建议选用 `resized` 版本（对应 128×128 尺寸下的统计量），其余情况使用原始尺寸版本即可。

---

## 🚀 使用方法

### 一、0-bit 水印

```bash
# 编码（不带密钥）
python main_0bit.py --data_dir ./workspace/input/input_images/ --model_path .\models\dino_r50_plus.pth --normlayer_path normlayers/out2048_yfcc_orig.pth --mode 1

# 编码（带密钥）
python main_0bit.py --data_dir ./workspace/input/input_images/ --model_path .\models\dino_r50_plus.pth --normlayer_path normlayers/out2048_yfcc_orig.pth --mode 1 --key "2026_Watermark"

# 解码（不带密钥）
python main_0bit.py --data_dir ./workspace/output/encoded_output/0bit/test/imgs/ --model_path .\models\dino_r50_plus.pth --normlayer_path normlayers/out2048_yfcc_orig.pth --mode 2

# 解码（带密钥）
python main_0bit.py --data_dir ./workspace/output/encoded_output/0bit/2026_Watermark/imgs/ --model_path .\models\dino_r50_plus.pth --normlayer_path normlayers/out2048_yfcc_orig.pth --mode 2 --key "2026_Watermark"

# 溯源（不带密钥）
python main_0bit.py --data_dir ./workspace/output/encoded_output/0bit/test/imgs/ --model_path .\models\dino_r50_plus.pth --normlayer_path normlayers/out2048_yfcc_orig.pth --mode 3

# 溯源（带密钥）
python main_0bit.py --data_dir ./workspace/output/encoded_output/0bit/2026_Watermark/imgs/ --model_path .\models\dino_r50_plus.pth --normlayer_path normlayers/out2048_yfcc_orig.pth --mode 3 --key "2026_Watermark"
```

### 二、多比特水印（BCH 纠错编码）

```bash
# 编码（不带密钥）
python main_multibit.py --data_dir ./workspace/input/input_images/ --model_path models/dino_r50_plus.pth --normlayer_path normlayers/out2048_coco_resized.pth --mode 1 --use_bch True

# 编码（带密钥）
python main_multibit.py --data_dir ./workspace/input/input_images/ --model_path models/dino_r50_plus.pth --normlayer_path normlayers/out2048_coco_resized.pth --mode 1 --use_bch True --key "2026_Watermark"

# 解码（不带密钥）
python main_multibit.py --data_dir ./workspace/output/encoded_output/multibit/test/imgs/ --model_path models/dino_r50_plus.pth --normlayer_path normlayers/out2048_coco_resized.pth --mode 2 --use_bch True

# 解码（带密钥）
python main_multibit.py --data_dir ./workspace/output/encoded_output/multibit/2026_Watermark/imgs/ --model_path models/dino_r50_plus.pth --normlayer_path normlayers/out2048_coco_resized.pth --mode 2 --use_bch True --key "2026_Watermark"

# 溯源（不带密钥）
python main_multibit.py --data_dir ./workspace/output/encoded_output/multibit/test/imgs/ --model_path models/dino_r50_plus.pth --normlayer_path normlayers/out2048_coco_resized.pth --mode 3

# 溯源（带密钥）
python main_multibit.py --data_dir ./workspace/output/encoded_output/multibit/2026_Watermark/imgs/ --model_path models/dino_r50_plus.pth --normlayer_path normlayers/out2048_coco_resized.pth --mode 3 --key "2026_Watermark"
```

### 三、视频水印

```bash
# 编码
python .\main_video.py --mode encode --msg_path .\workspace\input\input_videos\0\msgs.txt --input_video_path .\workspace\input\input_videos\0\0.mp4 --msg_path .\workspace\input\input_videos\0\msgs.txt --use_bch True --key "video_Watermark"

# 溯源
python .\main_video.py --mode trace --input_video_path .\workspace\output\video_output\0\0_encoded.mp4 --use_bch True
```

> **模式说明**：`--mode 1` 为编码，`--mode 2` 为解码，`--mode 3` 为溯源。

---

## 📁 项目结构

```
ssl_watermarking/
├── main_0bit.py                  # 0-bit 水印主入口（编码/解码/溯源）
├── main_multibit.py              # 多比特水印主入口（编码/解码/溯源）
├── main_video.py                 # 视频水印主入口（编码/溯源）
├── encode.py                     # 水印编码核心逻辑
├── decode.py                     # 水印解码核心逻辑
├── evaluate.py                   # 鲁棒性评估（多种图像攻击测试）
├── bch_codec.py                  # BCH 纠错编解码模块
├── data_augmentation.py          # 可微分数据增强模块
├── build_normalization_layer.py  # 归一化层构建脚本
├── utils.py                      # 通用工具函数（模型加载、载体生成等）
├── utils_img.py                  # 图像处理工具函数
├── video_utils.py                # 视频处理工具函数
├── requirements.txt              # Python 依赖清单
├── cmd.txt                       # 命令行示例参考
├── models/                       # 预训练模型存放目录
│   └── dino_r50_plus.pth
├── normlayers/                   # 归一化层权重存放目录
│   ├── out2048_yfcc_orig.pth
│   ├── out2048_yfcc_resized.pth
│   ├── out2048_coco_orig.pth
│   └── out2048_coco_resized.pth
├── extra/                        # 论文插图资源
│   ├── 0bit.png
│   ├── multibit.png
│   └── video.png
└── workspace/                    # 工作目录
    ├── input/                    # 输入数据（待水印图像/视频）
    ├── output/                   # 输出结果（编码后图像/视频）
    └── database/                 # 数据库文件
```

---

## 🛡️ 鲁棒性评估

`evaluate.py` 支持对水印图像施加多种攻击并评估解码效果，涵盖以下攻击类型：

- 无攻击（基线）
- 旋转（rotation）
- 灰度化（grayscale）
- 对比度调整（contrast）
- 亮度调整（brightness）
- 色调调整（hue）
- 水平翻转（hflip） / 垂直翻转（vflip）
- 高斯模糊（blur）
- JPEG 压缩（jpeg）
- 缩放（resize）
- 中心裁剪（center_crop）
- Meme 格式 / Emoji 叠加 / 截屏叠加等社交媒体变换

---

## 📋 依赖项

完整依赖列表见 [`requirements.txt`](./requirements.txt)，核心依赖包括：

- PyTorch / TorchVision — 深度学习框架
- NumPy / SciPy — 科学计算
- Pillow — 图像处理
- OpenCV — 视频读写
- timm — 预训练模型库
- galois — BCH 纠错码运算
- augly — 数据增强与攻击模拟