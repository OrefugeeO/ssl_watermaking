import os
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import root_scalar
from scipy.special import betainc
from scipy.stats import ortho_group
from torchvision import models
from torchvision.models import ResNet50_Weights
import time
import hashlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backbone(path, name):
    """Build a pretrained torchvision backbone from its name.

    Args:
        path: path to the checkpoint, can be an URL
        name: name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html)
        or timm (see https://rwightman.github.io/pytorch-image-models/models/).
        We highly recommand to use Resnet50 architecture as available in torchvision.
        Using other architectures (such as non-convolutional ones) might need changes in the implementation.
    """
    if hasattr(models, name):  # 检查models模块中是否有指定名称的模型
        # model = getattr(models, name)(pretrained=True)  # 获取预训练模型
        model = getattr(models, name)(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        import timm  # 导入timm库

        if name in timm.list_models():  # 检查timm库中是否有指定名称的模型
            model = timm.models.create_model(
                name, num_classes=0
            )  # 创建模型，num_classes设为0
        else:
            raise NotImplementedError(
                "Model %s does not exist in torchvision" % name
            )  # 如果模型不存在，抛出异常

    model.head = nn.Identity()  # 将模型的head层设置为Identity层
    model.fc = nn.Identity()  # 将模型的fc层设置为Identity层
    if path is not None:
        if path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(
                path, progress=False, map_location=device
            )  # 使用torch.hub.load_state_dict_from_url从URL加载检查点
        else:
            checkpoint = torch.load(
                path, map_location=device
            )  # 加载检查点，并将其映射到指定设备
        state_dict = checkpoint
        for ckpt_key in ["state_dict", "model_state_dict", "teacher"]:
            if ckpt_key in checkpoint:
                state_dict = checkpoint[ckpt_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(
            state_dict, strict=False
        )  # 将状态字典加载到模型中。strict=False表示允许状态字典中的某些键在模型中不存在
    return model.to(
        device, non_blocking=True
    )  # 将模型移动到指定的设备（例如GPU或CPU）上，并设置为非阻塞模式


def get_linear_layer(weight, bias):
    """Create a linear layer from weight and bias matrices"""
    dim_out, dim_in = weight.shape
    layer = nn.Linear(dim_in, dim_out)
    layer.weight = nn.Parameter(weight)
    layer.bias = nn.Parameter(bias)
    return layer


def load_normalization_layer(path, mode="whitening"):  # path:检查点，mode:模式
    """Loads the normalization layer from a checkpoint and returns the layer."""
    checkpoint = torch.load(
        path, map_location=device
    )  # 加载检查点，并将其映射到指定设备
    if mode == "whitening":
        # if PCA whitening is used scale the feature by the dimension of the latent space
        D = checkpoint["weight"].shape[
            1
        ]  # 获取权重矩阵的第二个维度大小D，表示潜在空间的维度
        weight = torch.nn.Parameter(
            D * checkpoint["weight"]
        )  # 将权重乘以D，并转换为torch.nn.Parameter类型
        bias = torch.nn.Parameter(
            D * checkpoint["bias"]
        )  # 将偏置乘以D，并转换为torch.nn.Parameter类型
    else:
        weight = checkpoint["weight"]  # 不处理
        bias = checkpoint["bias"]  # 不处理
    return get_linear_layer(weight, bias).to(
        device, non_blocking=True
    )  # 使用处理后的权重和偏置创建一个线性层


class NormLayerWrapper(nn.Module):
    """
    Wraps backbone model and normalization layer
    """

    def __init__(self, backbone, head):
        super(NormLayerWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        output = self.backbone(x)
        return self.head(output)


def cosine_pvalue(c, d, k=1):  # 计算随机单位向量的投影绝对值大于某个给定值的概率
    """
    Returns the probability that the absolute value of the projection between random unit vectors is higher than c
    Args:
        c: cosine value
        d: dimension of the features
        k: number of dimensions of the projection
    """
    assert k > 0
    a = (d - k) / 2.0
    b = k / 2.0
    if c < 0:
        return 1.0
    return betainc(a, b, 1 - c**2)


def pvalue_angle(dim, k=1, angle=None, proba=None):  # 将p值与高维空间中的角度联系起来
    """
    Links the pvalue to the angle of the hyperspace.
    If angle is input, the function returns the pvalue for the given angle.
    If proba is input, the function returns the angle for the given FPR.
    Args:
        dim: dimension of the latent space
        k: number of axes of the projection
        angle: angle of the hyperspace
        proba: target probability of false positive
    """

    def f(a):
        return cosine_pvalue(np.cos(a), dim, k) - proba

    a = root_scalar(f, x0=0.49 * np.pi, bracket=[0, np.pi / 2])
    return a.root


def generate_carriers(k, d, output_fpath=None):
    """
    Generate k random orthonormal vectors of size d.
    Args:
        k: number of bits to watermark
        d: dimension of the watermarking space
        output_fpath: path where the tensor is saved
    Returns:
        tensor KxD
    """
    assert k <= d
    np.random.seed(int(time.time()))
    if k == 1:

        carriers = torch.randn(1, d)  # 生成一个形状为 1xD 的随机张量
        carriers /= torch.norm(
            carriers, dim=1, keepdim=True
        )  # 归一化，使其成为单位向量
    else:
        # 生成一个形状为 DxD 的随机正交矩阵，并取其前 k 行
        carriers = ortho_group.rvs(d)[:k, :]
        print(carriers)
        # 转换为 PyTorch 张量，并设置数据类型为 torch.float
        carriers = torch.tensor(carriers, dtype=torch.float)
        
    if output_fpath is not None:
        torch.save(carriers, output_fpath)  # 将生成的 carriers 张量保存到指定路径
    return carriers


def generate_carriers_with_key(K, D, key: str, output_fpath=None):
    """
    Generate K normalized Gaussian random vectors deterministically from a key string.
    The same key always produces the same carriers; different keys produce
    independent carriers.

    Args:
        K: number of vectors to generate
        D: dimension of each vector
        key: user-supplied string that seeds the PRNG via SHA-256
        output_fpath: optional path to save the tensor

    Returns:
        tensor KxD
    """
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    seed = int(digest[:16], 16)

    # Use pure PyTorch random generation to avoid NumPy 2.x / PyTorch 1.x
    # compatibility issues (torch.from_numpy / torch.as_tensor both fail).
    generator = torch.Generator()
    generator.manual_seed(seed)
    carriers = torch.randn(K, D, generator=generator, dtype=torch.float32)

    norms = carriers.norm(dim=1, keepdim=True) + 1e-8
    carriers = carriers / norms

    if output_fpath is not None:
        torch.save(carriers, output_fpath)

    return carriers


def generate_messages(n, k):
    """
    Generate random original messages.
    Args:
        n: Number of messages to generate
        k: length of the message
    Returns:
        msgs: boolean tensor of size nxk
    """
    # 随机生成n*k的bool矩阵，true与false的值均为0.5
    return torch.rand((n, k)) > 0.5


def string_to_binary(st):
    """String to binary"""
    return "".join(format(ord(i), "08b") for i in st)


def binary_to_string(bi):
    """Binary to string"""
    return "".join(
        chr(int(byte, 2)) for byte in [bi[ii : ii + 8] for ii in range(0, len(bi), 8)]
    )


def get_num_bits(path, msg_type):
    """Get the number of bits of the watermark from the text file"""
    with open(path, "r") as f:
        lines = [line.strip() for line in f]
    if msg_type == "bit":
        return max([len(line) for line in lines])
    else:
        return 8 * max([len(line) for line in lines])


def load_messages(path, msg_type, N):
    """
    Load messages from a file
    path:信息路径
    msg_type:信息类型
    N:输入图片个数
    """
    # 只读打开文件，读取每行，删除前后空格
    with open(path, "r") as f:
        lines = [line.strip() for line in f]
    if msg_type == "bit":
        num_bit = max([len(line) for line in lines])  # 获取最大行
        lines = [line + "0" * (num_bit - len(line)) for line in lines]  # 用0补齐
        msgs = [[int(i) == 1 for i in line] for line in lines]  # 转化为bool矩阵
    else:
        num_byte = max([len(line) for line in lines])  # 获取最大行
        lines = [line + " " * (num_byte - len(line)) for line in lines]  # 用空格补齐
        msgs = [
            [int(i) == 1 for i in string_to_binary(line)] for line in lines
        ]  # 转化为2进制
    msgs = msgs * (N // len(msgs) + 1)  # 复制行
    return torch.tensor(msgs[:N])  # 取前N行


def save_messages(msgs, path):
    """Save messages to file"""
    txt_msgs = ["".join(map(str, x.type(torch.int).tolist())) for x in msgs]
    txt_msgs = "\n".join(txt_msgs)
    with open(os.path.join(path), "w") as f:
        f.write(txt_msgs)


def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example:
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(" ", "").split(",")
    params = {}
    params["name"] = s[0]
    for x in s[1:]:
        x = x.split("=")
        params[x[0]] = float(x[1])
    return params


def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected in args")


def compute_hamming_distance(bits1, bits2):
    """
    Compute the Hamming distance between two bit strings.
    
    Args:
        bits1: first bit string (e.g. "0101")
        bits2: second bit string (e.g. "0111")
    
    Returns:
        int: number of positions where bits1 and bits2 differ
    """
    return sum(c1 != c2 for c1, c2 in zip(bits1, bits2))


def append_message_record(key, index, filename, raw_msg, encoded_bits):
    """
    Append a message record to the message database for a given key.
    
    The record is stored in: workspace/database/message/<key>/records.csv
    Duplicate records (same raw_msg) are silently skipped.
    
    Args:
        key: key string identifying the carrier/message group
        index: image index (int)
        filename: original image filename (str)
        raw_msg: the original raw message text (str)
        encoded_bits: the final encoded bit string embedded (str)
    """
    import csv
    import datetime
    
    msg_dir = os.path.join("workspace", "database", "message", key)
    os.makedirs(msg_dir, exist_ok=True)
    
    records_path = os.path.join(msg_dir, "records.csv")
    
    # Check for duplicates: if file exists, check if raw_msg already present
    if os.path.exists(records_path):
        with open(records_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("raw_msg", "") == raw_msg:
                    return  # duplicate, skip
                if row.get("encoded_bits", "") == encoded_bits:
                    return  # duplicate, skip
    
    timestamp = datetime.datetime.now().isoformat()
    
    file_exists = os.path.exists(records_path)
    with open(records_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "raw_msg", "encoded_bits", "timestamp"])
        if not file_exists or os.path.getsize(records_path) == 0:
            writer.writeheader()
        writer.writerow({
            "index": index,
            "filename": filename,
            "raw_msg": raw_msg,
            "encoded_bits": encoded_bits,
            "timestamp": timestamp,
        })


def load_message_records(key):
    """
    Load all message records for a given key from the message database.
    
    Args:
        key: key string identifying the carrier/message group
    
    Returns:
        pandas.DataFrame with columns: index, filename, raw_msg, encoded_bits, timestamp
        Returns an empty DataFrame if the records file does not exist.
    """
    import pandas as pd
    
    records_path = os.path.join("workspace", "database", "message", key, "records.csv")
    
    if not os.path.exists(records_path):
        return pd.DataFrame(columns=["index", "filename", "raw_msg", "encoded_bits", "timestamp"])
    
    return pd.read_csv(records_path)
