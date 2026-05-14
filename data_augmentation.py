import numpy as np

import torch.nn.functional as F
from torchvision.transforms import functional


# 数据增强空模板
class DifferentiableDataAugmentation:
    def __init__(self):
        pass

    def sample_params(self, x, seed=None):
        """Sample parameters for a given data augmentation"""
        return 0

    def apply(self, x, params):
        """Apply data augmentation to image"""
        assert params == 0
        return x

    def __call__(self, x, params):
        return self.apply(x, params)


# 启用数据增强
class All(DifferentiableDataAugmentation):

    def __init__(
        self,
        degrees=30,  # 在 ±30 度范围内随机旋转
        crop_scale=(0.2, 1.0),  # 裁剪区域可以是原始图像的 20% 到 100%
        crop_ratio=(3 / 4, 4 / 3),  # 裁剪区域的宽高比可以在 3/4 到 4/3 之间
        resize_scale=(0.2, 1.0),  # 调整大小区域可以是原始图像的 20% 到 100%
        blur_size=17,  # 模糊核的大小可以在 1 到 17 之间（必须是奇数）
        flip=True,  # 有 50% 的概率被水平翻转
        interpolation="bilinear",  # 双线性插值
    ):
        """
        Apply a data augmentations, chosen at random between none, rotation, crop, resize, blur, with random parameters.

        Args:
            degrees (float): Amplitude of rotation augmentation (in ±degrees)
            crop_scale (tuple of float): Lower and upper bounds for the random area of the crop before resizing
            crop_ratio (tuple of float): Lower and upper bounds for the random aspect ratio of the crop, before resizing
            resize_scale (tuple of float): Lower and upper bounds for the random area of the resizing
            blur_size (int): Upper bound of the size of the blur kernel (sigma=ksize*0.15+0.35 and ksize=(sigma-0.35)/0.15)
            flip (boolean): whether to apply random horizontal flip
        """
        self.degrees = degrees
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.resize_scale = resize_scale
        self.blur_size = blur_size
        self.flip = flip
        self.interpolation = functional.InterpolationMode(
            interpolation
        )  # 选择对应的插值模式

    def sample_params(self, x):
        # randomly select one of augmentations
        ps = np.array([1, 1, 1, 1, 1])
        ps = ps / ps.sum()  # 归一化平分概率
        augm_type = np.random.choice(
            ["none", "rotation", "crop", "resize", "blur"], p=ps
        )  # 随机选择数据增强类型，选择概率由ps决定

        # flip param
        f = np.random.rand() > 0.5 if self.flip else 0  # 50%的概率为True(25%*50%)

        # sample params
        # none
        if augm_type == "none":
            return augm_type, 0, f
        # rotation
        elif augm_type == "rotation":
            d = (
                np.random.vonmises(0, 1) * self.degrees / np.pi
            )  # 生成[-degrees,+degrees]的基于冯米塞斯分布的角度
            return augm_type, d, f
        # crop or resize
        elif augm_type in ["crop", "resize"]:
            width, height = functional.get_image_size(x)  # 获取图片形状
            area = height * width  # 图片面积
            target_area = np.random.uniform(*self.crop_scale) * area  # 裁剪后图片面积
            aspect_ratio = np.exp(
                np.random.uniform(
                    np.log(self.crop_ratio[0]), np.log(self.crop_ratio[1])
                )
            )
            tw = int(np.round(np.sqrt(target_area * aspect_ratio)))  # 目标区域宽
            th = int(np.round(np.sqrt(target_area / aspect_ratio)))  # 目标区域高

            # crop
            if augm_type == "crop":
                i = np.random.randint(0, max(min(height - th + 1, height - 1), 0) + 1)
                j = np.random.randint(0, max(min(width - tw + 1, width - 1), 0) + 1)
                return augm_type, (i, j, th, tw), f
            # resize
            elif augm_type == "resize":
                s = np.random.uniform(*self.resize_scale)
                return augm_type, (s, th, tw), f
        # blur
        elif augm_type == "blur":
            b = np.random.randint(1, self.blur_size + 1)
            b = b - (1 - b % 2)  # make it odd
            return augm_type, b, f

    def apply(self, x, augmentation):
        augm_type, param, f = augmentation
        if augm_type == "blur":
            x = functional.gaussian_blur(x, param)  # 高斯滤波
        if augm_type == "rotation":
            x = functional.rotate(x, param, interpolation=self.interpolation)
            # x = functional.rotate(x, d, expand=True, interpolation=self.interpolation)
        elif augm_type == "crop":
            x = functional.crop(x, *param)
        elif augm_type == "resize":
            s, h, w = param
            x = functional.resize(
                x, int((s**0.5) * min(h, w)), interpolation=self.interpolation,antialias=True
            )
        x = functional.hflip(x) if f else x  # 水平翻转
        return x
