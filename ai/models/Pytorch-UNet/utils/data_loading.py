import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    """数据集加载，得到 PIL.Image 对象

    Args:
        filename (_type_): _description_

    Returns:
        PIL.Image: 高质量缩放，多种重采样算法
                   ！缩放支持 NEAREST 相邻取整，这样掩码图（分割标签图）的缩放不会改变像素的标签值！
    """    
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    """获取掩码图像的唯一标签值数组

    Args:
        idx (_type_): _description_
        mask_dir (_type_): _description_
        mask_suffix (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    # 找到图像对应的掩码文件
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # 掩码图像转为 numpy 数组
    mask = np.asarray(load_image(mask_file))
    #二维数组，灰度标签图，数组值就是掩码像素的分割标签
    if mask.ndim == 2:
        # np.unique 返回不重复的元素并排序
        return np.unique(mask)
    #三维数组，彩色标签图，数组值(R,G,B)就是掩码像素的分割标签
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        # axis=0 一行当一个元素
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    """把磁盘上的"图像 + 掩码"配对，逐样本加载、预处理，最后输出模型可直接用的 image/mask tensor

    Args:
        Dataset (_type_): _description_
    """    
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # 顺序扫描即可（数据量小时性能几乎无影响）。
        # 说明：原实现用 multiprocessing.Pool() 多进程扫描掩码，会按 os.cpu_count() fork 大量 worker；
        # 在本仓库与 DataLoader(num_workers>0) 的多进程交互下会残留空闲 worker 进程(实测)，
        # 且在 CPU 小内存机器上叠加 DataLoader worker 易触发 OOM 被系统 kill，故改为顺序处理。
        unique = [
            unique_mask_values(idx, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
            for idx in tqdm(self.ids, total=len(self.ids))
        ]

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # NEAREST 避免掩码图像缩放，像素插值
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            # pytorch 的图像维度排列顺序是 (C, H, W)，通道数在前面，后面是高宽
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # 归一化到 [0, 1]
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
