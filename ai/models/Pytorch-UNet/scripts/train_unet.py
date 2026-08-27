#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 train.py 的 U-Net 训练脚本：训练逻辑与 train.py 一致，
但【数据路径】和【训练超参数】全部从 config/config.json 读取。

config.json 结构：
{
  "imgs": "...",            # 原图目录（替代硬编码 ./data/imgs/）
  "masks": "...",           # 掩码目录（替代硬编码 ./data/masks/）
  "checkpoints": "...",     # 权重输出目录（替代硬编码 ./checkpoints/）
  "train": {
    "epochs": 5,
    "batch_size": 10,
    "learning_rate": 1e-4,
    "val_percent": 0.1,     # 验证集比例 (0~1)
    "save_checkpoint": true,
    "img_scale": 1.0,       # 图像缩放因子
    "amp": false,           # 混合精度（无 GPU 请保持 false）
    "weight_decay": 1e-8,
    "momentum": 0.9,
    "gradient_clipping": 1.0
  },
  "test_output": "...",     # 验证集测试结果（叠加图 + 训练报告）输出目录
  "mask_transparent": 0.5   # 分割掩码叠加到原图的透明度 (0~1)
}

用法：
    # 使用默认 config/config.json
    python scripts/train_unet.py

    # 指定配置文件
    python scripts/train_unet.py --config /path/to/config.json

    # 类别数 / 上采样方式 / 显式指定要加载的权重（config.json 中没有，需命令行指定）
    python scripts/train_unet.py --classes 2 --bilinear --load xxx.pth

    # 可选：命令行覆盖 config 中的参数
    python scripts/train_unet.py --epochs 20 --batch-size 4 --amp

特性：
    - checkpoints 输出的权重按时间命名：checkpoint_YYYYMMDD_HHMMSS_epochN.pth
    - 自动续训：checkpoints 目录已有 .pth 权重时自动加载最新的一个继续训练；
      目录为空或不存在时从零开始训练。也可用 --load 显式指定要加载的权重。
    - 验证集测试：训练结束后用最终模型在验证集上预测，按 mask_transparent 透明度
      将分割掩码叠加到原图，输出到 test_output；同时生成训练报告
      train_report_<时间戳>.txt（含开始/结束时间、耗时、设备、路径、数据量等）。
      分割掩码颜色定义在本脚本头部 MASK_COLORS（类别 0=黑色，1=绿色）。
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm

# 让脚本在任意 cwd 下都能 import 仓库内的 unet / utils / evaluate
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

DEFAULT_CONFIG = REPO_DIR / 'config' / 'config.json'

# 分割类型掩码颜色定义（类别索引 -> 十六进制颜色字符串），定义在脚本头部，便于统一修改
# 目前：类别 0 = 黑色（背景），类别 1 = 绿色（前景）
MASK_COLORS = [
    '#000000',      # 0: 黑色
    '#00FF00',      # 1: 绿色
]


def to_rgb_channels(img):
    """把 (C, H, W) 图像扩展为 3 通道。

    单通道 (1, H, W) -> 重复为 (3, H, W)；已是 3 通道保持不变。
    用于兼容灰度图输入与 n_channels=3 的 UNet。
    """
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img


class RGBDatasetWrapper:
    """数据集包装器：检测输入图像通道数，若为 1 通道则扩展为 3 通道。

    这样训练 DataLoader、验证 DataLoader（evaluate）以及验证集测试
    （save_validation_results）拿到的图像通道数始终与 UNet(n_channels=3) 匹配。
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = dict(item)
        item['image'] = to_rgb_channels(item['image'])
        return item

    @property
    def ids(self):
        return self.dataset.ids

    @property
    def mask_values(self):
        return self.dataset.mask_values


def load_config(path):
    """读取并校验 config.json，返回配置字典。"""
    config_path = Path(path)
    if not config_path.exists():
        sys.exit(f'[错误] 未找到配置文件：{config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for key in ('imgs', 'masks', 'checkpoints'):
        if not config.get(key):
            sys.exit(f'[错误] config.json 缺少必填字段：{key}')

    train = config.get('train', {})
    required = ('epochs', 'batch_size', 'learning_rate')
    for key in required:
        if key not in train:
            sys.exit(f'[错误] config.json 的 train 段缺少必填字段：{key}')

    return config


def find_latest_checkpoint(dir_checkpoint):
    """在 checkpoints 目录中查找最新的 .pth 权重文件；目录不存在或为空则返回 None。"""
    ckpt_dir = Path(dir_checkpoint)
    if not ckpt_dir.exists():
        return None
    files = [f for f in ckpt_dir.glob('*.pth') if f.is_file()]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def hex_to_rgb(color):
    """把十六进制颜色字符串 '#RRGGBB' 转成 (R, G, B) 元组。"""
    color = str(color).lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def colorize_mask(pred_mask, colors=MASK_COLORS):
    """把类别索引掩码 (H, W) 转成 RGB 彩色掩码 (H, W, 3)。

    colors : 每类颜色，支持十六进制字符串（如 '#00FF00'）或 (R,G,B) 元组。
    """
    pred_mask = np.asarray(pred_mask)
    h, w = pred_mask.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in enumerate(colors):
        rgb = hex_to_rgb(color) if isinstance(color, str) else color
        out[pred_mask == cls] = rgb
    return out


def overlay_mask_on_image(image, pred_mask, alpha, colors=MASK_COLORS):
    """将分割掩码按透明度 alpha 叠加到原图上，返回 PIL RGB 图像。

    image     : 原图，可为 (C,H,W) 或 (H,W,C) 的 float 张量/数组，取值 [0,1]
                （与 dataset 预处理后的格式一致）
    pred_mask : (H,W) 类别索引掩码
    alpha     : 掩码透明度 (0~1)，即 config 的 mask_transparent
    """
    if torch.is_tensor(image):
        img = image.detach().cpu().numpy()
    else:
        img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = img.transpose(1, 2, 0)  # CHW -> HWC
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = np.clip(img, 0.0, 1.0)

    color = colorize_mask(pred_mask, colors) / 255.0
    overlay = alpha * color + (1.0 - alpha) * img
    return Image.fromarray((overlay * 255).astype(np.uint8))


def save_validation_results(model, val_set, dataset, device, amp, test_output_dir,
                            alpha, n_classes, run_timestamp):
    """训练结束后用最终模型在验证集上预测，叠加分割掩码并保存到 test_output。

    输出：test_output/pred_<时间戳>/<原图名>_overlay.png
    返回保存的张数。
    """

    # model.train() 训练模式，Dropout 会随机丢弃一部分神经元，用来防止过拟合
    # model.eval() 推理模式，Dropout 关闭
    model.eval()
    out_dir = Path(test_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / f'pred_{run_timestamp}'
    pred_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    # torch.no_grad() 不需要记录梯度计算图
    with torch.no_grad():
        for pos, orig_idx in enumerate(val_set.indices):
            item = val_set[pos]
            # pytorch 输入要求知道 batch 大小，即 [B, C, H, W]（batch，通道数，高，宽），unsqueeze(0)在第 0 维度加一个维度
            # 
            image_t = item['image'].unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                output = model(image_t).cpu()
            if n_classes > 1:
                pred = output.argmax(dim=1)[0]
            else:
                pred = (torch.sigmoid(output) > 0.5).long()[0]

            img_np = item['image'].numpy()          # CHW [0,1]
            overlay = overlay_mask_on_image(img_np, pred.numpy(), alpha)
            name = dataset.ids[orig_idx] if hasattr(dataset, 'ids') else f'val_{pos:04d}'
            overlay.save(pred_dir / f'{name}_overlay.png')
            count += 1
    logging.info(f'验证集测试结果已保存到 {pred_dir}，共 {count} 张')
    model.train()
    return count


def write_training_report(test_output_dir, run_timestamp, info):
    """把本次训练报告写入 test_output/train_report_<时间戳>.txt。"""
    out_dir = Path(test_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f'train_report_{run_timestamp}.txt'

    mask_colors_str = ', '.join(
        f'{i}:{c}' for i, c in enumerate(MASK_COLORS)
    )
    lines = [
        '==================== U-Net 训练报告 ====================',
        f'训练开始时间     : {info.get("start_time", "")}',
        f'训练结束时间     : {info.get("end_time", "")}',
        f'训练耗时         : {info.get("duration", "")}',
        f'设备配置         : {info.get("device", "")}',
        f'  - 设备类型     : {info.get("device_type", "")}',
        f'  - CUDA 可用    : {info.get("cuda_available", "")}',
        f'  - CUDA 设备    : {info.get("cuda_device", "")}',
        f'  - PyTorch 版本 : {info.get("torch_version", "")}',
        f'输入源路径       : {info.get("imgs", "")}',
        f'  - 原图目录     : {info.get("imgs", "")}',
        f'  - 掩码目录     : {info.get("masks", "")}',
        f'输出路径         : {info.get("checkpoints", "")}',
        f'  - checkpoints  : {info.get("checkpoints", "")}',
        f'  - test_output  : {info.get("test_output", "")}',
        f'训练数据量       : {info.get("n_train", "")}',
        f'验证(测试)数据量 : {info.get("n_val", "")}',
        f'轮数 epochs      : {info.get("epochs", "")}',
        f'批大小 batch_size: {info.get("batch_size", "")}',
        f'学习率 lr        : {info.get("learning_rate", "")}',
        f'图像缩放 scale   : {info.get("img_scale", "")}',
        f'混合精度 amp     : {info.get("amp", "")}',
        f'类别数 classes   : {info.get("n_classes", "")}',
        f'掩码叠加透明度   : {info.get("mask_transparent", "")}',
        f'掩码颜色         : {mask_colors_str}',
        f'加载权重         : {info.get("load_path", "无（从零训练）")}',
        f'运行时间戳       : {info.get("run_timestamp", "")}',
        '=======================================================',
    ]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logging.info(f'训练报告已写入: {report_path}')


def train_model(
        model,
        device,
        config,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        dir_img=None,
        dir_mask=None,
        dir_checkpoint=None,
        dir_test_output='',
        mask_transparent=0.5,
        run_timestamp: str = '',
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 检测输入图通道数：若为 1 通道（灰度图），统一扩展为 3 通道，
    # 保证与 UNet(n_channels=3) 匹配；训练/验证(evaluate)/验证集测试全部走此包装。
    dataset = RGBDatasetWrapper(dataset)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # CPU 机器上 num_workers 必须为 0，原因有二（本机实测）：
    #   1) num_workers>0 会 fork 大量 worker 进程，加上 BasicDataset 内部用 Pool() 扫描掩码，
    #      多进程交互导致 Pool worker 进程残留(实测残留 os.cpu_count() 个空闲进程)并可能死锁；
    #   2) CPU 训练时 worker 过多会耗尽内存，被系统 OOM killer 直接 kill 进程。
    # 数据量不大(几百张)时，主进程(num_workers=0)顺序加载性能几乎无影响。
    # GPU 上仍需并行加载以喂饱显卡，保留少量 worker 并启用 pin_memory。
    if device.type == 'cuda':
        num_workers = min(os.cpu_count(), 4)
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            filename = 'checkpoint_{}_epoch{}.pth'.format(run_timestamp, epoch)
            torch.save(state_dict, str(Path(dir_checkpoint) / filename))
            logging.info(f'Checkpoint saved: {filename}')

    # 训练结束后：在验证集上预测，叠加分割掩码保存到 test_output
    if dir_test_output:
        save_validation_results(
            model=model,
            val_set=val_set,
            dataset=dataset,
            device=device,
            amp=amp,
            test_output_dir=dir_test_output,
            alpha=mask_transparent,
            n_classes=model.n_classes,
            run_timestamp=run_timestamp,
        )

    return {
        'n_train': n_train,
        'n_val': n_val,
    }


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks (parameters from config.json)'
    )
    parser.add_argument('--config', default=str(DEFAULT_CONFIG),
                        help='Path to config.json (default: config/config.json)')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    # 以下参数默认 None：不传则使用 config.json 的值；传了则覆盖 config
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=None, help='Number of epochs (override config)')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=None,
                        help='Batch size (override config)')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=None,
                        help='Learning rate (override config)', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=None,
                        help='Downscaling factor of the images (override config)')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=None,
                        help='Percent of the data used as validation 0-100 (override config)')
    parser.add_argument('--amp', action='store_true', default=None,
                        help='Use mixed precision (override config)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    config = load_config(args.config)
    train_cfg = config['train']

    # 训练参数：优先命令行覆盖，否则用 config.json
    epochs = args.epochs if args.epochs is not None else train_cfg.get('epochs', 5)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get('batch_size', 1)
    learning_rate = args.lr if args.lr is not None else train_cfg.get('learning_rate', 1e-5)
    val_percent = (args.val / 100) if args.val is not None else train_cfg.get('val_percent', 0.1)
    save_checkpoint = train_cfg.get('save_checkpoint', True)
    img_scale = args.scale if args.scale is not None else train_cfg.get('img_scale', 0.5)
    amp = args.amp if args.amp is not None else bool(train_cfg.get('amp', False))
    weight_decay = train_cfg.get('weight_decay', 1e-8)
    momentum = train_cfg.get('momentum', 0.999)
    gradient_clipping = train_cfg.get('gradient_clipping', 1.0)

    dir_img = config['imgs']
    dir_mask = config['masks']
    dir_checkpoint = config['checkpoints']
    dir_test_output = config.get('test_output', '')
    mask_transparent = float(config.get('mask_transparent', 0.5))

    logging.info(f'数据路径: imgs={dir_img} masks={dir_mask} checkpoints={dir_checkpoint} '
                 f'test_output={dir_test_output}')
    logging.info(f'训练参数: epochs={epochs} batch_size={batch_size} lr={learning_rate} '
                 f'val_percent={val_percent} img_scale={img_scale} amp={amp} '
                 f'mask_transparent={mask_transparent}')

    # 网络结构：n_channels=3 for RGB images
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 自动续训：未显式指定 --load 时，若 checkpoints 目录已有模型则加载最新的一个继续训练
    load_path = args.load
    if not load_path:
        latest = find_latest_checkpoint(dir_checkpoint)
        if latest is not None:
            load_path = str(latest)
            logging.info(f'检测到已有模型，自动续训：{load_path}')
        else:
            logging.info('checkpoints 目录没有已有模型，从零开始训练')

    if load_path:
        try:
            state_dict = torch.load(load_path, map_location=device)
            state_dict.pop('mask_values', None)
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {load_path}')
        except RuntimeError as e:
            logging.warning(f'加载 {load_path} 失败（可能与当前模型结构/类别数不匹配）：{e}')
            logging.warning('已回退为从零开始训练')

    model.to(device=device)

    # 本次运行的统一时间戳，用于 checkpoints 输出命名
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.info(f'本次运行时间戳: {run_timestamp}')

    # 记录训练起止时间，用于训练报告
    start_time = datetime.now()
    train_stats = None
    try:
        train_stats = train_model(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            config=config,
            img_scale=img_scale,
            val_percent=val_percent,
            amp=amp,
            save_checkpoint=save_checkpoint,
            weight_decay=weight_decay,
            momentum=momentum,
            gradient_clipping=gradient_clipping,
            dir_img=dir_img,
            dir_mask=dir_mask,
            dir_checkpoint=dir_checkpoint,
            dir_test_output=dir_test_output,
            mask_transparent=mask_transparent,
            run_timestamp=run_timestamp,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_stats = train_model(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            config=config,
            img_scale=img_scale,
            val_percent=val_percent,
            amp=amp,
            save_checkpoint=save_checkpoint,
            weight_decay=weight_decay,
            momentum=momentum,
            gradient_clipping=gradient_clipping,
            dir_img=dir_img,
            dir_mask=dir_mask,
            dir_checkpoint=dir_checkpoint,
            dir_test_output=dir_test_output,
            mask_transparent=mask_transparent,
            run_timestamp=run_timestamp,
        )

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f'训练完成，耗时 {str(duration).split(".")[0]}')

    # 训练报告输出到 test_output
    if dir_test_output:
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
        write_training_report(
            test_output_dir=dir_test_output,
            run_timestamp=run_timestamp,
            info={
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': str(duration).split('.')[0],
                'device': device,
                'device_type': device.type,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': cuda_name,
                'torch_version': torch.__version__,
                'imgs': dir_img,
                'masks': dir_mask,
                'checkpoints': dir_checkpoint,
                'test_output': dir_test_output,
                'n_train': train_stats.get('n_train') if train_stats else '',
                'n_val': train_stats.get('n_val') if train_stats else '',
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'img_scale': img_scale,
                'amp': amp,
                'n_classes': args.classes,
                'mask_transparent': mask_transparent,
                'load_path': load_path if load_path else '无（从零训练）',
                'run_timestamp': run_timestamp,
            },
        )
