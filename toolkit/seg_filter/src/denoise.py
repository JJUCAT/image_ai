"""基于 PyTorch 的投影面积降噪类（核尺寸计算）。

Denoise 类加载 denoise.json 配置与投影面积表 bin（格式协议见 README.md，
读写复用 scripts/distance2area.py 的 load_area_table_bin，避免协议分叉），
并加载 seg_img.json 的 classes 类别定义；labels_to_class_channels 把
(H, W) 标签张量（像素值 = classes 数组下标）拆分为类别多通道 (C, H, W)。
配置里 row_ave_area 为每行平均投影面积：一个 k×k 核覆盖 k 行、每行 k 个
像素，核的总投影面积 ≈ k × Σ(核覆盖的 k 行行平均面积)；取 k 个最小的行
平均面积，得到任意 k×k 核的最小总投影面积（核放在投影面积最小的行上）。

核尺寸从 row_ave_area 底部（最小投影面积）向上检查，按公式

    k * Σ_{i=0}^{k-1} row_ave_area_sorted[i] > search_area

求满足条件的理论最小核尺寸 k；k 必须是奇数（偶数时 +1）。因 k * Σ 随 k
单调递增，只检查奇数 k 与「先找最小满足的 k 再取奇数」等价。实际使用的
核尺寸受 max_kernel_size 上限约束：actual_k = min(theoretical_k, max_kernel_size)。
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from distance2area import load_area_table_bin  # noqa: E402

_LOGGER = logging.getLogger("seg_filter.denoise")


def load_config(config_path):
    """读取 denoise.json 配置，table 相对路径按配置所在目录解析为绝对路径。

    返回 dict，键与 json 一致：table / devices / max_kernel_size /
    search_area / denoise_rows。
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["table"] = (config_path.parent / cfg["table"]).resolve()
    return cfg


def load_seg_config(config_path):
    """读取 seg_img.json 配置，table 相对路径按配置所在目录解析为绝对路径。

    返回 dict，键与 json 一致：table / classes / classes_color。
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["table"] = (config_path.parent / cfg["table"]).resolve()
    return cfg


class Denoise:
    """投影面积降噪：加载配置与投影面积表，计算降噪核尺寸。

    属性（均来自 denoise.json / seg_img.json）：
    - table            投影面积表 bin 绝对路径
    - devices          计算设备（torch.device）
    - search_area       核尺寸搜索的面积阈值，核的理论总投影面积需大于该值
    - max_kernel_size  核尺寸上限（实际 k = min(理论 k, 上限)）
    - denoise_rows     从最小投影面积行向上参与搜索的行数上限
    - area_table       投影面积表 (H, W) torch float32，无效像素为 +inf
    - row_ave_area     每行平均投影面积 (H,) torch float32，行内含无效
                       像素时为 +inf（不计入求和）
    - seg_config       来自 seg_img.json 的配置 dict
    - classes          类别名列表（seg_img.json -> "classes"），标签像素
                       值为该数组下标，类别通道按下标对应
    """

    def __init__(self, config_path, seg_config_path=None, logger=None):
        self.logger = logger or _LOGGER
        self.config = load_config(config_path)
        if seg_config_path is None:
            # 默认取与 denoise.json 同目录下的 seg_img.json
            seg_config_path = Path(config_path).resolve().parent / "seg_img.json"
        self.seg_config = load_seg_config(seg_config_path)
        self.classes = self.seg_config["classes"]
        self.device = torch.device(self.config["devices"])
        self.search_area = float(self.config["search_area"])
        self.max_kernel_size = int(self.config["max_kernel_size"])
        self.denoise_rows = int(self.config["denoise_rows"])

        area, row_ave = load_area_table_bin(self.config["table"], return_row_ave=True)
        self.area_table = torch.from_numpy(area).to(self.device)
        self.row_ave_area = torch.from_numpy(row_ave).to(self.device)

    def compute_kernel_size(self):
        """计算理论最小奇数核尺寸并输出日志。

        从 row_ave_area 底部（最小投影面积）向上检查：先过滤无效行（inf），
        升序排列取前 denoise_rows 行作为候选；只检查奇数 k，第一个满足
        k * cumsum[k-1] > search_area 的奇数 k 即为理论核尺寸。

        返回 (theoretical_k, kernel_area, actual_k)：
        - theoretical_k  理论最小奇数核尺寸（候选行内无解时为 None）
        - kernel_area    该理论核的总投影面积 k * Σ（float）
        - actual_k       实际核尺寸 = min(theoretical_k, max_kernel_size)
        """
        finite = self.row_ave_area[torch.isfinite(self.row_ave_area)]
        if finite.numel() == 0:
            raise ValueError("row_ave_area 没有有效行（全部为 +inf）")
        # 底部（最小投影面积）开始向上：升序排列，只取前 denoise_rows 行
        ascending = torch.sort(finite).values[: self.denoise_rows]
        if ascending.numel() == 0:
            raise ValueError(f"denoise_rows={self.denoise_rows} 无候选行（有效行 "
                             f"{finite.numel()} 行）")
        cumsum = torch.cumsum(ascending, dim=0)

        ks = torch.arange(1, ascending.numel() + 1, step=2, device=self.device)
        hit = ks * cumsum[ks - 1] > self.search_area

        # 日志：配置 + row_ave_area 数组 + 升序候选数组
        self.logger.info("降噪配置: table=%s devices=%s search_area=%.6g "
                         "max_kernel_size=%d denoise_rows=%d",
                         self.config["table"], self.device,
                         self.search_area, self.max_kernel_size, self.denoise_rows)
        raw = self.row_ave_area.cpu().numpy()
        self.logger.info("row_ave_area（每行平均投影面积, 长度=%d, 无效行为 inf）:\n%s",
                         raw.size, np.array2string(raw, precision=4, threshold=raw.size,
                                                   max_line_width=120, separator=", "))
        cand = ascending.cpu().numpy()
        self.logger.info("升序候选（底部=最小投影面积, 有效行 %d 取前 %d 行）:\n%s",
                         finite.numel(), self.denoise_rows,
                         np.array2string(cand, precision=4, threshold=cand.size,
                                         max_line_width=120, separator=", "))

        if not hit.any():
            self.logger.warning("候选 %d 行内没有满足 k*Σ > search_area=%.6g 的奇数 k，"
                                "实际核尺寸取 max_kernel_size=%d",
                                ascending.numel(), self.search_area, self.max_kernel_size)
            return None, float("nan"), self.max_kernel_size

        k = int(ks[hit][0])
        row_sum = float(cumsum[k - 1])
        kernel_area = float(k * cumsum[k - 1])
        actual_k = min(k, self.max_kernel_size)

        self.logger.info("理论 k=%d（奇数）, Σ(前 %d 行)=%.6g, 核面积 k*Σ=%.6g > "
                         "search_area=%.6g", k, k, row_sum, kernel_area, self.search_area)
        self.logger.info("max_kernel_size（denoise.json）=%d", self.max_kernel_size)
        self.logger.info("实际 k = min(理论 k=%d, max_kernel_size=%d) = %d",
                         k, self.max_kernel_size, actual_k)
        return k, kernel_area, actual_k

    def labels_to_class_channels(self, labels):
        """标签张量 (H, W) 拆分为类别多通道 (C, H, W) torch float32。

        像素值为 self.classes（seg_img.json -> "classes"）数组下标；第 c 个
        通道为类别 c 的掩码（真值 1、其余 0），通道与 classes 按下标一一对应，
        布局与 UNet 输出 (B, C, H, W) 去掉 batch 维后一致。标签值必须在
        [0, C) 内，越界（含负值）抛 ValueError，避免类别外的像素被静默丢弃。
        """
        n_classes = len(self.classes)
        if labels.dim() != 2:
            raise ValueError(f"标签张量应为 (H, W)，实际 shape {tuple(labels.shape)}")
        if labels.numel() == 0:
            return torch.zeros((n_classes, *labels.shape), dtype=torch.float32,
                               device=labels.device)
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= n_classes:
            raise ValueError(f"标签值应为 classes 数组下标 [0, {n_classes})，"
                             f"实际范围 [{lo}, {hi}]")
        return torch.stack([labels == c for c in range(n_classes)]).float()
