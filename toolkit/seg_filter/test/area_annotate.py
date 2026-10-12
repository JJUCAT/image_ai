"""连通域投影面积标注（后处理阶段）。

在已生成的随机分割矩阵/PNG 基础上做标注，步骤分离清晰：
1. connected_components   对每个类别求 8 连通域，跨类别统一编号
2. component_areas        查投影面积表 project_area_table.bin，
                          累加每个连通域内有效像素的投影面积
3. annotate_area_png      在每个连通域质心标注投影面积，保存 PNG

依赖 scipy.ndimage（连通域）与 Pillow（文本绘制）；投影面积表 bin 协议
复用 scripts/distance2area.py 的 load_area_table_bin（魔数/版本/宽高校验，
见 README.md），避免协议分叉。
"""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from distance2area import load_area_table_bin

# 标注用字体：优先 DejaVu（可读性优于 PIL 默认位图字体）
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _load_font(font_size):
    """加载标注字体，找不到系统字体时退回 PIL 默认字体。"""
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, font_size)
    return ImageFont.load_default()


def connected_components(labels, n_classes):
    """对每个类别求 8 连通域，跨类别统一编号。

    返回 (cc_labels, n_components, comp_class)：
    - cc_labels:   (H, W) int，0 为背景，1..n_components 为连通域编号
    - n_components: int 连通域总数
    - comp_class:  (n_components+1,) int，comp_class[cid] 为连通域类别下标
    """
    structure = ndimage.generate_binary_structure(2, 2)   # 8 连通
    cc_labels = np.zeros(labels.shape, dtype=int)
    comp_class = [0]
    next_id = 1
    for c in range(n_classes):
        lab, n = ndimage.label(labels == c, structure=structure)
        if n:
            sel = lab > 0
            cc_labels[sel] = lab[sel] + next_id - 1       # 跨类别唯一编号
            comp_class.extend([c] * n)
            next_id += n
    return cc_labels, next_id - 1, np.array(comp_class, dtype=int)


def component_areas(cc_labels, n_components, area_table):
    """查投影面积表，累加每个连通域内有效像素的投影面积。

    面积表无效像素为 +inf（投影范围外）。连通域内一旦存在无效像素，
    该连通域的投影面积无法完整计算，标注为 inf；全部有效时才返回
    有效像素面积和。
    返回 (areas, valid_counts)：
    - areas:        (n_components+1,) float，areas[cid] 为连通域 cid 投影面积
                    （可能为 inf；下标 0 为背景，恒为 0）
    - valid_counts: (n_components+1,) int，连通域 cid 内有效像素数
    """
    finite = np.isfinite(area_table)
    areas = np.zeros(n_components + 1, dtype=float)
    valid_counts = np.zeros(n_components + 1, dtype=int)
    for cid in range(1, n_components + 1):
        mask = cc_labels == cid
        valid_counts[cid] = int((mask & finite).sum())
        if (mask & ~finite).any():
            areas[cid] = np.inf        # 含投影范围外像素，面积不可计算
        else:
            areas[cid] = float(area_table[mask & finite].sum())
    return areas, valid_counts


def annotate_area_png(rgb, cc_labels, areas, out_path, font_size=8, stroke_width=1):
    """在 rgb 图上每个连通域质心标注投影面积，保存 PNG。

    rgb 为 (H, W, 3) uint8；cc_labels 为连通域编号图；
    areas[cid] 为连通域 cid 的投影面积（可为 inf，标注为 "inf"；
    下标 0 为背景，不标注）。
    文字带黑色描边白字，保证在任意底色下可读。
    """
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    for cid in range(1, len(areas)):
        ys, xs = np.where(cc_labels == cid)
        if ys.size == 0:
            continue
        text = f"{areas[cid]:.4g}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = float(xs.mean()) - tw / 2
        cy = float(ys.mean()) - th / 2
        draw.text((cx, cy), text, font=font, fill=(255, 255, 255),
                  stroke_width=stroke_width, stroke_fill=(0, 0, 0))
    img.save(out_path)
