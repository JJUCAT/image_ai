"""随机分割图生成器（加权 Voronoi 扩展 + 噪声扰动）。

思路：在空图上随机撒种子（种子数 > 类别数），每个种子随机取一个类别，
按加权距离自动扩展成整幅标签图；扩展用椭圆距离 + 每类不同的扩展速度
与噪声扰动，保证非圆形、各区域纹理不同。

两阶段输出：
- 阶段一：生成 HxW 的 numpy 分割矩阵，像素值 = classes 数组对应的下标
  （类别 id），与 classes_color 无关；
- 阶段二：由 numpy 分割矩阵 + classes_color 渲染成 PNG 图像。

生成步骤分离清晰，便于按需修改（每步一个独立函数）：
1. place_seeds           随机撒种子（种子数 > 类别数），随机取类别
2. sample_class_params   为每个种子按类别采样扩展参数
                         （速度 / 边界噪声幅度 / 噪声平滑半径 / 椭圆拉伸 / 旋转）
3. expand_regions        按加权距离自动扩展 -> HxW 类别下标矩阵
4. render_color          类别下标矩阵 -> 类别颜色 RGB
5. save_seg_png          保存 RGB 为 PNG
6. save_labels_png       由类别下标矩阵直接保存 PNG（= 4 + 5）
7. save_labels_npy       保存类别下标矩阵为 .npy

核心函数均为纯 numpy 计算、入参出参明确，测试后可按需改动任意步骤。
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 复用 scripts/distance2area.py 的投影面积表 bin 读写（魔数/版本/宽高校验
# 与 bin 格式协议只在 distance2area.py 定义，见 README.md），避免协议分叉
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from distance2area import load_area_table_bin

# 每类的扩展参数采样区间 [low, high]，在区间内均匀随机取值：
#   speed        扩展速度，越大区域扩张越快（代价越小、占得越多）
#   wobble       边界噪声幅度，越大边界越扭曲（乘性扰动，相对比例）
#   noise_radius 噪声平滑半径，越大纹理越粗
#   elong        椭圆拉伸，>1 时沿旋转方向拉长，保证非圆形
#   rot_deg      椭圆旋转角（度）
_CLASS_PARAMS = {
    "background": dict(speed=(0.5, 1.0),  wobble=(0.03, 0.10), noise_radius=(1, 3), elong=(1.0, 1.8), rot_deg=(0, 180)),
    "grass":      dict(speed=(1.0, 2.0),  wobble=(0.10, 0.28), noise_radius=(3, 7), elong=(1.2, 2.5), rot_deg=(0, 180)),
    "person":     dict(speed=(0.3, 0.8),  wobble=(0.02, 0.10), noise_radius=(1, 2), elong=(1.0, 1.5), rot_deg=(0, 180)),
    "bare":       dict(speed=(0.8, 1.5),  wobble=(0.06, 0.18), noise_radius=(2, 5), elong=(1.1, 2.0), rot_deg=(0, 180)),
    "leaves":     dict(speed=(0.8, 1.6),  wobble=(0.12, 0.32), noise_radius=(4, 8), elong=(1.2, 2.5), rot_deg=(0, 180)),
    "road":       dict(speed=(1.5, 3.0),  wobble=(0.04, 0.14), noise_radius=(2, 4), elong=(1.5, 3.5), rot_deg=(0, 180)),
}
# 未在 _CLASS_PARAMS 中登记的类别的兜底参数区间
_DEFAULT_PARAMS = dict(speed=(0.5, 2.0), wobble=(0.05, 0.2), noise_radius=(2, 5), elong=(1.0, 2.5), rot_deg=(0, 180))


def load_config(config_path):
    """读取 seg_img.json 配置，返回 dict。

    配置含 classes / classes_color / table（bin 相对路径）；
    table 相对配置所在目录解析为绝对路径，图像 H/W 由 bin 头部决定
    （见 load_table_size），不再由配置的 img 字段指定。
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["table"] = (config_path.parent / cfg["table"]).resolve()
    return cfg


def load_table_size(bin_path):
    """读取投影面积表 bin，返回 (H, W) 图像尺寸。

    复用 load_area_table_bin（校验魔数/版本/宽高），bin 协议见 README.md；
    返回加载后的表 shape，即 bin 文件头声明的图像高宽。
    """
    area = load_area_table_bin(bin_path)
    return area.shape


# ---------------------------------------------------------------------------
# 噪声工具
# ---------------------------------------------------------------------------

def box_blur(field, radius):
    """可分离盒式滤波：先沿行再沿列，得到各向同性的平滑噪声场。

    用轴内累加和实现滑动窗口均值，O(H*W) 复杂度，边界用边缘值延拓。
    """
    if radius <= 0:
        return field.astype(np.float32)
    w = 2 * radius + 1
    pad = np.pad(field, ((radius, radius), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=0)
    cs = np.concatenate([np.zeros((1, cs.shape[1]), dtype=cs.dtype), cs], axis=0)
    out = (cs[w:] - cs[:-w]) / w                    # 沿行平滑
    pad = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
    cs = np.cumsum(pad, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1), dtype=cs.dtype), cs], axis=1)
    out = (cs[:, w:] - cs[:, :-w]) / w              # 沿列平滑
    return out.astype(np.float32)


def smooth_noise(shape, radius, rng):
    """生成平滑随机噪声场：白噪声 + 盒式滤波，值域约 [0,1)。

    radius 越大噪声场越平滑（纹理越粗）。
    """
    field = rng.random(shape, dtype=np.float32)
    if radius > 0:
        field = box_blur(field, radius)
    return field


# ---------------------------------------------------------------------------
# 生成步骤
# ---------------------------------------------------------------------------

def place_seeds(H, W, n_classes, n_seeds, rng, margin=4, ensure_all_classes=True):
    """在空图上随机撒 n_seeds 个种子，并随机指定类别。

    种子位置在 [margin, W-margin) x [margin, H-margin) 内均匀随机，避免贴边。
    ensure_all_classes=True 时前 n_classes 个种子打乱分配覆盖全部类别，
    其余种子随机取类别（保证每张图各类别都出现）。
    返回 (positions, class_ids)：
    - positions: (n_seeds, 2) float 数组，列为 [x, y]
    - class_ids: (n_seeds,) int 数组
    """
    if n_seeds < n_classes:
        raise ValueError(f"种子数 {n_seeds} 应不少于类别数 {n_classes}")
    positions = rng.uniform(low=margin, high=[W - margin, H - margin], size=(n_seeds, 2))
    if ensure_all_classes:
        class_ids = np.empty(n_seeds, dtype=int)
        class_ids[:n_classes] = rng.permutation(n_classes)
        class_ids[n_classes:] = rng.integers(0, n_classes, size=n_seeds - n_classes)
    else:
        class_ids = rng.integers(0, n_classes, size=n_seeds)
    return positions, class_ids


def sample_class_params(classes, class_ids, rng):
    """按每个种子的类别采样扩展参数。

    在 _CLASS_PARAMS[类别]（未登记用 _DEFAULT_PARAMS）的区间内均匀随机取值，
    使不同类别的扩展速度与纹理各不相同。
    返回 dict，每个值为 (n_seeds,) 的 float32 数组：
    speed / wobble / noise_radius / elong / rot_deg
    """
    n = len(class_ids)
    speeds = np.empty(n, dtype=np.float32)
    wobbles = np.empty(n, dtype=np.float32)
    radii = np.empty(n, dtype=np.float32)
    elongs = np.empty(n, dtype=np.float32)
    rots = np.empty(n, dtype=np.float32)
    for i, cid in enumerate(class_ids):
        p = _CLASS_PARAMS.get(classes[cid], _DEFAULT_PARAMS)
        speeds[i] = rng.uniform(*p["speed"])
        wobbles[i] = rng.uniform(*p["wobble"])
        radii[i] = rng.uniform(*p["noise_radius"])
        elongs[i] = rng.uniform(*p["elong"])
        rots[i] = rng.uniform(*p["rot_deg"])
    return {"speed": speeds, "wobble": wobbles, "noise_radius": radii,
            "elong": elongs, "rot_deg": rots}


def expand_regions(H, W, seeds, class_ids, params, rng):
    """按加权距离自动扩展种子，得到整幅类别标签图。

    每个像素归属代价最小的种子：
        cost_k(p) = dist_k(p) / speed_k * (1 + wobble_k * noise_k(p))
    - dist_k 为椭圆距离（先按 rot_deg 旋转、再按 elong 拉伸），
      保证扩展不是固定圆形；
    - speed_k 越大扩张越快（代价越小）；
    - noise_k 为该种子的平滑随机场，使边界有机扭曲，
      noise_radius 越大纹理越粗。
    返回 shape=(H, W) 的 int 标签图（值为类别 id）。
    """
    K = len(seeds)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cost = np.empty((H, W, K), dtype=np.float32)
    speeds = params["speed"]
    wobbles = params["wobble"]
    radii = params["noise_radius"]
    elongs = params["elong"]
    rots = np.deg2rad(params["rot_deg"])
    for k in range(K):
        dx = xx - seeds[k, 0]
        dy = yy - seeds[k, 1]
        c, s = np.cos(rots[k]), np.sin(rots[k])
        u = (c * dx + s * dy) / elongs[k]
        v = -s * dx + c * dy
        dist = np.sqrt(u * u + v * v)
        # 噪声场中心化到 0 附近，做乘性扰动（相对代价的比例）
        noise = smooth_noise((H, W), int(round(radii[k])), rng) - 0.5
        cost[..., k] = dist / speeds[k] * (1.0 + wobbles[k] * noise)
    seed_of = np.argmin(cost, axis=-1)
    return class_ids[seed_of]


def add_fine_noise(labels, n_classes, rng, ratio=0.03, radius=1):
    """叠加小范围噪点：把少量像素替换为随机类别。

    用平滑噪声场按阈值挑出约 ratio 比例的像素；由于场是平滑的，
    被替换像素会聚成小斑块而非纯椒盐噪点。radius 控制噪点块大小。
    返回新标签图（不改动入参 labels）。

    注：雪花噪点已从完整链路中移除，此函数保留供需要时单独调用。
    """
    field = smooth_noise(labels.shape, radius, rng)
    thr = np.quantile(field, 1.0 - ratio)
    mask = field >= thr
    out = labels.copy()
    out[mask] = rng.integers(0, n_classes, size=int(mask.sum()))
    return out


# ---------------------------------------------------------------------------
# 渲染与保存
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color):
    """#RRGGBB -> (r, g, b) uint8 元组。"""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"颜色 {hex_color!r} 不是 6 位 16 进制")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render_color(labels, classes_color):
    """类别下标矩阵（像素值=类别下标）-> 类别颜色 RGB 图，返回 (H, W, 3) uint8。"""
    H, W = labels.shape
    palette = np.array([hex_to_rgb(c) for c in classes_color], dtype=np.uint8)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(palette.shape[0]):
        rgb[labels == c] = palette[c]
    return rgb


def save_seg_png(rgb, png_path):
    """保存分割图 PNG（RGB，与输入同分辨率，不绘制坐标轴）。"""
    plt.imsave(png_path, rgb)


# ---------------------------------------------------------------------------
# 完整链路
# ---------------------------------------------------------------------------

def save_labels_npy(labels, npy_path):
    """保存 HxW 类别下标矩阵为 .npy（阶段一结果，供后续复用/测试）。"""
    np.save(npy_path, labels)


def save_labels_png(labels, classes_color, png_path):
    """由 HxW 类别下标矩阵 + classes_color 渲染并保存 PNG（阶段二）。"""
    save_seg_png(render_color(labels, classes_color), png_path)


# ---------------------------------------------------------------------------
# 完整链路
# ---------------------------------------------------------------------------

def generate_random_seg(H, W, classes, rng, n_seeds=None,
                        ensure_all_classes=True):
    """生成一张随机分割矩阵（阶段一）。

    步骤：place_seeds -> sample_class_params -> expand_regions
    返回 shape=(H, W) 的 int 矩阵，像素值 = classes 数组对应的下标
    （类别 id）。PNG 渲染由阶段二 save_labels_png / render_color 完成。
    n_seeds 默认 = 类别数 * 3（保证种子数多于类别数）。
    """
    n_classes = len(classes)
    n_seeds = n_seeds if n_seeds is not None else n_classes * 3
    positions, class_ids = place_seeds(H, W, n_classes, n_seeds, rng,
                                       ensure_all_classes=ensure_all_classes)
    params = sample_class_params(classes, class_ids, rng)
    return expand_regions(H, W, positions, class_ids, params, rng)


if __name__ == "__main__":
    # 快速自测：加载配置与 table bin，生成一张分割矩阵，打印各类占比（不写文件）
    _here = Path(__file__).resolve().parent
    _cfg = load_config(_here.parent / "config" / "seg_img.json")
    _H, _W = load_table_size(_cfg["table"])
    _labels = generate_random_seg(_H, _W, _cfg["classes"], np.random.default_rng(0))
    _counts = np.bincount(_labels.ravel(), minlength=len(_cfg["classes"]))
    _pct = 100.0 * _counts / _labels.size
    for _name, _c, _p in zip(_cfg["classes"], _counts, _pct):
        print(f"{_name:<12s} {_c:>7d} 像素  {_p:5.2f}%")
