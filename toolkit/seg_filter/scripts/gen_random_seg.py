"""生成随机分割图并保存为 PNG + npy。

调用 test/random_seg_gen.py 的生成链路，分两阶段输出：
- 阶段一：生成 HxW 的 numpy 分割矩阵（像素值 = classes 数组下标），
  保存为 .npy；
- 阶段二：由分割矩阵 + classes_color 渲染成 PNG，保存在 test_output/。
图像高宽由配置 table 字段指向的 bin 文件头部决定（不再用 img 字段）。
路径基于脚本位置解析，任意目录均可运行。

用法：
    python3 toolkit/seg_filter/scripts/gen_random_seg.py            # 默认 10 张
    python3 toolkit/seg_filter/scripts/gen_random_seg.py --n 20
    python3 toolkit/seg_filter/scripts/gen_random_seg.py --seed 42  # 固定随机种子
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "test"))
from random_seg_gen import (load_config, load_table_size, generate_random_seg,
                            save_labels_npy, save_labels_png)

_CONFIG_PATH = (_script_dir.parent / "config" / "seg_img.json").resolve()
_OUT_DIR = (_script_dir.parent / "test_output").resolve()


def main():
    parser = argparse.ArgumentParser(description="生成随机分割矩阵并渲染 PNG")
    parser.add_argument("--n", type=int, default=10, help="生成张数（默认 10）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（不传则每次结果随机）")
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH,
                        help="seg_img.json 配置路径")
    parser.add_argument("--out", type=Path, default=_OUT_DIR, help="输出目录")
    args = parser.parse_args()

    config = load_config(args.config)
    H, W = load_table_size(config["table"])     # 图像高宽取自 table bin 头部
    classes = config["classes"]
    classes_color = config["classes_color"]
    n_classes = len(classes)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"图像尺寸 {H}x{W}（来自 {config['table']}）")

    rng = np.random.default_rng(args.seed)
    paths = []
    for i in range(1, args.n + 1):
        # 阶段一：HxW 分割矩阵（像素值 = classes 下标）
        labels = generate_random_seg(H, W, classes, rng)
        npy_path = args.out / f"random_seg_{i:02d}.npy"
        save_labels_npy(labels, npy_path)
        # 阶段二：分割矩阵 -> 类别颜色 PNG
        png_path = args.out / f"random_seg_{i:02d}.png"
        save_labels_png(labels, classes_color, png_path)
        paths.append(png_path)
        counts = np.bincount(labels.ravel(), minlength=n_classes)
        pct = 100.0 * counts / labels.size
        summary = " ".join(f"{name}{p:.0f}%" for name, p in zip(classes, pct))
        print(f"{png_path.name}: {summary}")

    print(f"\n共生成 {len(paths)} 张 PNG（及对应 .npy 分割矩阵），保存在 {args.out}")


if __name__ == "__main__":
    main()
