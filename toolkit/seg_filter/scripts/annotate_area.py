"""对已生成的随机分割图标注连通域投影面积。

读取 test_output 下生成的 random_seg_XX.npy 分割矩阵，对每个类别求
8 连通域；查表 project_area_table.bin（配置 table 字段）累加每个连通域
的投影面积；在原 random_seg_XX.png 上标注面积，保存 random_seg_XX_area.png。

用法：
    python3 toolkit/seg_filter/scripts/annotate_area.py
    python3 toolkit/seg_filter/scripts/annotate_area.py --font-size 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "test"))
from random_seg_gen import load_config, render_color  # noqa: E402
from area_annotate import (connected_components, component_areas,
                           annotate_area_png)  # noqa: E402
from distance2area import load_area_table_bin  # noqa: E402

_CONFIG_PATH = (_script_dir.parent / "config" / "seg_img.json").resolve()
_OUT_DIR = (_script_dir.parent / "test_output").resolve()


def main():
    parser = argparse.ArgumentParser(description="标注连通域投影面积")
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH,
                        help="seg_img.json 配置路径")
    parser.add_argument("--out", type=Path, default=_OUT_DIR,
                        help="输入/输出目录（random_seg_XX.npy / .png）")
    parser.add_argument("--font-size", type=int, default=8, help="标注字号（默认 8）")
    args = parser.parse_args()

    config = load_config(args.config)
    classes = config["classes"]
    n_classes = len(classes)
    area_table = load_area_table_bin(config["table"])   # 投影面积表

    npy_paths = sorted(args.out.glob("random_seg_*.npy"))
    if not npy_paths:
        raise SystemExit(f"目录下没有 random_seg_*.npy: {args.out}")
    for npy_path in npy_paths:
        labels = np.load(str(npy_path))
        cc_labels, n_comp, comp_class = connected_components(labels, n_classes)
        areas, valid_counts = component_areas(cc_labels, n_comp, area_table)

        # 底图：优先用已生成的 png，缺失时由矩阵渲染
        png_path = npy_path.with_suffix(".png")
        if png_path.exists():
            rgb = (mpimg.imread(str(png_path))[..., :3] * 255).round().astype(np.uint8)
        else:
            rgb = render_color(labels, config["classes_color"])
        out_path = npy_path.with_name(npy_path.stem + "_area.png")
        annotate_area_png(rgb, cc_labels, areas, out_path,
                          font_size=args.font_size)

        print(f"{npy_path.name}: {n_comp} 个连通域 -> {out_path.name}")
        for cid in range(1, n_comp + 1):
            print(f"  #{cid:<2d} {classes[comp_class[cid]]:<10s} "
                  f"有效像素{valid_counts[cid]:>6d}  投影面积 {areas[cid]:.4g}")

    print(f"\n标注完成，保存在 {args.out}")


if __name__ == "__main__":
    main()
