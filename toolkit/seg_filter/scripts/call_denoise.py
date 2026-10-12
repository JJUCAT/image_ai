#!/home/bg11738/anaconda3/envs/unet/bin/python
"""调用测试 Denoise 类（toolkit/seg_filter/scripts）。

加载 denoise.json 与投影面积表 bin，实例化 src/denoise.py 的 Denoise 类，
复用 test/test_denoise.py 的 check_kernel_size 校验核尺寸计算：
- 理论 k 为奇数；
- 满足阈值条件 k*Σ(前 k 个最小行平均面积) > search_area；
- 最小性：上一个奇数 k-2 不满足条件；
- 实际 k = min(理论 k, max_kernel_size)。

运行即打印完整日志（row_ave_area 数组、升序候选、理论 k 与对应核面积、
max_kernel_size、实际 k），校验通过输出 [PASS] 一行。校验项均与
search_area / max_kernel_size 配置无关，调整 denoise.json 后可直接重跑。
路径基于脚本位置解析，任意目录均可运行。脚本 shebang 指向 unet conda 环境
（/home/bg11738/anaconda3/envs/unet/bin/python），该环境装有 torch；用
python3 手动运行时需确保当前环境已装 torch。

用法：
    python3 toolkit/seg_filter/scripts/call_denoise.py
    python3 toolkit/seg_filter/scripts/call_denoise.py --config .../denoise.json
"""

import argparse
import logging
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
for _d in (_script_dir.parent / "src", _script_dir.parent / "test"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
from denoise import Denoise  # noqa: E402
from test_denoise import check_kernel_size  # noqa: E402

_CONFIG_PATH = (_script_dir.parent / "config" / "denoise.json").resolve()


def main():
    parser = argparse.ArgumentParser(description="调用测试 Denoise 核尺寸计算")
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH,
                        help="denoise.json 配置路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    denoise = Denoise(args.config)
    theoretical_k, kernel_area, actual_k = check_kernel_size(denoise)

    print(f"\n配置: {args.config}")
    print(f"理论 k={theoretical_k}（奇数）, 核面积 k*Σ={kernel_area:.6g}, "
          f"实际 k={actual_k}（max_kernel_size={denoise.max_kernel_size}）")


if __name__ == "__main__":
    main()
