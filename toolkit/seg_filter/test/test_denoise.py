"""Denoise 核尺寸计算测试（toolkit/seg_filter/test）。

加载 denoise.json 与投影面积表 bin，实例化 src/denoise.py 的 Denoise 类，
运行核尺寸计算并校验：
- 理论 k 为奇数；
- 满足阈值条件 k*Σ(前 k 个最小行平均面积) > search_area；
- 最小性：上一个奇数 k-2 不满足条件；
- 实际 k = min(理论 k, max_kernel_size)。

运行即打印完整日志（row_ave_area 数组、升序候选、理论 k 与对应核面积、
max_kernel_size、实际 k），校验通过输出 [PASS] 一行。
校验项均与 search_area / max_kernel_size 配置无关（随配置取值变化），
因此调整 denoise.json 后可直接重跑，无需改动测试。
"""

import logging
import sys
from pathlib import Path

import torch

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
from denoise import Denoise  # noqa: E402

_CONFIG_PATH = (Path(__file__).resolve().parent.parent / "config" / "denoise.json").resolve()


def check_kernel_size(denoise):
    """运行核尺寸计算，校验奇偶性 / 阈值 / 最小性 / 上限，返回计算结果。"""
    theoretical_k, kernel_area, actual_k = denoise.compute_kernel_size()

    assert theoretical_k is not None, "候选行内应能找到满足条件的奇数 k"
    assert isinstance(theoretical_k, int) and theoretical_k >= 1
    assert theoretical_k % 2 == 1, f"理论 k 必须是奇数，实际 {theoretical_k}"

    # 重新按公式算一遍，交叉核对返回的核面积
    finite = denoise.row_ave_area[torch.isfinite(denoise.row_ave_area)]
    ascending = torch.sort(finite).values[: denoise.denoise_rows]
    cumsum = torch.cumsum(ascending, dim=0)
    area = float(theoretical_k * cumsum[theoretical_k - 1])
    assert area > denoise.search_area, (
        f"理论 k={theoretical_k} 核面积 {area:.6g} 应大于 search_area {denoise.search_area:.6g}")
    assert area == kernel_area, f"返回核面积 {kernel_area:.6g} 与重算 {area:.6g} 不一致"

    # 最小性：上一个奇数 k-2 必须不满足条件（k=1 时无前一个奇数）
    if theoretical_k > 1:
        prev_area = float((theoretical_k - 2) * cumsum[theoretical_k - 3])
        assert prev_area <= denoise.search_area, (
            f"k={theoretical_k - 2} 核面积 {prev_area:.6g} 不应超过 search_area "
            f"{denoise.search_area:.6g}（理论 k 应为最小）")

    # 实际 k = min(理论 k, max_kernel_size)
    assert actual_k == min(theoretical_k, denoise.max_kernel_size), (
        f"实际 k={actual_k} 应等于 min(理论 k={theoretical_k}, "
        f"max_kernel_size={denoise.max_kernel_size})")

    print(f"\n[PASS] 理论 k={theoretical_k}（奇数）, 核面积 k*Σ={kernel_area:.6g}, "
          f"实际 k={actual_k}（max_kernel_size={denoise.max_kernel_size}）")
    return theoretical_k, kernel_area, actual_k


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    denoise = Denoise(_CONFIG_PATH)
    check_kernel_size(denoise)
