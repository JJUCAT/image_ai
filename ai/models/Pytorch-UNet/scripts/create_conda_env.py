#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 Pytorch-UNet 项目的 requirements.txt 创建 conda 环境并安装依赖。

说明：
- requirements.txt 里【不含 PyTorch】（README 要求 PyTorch >= 1.13），
  本脚本会先单独安装 torch + torchvision，再安装 requirements.txt 里的依赖。
- 默认自动检测本机是否具有 NVIDIA GPU：
    * 有 GPU  -> 安装默认（自带 CUDA）的 torch
    * 无 GPU  -> 安装 CPU 版 torch
- 可通过 --torch 手动指定安装方式。

用法示例：
    # 自动检测（有 GPU 装 CUDA 版，无 GPU 装 CPU 版）
    python create_conda_env.py

    # 指定环境名与 Python 版本
    python create_conda_env.py --name unet --python 3.10

    # 无 GPU 时显式安装 CPU 版 PyTorch
    python create_conda_env.py --torch cpu

    # 指定 CUDA 版本（例如 cu121 对应 CUDA 12.1）
    python create_conda_env.py --torch cu121

    # 指定 torch 版本号
    python create_conda_env.py --pytorch-version 2.2.0

    # 放宽 requirements.txt 中过旧的版本锁定（避免与新版 torch/numpy 冲突）
    python create_conda_env.py --relax-pins
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_TXT = REPO_DIR / 'requirements.txt'

DEFAULT_ENV_NAME = 'unet'
DEFAULT_PYTHON = '3.10'

# PyTorch 官方 wheel 源（按 CUDA 版本区分）
TORCH_INDEX_URLS = {
    'cpu': 'https://download.pytorch.org/whl/cpu',
    'cu118': 'https://download.pytorch.org/whl/cu118',
    'cu121': 'https://download.pytorch.org/whl/cu121',
    'cu124': 'https://download.pytorch.org/whl/cu124',
}

# requirements.txt 中与新版 PyTorch/numpy 可能冲突的“过旧”依赖（relax-pins 时替换）
PINNED_DEPENDENCIES = {
    'matplotlib',
    'numpy',
    'Pillow',
    'tqdm',
    'wandb',
}


def has_nvidia_gpu() -> bool:
    """通过 nvidia-smi 判断是否存在可用 NVIDIA GPU。"""
    return shutil.which('nvidia-smi') is not None


def run(cmd):
    """打印并执行命令，失败时抛出异常终止脚本。"""
    print(f'\n>>> {" ".join(cmd)}')
    subprocess.check_call(cmd)


def build_torch_install_cmd(pip_prefix, torch_mode, pytorch_version):
    """构造安装 torch/torchvision 的命令。"""
    if pytorch_version:
        base = [*pip_prefix, f'torch=={pytorch_version}', 'torchvision']
    else:
        base = [*pip_prefix, 'torch', 'torchvision']

    if torch_mode == 'cpu':
        return [*base, '--index-url', TORCH_INDEX_URLS['cpu']]
    if torch_mode in TORCH_INDEX_URLS:
        return [*base, '--index-url', TORCH_INDEX_URLS[torch_mode]]
    # 'default'：pip 默认源（自带 CUDA）
    return base


def resolve_torch_mode(mode):
    """把 auto / cpu / default / cuXXX 解析为最终安装模式。"""
    if mode == 'auto':
        return 'cpu' if not has_nvidia_gpu() else 'default'
    return mode


CONDA_CANDIDATES = [
    '~/anaconda3/bin/conda',
    '~/miniconda3/bin/conda',
    '~/anaconda/bin/conda',
    '~/miniconda/bin/conda',
    '/opt/conda/bin/conda',
    '/usr/local/anaconda3/bin/conda',
]


def find_conda():
    """查找 conda 可执行文件。

    优先使用 PATH 上的 conda；若未初始化（如非登录 shell），
    再尝试常见安装路径。返回绝对路径或 None。
    """
    which = shutil.which('conda')
    if which:
        return which
    for cand in CONDA_CANDIDATES:
        p = Path(cand).expanduser()
        if p.is_file():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='根据 Pytorch-UNet 的 requirements.txt 创建 conda 环境'
    )
    parser.add_argument(
        '--name', '-n', default=DEFAULT_ENV_NAME,
        help=f'conda 环境名（默认 {DEFAULT_ENV_NAME}）'
    )
    parser.add_argument(
        '--python', '-p', default=DEFAULT_PYTHON,
        help=f'Python 版本（默认 {DEFAULT_PYTHON}）'
    )
    parser.add_argument(
        '--torch', choices=['auto', 'cpu', 'default', 'cu118', 'cu121', 'cu124'],
        default='auto',
        help='PyTorch 安装方式：auto=自动检测 GPU；cpu=仅 CPU；default=默认源(含CUDA)；'
             'cu118/cu121/cu124=指定 CUDA 版本的官方源（默认 auto）'
    )
    parser.add_argument(
        '--pytorch-version', default=None,
        help='指定 torch 版本号（默认安装官方源当前最新版，例如 2.2.0）'
    )
    parser.add_argument(
        '--relax-pins', action='store_true',
        help='放宽 requirements.txt 中的版本锁定，安装这些依赖的最新版（避免与新版 PyTorch 冲突）'
    )
    parser.add_argument(
        '--requirements', default=None,
        help='自定义 requirements.txt 路径（默认取仓库根目录的 requirements.txt）'
    )
    args = parser.parse_args()

    # ---------- 1. 前置检查 ----------
    conda_bin = find_conda()
    if conda_bin is None:
        sys.exit('[错误] 未检测到 conda，请先安装 Miniconda 或 Anaconda：'
                 'https://docs.conda.io/en/latest/miniconda.html')
    print(f'[配置] 使用 conda: {conda_bin}')

    req_path = Path(args.requirements) if args.requirements else REQUIREMENTS_TXT
    if not req_path.exists():
        sys.exit(f'[错误] 未找到 requirements.txt：{req_path}')

    torch_mode = resolve_torch_mode(args.torch)
    gpu_note = f'检测到 NVIDIA GPU，将安装默认(CUDA)版 PyTorch' if has_nvidia_gpu() \
        else '未检测到 NVIDIA GPU，将安装 CPU 版 PyTorch'
    print(f'[配置] 环境名={args.name} Python={args.python} '
          f'PyTorch={torch_mode}（{gpu_note}）')

    # ---------- 2. 创建 conda 环境 ----------
    run([conda_bin, 'create', '-y', '-n', args.name, f'python={args.python}'])

    # 后续所有 pip 安装都在该环境内执行
    pip_prefix = [conda_bin, 'run', '-n', args.name, 'pip', 'install']

    # ---------- 3. 安装 PyTorch（requirements.txt 中不包含） ----------
    run(build_torch_install_cmd(pip_prefix, torch_mode, args.pytorch_version))

    # ---------- 4. 安装 requirements.txt 中的依赖 ----------
    if args.relax_pins:
        # 去掉版本锁定，安装最新版
        names = [line.strip().split('==')[0]
                 for line in req_path.read_text().splitlines()
                 if line.strip() and not line.strip().startswith('#')]
        run([*pip_prefix, *names])
    else:
        run([*pip_prefix, '-r', str(req_path)])

    # ---------- 4.5 兼容修复：旧版 wandb 依赖 pkg_resources ----------
    # requirements.txt 锁定 wandb==0.13.5，其导入需要 pkg_resources；
    # 而 setuptools>=81 已移除 pkg_resources，会导致 import wandb 失败。
    # 因此把 setuptools 固定为 <81（仍带 pkg_resources）。
    run([*pip_prefix, 'setuptools<81'])

    # ---------- 5. 完成提示 ----------
    print()
    print('=' * 66)
    print(f'conda 环境创建完成！')
    print(f'  激活环境   : conda activate {args.name}')
    print(f'  验证 PyTorch: conda run -n {args.name} '
          f'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"')
    print('  开始训练   : conda activate {0} && cd {1} && python train.py --amp'.format(
        args.name, REPO_DIR))
    print('=' * 66)

    if not args.relax_pins:
        print()
        print('[提示] requirements.txt 里 numpy==1.23.5 / Pillow==9.3.0 等版本较旧，')
        print('       若安装新版 PyTorch 时报 numpy/Pillow 兼容性错误，可重跑本脚本加 --relax-pins。')


if __name__ == '__main__':
    main()
