#!/usr/bin/env bash
#
# 进入 conda 的 unet 环境，执行 train_unet.py 训练。
# 训练数据路径与超参数均从 config/config.json 读取。
#
# 用法：
#     bash scripts/train.sh                # 直接按 config.json 训练
#     bash scripts/train.sh --epochs 20    # 可附加 train_unet.py 的参数
#     ./scripts/train.sh
#
# 可通过环境变量覆盖：
#     CONDA_BIN     conda 可执行文件路径（默认自动探测）
#     ENV_NAME      conda 环境名（默认 unet）
#     WANDB_MODE    wandb 模式（默认 disabled，离线/内网不联网；可设 online）
#
set -euo pipefail

# 脚本所在目录（scripts/）的上一级 = 仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---------- 1. 定位 conda ----------
CONDA_BIN="${CONDA_BIN:-}"
if [ -z "$CONDA_BIN" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
    elif [ -x "$HOME/anaconda3/bin/conda" ]; then
        CONDA_BIN="$HOME/anaconda3/bin/conda"
    elif [ -x "$HOME/miniconda3/bin/conda" ]; then
        CONDA_BIN="$HOME/miniconda3/bin/conda"
    elif [ -x /opt/conda/bin/conda ]; then
        CONDA_BIN="/opt/conda/bin/conda"
    else
        echo "[错误] 未找到 conda，请安装 Miniconda/Anaconda 或设置 CONDA_BIN 环境变量" >&2
        exit 1
    fi
fi

# ---------- 2. conda 环境名 ----------
ENV_NAME="${ENV_NAME:-unet}"

# ---------- 3. 离线/内网环境默认禁用 wandb 联网日志（可用 WANDB_MODE=online 覆盖） ----------
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "[info] conda     : $CONDA_BIN"
echo "[info] 环境名    : $ENV_NAME"
echo "[info] 仓库目录  : $REPO_DIR"
echo "[info] wandb     : $WANDB_MODE"

# ---------- 4. 在 unet 环境中执行训练（conda run 无需 activate） ----------
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/train_unet.py "$@"
