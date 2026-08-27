#!/bin/bash
# =============================================================================
# run.sh - 执行数据清理
#
# 功能：
#   - 运行 bin/ 目录下的 preprocess 程序清理样本图像；
#   - 默认读取 config/config.json，可通过命令行参数覆盖。
#
# 用法：
#   ./script/run.sh                      # 使用默认 config/config.json
#   ./script/run.sh --config <路径>      # 指定配置文件
#   ./script/run.sh --help               # 查看程序帮助
#
# 说明：
#   - 若 bin/preprocess 不存在，提示先运行 script/build.sh；
#   - 以项目根目录为工作目录运行，保证默认配置路径可被找到。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
BIN="${PROJECT_DIR}/bin/preprocess"

if [[ ! -x "${BIN}" ]]; then
  echo "[run] 错误: 未找到可执行文件 ${BIN}" >&2
  echo "[run] 请先执行: ./script/build.sh" >&2
  exit 1
fi

# 以项目根目录为工作目录运行程序，透传全部参数。
cd "${PROJECT_DIR}"
"${BIN}" "$@"
