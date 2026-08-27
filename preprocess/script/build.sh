#!/bin/bash
# =============================================================================
# build.sh - 编译数据清理工程
#
# 功能：
#   - 配置并编译 preprocess 工程（C++17 + OpenCV 4.2）；
#   - 可执行文件输出到 bin 目录：preprocess/bin/preprocess。
#
# 用法：
#   ./script/build.sh
#
# 说明：
#   - 中间编译产物位于 build/ 目录，可执行文件位于 bin/ 目录；
#   - 脚本幂等，重复执行只增量编译。
# =============================================================================
set -euo pipefail

# 切换到项目根目录，保证相对路径稳定。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

BIN_DIR="${PROJECT_DIR}/bin"
BUILD_DIR="${PROJECT_DIR}/build"

mkdir -p "${BIN_DIR}" "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[build] 配置 CMake (Release, 输出目录=${BIN_DIR})..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${BIN_DIR}"

echo "[build] 编译..."
cmake --build . -j"$(nproc)"

echo "[build] 完成: ${BIN_DIR}/preprocess"
