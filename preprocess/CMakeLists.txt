# =============================================================================
# CMakeLists.txt - 图像预处理项目构建脚本
#
# 功能：
#   - 编译链接 OpenCV 4.2（core / imgproc / imgcodecs 等）；
#   - 链接 nlohmann/json（若以 CMake 包形式安装则使用之，否则依赖系统头文件）；
#   - 链接 spdlog 日志库；
#   - 导出 compile_commands.json 以支持 IDE 智能提示。
#
# 用法：
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j
# =============================================================================
cmake_minimum_required(VERSION 3.10)

project(preprocess
        VERSION 1.1.0
        DESCRIPTION "基于 OpenCV 4.2 的图像预处理工具（时间间隔 + 感知哈希去重 + mask 像素映射）"
        LANGUAGES CXX)

# ---- 编译选项 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 导出编译命令，便于 IDE 智能提示与静态分析。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- 依赖 ----
find_package(OpenCV 4.2 REQUIRED)
find_package(Threads REQUIRED)
find_package(spdlog 1.5 REQUIRED)
# nlohmann/json 通常为单头文件，位于系统 include 路径；若以 CMake 包形式
# 安装则通过 target 链接，否则依赖系统头文件 <nlohmann/json.hpp>。
find_package(nlohmann_json QUIET)

# ---- 可执行目标 ----
add_executable(preprocess
  src/main.cpp
  src/config_loader.cpp
  src/image_cleaner.cpp
  src/logging.cpp
  src/mask_processor.cpp
  src/perceptual_hash.cpp
  src/time_utils.cpp
)

target_include_directories(preprocess PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(preprocess PRIVATE
  ${OpenCV_LIBS}
  Threads::Threads
  spdlog::spdlog
)

if(nlohmann_json_FOUND)
  target_link_libraries(preprocess PRIVATE nlohmann_json::nlohmann_json)
endif()

# ---- 安装 ----
install(TARGETS preprocess RUNTIME DESTINATION bin)
