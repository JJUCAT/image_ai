# preprocess —— 图像预处理工具

基于 **C++17 + OpenCV 4.2** 的图像预处理工具，包含**样本图片清理**与**标注 mask 像素映射**两大任务，用于在深度学习数据集标注与训练前对原始数据进行去冗余和标签归一化处理。

## 主要功能

1. **时间间隔清理**：图片文件名视为时间戳，仅保留与上一张已保留图片时间间隔不小于 `time_min_interval` 秒的图像，剔除时间上过于密集的帧。
2. **感知哈希去重**：使用基于 DCT 变换的感知哈希（pHash，思路同 OpenCV `img_hash::PHash`），对图像内容进行相似度判定，汉明距离小于 `hamming_distance` 的图像视为重复并丢弃。
3. **批量内存控制**：按 `batch_szie` 分批解码图像，限制单次内存占用上界。
4. **完整统计输出**：打印扫描总数、保留数、各类丢弃/失败计数，便于核查清理效果。
5. **spdlog 日志与处理报告**：使用 spdlog 记录控制台与文件日志；每项任务结束生成报告，记录时间、路径与数量等。
6. **灰度图输出**：清理完成后，将保留图像经色彩转换（BGR → GRAY）提取亮度，以同名 PNG 保存到 `gray` 目录。
7. **mask 像素映射**：在“清理 + 灰度化”之后，以 `gray` 目录（清理 + 灰度化后的保留图片）为匹配基准，对每个文件名取 `raw_masks` 目录中**同名**的标注 mask PNG，按 `grass_mapping` 映射表将指定像素值统一映射为目标标签，其余像素值一律置为 `0`，以同名保存到 `gray_masks` 目录。

## 目录结构

```
preprocess/
├── CMakeLists.txt                 # CMake 构建脚本
├── config/
│   └── config.json                # 预处理配置文件
├── include/preprocess/
│   ├── config_loader.h            # 配置结构体与 JSON 加载接口
│   ├── perceptual_hash.h          # 可配置 DCT 感知哈希接口
│   ├── image_cleaner.h            # 样本清理主流程接口
│   ├── mask_processor.h           # 标注 mask 像素映射主流程接口
│   ├── logging.h                  # spdlog 日志初始化接口
│   └── time_utils.h               # 文件名时间解析接口
├── src/
│   ├── main.cpp                   # 程序入口
│   ├── config_loader.cpp          # 配置加载实现
│   ├── perceptual_hash.cpp        # 感知哈希实现
│   ├── image_cleaner.cpp          # 样本清理主流程实现
│   ├── mask_processor.cpp         # 标注 mask 像素映射实现
│   ├── logging.cpp                # spdlog 日志初始化实现
│   └── time_utils.cpp             # 时间解析实现
├── script/
│   ├── build.sh                   # 编译脚本（可执行文件输出到 bin/）
│   └── run.sh                     # 运行脚本（执行 bin/preprocess 处理数据）
├── bin/                           # 可执行文件输出目录（编译后生成）
├── logs/                          # 日志文件目录（运行时生成，默认 clean.log）
└── readme.md
```

## 环境要求

- Linux（本机 Ubuntu 20.04 / g++ 9.4 验证通过）
- CMake ≥ 3.10
- OpenCV ≥ 4.2（需包含 `core`、`imgproc`、`imgcodecs`，本机已安装 4.2.0）
- nlohmann/json（单头文件，位于系统 include 路径或通过 CMake 包提供）
- spdlog ≥ 1.5（日志库，本机已安装 1.5.0）

## 编译构建

方式一：一键编译脚本（推荐，可执行文件输出到 `bin/` 目录）

```bash
cd preprocess
./script/build.sh
```

方式二：手动 CMake

```bash
cd preprocess
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

编译完成后，脚本方式的可执行文件位于 `bin/preprocess`，手动方式位于 `build/preprocess`。

## 使用方法

在 `preprocess` 目录下执行：

```bash
./script/run.sh
# 或指定配置文件
./script/run.sh --config /path/to/config.json
# 也可直接运行可执行文件
./bin/preprocess --config /path/to/config.json
```

命令行参数：

| 参数 | 说明 |
| ---- | ---- |
| `-c, --config <path>` | 指定 config.json 路径（默认 `config/config.json`，相对于运行目录） |
| `--log <path>` | 指定日志文件路径（默认 `logs/clean.log`） |
| `-h, --help` | 显示帮助信息 |

运行示例输出（spdlog 格式，含清理报告与 mask 预处理报告）：

```
[2026-08-18 17:06:13.102] [info] 加载配置: config/config.json
[2026-08-18 17:06:13.102] [info] 原始目录: /home/bg11738/ws/label_studio_database/seg_ws/raw/
...
[2026-08-18 17:06:15.805] [info] ==== 清理报告 ====
时间: 2026-08-18 17:06:18
数据源路径: /home/bg11738/ws/label_studio_database/seg_ws/raw/
清理后路径: /home/bg11738/ws/label_studio_database/seg_ws/clean/
灰度图路径: /home/bg11738/ws/label_studio_database/seg_ws/gray/
总图像数: 172
保留图像数: 163
时间间隔丢弃: 0
感知哈希重复丢弃: 9
读取失败: 0
写入失败: 0
灰度写入失败: 0
耗时: 2.702 秒
[2026-08-18 17:06:16.120] [info] 灰度图目录图片 163 张（作为 mask 匹配基准）
[2026-08-18 17:06:16.880] [info] ==== mask 预处理报告 ====
时间: 2026-08-18 17:06:18
匹配基准目录: /home/bg11738/ws/label_studio_database/seg_ws/gray/
mask 源路径: /home/bg11738/ws/label_studio_database/seg_ws/raw_masks/
mask 输出路径: /home/bg11738/ws/label_studio_database/seg_ws/gray_masks/
像素映射表: 1 -> 1, 7 -> 1
总 mask 数: 163
处理成功: 163
未找到同名 mask: 0
读取失败: 0
写入失败: 0
耗时: 0.760 秒
```

**样本清理任务**：保留的图像以**原文件名**写入配置的 `clean` 目录；同时经色彩转换（BGR → GRAY）提取亮度后，以同名 **PNG** 写入配置的 `gray` 目录（目录不存在时自动创建）。

**mask 预处理任务**：在清理 + 灰度化之后，以 `gray` 目录为匹配基准，对每个文件名取 `raw_masks` 中**同名**的 mask PNG，按 `grass_mapping` 映射像素值（不在映射中的像素值置为 `0`），以同名写入配置的 `gray_masks` 目录（目录不存在时自动创建）。

## 日志与报告

- **日志**：程序使用 spdlog 记录日志，同时输出到**控制台**与**滚动文件**（默认 `logs/clean.log`，单文件上限 10MB、保留最近 5 个文件）。日志格式为 `[时间戳] [级别] 消息`。
- **清理报告**：样本清理结束后输出并写入日志一份报告，包含：
  - 时间（任务完成时刻，本地时间）；
  - 数据源路径（`raw`）；
  - 清理后路径（`clean`）；
  - 灰度图路径（`gray`）；
  - 清理数量：总图像数、保留数、时间间隔丢弃、感知哈希重复丢弃、读取/写入失败、灰度写入失败；
  - 耗时（清理执行耗时，秒）。
- **mask 预处理报告**：mask 像素映射结束后输出并写入日志一份报告，包含：
  - 时间（任务完成时刻，本地时间）；
  - 匹配基准目录（`gray`）；
  - mask 源路径（`raw_masks`）；
  - mask 输出路径（`gray_masks`）；
  - 像素映射表（`grass_mapping` 的内容，如 `1 -> 1, 7 -> 1`）；
  - 处理数量：总 mask 数、处理成功、未找到同名 mask、读取/写入失败；
  - 耗时（mask 处理耗时，秒）。
- 日志文件路径可通过 `--log <path>` 自定义；目录不存在时自动创建。

## 配置文件说明

`config/config.json` 各字段含义如下：

```json
{
  "raw": "/home/bg11738/ws/label_studio_database/seg_ws/raw/",
  "clean": "/home/bg11738/ws/label_studio_database/seg_ws/clean/",
  "gray": "/home/bg11738/ws/label_studio_database/seg_ws/gray/",
  "clean_config": {
    "batch_szie": 1000,
    "time_min_interval": 1,
    "perceptual_hash": {
      "scaled_size": 32,
      "DCT_szie": 32,
      "hash_size": 8,
      "hamming_distance": 6
    }
  },
  "raw_masks": "/home/bg11738/ws/label_studio_database/seg_ws/raw_masks/",
  "gray_masks": "/home/bg11738/ws/label_studio_database/seg_ws/gray_masks/",
  "grass_mapping": {
    "1": 1,
    "7": 1
  }
}
```

| 键 | 含义 |
| --- | ---- |
| `raw` | 原始样本图片所在目录路径 |
| `clean` | 数据清理后的图片输出目录路径 |
| `gray` | 清理图片转灰度 PNG 的输出目录路径 |
| `batch_szie` | 批量处理时单次加载（解码）的图像数量（清理与 mask 任务共用） |
| `time_min_interval` | 图片名视为时间，保留图像间的最小时间间隔（秒） |
| `scaled_size` | 感知哈希：哈希计算前的缩小尺寸（缩小降噪） |
| `DCT_szie` | 感知哈希：DCT 变换维度 |
| `hash_size` | 感知哈希：从 DCT 系数矩阵左上角取出的低频块尺寸 |
| `hamming_distance` | 感知哈希：两图像汉明距离**小于**该值判为重复 |
| `raw_masks` | 原始标注 mask PNG 所在目录路径 |
| `gray_masks` | mask 像素映射处理后的输出目录路径 |
| `grass_mapping` | mask 像素值映射表（键为原像素值，值为映射后像素值）；不在映射中的像素值一律置为 `0` |

> 注意：键名 `batch_szie` 与 `DCT_szie` 为约定拼写，请勿修改，否则加载会因键名不匹配而抛出异常。`grass_mapping` 的键为 JSON 对象的字符串键，须可解析为 `[0, 255]` 内的整数。

## 算法说明

### 时间间隔清理

- 依次对每个图像执行：`当前时间 - 上一张保留时间 < time_min_interval` 则丢弃；首张图像始终保留。
- 时间从**文件名**解析，支持：纯数字（按 Unix 秒，可为小数，如 `34804.png`）、常见日期时间格式（如 `20230101120000`、`2023-01-01_12-00-00`）；解析失败时回退到文件系统修改时间。

### 感知哈希（pHash）清理

基于 DCT 的感知哈希计算流程：

1. 图像转灰度；
2. 缩放到 `scaled_size × scaled_size`；
3. 按需再缩放到 `dct_size × dct_size` 作为 DCT 输入；
4. 执行二维 DCT，得到 `dct_size × dct_size` 系数矩阵；
5. 截取矩阵**左上角** `hash_size × hash_size` 的低频系数块；
6. 以块内均值为阈值，系数大于均值记 1，否则记 0，得到 `hash_size²` 位的哈希向量；
7. 与所有已保留图像的哈希计算汉明距离，距离 `< hamming_distance` 判为重复并丢弃。

### mask 像素映射预处理

将标注 mask 中的多类别标签统一归一化，该任务在“清理 + 灰度化”之后执行：

1. 以 `gray` 目录（清理 + 灰度化后的保留图片）为匹配基准，收集全部图片名；
2. 对每个基准文件名，在 `raw_masks` 目录中查找**同名**的 mask PNG；
3. 按 `batch_szie` 分批读取 mask（限制单次解码内存占用上界）；
4. 构建 256 项查找表（LUT）：`grass_mapping` 中键对应的像素值替换为目标值，**其余像素值一律置为 `0`**；
5. 应用 `cv::LUT` 完成像素重映射；
6. 以同名（与 `gray` 一致）写入 `gray_masks` 目录。

典型场景：Label Studio 导出 mask 中 `1`、`7` 均代表草地类别，配置 `"grass_mapping": {"1": 1, "7": 1}` 后，两类统一映射为标签 `1`，其余像素（背景等）置为 `0`，便于后续训练与评估；只有被清理 + 灰度化保留的图片（即 `gray` 目录中存在的文件名）才会处理对应的 mask，保证与训练样本一一对应。

## 注意事项

1. **文件名时间语义**：时间间隔清理依赖文件名可解析为时间；纯数字文件名按 Unix 秒解释，相邻文件数值差即为间隔（秒）。若文件名不可解析且文件修改时间相同，可能被大量判为过近而丢弃，请保证文件命名含时间信息。
2. **感知哈希为全局去重**：每张候选图像会与**所有已保留**图像的哈希比较；`batch_szie` 仅约束单批解码图像的内存上界，不限制去重范围。
3. **双重规则**：图像须同时通过时间间隔与感知哈希两项规则才被保留，任一规则不满足即被丢弃。
4. **覆盖式输出**：`clean` 目录中的同名文件会被覆盖；重复运行会基于当前 `raw` 内容重新生成结果。
5. **阈值语义**：`hamming_distance` 越大越容易判为重复（0 表示完全禁用哈希去重，最大有效值不超过哈希位数 `hash_size²`）。
6. **参数约束**：`scaled_size`、`dct_size`、`hash_size` 均须为正整数，且 `hash_size ≤ dct_size`；`raw_masks`、`gray_masks` 目录与 `grass_mapping` 映射表不得为空，映射键值均须在 `[0, 255]` 内；非法配置会在加载阶段抛出异常并终止。
7. **mask 任务以 gray 为基准**：mask 映射只处理与 `gray` 目录**同名**（清理 + 灰度化后保留）的 `raw_masks`；若 `raw_masks` 中缺少同名 mask，该文件被跳过并计入“未找到同名 mask”。
8. **mask 输出目录清空**：每次运行前会清空 `gray_masks` 目录，确保其与 `gray` 基准严格一一对应；重复运行会基于当前 `gray` 与 `raw_masks` 内容重新生成结果。
9. **mask 读取语义**：mask 以原始格式读取，非 8 位单通道图像会先转为 8 位灰度以保证 `grass_mapping` 像素映射正确。
