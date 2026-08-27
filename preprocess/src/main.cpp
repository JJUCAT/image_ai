/**
 * @file main.cpp
 * @brief 图像预处理程序入口
 *
 * @details
 * 程序启动流程：
 *  1. 解析命令行参数（--config 指定配置文件、--log 指定日志文件）；
 *  2. 初始化 spdlog 日志（控制台 + 滚动文件）；
 *  3. 加载并校验清理配置；
 *  4. 构造 ImageCleaner 并执行样本图片清理，统计耗时；
 *  5. 生成并记录清理报告（时间、数据源路径、清理后路径、清理数量、耗时）；
 *  6. 构造 MaskProcessor 并执行标注 mask 像素映射预处理，统计耗时；
 *  7. 生成并记录 mask 预处理报告。
 *
 * 用法示例：
 *  ./preprocess
 *  ./preprocess --config /path/config.json
 *  ./preprocess --log /path/clean.log
 *  ./preprocess --help
 *
 * @note 默认配置文件路径 "config/config.json"、默认日志文件
 *       "logs/clean.log" 均相对于运行目录，建议在 preprocess 目录下运行。
 *
 * @author bg11738
 * @version 1.2.0
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "preprocess/config_loader.h"
#include "preprocess/image_cleaner.h"
#include "preprocess/logging.h"
#include "preprocess/mask_processor.h"

namespace {

/// 命令行参数集合。
struct ProgramArgs {
  std::string config_path = "config/config.json";  ///< 配置文件路径
  std::string log_path = "logs/clean.log";         ///< 日志文件路径
};

/**
 * @brief 解析命令行参数
 *
 * @param argc 参数个数
 * @param argv 参数数组
 * @return 解析得到的 ProgramArgs
 *
 * @note 支持 --config/-c <path>、--config=<path>、--log <path>、
 *       --log=<path>、--help/-h；未知参数忽略并告警。
 */
ProgramArgs ParseArgs(int argc, char* argv[]) {
  ProgramArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" || arg == "-c") {
      if (i + 1 < argc) {
        args.config_path = argv[++i];
      }
    } else if (arg.rfind("--config=", 0) == 0) {
      args.config_path = arg.substr(std::string("--config=").size());
    } else if (arg == "--log") {
      if (i + 1 < argc) {
        args.log_path = argv[++i];
      }
    } else if (arg.rfind("--log=", 0) == 0) {
      args.log_path = arg.substr(std::string("--log=").size());
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "用法: preprocess [选项]\n"
                << "选项:\n"
                << "  -c, --config <path>  指定 config.json 路径\n"
                << "                        (默认: config/config.json)\n"
                << "      --log <path>     指定日志文件路径\n"
                << "                        (默认: logs/clean.log)\n"
                << "  -h, --help           显示帮助信息\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "警告: 忽略未知参数: " << arg << "\n";
    }
  }
  return args;
}

}  // namespace

int main(int argc, char* argv[]) {
  const ProgramArgs args = ParseArgs(argc, argv);

  // 初始化日志：控制台 + 滚动文件（失败仅回退控制台，不阻断运行）。
  preprocess::InitLogger(args.log_path);
  spdlog::info("加载配置: {}", args.config_path);

  try {
    const preprocess::CleanConfig config = preprocess::LoadConfig(args.config_path);
    spdlog::info("原始目录: {}", config.raw_dir);
    spdlog::info("输出目录: {}", config.clean_dir);
    spdlog::info("灰度图目录: {}", config.gray_dir);
    spdlog::info("批量大小: {}", config.batch_size);
    spdlog::info("最小时间间隔(秒): {}", config.time_min_interval);
    spdlog::info("感知哈希(scaled/DCT/hash): {}/{}/{}", config.scaled_size,
                 config.dct_size, config.hash_size);
    spdlog::info("汉明距离阈值: {}", config.hamming_distance);
    spdlog::info("原始 mask 目录: {}", config.raw_masks_dir);
    spdlog::info("mask 输出目录: {}", config.gray_masks_dir);
    std::ostringstream mapping_text;
    for (auto it = config.grass_mapping.begin(); it != config.grass_mapping.end();
         ++it) {
      if (it != config.grass_mapping.begin()) {
        mapping_text << ", ";
      }
      mapping_text << it->first << "->" << it->second;
    }
    spdlog::info("mask 像素映射表: {}", mapping_text.str());
    spdlog::info("---- 开始清理 ----");

    // 执行清理并统计耗时。
    const auto start = std::chrono::steady_clock::now();
    preprocess::ImageCleaner cleaner(config);
    const preprocess::CleanReport report = cleaner.Clean();
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(end - start).count();

    // 生成并输出清理报告（含时间、路径、数量与耗时）。
    const std::string report_text =
        preprocess::BuildCleanReport(config, report, elapsed_seconds);
    spdlog::info("==== 清理报告 ====\n{}", report_text);

    // 执行标注 mask 像素映射预处理并统计耗时。
    spdlog::info("---- 开始 mask 预处理 ----");
    const auto mask_start = std::chrono::steady_clock::now();
    preprocess::MaskProcessor mask_processor(config);
    const preprocess::MaskReport mask_report = mask_processor.Process();
    const auto mask_end = std::chrono::steady_clock::now();
    const double mask_elapsed_seconds =
        std::chrono::duration<double>(mask_end - mask_start).count();

    // 生成并输出 mask 预处理报告。
    const std::string mask_report_text =
        preprocess::BuildMaskReport(config, mask_report, mask_elapsed_seconds);
    spdlog::info("==== mask 预处理报告 ====\n{}", mask_report_text);
  } catch (const std::exception& e) {
    spdlog::error("错误: {}", e.what());
    preprocess::ShutdownLogger();
    return EXIT_FAILURE;
  }

  preprocess::ShutdownLogger();
  return EXIT_SUCCESS;
}
