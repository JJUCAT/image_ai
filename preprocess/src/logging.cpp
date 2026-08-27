/**
 * @file logging.cpp
 * @brief spdlog 日志初始化实现
 *
 * @details
 * 实现细节：
 *  - 创建彩色控制台 sink（stdout_color_sink_mt）；
 *  - 创建滚动文件 sink（rotating_file_sink_mt），单文件上限 10MB、
 *    保留最近 5 个文件，防止日志无限增长；
 *  - 组合二者生成默认日志器并注册为 spdlog 默认日志器；
 *  - 设置 info 级别起自动刷新，保证日志及时落盘；
 *  - 日志目录不存在时调用 create_directories 自动创建。
 */

#include "preprocess/logging.h"

#include <filesystem>
#include <vector>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace preprocess {

namespace {
namespace fs = std::filesystem;
}  // namespace

bool InitLogger(const std::string& log_file, spdlog::level::level_enum level) {
  try {
    // 确保日志目录存在。
    const fs::path log_path(log_file);
    if (log_path.has_parent_path()) {
      fs::create_directories(log_path.parent_path());
    }

    // 控制台 sink：彩色输出，便于实时查看。
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // 文件 sink：单文件 10MB，保留最近 5 个滚动文件。
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file, /*max_size=*/10 * 1024 * 1024, /*max_files=*/5);

    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger =
        std::make_shared<spdlog::logger>("clean", sinks.begin(), sinks.end());
    logger->set_level(level);
    // info 及以上级别即时刷新，避免程序异常退出时丢失日志。
    logger->flush_on(spdlog::level::info);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    spdlog::set_default_logger(logger);
    return true;
  } catch (const spdlog::spdlog_ex& e) {
    // 初始化失败时退回控制台输出，不阻断程序运行。
    spdlog::error("初始化日志失败: {}", e.what());
    return false;
  }
}

void ShutdownLogger() { spdlog::shutdown(); }

}  // namespace preprocess
