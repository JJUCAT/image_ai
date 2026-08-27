/**
 * @file logging.h
 * @brief spdlog 日志初始化接口
 *
 * @details
 * 本模块封装 spdlog 的初始化，提供：
 *  - 控制台彩色输出 + 滚动文件日志（单文件上限、保留文件数）；
 *  - 统一的日志格式（时间戳 + 级别 + 消息）；
 *  - 默认日志级别与刷新策略。
 * 初始化后，全工程可通过 spdlog::info/warn/error 等直接写日志。
 *
 * @note 日志文件所在目录不存在时会被自动创建。
 *
 * @author bg11738
 * @version 1.0.0
 */

#ifndef PREPROCESS_LOGGING_H_
#define PREPROCESS_LOGGING_H_

#include <string>

#include <spdlog/spdlog.h>

namespace preprocess {

/**
 * @brief 初始化 spdlog 默认日志器（控制台 + 滚动文件）
 *
 * @param log_file 日志文件路径（可含目录，目录不存在时自动创建）
 * @param level    日志级别，默认 info
 * @return true 初始化成功；false 初始化失败（此时仍可调用 spdlog 函数，
 *         但仅输出到控制台）
 *
 * @note 该函数应只调用一次；日志格式为
 *       "[YYYY-MM-DD HH:MM:SS.mmm] [级别] 消息"。
 */
bool InitLogger(const std::string& log_file,
                spdlog::level::level_enum level = spdlog::level::info);

/**
 * @brief 刷新并关闭日志系统
 *
 * 程序结束前调用，确保所有缓冲日志写入文件。
 */
void ShutdownLogger();

}  // namespace preprocess

#endif  // PREPROCESS_LOGGING_H_
