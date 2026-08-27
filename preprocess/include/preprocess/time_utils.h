/**
 * @file time_utils.h
 * @brief 图像文件名时间解析工具接口
 *
 * @details
 * 本项目按时间间隔清理图像，因此需要从图像文件名中解析时间。
 * 支持的命名形式（按优先级尝试）：
 *  1. 纯数字（Unix 秒，可为小数），例如 "1672545600.123"、"34804"；
 *  2. 常见日期时间格式，例如 "20230101120000"、"2023-01-01_12-00-00"、
 *     "2023-01-01 12:00:00"、"2023-01-01T12:00:00"；
 *  3. 上述均失败时回退到文件系统修改时间。
 *
 * @note 文件名若为纯递增数字（如 34804.png），将按 Unix 秒解释，
 *       相邻文件的数值差即视为时间间隔（秒）。
 *
 * @author bg11738
 * @version 1.0.0
 */

#ifndef PREPROCESS_TIME_UTILS_H_
#define PREPROCESS_TIME_UTILS_H_

#include <filesystem>
#include <string>

namespace preprocess {

/**
 * @brief 从图像文件解析时间（Unix 秒，可含小数部分）
 *
 * 解析顺序见文件头注释：数字名 -> 日期时间格式 -> 文件修改时间。
 *
 * @param file_path 图像文件完整路径
 * @param seconds   [out] 解析得到的时间，单位秒
 * @return true 解析成功；false 解析失败（此时 *seconds 不保证有效）
 */
bool ParseImageTime(const std::filesystem::path& file_path, double* seconds);

/**
 * @brief 将文件系统时间点转换为 Unix 秒
 *
 * @param t 文件系统时间点（std::filesystem::file_time_type）
 * @return 以秒为单位的 Unix 时间戳（可含小数）
 */
double ToUnixSeconds(const std::filesystem::file_time_type& t);

}  // namespace preprocess

#endif  // PREPROCESS_TIME_UTILS_H_
