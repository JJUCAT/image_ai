/**
 * @file time_utils.cpp
 * @brief 图像文件名时间解析实现
 *
 * @details
 * 时间解析按以下优先级进行：
 *  1. 文件主名（不含扩展名）为纯数字时，按 Unix 秒（可为小数）解析，
 *     例如 "34804" -> 34804.0 秒；
 *  2. 尝试一组常见日期时间格式（strptime 逐个匹配）：
 *     "%Y%m%d%H%M%S"、"%Y-%m-%d %H:%M:%S"、"%Y-%m-%d_%H-%M-%S"、
 *     "%Y-%m-%dT%H:%M:%S"、"%Y/%m/%d %H:%M:%S"、"%%Y-%m-%d"；
 *  3. 以上均失败时，回退到文件系统修改时间。
 */

#include "preprocess/time_utils.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <system_error>

namespace preprocess {

namespace {
namespace fs = std::filesystem;

// 常见日期时间格式列表（strptime 格式串）。
constexpr const char* kTimeFormats[] = {
    "%Y%m%d%H%M%S",        // 20230101120000
    "%Y-%m-%d %H:%M:%S",   // 2023-01-01 12:00:00
    "%Y-%m-%d_%H-%M-%S",   // 2023-01-01_12-00-00
    "%Y-%m-%dT%H:%M:%S",   // 2023-01-01T12:00:00
    "%Y/%m/%d %H:%M:%S",   // 2023/01/01 12:00:00
    "%Y-%m-%d",            // 2023-01-01
};
constexpr int kTimeFormatsCount = sizeof(kTimeFormats) / sizeof(kTimeFormats[0]);
}  // namespace

double ToUnixSeconds(const fs::file_time_type& t) {
  // 在主流 Linux (libstdc++) 上 file_time_type 的纪元与 Unix 纪元一致，
  // 直接取 time_since_epoch 即可得到 Unix 秒。
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             t.time_since_epoch())
      .count();
}

bool ParseImageTime(const fs::path& file_path, double* seconds) {
  if (seconds == nullptr) {
    return false;
  }
  const std::string stem = file_path.stem().string();

  // 1. 纯数字主名 -> Unix 秒。
  if (!stem.empty()) {
    char* end = nullptr;
    const double value = std::strtod(stem.c_str(), &end);
    if (end != stem.c_str() && *end == '\0') {
      *seconds = value;
      return true;
    }
  }

  // 2. 常见日期时间格式。
  for (int i = 0; i < kTimeFormatsCount; ++i) {
    struct std::tm tmv = {};
    char* end = strptime(stem.c_str(), kTimeFormats[i], &tmv);
    if (end != nullptr && *end == '\0') {
      *seconds = static_cast<double>(std::mktime(&tmv));
      return true;
    }
  }

  // 3. 回退到文件系统修改时间。
  std::error_code ec;
  const fs::file_time_type ftime = fs::last_write_time(file_path, ec);
  if (!ec) {
    *seconds = ToUnixSeconds(ftime);
    return true;
  }
  return false;
}

}  // namespace preprocess
