/**
 * @file mask_processor.cpp
 * @brief 标注 mask 像素映射预处理主流程实现
 *
 * @details
 * 本任务应在样本“清理 + 灰度化”之后执行，处理流程：
 *  1. 以 gray 目录（清理 + 灰度化后的保留图片）为匹配基准，收集全部图片名；
 *  2. 对每个基准文件名，在 raw_masks 目录中查找同名 mask PNG；
 *  3. 构建 256 项查找表（LUT）：grass_mapping 中键对应的像素值映射为
 *     目标值，其余像素值一律置为 0；
 *  4. 按 batch_size 分批加载 mask（限制单次解码内存占用上界）；
 *  5. 对每张 mask 应用 cv::LUT 完成像素重映射；
 *  6. 映射结果以同名（与 gray 一致）写入 gray_masks 目录（自动创建）。
 *
 * 日志通过 spdlog 输出（控制台 + 文件）；处理结束后由 BuildMaskReport()
 * 生成包含时间、路径、映射表与统计数量的报告文本。
 */

#include "preprocess/mask_processor.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace preprocess {

namespace {
namespace fs = std::filesystem;

// 支持处理的图像扩展名（小写）。gray 与 raw_masks 通常均为 PNG，可按需扩展。
constexpr const char* kMaskExtensions[] = {
    ".png",
};

// 判断文件扩展名是否为受支持的图像格式。
bool IsMaskFile(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  for (const char* supported : kMaskExtensions) {
    if (ext == supported) {
      return true;
    }
  }
  return false;
}
}  // namespace

MaskProcessor::MaskProcessor(const CleanConfig& config) : config_(config) {}

MaskReport MaskProcessor::Process() {
  MaskReport report;
  const fs::path gray_dir(config_.gray_dir);         // 匹配基准（清理+灰度化输出）
  const fs::path mask_dir(config_.raw_masks_dir);    // 原始标注 mask 目录

  // 校验输入目录并创建输出目录。
  if (!fs::exists(gray_dir) || !fs::is_directory(gray_dir)) {
    throw std::runtime_error("灰度图目录不存在或不可访问: " + config_.gray_dir);
  }
  if (!fs::exists(mask_dir) || !fs::is_directory(mask_dir)) {
    throw std::runtime_error("原始 mask 目录不存在或不可访问: " +
                             config_.raw_masks_dir);
  }
  std::error_code ec;
  fs::create_directories(config_.gray_masks_dir, ec);
  if (ec) {
    throw std::runtime_error("创建 mask 输出目录失败 (" + config_.gray_masks_dir +
                             "): " + ec.message());
  }
  // 清空输出目录，确保 gray_masks 与 gray 基准严格一一对应，
  // 避免上次运行残留的不匹配文件被保留。
  for (const auto& entry : fs::directory_iterator(config_.gray_masks_dir)) {
    fs::remove_all(entry.path(), ec);
    if (ec) {
      spdlog::warn("清理 mask 输出目录失败 ({}): {}", entry.path().string(),
                   ec.message());
      ec.clear();
    }
  }

  // 1. 遍历 gray 目录收集基准图片（即清理 + 灰度化后保留的图片）。
  std::vector<fs::path> gray_files;
  for (const auto& entry : fs::directory_iterator(gray_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path& path = entry.path();
    if (!IsMaskFile(path)) {
      continue;
    }
    gray_files.push_back(path);
  }
  report.total_masks = gray_files.size();
  spdlog::info("灰度图目录图片 {} 张（作为 mask 匹配基准）", report.total_masks);
  if (gray_files.empty()) {
    spdlog::warn("灰度图目录无图片，跳过 mask 映射: {}", config_.gray_dir);
    return report;
  }

  // 2. 构建 8 位查找表：映射表中的原值替换为目标值，其余一律置 0。
  std::vector<uchar> lut(256, 0);
  for (const auto& [from, to] : config_.grass_mapping) {
    if (from >= 0 && from < 256) {
      lut[static_cast<std::size_t>(from)] = static_cast<uchar>(to);
    }
  }
  std::ostringstream mapping_text;
  for (auto it = config_.grass_mapping.begin(); it != config_.grass_mapping.end(); ++it) {
    if (it != config_.grass_mapping.begin()) {
      mapping_text << ", ";
    }
    mapping_text << it->first << "->" << it->second;
  }
  spdlog::info("像素映射表: {}", mapping_text.str());

  // 3. 按 batch_size 分批处理：取 gray 中同名 raw_masks 映射并写出。
  const std::size_t batch_size = static_cast<std::size_t>(config_.batch_size);
  for (std::size_t start = 0; start < gray_files.size(); start += batch_size) {
    const std::size_t end = std::min(start + batch_size, gray_files.size());
    for (std::size_t i = start; i < end; ++i) {
      const fs::path& gray_path = gray_files[i];
      const std::string name = gray_path.filename().string();

      // 在 raw_masks 中查找同名 mask。
      const fs::path mask_src = mask_dir / gray_path.filename();
      if (!fs::exists(mask_src) || !fs::is_regular_file(mask_src)) {
        ++report.no_matching_mask;
        // spdlog::warn("raw_masks 中未找到同名 mask: {}", name);
        continue;
      }

      // 保留原始格式读取（8 位 PNG 得到 CV_8UC1），避免颜色转换破坏标签值。
      cv::Mat mask = cv::imread(mask_src.string(), cv::IMREAD_UNCHANGED);
      if (mask.empty()) {
        ++report.failed_to_read;
        spdlog::warn("mask 读取失败: {}", mask_src.string());
        continue;
      }
      // 非 8 位单通道（如 16 位 PNG 或多通道图像）先转为 8 位灰度，
      // 保证 LUT 语义正确。
      if (mask.type() != CV_8U) {
        cv::Mat converted;
        if (mask.channels() == 1) {
          cv::Mat tmp;
          mask.convertTo(tmp, CV_8U);
          converted = tmp;
        } else if (mask.channels() == 3) {
          cv::cvtColor(mask, converted, cv::COLOR_BGR2GRAY);
        } else if (mask.channels() == 4) {
          cv::cvtColor(mask, converted, cv::COLOR_BGRA2GRAY);
        } else {
          ++report.failed_to_read;
          spdlog::warn("mask 通道数不支持: {}", mask_src.string());
          continue;
        }
        mask = converted;
      }
      if (mask.type() != CV_8U) {
        ++report.failed_to_read;
        spdlog::warn("mask 格式不支持（仅支持 8 位单通道）: {}", mask_src.string());
        continue;
      }

      // 应用查找表完成像素重映射。
      cv::Mat mapped;
      cv::LUT(mask, lut, mapped);

      // 以同名（与 gray 一致）写入 gray_masks 目录。
      const fs::path out_path = fs::path(config_.gray_masks_dir) / name;
      if (!cv::imwrite(out_path.string(), mapped)) {
        ++report.failed_to_write;
        spdlog::error("mask 写入失败: {}", out_path.string());
        continue;
      }
      ++report.processed_masks;
    }
  }

  return report;
}

std::string BuildMaskReport(const CleanConfig& config,
                            const MaskReport& report,
                            double elapsed_seconds) {
  std::ostringstream oss;

  // 报告时间（本地时间，格式 YYYY-MM-DD HH:MM:SS）。
  const std::time_t now = std::time(nullptr);
  char time_buf[32] = {0};
  std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                std::localtime(&now));

  std::ostringstream mapping_text;
  for (auto it = config.grass_mapping.begin(); it != config.grass_mapping.end(); ++it) {
    if (it != config.grass_mapping.begin()) {
      mapping_text << ", ";
    }
    mapping_text << it->first << " -> " << it->second;
  }

  oss << "时间: " << time_buf << "\n"
      << "匹配基准目录: " << config.gray_dir << "\n"
      << "mask 源路径: " << config.raw_masks_dir << "\n"
      << "mask 输出路径: " << config.gray_masks_dir << "\n"
      << "像素映射表: " << mapping_text.str() << "\n"
      << "总 mask 数: " << report.total_masks << "\n"
      << "处理成功: " << report.processed_masks << "\n"
      << "未找到同名 mask: " << report.no_matching_mask << "\n"
      << "读取失败: " << report.failed_to_read << "\n"
      << "写入失败: " << report.failed_to_write << "\n"
      << "耗时: " << std::fixed << std::setprecision(3) << elapsed_seconds
      << " 秒";
  return oss.str();
}

}  // namespace preprocess
