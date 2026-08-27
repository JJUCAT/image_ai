/**
 * @file image_cleaner.cpp
 * @brief 图像清理主流程实现（时间间隔 + 感知哈希去重）
 *
 * @details
 * 处理流程：
 *  1. 扫描 raw 目录，过滤出支持扩展名的图像文件；
 *  2. 用 ParseImageTime 解析每张图像的时间，解析失败的跳过并计数；
 *  3. 按时间升序稳定排序；
 *  4. 按 batch_size 分批加载解码图像（限制单次内存占用）；
 *  5. 对每张图像按序执行两种规则：
 *     - 时间间隔：与上一张保留图像时间差 < time_min_interval 则丢弃；
 *     - 感知哈希：与全局已保留哈希汉明距离 < hamming_distance 则丢弃；
 *  6. 保留图像以原文件名写入 clean 目录（自动创建）；
 *  7. 保留图像经色彩转换（BGR -> GRAY）提取亮度后，
 *     以同名 PNG 写入 gray 目录。
 *
 * 感知哈希去重为全局判定；batch_size 仅约束单批解码图像的内存上界。
 * 日志通过 spdlog 输出（控制台 + 文件）；清理结束后由
 * BuildCleanReport() 生成包含时间、路径与统计数量的报告文本。
 */

#include "preprocess/image_cleaner.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "preprocess/time_utils.h"

namespace preprocess {

namespace {
namespace fs = std::filesystem;

// 支持处理的图像扩展名（小写）。
constexpr const char* kSupportedExtensions[] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
};

// 判断文件扩展名是否为受支持的图像格式。
bool IsImageFile(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  for (const char* supported : kSupportedExtensions) {
    if (ext == supported) {
      return true;
    }
  }
  return false;
}
}  // namespace

ImageCleaner::ImageCleaner(const CleanConfig& config)
    : config_(config),
      hash_(PerceptualHash::Params{config.scaled_size, config.dct_size,
                                   config.hash_size}) {}

CleanReport ImageCleaner::Clean() {
  CleanReport report;
  const fs::path raw_dir(config_.raw_dir);

  if (!fs::exists(raw_dir) || !fs::is_directory(raw_dir)) {
    throw std::runtime_error("原始图片目录不存在或不可访问: " + config_.raw_dir);
  }
  std::error_code ec;
  fs::create_directories(config_.clean_dir, ec);
  if (ec) {
    throw std::runtime_error("创建输出目录失败 (" + config_.clean_dir + "): " +
                             ec.message());
  }
  fs::create_directories(config_.gray_dir, ec);
  if (ec) {
    throw std::runtime_error("创建灰度输出目录失败 (" + config_.gray_dir + "): " +
                             ec.message());
  }

  // 1. 扫描目录并解析每张图像的时间。
  std::vector<std::pair<fs::path, double>> items;  // (路径, 时间秒)
  for (const auto& entry : fs::directory_iterator(raw_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path& path = entry.path();
    if (!IsImageFile(path)) {
      continue;
    }
    double time = 0.0;
    if (!ParseImageTime(path, &time)) {
      spdlog::warn("无法解析图像时间, 跳过: {}", path.string());
      continue;
    }
    items.emplace_back(path, time);
  }
  report.total_images = items.size();
  spdlog::info("扫描到图像 {} 张", report.total_images);

  // 2. 按时间升序稳定排序（时间相同则保持原顺序）。
  std::stable_sort(items.begin(), items.end(),
                   [](const std::pair<fs::path, double>& lhs,
                      const std::pair<fs::path, double>& rhs) {
                     return lhs.second < rhs.second;
                   });

  // 3. 分批加载并清理。
  std::vector<std::vector<std::uint8_t>> kept_hashes;  // 全局已保留哈希
  bool has_last_time = false;                          // 是否已有上一保留时间
  double last_kept_time = 0.0;                         // 上一保留图像时间(秒)
  const std::size_t batch_size = static_cast<std::size_t>(config_.batch_size);

  for (std::size_t start = 0; start < items.size(); start += batch_size) {
    const std::size_t end = std::min(start + batch_size, items.size());

    // 3.1 一次性解码当前批次的图像（内存上界 = batch_size 张）。
    struct LoadedImage {
      fs::path path;                // 原始文件路径
      cv::Mat mat;                  // 解码后的图像
      double time;                  // 解析得到的时间(秒)
      std::vector<std::uint8_t> hash;  // 感知哈希（延迟计算，初始为空）
    };
    std::vector<LoadedImage> loaded;
    loaded.reserve(end - start);
    for (std::size_t i = start; i < end; ++i) {
      LoadedImage item;
      item.path = items[i].first;
      item.time = items[i].second;
      item.mat = cv::imread(item.path.string(), cv::IMREAD_COLOR);
      if (item.mat.empty()) {
        ++report.failed_to_read;
        spdlog::warn("图像读取失败: {}", item.path.string());
        continue;
      }
      loaded.push_back(std::move(item));
    }

    // 3.2 依次应用时间间隔与感知哈希规则。
    for (LoadedImage& item : loaded) {
      // a) 时间间隔规则：与上一张保留图像间隔过近则丢弃。
      if (has_last_time && (item.time - last_kept_time) < config_.time_min_interval) {
        ++report.dropped_by_time;
        continue;
      }
      // b) 感知哈希规则：与任意已保留图像重复则丢弃。
      if (item.hash.empty()) {
        item.hash = hash_.Compute(item.mat);
      }
      bool duplicate = false;
      for (const auto& kept : kept_hashes) {
        if (PerceptualHash::HammingDistance(item.hash, kept) <
            config_.hamming_distance) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        ++report.dropped_by_hash;
        continue;
      }
      // c) 通过全部规则：写入 clean 目录并记录状态。
      const fs::path out_path = fs::path(config_.clean_dir) / item.path.filename();
      if (!cv::imwrite(out_path.string(), item.mat)) {
        ++report.failed_to_write;
        spdlog::error("图像写入失败: {}", out_path.string());
        continue;
      }
      // d) 色彩转换提取亮度（BGR -> GRAY），以同名 PNG 写入 gray 目录。
      cv::Mat gray;
      cv::cvtColor(item.mat, gray, cv::COLOR_BGR2GRAY);
      const fs::path gray_path =
          fs::path(config_.gray_dir) / (item.path.stem().string() + ".png");
      if (!cv::imwrite(gray_path.string(), gray)) {
        ++report.failed_to_write_gray;
        spdlog::error("灰度图写入失败: {}", gray_path.string());
      }
      kept_hashes.push_back(item.hash);
      has_last_time = true;
      last_kept_time = item.time;
      ++report.kept_images;
    }
  }

  return report;
}

std::string BuildCleanReport(const CleanConfig& config,
                             const CleanReport& report,
                             double elapsed_seconds) {
  std::ostringstream oss;

  // 报告时间（本地时间，格式 YYYY-MM-DD HH:MM:SS）。
  const std::time_t now = std::time(nullptr);
  char time_buf[32] = {0};
  std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                std::localtime(&now));

  oss << "时间: " << time_buf << "\n"
      << "数据源路径: " << config.raw_dir << "\n"
      << "清理后路径: " << config.clean_dir << "\n"
      << "灰度图路径: " << config.gray_dir << "\n"
      << "总图像数: " << report.total_images << "\n"
      << "保留图像数: " << report.kept_images << "\n"
      << "时间间隔丢弃: " << report.dropped_by_time << "\n"
      << "感知哈希重复丢弃: " << report.dropped_by_hash << "\n"
      << "读取失败: " << report.failed_to_read << "\n"
      << "写入失败: " << report.failed_to_write << "\n"
      << "灰度写入失败: " << report.failed_to_write_gray << "\n"
      << "耗时: " << std::fixed << std::setprecision(3) << elapsed_seconds
      << " 秒";
  return oss.str();
}

}  // namespace preprocess
