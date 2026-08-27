/**
 * @file image_cleaner.h
 * @brief 图像清理主流程接口（时间间隔 + 感知哈希去重）
 *
 * @details
 * 本模块实现图像清理主流程：
 *  1. 扫描 raw 目录，收集全部图像并解析文件名时间；
 *  2. 按时间升序排序；
 *  3. 按 batch_size 分批加载图像（内存占用上界由批大小决定）；
 *  4. 对每张图像依次执行两种清理规则，均通过才保留：
 *     a) 时间间隔规则：与上一张保留图像的时间差小于
 *        time_min_interval 秒的图像被丢弃；
 *     b) 感知哈希规则：与任意已保留图像汉明距离小于
 *        hamming_distance 的图像判为重复并丢弃；
 *  5. 保留图像以原文件名写入 clean 目录。
 *
 * @note 感知哈希去重为全局判定（与所有已保留图像比较），
 *       batch_size 仅限制单次解码图像的内存占用。
 *
 * @author bg11738
 * @version 1.0.0
 */

#ifndef PREPROCESS_IMAGE_CLEANER_H_
#define PREPROCESS_IMAGE_CLEANER_H_

#include <cstddef>

#include "preprocess/config_loader.h"
#include "preprocess/perceptual_hash.h"

namespace preprocess {

/**
 * @brief 清理结果统计报告
 *
 * 记录各类丢弃/失败计数，供主程序打印与后续分析。
 */
struct CleanReport {
  std::size_t total_images = 0;        ///< 扫描到的图像总数
  std::size_t kept_images = 0;         ///< 成功保留并写出的图像数
  std::size_t dropped_by_time = 0;     ///< 因时间间隔过近被丢弃
  std::size_t dropped_by_hash = 0;     ///< 因感知哈希重复被丢弃
  std::size_t failed_to_read = 0;      ///< 读取失败的图像数
  std::size_t failed_to_write = 0;     ///< clean 图像写入失败的图像数
  std::size_t failed_to_write_gray = 0;  ///< 灰度 PNG 写入失败的图像数
};

/**
 * @brief 图像清理器
 *
 * 使用给定的 CleanConfig 构造，调用 Clean() 执行完整清理流程。
 * Clean() 可被多次调用；每次调用都会重新扫描并覆盖式写入 clean 目录。
 */
class ImageCleaner {
 public:
  /**
   * @brief 构造图像清理器
   *
   * @param config 清理配置；内部据此构造感知哈希计算器。
   */
  explicit ImageCleaner(const CleanConfig& config);

  /**
   * @brief 执行完整清理流程
   *
   * @return CleanReport 清理统计结果
   *
   * @exception std::runtime_error raw 目录不存在或不可访问时抛出。
   */
  CleanReport Clean();

 private:
  CleanConfig config_;      ///< 清理配置
  PerceptualHash hash_;     ///< 感知哈希计算器（按配置参数构造）
};

/**
 * @brief 构建清理报告文本
 *
 * @param config          清理配置（用于报告中的路径信息）
 * @param report          清理统计结果
 * @param elapsed_seconds 清理耗时（秒，由调用方计时）
 * @return 多行报告文本，包含：时间、数据源路径、清理后路径、
 *         总图像数、保留数、各类丢弃/失败计数与耗时。
 */
std::string BuildCleanReport(const CleanConfig& config,
                             const CleanReport& report,
                             double elapsed_seconds);

}  // namespace preprocess

#endif  // PREPROCESS_IMAGE_CLEANER_H_
