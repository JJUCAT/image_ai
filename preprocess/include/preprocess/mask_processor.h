/**
 * @file mask_processor.h
 * @brief 标注 mask 像素映射预处理主流程接口
 *
 * @details
 * 本模块实现标注 mask 的像素重映射预处理，应在样本“清理 + 灰度化”之后执行：
 *  1. 以 gray 目录（清理 + 灰度化后的保留图片）为匹配基准，收集全部图片名；
 *  2. 对每个基准文件名，在 raw_masks 目录中查找同名 mask PNG；
 *  3. 构建 8 位查找表（LUT），将 grass_mapping 中的像素值映射为目标值，
 *     其余像素值一律置为 0；
 *  4. 按 batch_size 分批加载 mask（限制单次解码内存占用上界）；
 *  5. 对每张 mask 应用 LUT 映射，以同名（与 gray 一致）写入 gray_masks 目录。
 *
 * 典型用途：标注工具（如 Label Studio）导出的 mask 中多种类别共用同一
 * 标签（如 1、7 均代表草地），通过 grass_mapping 统一映射后再训练；
 * 仅处理与 gray 目录同名（即清理后保留）的 mask。
 *
 * @author bg11738
 * @version 1.1.0
 */

#ifndef PREPROCESS_MASK_PROCESSOR_H_
#define PREPROCESS_MASK_PROCESSOR_H_

#include <cstddef>
#include <string>

#include "preprocess/config_loader.h"

namespace preprocess {

/**
 * @brief mask 预处理统计报告
 *
 * 记录各类成功/失败计数，供主程序打印与后续分析。
 */
struct MaskReport {
  std::size_t total_masks = 0;       ///< gray 目录中的图片总数（匹配基准）
  std::size_t processed_masks = 0;   ///< 成功处理并写出的 mask 数
  std::size_t no_matching_mask = 0;  ///< raw_masks 中未找到同名 mask 的数量
  std::size_t failed_to_read = 0;    ///< 读取失败的 mask 数
  std::size_t failed_to_write = 0;   ///< mask 写入失败的图像数
};

/**
 * @brief 标注 mask 处理器
 *
 * 使用给定的 CleanConfig 构造，调用 Process() 执行完整预处理流程。
 * Process() 可被多次调用；每次调用都会重新扫描并覆盖式写入 gray_masks 目录。
 */
class MaskProcessor {
 public:
  /**
   * @brief 构造标注 mask 处理器
   *
   * @param config 清理配置；使用 gray_dir（匹配基准）、raw_masks_dir（mask
   *               源）、gray_masks_dir（输出）、grass_mapping 与 batch_size 字段。
   */
  explicit MaskProcessor(const CleanConfig& config);

  /**
   * @brief 执行完整 mask 预处理流程
   *
   * @return MaskReport mask 预处理统计结果
   *
   * @exception std::runtime_error raw_masks 目录不存在或不可访问时抛出。
   */
  MaskReport Process();

 private:
  CleanConfig config_;  ///< 清理配置
};

/**
 * @brief 构建 mask 预处理报告文本
 *
 * @param config          清理配置（用于报告中的路径信息）
 * @param report          mask 预处理统计结果
 * @param elapsed_seconds 处理耗时（秒，由调用方计时）
 * @return 多行报告文本，包含：时间、匹配基准路径（gray）、mask 源路径、
 *         mask 输出路径、mask 映射表、总数/成功/无同名/失败计数与耗时。
 */
std::string BuildMaskReport(const CleanConfig& config,
                            const MaskReport& report,
                            double elapsed_seconds);

}  // namespace preprocess

#endif  // PREPROCESS_MASK_PROCESSOR_H_
