/**
 * @file config_loader.h
 * @brief 清理配置结构体定义与 JSON 配置加载接口
 *
 * @details
 * 本模块负责定义图像清理所需的全部配置项，并从 JSON 配置文件
 * （config/config.json）中解析、校验并加载这些配置。
 * 配置项语义如下：
 *  - "raw"               原始样本图片所在目录
 *  - "clean"             数据清理后图片的输出目录
 *  - "gray"              清理图片转灰度 PNG 的输出目录
 *  - "batch_szie"        批量清理时单次加载的图像数量
 *  - "time_min_interval" 图像名视为时间，保留图像间的最小时间间隔（秒）
 *  - "scaled_size"       感知哈希：哈希计算前的缩小尺寸
 *  - "DCT_szie"          感知哈希：DCT 变换维度
 *  - "hash_size"         感知哈希：从 DCT 系数矩阵左上角取出的块尺寸
 *  - "hamming_distance"  感知哈希：两图像汉明距离小于该值判为重复
 *  - "raw_masks"         原始标注 mask PNG 所在目录
 *  - "gray_masks"        mask 像素映射处理后的输出目录
 *  - "grass_mapping"     mask 像素值映射表（原像素值 -> 新像素值，
 *                        不在映射中的像素值一律置为 0）
 *
 * @note 配置文件中键名 "batch_szie" 与 "DCT_szie" 为约定拼写，请勿修改，
 *       否则加载会因键名不匹配而抛出异常。
 *
 * @author bg11738
 * @version 1.0.0
 */

#ifndef PREPROCESS_CONFIG_LOADER_H_
#define PREPROCESS_CONFIG_LOADER_H_

#include <map>
#include <string>

namespace preprocess {

/**
 * @brief 图像清理配置结构体
 *
 * 保存从 config/config.json 中解析出的全部清理参数，
 * 字段默认值用于 JSON 缺省时提供安全兜底。
 */
struct CleanConfig {
  std::string raw_dir;                    ///< 原始样本图片目录路径
  std::string clean_dir;                  ///< 清理后图片输出目录路径
  std::string gray_dir;                   ///< 清理图片转灰度 PNG 的输出目录路径
  int batch_size = 1000;                  ///< 批量加载的图像数量（>0）
  double time_min_interval = 1.0;         ///< 保留图像间的最小时间间隔（秒，>=0）
  int scaled_size = 32;                   ///< 感知哈希缩小尺寸（>0）
  int dct_size = 32;                      ///< 感知哈希 DCT 变换维度（>0）
  int hash_size = 8;                      ///< 从 DCT 左上角取出的哈希块尺寸（>0 且 <= dct_size）
  int hamming_distance = 6;               ///< 感知哈希汉明距离阈值（>=0）
  std::string raw_masks_dir;              ///< 原始标注 mask PNG 所在目录路径
  std::string gray_masks_dir;             ///< mask 像素映射处理后的输出目录路径
  std::map<int, int> grass_mapping;       ///< mask 像素值映射表（原像素值 -> 新像素值）
};

/**
 * @brief 从 JSON 配置文件加载并校验清理配置
 *
 * @param config_path 配置文件路径（例如 "config/config.json"）
 * @return 解析并校验通过的 CleanConfig 结构体
 *
 * @exception std::runtime_error 文件无法打开、键缺失或参数非法时抛出，
 *            异常消息中包含具体原因。
 */
CleanConfig LoadConfig(const std::string& config_path);

}  // namespace preprocess

#endif  // PREPROCESS_CONFIG_LOADER_H_
