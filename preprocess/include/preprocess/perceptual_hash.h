/**
 * @file perceptual_hash.h
 * @brief 可配置参数的 DCT 感知哈希（pHash）接口
 *
 * @details
 * 本模块实现基于 DCT 变换的感知哈希算法，其思路与 OpenCV 4.2
 * img_hash::PHash 一致，但缩放尺寸、DCT 维度、哈希块尺寸均可配置，
 * 便于针对不同数据集调整重复判定灵敏度。
 *
 * 算法流程：
 *  1. 图像转灰度；
 *  2. 缩放到 scaled_size × scaled_size（缩小降噪、去高频细节）；
 *  3. 按需再缩放到 dct_size × dct_size 作为 DCT 输入；
 *  4. 对输入执行二维 DCT，得到 dct_size × dct_size 系数矩阵；
 *  5. 截取矩阵左上角 hash_size × hash_size 的低频系数块；
 *  6. 以该块均值为阈值，系数大于均值记 1，否则记 0，得到位向量哈希。
 *
 * 汉明距离小于 hamming_distance 的两图像判定为重复。
 *
 * @note 哈希位数为 hash_size × hash_size（默认 8×8 = 64 位）。
 *
 * @author bg11738
 * @version 1.0.0
 */

#ifndef PREPROCESS_PERCEPTUAL_HASH_H_
#define PREPROCESS_PERCEPTUAL_HASH_H_

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace preprocess {

/**
 * @brief 可配置参数的 DCT 感知哈希计算器
 *
 * 线程安全：Compute() 为只读操作，可在多线程中并发调用同一实例。
 */
class PerceptualHash {
 public:
  /**
   * @brief 感知哈希参数集合
   *
   * 各参数含义与 config.json 中 perceptual_hash 节点一一对应。
   */
  struct Params {
    int scaled_size = 32;  ///< 缩小后的尺寸（缩小降噪）
    int dct_size = 32;     ///< DCT 变换维度
    int hash_size = 8;     ///< 从 DCT 系数矩阵左上角取出的块尺寸
  };

  /**
   * @brief 构造感知哈希计算器
   *
   * @param params 哈希参数，要求 hash_size <= dct_size 且各项 > 0，
   *               否则抛出 std::invalid_argument。
   */
  explicit PerceptualHash(const Params& params);

  /**
   * @brief 计算单张图像的感知哈希
   *
   * @param image 输入图像，支持 1（灰度）、3（BGR）、4（BGRA）通道。
   * @return 哈希位向量，每个元素取值为 0 或 1，长度为 hash_size²。
   *
   * @exception std::invalid_argument 图像为空或不支持的通道数时抛出。
   */
  std::vector<std::uint8_t> Compute(const cv::Mat& image) const;

  /**
   * @brief 计算两个哈希之间的汉明距离
   *
   * 对应位不同则计数加一；长度不同时，多出的位全部计入距离，
   * 以避免长度不一致的哈希被判为相同。
   *
   * @param lhs 第一个哈希位向量
   * @param rhs 第二个哈希位向量
   * @return 汉明距离
   */
  static int HammingDistance(const std::vector<std::uint8_t>& lhs,
                             const std::vector<std::uint8_t>& rhs);

  /**
   * @brief 返回单个哈希的位数（hash_size²）
   * @return 位数
   */
  std::size_t bit_count() const;

 private:
  Params params_;  ///< 哈希参数副本
};

}  // namespace preprocess

#endif  // PREPROCESS_PERCEPTUAL_HASH_H_
