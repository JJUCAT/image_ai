/**
 * @file perceptual_hash.cpp
 * @brief 可配置 DCT 感知哈希的实现
 *
 * @details
 * 实现细节：
 *  - 灰度化：3 通道 BGR -> GRAY，4 通道 BGRA -> GRAY，1 通道直接使用；
 *  - 缩放：先缩放到 scaled_size × scaled_size（INTER_AREA 适合缩小降噪），
 *    若 DCT 维度不同，再缩放到 dct_size × dct_size 作为 DCT 输入；
 *  - DCT：对单通道 CV_32F 矩阵调用 cv::dct；
 *  - 取块：截取系数矩阵左上角 hash_size × hash_size 低频块；
 *  - 阈值化：以块内均值为阈值，系数大于均值置 1，否则置 0。
 */

#include "preprocess/perceptual_hash.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace preprocess {

PerceptualHash::PerceptualHash(const Params& params) : params_(params) {
  if (params_.scaled_size <= 0 || params_.dct_size <= 0 || params_.hash_size <= 0) {
    throw std::invalid_argument("感知哈希参数必须为正整数");
  }
  if (params_.hash_size > params_.dct_size) {
    throw std::invalid_argument("感知哈希 hash_size 不能大于 dct_size");
  }
}

std::vector<std::uint8_t> PerceptualHash::Compute(const cv::Mat& image) const {
  if (image.empty()) {
    throw std::invalid_argument("输入图像为空, 无法计算感知哈希");
  }

  // 1. 转灰度。
  cv::Mat gray;
  switch (image.channels()) {
    case 1:
      gray = image;
      break;
    case 3:
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
      break;
    case 4:
      cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
      break;
    default:
      throw std::invalid_argument("不支持的图像通道数: " +
                                  std::to_string(image.channels()));
  }

  // 2. 缩小到 scaled_size，并转为单精度浮点。
  cv::Mat small;
  cv::resize(gray, small,
             cv::Size(params_.scaled_size, params_.scaled_size), 0, 0,
             cv::INTER_AREA);
  small.convertTo(small, CV_32F);

  // 3. 若 DCT 维度与缩放尺寸不同，则再缩放到 dct_size。
  cv::Mat dct_input;
  if (params_.scaled_size != params_.dct_size) {
    cv::resize(small, dct_input,
               cv::Size(params_.dct_size, params_.dct_size), 0, 0,
               cv::INTER_AREA);
  } else {
    dct_input = small;
  }

  // 4. 二维 DCT，得到 dct_size × dct_size 系数矩阵。
  cv::Mat dct_out;
  cv::dct(dct_input, dct_out);

  // 5. 截取左上角 hash_size × hash_size 低频块，并计算均值。
  const int n = params_.hash_size;
  double sum = 0.0;
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      sum += dct_out.at<float>(r, c);
    }
  }
  const double mean = sum / static_cast<double>(n * n);

  // 6. 系数大于均值记 1，否则记 0，生成位向量哈希。
  std::vector<std::uint8_t> bits;
  bits.reserve(static_cast<std::size_t>(n * n));
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      bits.push_back(dct_out.at<float>(r, c) > mean ? 1u : 0u);
    }
  }
  return bits;
}

int PerceptualHash::HammingDistance(const std::vector<std::uint8_t>& lhs,
                                    const std::vector<std::uint8_t>& rhs) {
  const std::size_t common = std::min(lhs.size(), rhs.size());
  int distance = 0;
  for (std::size_t i = 0; i < common; ++i) {
    if (lhs[i] != rhs[i]) {
      ++distance;
    }
  }
  // 长度不同的部分全部计入距离，避免长度不一致被判为相同。
  const std::size_t extra =
      lhs.size() > rhs.size() ? lhs.size() - rhs.size() : rhs.size() - lhs.size();
  distance += static_cast<int>(extra);
  return distance;
}

std::size_t PerceptualHash::bit_count() const {
  return static_cast<std::size_t>(params_.hash_size) * params_.hash_size;
}

}  // namespace preprocess
