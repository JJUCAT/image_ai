/**
 * @file config_loader.cpp
 * @brief 清理配置的 JSON 加载与校验实现
 *
 * @details
 * 使用 nlohmann/json 解析 config/config.json：
 *  - 读取顶层 "raw"、"clean"、"gray"、"raw_masks"、"gray_masks"；
 *  - 读取 "clean_config" 节点下的 "batch_szie"、"time_min_interval"；
 *  - 读取 "clean_config.perceptual_hash" 节点下的
 *    "scaled_size"、"DCT_szie"、"hash_size"、"hamming_distance"；
 *  - 读取顶层 "grass_mapping" 对象，解析为像素值映射表（原值 -> 新值）。
 * 解析完成后对全部参数进行合理性校验，非法时抛出异常并给出原因。
 */

#include "preprocess/config_loader.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace preprocess {

namespace {
namespace fs = std::filesystem;
}  // namespace

CleanConfig LoadConfig(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开配置文件: " + config_path);
  }

  nlohmann::json root;
  try {
    file >> root;
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error("配置文件 JSON 解析失败 (" + config_path + "): " + e.what());
  }

  CleanConfig config;

  // 顶层：原始目录、清理输出目录与灰度输出目录。
  config.raw_dir = root.at("raw").get<std::string>();
  config.clean_dir = root.at("clean").get<std::string>();
  config.gray_dir = root.at("gray").get<std::string>();

  // 顶层：原始 mask 目录、mask 映射输出目录与像素映射表。
  config.raw_masks_dir = root.at("raw_masks").get<std::string>();
  config.gray_masks_dir = root.at("gray_masks").get<std::string>();
  const nlohmann::json& gm = root.at("grass_mapping");
  if (!gm.is_object()) {
    throw std::runtime_error("配置错误: \"grass_mapping\" 必须为对象");
  }
  for (auto it = gm.begin(); it != gm.end(); ++it) {
    int from = 0;
    try {
      from = std::stoi(it.key());
    } catch (const std::exception&) {
      throw std::runtime_error("配置错误: \"grass_mapping\" 键 '" + it.key() +
                               "' 无法解析为整数像素值");
    }
    config.grass_mapping[from] = it.value().get<int>();
  }

  // clean_config 节点：批量大小与时间间隔。
  const nlohmann::json& cc = root.at("clean_config");
  config.batch_size = cc.at("batch_szie").get<int>();
  config.time_min_interval = cc.at("time_min_interval").get<double>();

  // perceptual_hash 节点：感知哈希参数。
  const nlohmann::json& ph = cc.at("perceptual_hash");
  config.scaled_size = ph.at("scaled_size").get<int>();
  config.dct_size = ph.at("DCT_szie").get<int>();
  config.hash_size = ph.at("hash_size").get<int>();
  config.hamming_distance = ph.at("hamming_distance").get<int>();

  // ---- 参数校验 ----
  if (config.raw_dir.empty()) {
    throw std::runtime_error("配置错误: \"raw\" 原始目录不能为空");
  }
  if (config.clean_dir.empty()) {
    throw std::runtime_error("配置错误: \"clean\" 输出目录不能为空");
  }
  if (config.gray_dir.empty()) {
    throw std::runtime_error("配置错误: \"gray\" 灰度输出目录不能为空");
  }
  if (config.raw_masks_dir.empty()) {
    throw std::runtime_error("配置错误: \"raw_masks\" 原始 mask 目录不能为空");
  }
  if (config.gray_masks_dir.empty()) {
    throw std::runtime_error("配置错误: \"gray_masks\" mask 输出目录不能为空");
  }
  if (config.grass_mapping.empty()) {
    throw std::runtime_error("配置错误: \"grass_mapping\" 映射表不能为空");
  }
  for (const auto& [from, to] : config.grass_mapping) {
    if (from < 0 || from > 255) {
      throw std::runtime_error(
          "配置错误: \"grass_mapping\" 原像素值 " + std::to_string(from) +
          " 超出 8 位灰度范围 [0, 255]");
    }
    if (to < 0 || to > 255) {
      throw std::runtime_error(
          "配置错误: \"grass_mapping\" 映射值 " + std::to_string(to) +
          " 超出 8 位灰度范围 [0, 255]");
    }
  }
  if (config.batch_size <= 0) {
    throw std::runtime_error("配置错误: \"batch_szie\" 必须为正整数, 当前: " +
                             std::to_string(config.batch_size));
  }
  if (config.time_min_interval < 0.0) {
    throw std::runtime_error("配置错误: \"time_min_interval\" 不能为负, 当前: " +
                             std::to_string(config.time_min_interval));
  }
  if (config.scaled_size <= 0 || config.dct_size <= 0 || config.hash_size <= 0) {
    throw std::runtime_error(
        "配置错误: \"scaled_size\"、\"DCT_szie\"、\"hash_size\" 必须为正整数");
  }
  if (config.hash_size > config.dct_size) {
    throw std::runtime_error(
        "配置错误: \"hash_size\"(" + std::to_string(config.hash_size) +
        ") 不能大于 \"DCT_szie\"(" + std::to_string(config.dct_size) + ")");
  }
  if (config.hamming_distance < 0) {
    throw std::runtime_error("配置错误: \"hamming_distance\" 不能为负, 当前: " +
                             std::to_string(config.hamming_distance));
  }

  return config;
}

}  // namespace preprocess
