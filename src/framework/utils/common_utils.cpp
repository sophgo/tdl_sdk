#include "utils/common_utils.hpp"
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>  // for PATH_MAX
#include <unistd.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include "utils/tdl_log.hpp"

uint32_t CommonUtils::getDataTypeSize(TDLDataType data_type) {
  switch (data_type) {
    case TDLDataType::FP32:
      return 4;
    case TDLDataType::INT32:
      return 4;
    case TDLDataType::UINT32:
      return 4;
    case TDLDataType::FP16:
      return 2;
    case TDLDataType::BF16:
      return 2;
    case TDLDataType::INT16:
      return 2;
    case TDLDataType::UINT16:
      return 2;
    case TDLDataType::INT8:
      return 1;
    case TDLDataType::UINT8:
      return 1;
    default:
      return 0;
  }
}

InferencePlatform CommonUtils::getPlatform() {
#if defined(__BM168X__)
  return InferencePlatform::BM168X;
#elif defined(__CV186X__)
  return InferencePlatform::CV186X;
#elif defined(__CV184X__)
  return InferencePlatform::CV184X;
#elif defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__)
  return InferencePlatform::CVITEK;
#elif defined(__CMODEL_CV181X__)
  return InferencePlatform::CMODEL_CV181X;
#elif defined(__CMODEL_CV184X__)
  return InferencePlatform::CMODEL_CV184X;
#else
  return InferencePlatform::UNKOWN;
#endif
}

bool CommonUtils::readBinaryFile(const std::string& strf,
                                 std::vector<uint8_t>& buffer) {
  FILE* fp = fopen(strf.c_str(), "rb");
  if (fp == nullptr) {
    LOGE("read file failed,%s\n", strf.c_str());
    return false;
  }
  fseek(fp, 0, SEEK_END);
  int len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  buffer.resize(len);
  fread(buffer.data(), len, 1, fp);
  fclose(fp);
  return true;
}

std::string CommonUtils::getLibraryPath() {
  Dl_info info{};
  // use the address of any function or static in your .so:
  if (dladdr((void*)&CommonUtils::getLibraryPath, &info) && info.dli_fname) {
    // info.dli_fname is the path as the dynamic loader saw it
    char resolved[PATH_MAX];
    if (realpath(info.dli_fname, resolved))
      return std::string(resolved);
    else
      return std::string(info.dli_fname);
  }
  return {};
}

std::string CommonUtils::getLibraryDir() {
  auto full = getLibraryPath();
  auto pos = full.find_last_of('/');
  return (pos != std::string::npos ? full.substr(0, pos) : std::string{});
}

std::string CommonUtils::getParentDir(const std::string& path) {
  auto pos = path.find_last_of('/');
  return (pos != std::string::npos ? path.substr(0, pos) : std::string{});
}

std::string CommonUtils::getExecutableDir() {
  char buffer[PATH_MAX];
  auto len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len == -1) {
    return "";
  }
  buffer[len] = '\0';
  char dir[PATH_MAX];
  strncpy(dir, buffer, PATH_MAX);
  return std::string(dirname(dir));
}

int32_t CommonUtils::randomFill(uint8_t* data, uint32_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  for (uint32_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>(dis(gen));
  }
  return 0;
}

float CommonUtils::dot_product(const std::vector<float>& a,
                               const std::vector<float>& b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

// L2归一化
void CommonUtils::normalize(std::vector<float>& v) {
  float norm = 0.0f;
  for (float f : v) norm += f * f;
  norm = std::sqrt(norm);
  if (norm > 1e-6)
    for (float& f : v) f /= norm;
}

std::vector<float> CommonUtils::softmax(const std::vector<float>& logits) {
  std::vector<float> result(logits.size());
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    result[i] = std::exp(logits[i] - max_logit);  // for numerical stability
    sum += result[i];
  }
  for (float& v : result) v /= sum;
  return result;
}

void CommonUtils::visualizeMask(
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta,
    const std::string& str_img_name) {
  if (!obj_meta || obj_meta->box_seg.empty()) {
    return;
  }

  int proto_h = obj_meta->mask_height;
  int proto_w = obj_meta->mask_width;

  cv::Mat dst;
  for (uint32_t i = 0; i < obj_meta->box_seg.size(); i++) {
    cv::Mat src(proto_h, proto_w, CV_8UC1, obj_meta->box_seg[i].mask,
                proto_w * sizeof(uint8_t));

    if (i == 0) {
      dst = src.clone();
    } else {
      cv::bitwise_or(dst, src, dst);
    }
  }

  if (obj_meta->box_seg.size() > 0) {
    cv::imwrite(str_img_name, dst);
  }
}

std::pair<int, int> CommonUtils::findMaskOrSmallestBbox(
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta, uint32_t image_height,
    uint32_t image_width, cv::Point point) {
  if (!obj_meta || obj_meta->box_seg.empty()) {
    return std::make_pair(-1, -1);
  }

  const int proto_h = static_cast<int>(obj_meta->mask_height);
  const int proto_w = static_cast<int>(obj_meta->mask_width);

  // === Step 1: 查找包含该点的面积最小bbox ===
  int smallest_bbox_index = -1;
  float min_area = std::numeric_limits<float>::max();

  // 缓存 box_seg 引用以提高性能
  const auto& box_seg = obj_meta->box_seg;
  for (uint32_t i = 0; i < box_seg.size(); ++i) {
    const auto& seg = box_seg[i];
    float x1 = seg.x1, y1 = seg.y1, x2 = seg.x2, y2 = seg.y2;

    // 跳过无效框（宽/高为0或负数）
    if (x2 <= x1 || y2 <= y1) continue;

    // 检查点是否在bbox内（闭区间）
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) {
      float area = (x2 - x1) * (y2 - y1);
      if (area < min_area) {
        min_area = area;
        smallest_bbox_index = static_cast<int>(i);
      }
    }
  }

  // === Step 2: 计算proto坐标（仅在找到最小框后才需要）===
  int proto_x = -1, proto_y = -1;
  if (smallest_bbox_index >= 0) {
    float ratio_height = static_cast<float>(proto_h) / image_height;
    float ratio_width = static_cast<float>(proto_w) / image_width;

    int source_y_offset = 0, source_x_offset = 0;
    if (ratio_height > ratio_width) {
      source_y_offset = static_cast<int>(
          (proto_h - image_height * ratio_width) / 2.0f + 0.5f);
      source_x_offset = 0;
    } else {
      source_x_offset = static_cast<int>(
          (proto_w - image_width * ratio_height) / 2.0f + 0.5f);
      source_y_offset = 0;
    }

    int source_region_height = proto_h - 2 * source_y_offset;
    int source_region_width = proto_w - 2 * source_x_offset;

    float height_scale =
        static_cast<float>(source_region_height) / image_height;
    float width_scale = static_cast<float>(source_region_width) / image_width;

    proto_x = static_cast<int>(point.x * width_scale + source_x_offset + 0.5f);
    proto_y = static_cast<int>(point.y * height_scale + source_y_offset + 0.5f);

    proto_x = std::max(0, std::min(proto_x, proto_w - 1));
    proto_y = std::max(0, std::min(proto_y, proto_h - 1));
  }

  // === Step 3: 仅检查最小框对应的mask（而非所有mask）===
  int mask_index = -1;
  if (smallest_bbox_index >= 0) {
    const auto& seg = box_seg[smallest_bbox_index];
    cv::Mat proto_mask(proto_h, proto_w, CV_8UC1, seg.mask,
                       proto_w * sizeof(uint8_t));

    if (proto_mask.at<uint8_t>(proto_y, proto_x) > 0) {
      mask_index = smallest_bbox_index;  // mask验证通过
    }
  }

  // 返回：mask_index（最小框的mask验证结果），smallest_bbox_index（找到的最小框索引）
  return std::make_pair(mask_index, smallest_bbox_index);
}