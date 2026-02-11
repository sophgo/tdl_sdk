#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

void printMaskForBboxIndex(std::shared_ptr<ModelBoxSegmentationInfo> obj_meta,
                           int bbox_index) {
  if (bbox_index < 0 ||
      bbox_index >= static_cast<int>(obj_meta->box_seg.size())) {
    std::cout << "Invalid bbox index." << std::endl;
    return;
  }

  auto& seg = obj_meta->box_seg[bbox_index];
  int proto_h = obj_meta->mask_height;  // 特征图高度
  int proto_w = obj_meta->mask_width;   // 特征图宽度

  cv::Mat mask(proto_h, proto_w, CV_8UC1, seg.mask, proto_w * sizeof(uint8_t));

  std::cout << "Mask for bbox index " << bbox_index << ":" << std::endl;
  for (int y = 0; y < mask.rows; ++y) {
    for (int x = 0; x < mask.cols; ++x) {
      std::cout << static_cast<int>(mask.at<uint8_t>(y, x)) << " ";
    }
    std::cout << std::endl;
  }

  cv::imwrite("mask_" + std::to_string(bbox_index) + ".png", mask);
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_path> <point_x> "
        "<point_y>\n",
        argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_path = argv[3];
  int point_x = std::stoi(argv[4]);
  int point_y = std::stoi(argv[5]);

  const int kCropSize = 320;

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  const int img_w = static_cast<int>(image1->getWidth());
  const int img_h = static_cast<int>(image1->getHeight());
  // 以种子点为中心抠 320x320，边界处截断到图像内
  int crop_x1 = point_x - kCropSize / 2;
  int crop_y1 = point_y - kCropSize / 2;
  crop_x1 = std::max(0, crop_x1);
  crop_y1 = std::max(0, crop_y1);
  int crop_x2 = std::min(img_w, crop_x1 + kCropSize);
  int crop_y2 = std::min(img_h, crop_y1 + kCropSize);
  int crop_w = crop_x2 - crop_x1;
  int crop_h = crop_y2 - crop_y1;
  if (crop_w <= 0 || crop_h <= 0) {
    printf("Crop region invalid (seed point or image size error).\n");
    return -1;
  }

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_od = model_factory.getModel(model_id_name);
  if (!model_od) {
    printf("Failed to create model_od\n");
    return -1;
  }

  std::shared_ptr<BasePreprocessor> preprocessor = model_od->getPreprocessor();
  if (!preprocessor) {
    printf("Failed to get preprocessor\n");
    return -1;
  }
  std::shared_ptr<BaseImage> crop_image =
      preprocessor->crop(image1, crop_x1, crop_y1, crop_w, crop_h);
  if (!crop_image) {
    printf("Failed to crop image\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {crop_image};
  model_od->inference(input_images, out_datas);

  // 种子点在裁剪图内的坐标
  cv::Point check_point_in_crop(point_x - crop_x1, point_y - crop_y1);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta =
        std::static_pointer_cast<ModelBoxSegmentationInfo>(out_datas[i]);

    uint32_t image_width = input_images[i]->getWidth();
    uint32_t image_height = input_images[i]->getHeight();

    printf("Crop dimensions (inference input) - height: %u, width: %u\n",
           image_height, image_width);
    CommonUtils::visualizeMask(obj_meta, "yolov8_seg_mask.png");

    // 在裁剪图坐标系下查找 mask 或最小 bbox
    std::pair<int, int> result_pair = CommonUtils::findMaskOrSmallestBbox(
        obj_meta, image_height, image_width, check_point_in_crop);
    int mask_index = result_pair.first;
    int smallest_bbox_index = result_pair.second;

    cv::Mat vis_image;
    bool is_rgb;
    int32_t ret =
        ImageFactory::convertToMat(input_images[i], vis_image, is_rgb);
    if (ret != 0) {
      std::cout << "Failed to convert image to Mat for visualization."
                << std::endl;
      continue;
    }

    if (is_rgb) {
      cv::cvtColor(vis_image, vis_image, cv::COLOR_RGB2BGR);
    }

    // 在裁剪图上绘制种子点（裁剪图内坐标）
    cv::circle(vis_image, check_point_in_crop, 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(vis_image, check_point_in_crop, 5, cv::Scalar(255, 255, 255), 1);

    if (mask_index >= 0) {
      auto& seg = obj_meta->box_seg[mask_index];
      cv::rectangle(vis_image,
                    cv::Rect(static_cast<int>(seg.x1), static_cast<int>(seg.y1),
                             static_cast<int>(seg.x2 - seg.x1),
                             static_cast<int>(seg.y2 - seg.y1)),
                    cv::Scalar(0, 255, 0), 2);
    } else if (smallest_bbox_index >= 0) {
      auto& seg = obj_meta->box_seg[smallest_bbox_index];
      cv::rectangle(vis_image,
                    cv::Rect(static_cast<int>(seg.x1), static_cast<int>(seg.y1),
                             static_cast<int>(seg.x2 - seg.x1),
                             static_cast<int>(seg.y2 - seg.y1)),
                    cv::Scalar(0, 255, 255), 2);
    }

    std::string output_path = "highlighted_result.jpg";
    if (cv::imwrite(output_path, vis_image)) {
      std::cout << "Saved result to " << output_path << std::endl;
    } else {
      std::cout << "Failed to save result image." << std::endl;
    }
  }

  return 0;
}