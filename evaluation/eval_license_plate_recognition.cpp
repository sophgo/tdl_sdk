#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

void save_all_results(
    std::ofstream& output_file, const std::string& img_path,
    const std::string& plate_text,
    const std::shared_ptr<ModelLandmarksInfo>& landmarks_meta) {
  if (output_file.is_open()) {
    output_file << plate_text << "\n";
  }
}

std::vector<std::shared_ptr<ModelOutputInfo>>
extract_crop_license_plate_landmark(
    std::shared_ptr<BaseModel> model_hk,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<BaseImage>>& license_plate_crops,
    std::vector<std::shared_ptr<ModelOutputInfo>>& license_plate_metas) {
  std::shared_ptr<BasePreprocessor> preprocessor = model_hk->getPreprocessor();

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  for (size_t i = 0; i < images.size(); i++) {
    std::shared_ptr<ModelBoxInfo> license_plate_meta =
        std::static_pointer_cast<ModelBoxInfo>(license_plate_metas[i]);
    for (size_t j = 0; j < license_plate_meta->bboxes.size() && j < 1; j++) {
      int x1 = license_plate_meta->bboxes[j].x1;
      int y1 = license_plate_meta->bboxes[j].y1;
      int x2 = license_plate_meta->bboxes[j].x2;
      int y2 = license_plate_meta->bboxes[j].y2;

      int width = x2 - x1;
      int height = y2 - y1;

      float expansion_factor = 1.25f;
      int new_width = static_cast<int>(width * expansion_factor);
      int new_height = static_cast<int>(height * expansion_factor);

      int crop_x1 = x1 - (new_width - width) / 2;
      int crop_y1 = y1 - (new_height - height) / 2;
      int crop_x2 = crop_x1 + new_width;
      int crop_y2 = crop_y1 + new_height;

      std::shared_ptr<BaseImage> license_plate_crop = preprocessor->crop(
          images[i], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

      license_plate_crops.push_back(license_plate_crop);
    }
  }
  model_hk->inference(license_plate_crops, out_datas);

  return out_datas;
}

std::vector<std::shared_ptr<ModelOutputInfo>> license_plate_recognition(
    std::shared_ptr<BaseModel> model_hr,
    std::vector<std::shared_ptr<BaseImage>>& license_plate_crops,
    std::vector<std::shared_ptr<BaseImage>>& license_plate_aligns,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_hk) {
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;

  for (size_t i = 0; i < out_hk.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);

    float landmarks[8];

    for (int k = 0; k < 4; k++) {
      landmarks[2 * k] = landmarks_meta->landmarks_x[k];
      landmarks[2 * k + 1] = landmarks_meta->landmarks_y[k];
    }
    std::shared_ptr<BaseImage> license_plate_align =
        ImageFactory::alignLicensePlate(license_plate_crops[i], landmarks,
                                        nullptr, 4, nullptr);

    license_plate_aligns.push_back(license_plate_align);
  }

  model_hr->inference(license_plate_aligns, out_datas);

  return out_datas;
}

// 批量处理所有图片
void process_all_images(const std::string& image_list,
                        const std::string& image_dir,
                        const std::string& result_file_path,
                        std::shared_ptr<BaseModel> model_hd,
                        std::shared_ptr<BaseModel> model_hk,
                        std::shared_ptr<BaseModel> model_hr) {
  if (model_hd == nullptr) {
    std::cerr << "模型未初始化成功，请检查模型文件路径和参数设置！\n";
  }
  std::cout << "开始处理所有图片...\n";
  std::ofstream output_file(result_file_path);
  if (!output_file.is_open()) {
    std::cerr << "无法打开结果文件: " << result_file_path << "\n";
    return;
  }

  std::ifstream bench_fstream(image_list);
  if (!bench_fstream.is_open()) {
    std::cerr << "打开 benchmark 文件失败: " << image_list << "\n";
    output_file.close();
    return;
  }

  std::string img_rel_path;
  while (bench_fstream >> img_rel_path) {
    const std::string img_path = image_dir + "/" + img_rel_path;
    std::cout << "处理图片: " << img_path << "\n";

    std::shared_ptr<BaseImage> image = ImageFactory::readImage(img_path);
    if (!image) {
      std::cerr << "读取图像失败: " << img_path << "\n";
      continue;
    }

    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    std::vector<std::shared_ptr<ModelOutputInfo>> out_hd;
    model_hd->inference(input_images, out_hd);

    std::vector<std::shared_ptr<BaseImage>> license_plate_crops;
    std::vector<std::shared_ptr<ModelOutputInfo>> out_hk =
        extract_crop_license_plate_landmark(model_hk, input_images,
                                            license_plate_crops, out_hd);

    std::vector<std::shared_ptr<BaseImage>> license_plate_aligns;
    std::vector<std::shared_ptr<ModelOutputInfo>> out_hr =
        license_plate_recognition(model_hr, license_plate_crops,
                                  license_plate_aligns, out_hk);

    for (size_t i = 0; i < out_hk.size(); i++) {
      std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
          std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);
      std::shared_ptr<ModelOcrInfo> text_meta =
          std::static_pointer_cast<ModelOcrInfo>(out_hr[i]);

      save_all_results(output_file, img_path, text_meta->text_info,
                       landmarks_meta);
    }

    input_images.clear();
    out_hd.clear();
    license_plate_crops.clear();
    out_hk.clear();
    license_plate_aligns.clear();
    out_hr.clear();
  }

  output_file.close();
  bench_fstream.close();
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf(
        "\n用法: %s MODEL_DIR IMAGE_DIR IMAGE_LIST  RESULT_FILE\n\n"
        "\tMODEL_DIR, 模型文件或目录路径\n"
        "\tIMAGE_LIST, 存储图片相对路径的txt文件\n"
        "\tIMAGE_DIR, 图片根目录\n"
        "\tRESULT_FILE, "
        "最终结果保存的txt文件路径（如：./all_license_plates.txt）\n",
        argv[0]);
    return -1;
  }

  std::string model_dir = argv[1];
  std::string image_list = argv[2];
  std::string image_dir = argv[3];
  std::string result_file_path = argv[4];  // 结果文件路径

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_hd =
      model_factory.getModel(ModelType::YOLOV8N_DET_LICENSE_PLATE);
  if (!model_hd) {
    printf("创建检测模型失败\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hk =
      model_factory.getModel(ModelType::KEYPOINT_LICENSE_PLATE);
  if (!model_hk) {
    printf("创建关键点模型失败\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hr =
      model_factory.getModel(ModelType::RECOGNITION_LICENSE_PLATE);
  if (!model_hr) {
    printf("创建识别模型失败\n");
    return -1;
  }

  std::cout << "模型加载成功，开始处理图片..." << std::endl;
  process_all_images(image_list, image_dir, result_file_path, model_hd,
                     model_hk, model_hr);

  std::cout << "所有图片处理完成，结果已保存至: " << result_file_path
            << std::endl;

  // 释放模型资源
  model_hd.reset();
  model_hk.reset();
  model_hr.reset();

  return 0;
}