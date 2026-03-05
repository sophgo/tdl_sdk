#include <fstream>  // 新增：用于文件输出
#include <iomanip>  // 新增：用于控制输出精度
#include "tdl_model_factory.hpp"

// 保存深度数据到 TXT 文件
void save_depth_to_txt(std::shared_ptr<ModelDepthInfo> depth_meta,
                       const std::string &txt_path) {
  std::ofstream outfile(txt_path);
  if (!outfile.is_open()) {
    printf("Failed to open %s for writing\n", txt_path.c_str());
    return;
  }

  // 设置输出精度，保留4位小数
  outfile << std::fixed << std::setprecision(4);

  uint32_t width = depth_meta->w;
  uint32_t height = depth_meta->h;
  float *logits = depth_meta->logits;

  // 写入数据：按行遍历
  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j++) {
      // logits 是一维数组，需计算线性索引
      float val = logits[i * width + j];
      outfile << val;

      // 列间用空格分隔，最后一列不加空格
      if (j < width - 1) {
        outfile << " ";
      }
    }
    // 每行结束换行
    outfile << "\n";
  }

  outfile.close();
  printf("Depth data saved to %s (Size: %dx%d)\n", txt_path.c_str(), width,
         height);
}

void visualize_depth(std::shared_ptr<ModelDepthInfo> depth_meta,
                     const std::string &str_img_name) {
  uint32_t width = depth_meta->w;
  uint32_t height = depth_meta->h;

  // 1. 创建 float 类型的 Mat，包装原始数据
  cv::Mat depth_map(height, width, CV_32FC1, depth_meta->logits);

  // 2. 归一化到 0-255
  cv::Mat depth_map_vis;
  cv::normalize(depth_map, depth_map_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  // 3. 保存图像
  cv::imwrite(str_img_name, depth_map_vis);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_dir> <left_image_path> <right_image_path>\n",
           argv[0]);
    return -1;
  }
  std::string model_dir = argv[1];
  std::string left_image_path = argv[2];
  std::string right_image_path = argv[3];

  std::shared_ptr<BaseImage> left_image =
      ImageFactory::readImage(left_image_path);
  if (!left_image) {
    printf("Failed to create left image\n");
    return -1;
  }

  std::shared_ptr<BaseImage> right_image =
      ImageFactory::readImage(right_image_path);
  if (!right_image) {
    printf("Failed to create right image\n");
    return -1;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_stereo =
      model_factory.getModel(ModelType::DEPTH_ESTIMATION_STEREO);
  if (!model_stereo) {
    printf("Failed to create model_stereo\n");
    return -1;
  }

  std::vector<std::vector<std::shared_ptr<BaseImage>>> input_images;
  std::vector<std::shared_ptr<BaseImage>> stereo_pair = {left_image,
                                                         right_image};
  input_images.push_back(stereo_pair);

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  model_stereo->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelDepthInfo> depth_meta =
        std::static_pointer_cast<ModelDepthInfo>(out_datas[i]);

    printf("Depth map size: %d x %d\n", depth_meta->w, depth_meta->h);

    visualize_depth(depth_meta, "stereo_depth.png");
    // 2. 保存原始数据到 TXT
    save_depth_to_txt(depth_meta, "stereo_depth_data.txt");
    int pix_size = depth_meta->w * depth_meta->h;
    printf("First 10 depth values:\n");
    for (int i = 0; i < std::min(10, pix_size); i++) {
      printf("%.2f ", depth_meta->logits[i]);
    }
    printf("\n");
  }

  return 0;
}
