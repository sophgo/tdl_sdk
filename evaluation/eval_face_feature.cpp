#include <sys/stat.h>
#include <algorithm>  // 用于trim函数
#include <fstream>
#include <iomanip>  // 用于格式化输出
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "evaluator.hpp"
#include "model/base_model.hpp"
#include "tdl_model_factory.hpp"

template <typename T>
std::string embeddingToString(void *embedding, size_t num) {
  T *feature = reinterpret_cast<T *>(embedding);
  std::string result;
  for (size_t i = 0; i < num; ++i) {
    result += std::to_string(feature[i]);
    if (i < num - 1) {
      result += " ";
    }
  }
  return result;
}

class FaceFeatureEvaluator : public Evaluator {
 public:
  FaceFeatureEvaluator(std::shared_ptr<BaseModel> feature_model);
  ~FaceFeatureEvaluator();

  void evaluate(const std::vector<std::string> &eval_files,
                const std::string &output_dir);

  // 重写基类的packOutput方法，处理特征信息
  std::string packOutput(std::shared_ptr<ModelOutputInfo> model_output);

  // 添加特征信息的格式化方法
  std::string formatFeatureOutput(
      const std::string &image_path,
      std::shared_ptr<ModelFeatureInfo> feature_info);

 private:
  void process_label_file(const std::string &label_file,
                          const std::string &output_file);
  // 去除字符串右侧的空白字符
  std::string rtrim(const std::string &s);
  // 显示进度条
  void showProgressBar(int current, int total, int bar_width = 50);
  std::shared_ptr<BaseModel> feature_model_;
};

FaceFeatureEvaluator::FaceFeatureEvaluator(
    std::shared_ptr<BaseModel> feature_model)
    : feature_model_(feature_model) {}

FaceFeatureEvaluator::~FaceFeatureEvaluator() {}

std::string FaceFeatureEvaluator::packOutput(

    std::shared_ptr<ModelOutputInfo> model_output) {
  if (model_output->getType() == ModelOutputType::FEATURE_EMBEDDING) {
    auto feature_info =
        std::static_pointer_cast<ModelFeatureInfo>(model_output);
    std::string result;

    switch (feature_info->embedding_type) {
      case TDLDataType::INT8:
        return embeddingToString<int8_t>(feature_info->embedding,
                                         feature_info->embedding_num);
      case TDLDataType::UINT8:
        return embeddingToString<uint8_t>(feature_info->embedding,
                                          feature_info->embedding_num);
      case TDLDataType::FP32:
        return embeddingToString<float>(feature_info->embedding,
                                        feature_info->embedding_num);
      default:
        assert(false && "Unsupported embedding_type");
        return "";
    }

  } else {
    std::cout << "model_output->getType() is not FEATURE_EMBEDDING"
              << std::endl;
    assert(false);
    return "";
  }
}

std::string FaceFeatureEvaluator::formatFeatureOutput(
    const std::string &image_path,
    std::shared_ptr<ModelFeatureInfo> feature_info) {
  return image_path + "#" + packOutput(feature_info) + "\n";
}

void FaceFeatureEvaluator::evaluate(const std::vector<std::string> &eval_files,
                                    const std::string &output_dir) {
  if (eval_files.empty()) {
    printf("No evaluation files provided.\n");
    return;
  }

  for (const auto &label_file : eval_files) {
    process_label_file(label_file, output_dir);
  }
}

// 去除字符串右侧的空白字符
std::string FaceFeatureEvaluator::rtrim(const std::string &s) {
  std::string result = s;
  result.erase(std::find_if(result.rbegin(), result.rend(),
                            [](unsigned char ch) { return !std::isspace(ch); })
                   .base(),
               result.end());
  return result;
}

// 显示进度条
void FaceFeatureEvaluator::showProgressBar(int current, int total,
                                           int bar_width) {
  float progress = float(current) / total;
  int pos = bar_width * progress;

  std::cout << "[";
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total
            << ")\r";
  std::cout.flush();

  // 当进度完成时，添加换行
  if (current == total) {
    std::cout << std::endl;
  }
}

void FaceFeatureEvaluator::process_label_file(const std::string &label_file,
                                              const std::string &output_file) {
  printf("Processing label file: %s\n", label_file.c_str());

  std::ifstream label_stream(label_file);
  if (!label_stream.is_open()) {
    printf("Failed to open label file: %s\n", label_file.c_str());
    return;
  }

  // 获取label文件所在的基础路径
  std::string base_path =
      label_file.substr(0, label_file.find_last_of("/\\") + 1);
  if (base_path.empty()) {
    base_path = "./";
  }

  // 计算总行数用于进度条显示
  int total_lines = 0;
  std::string temp_line;
  while (std::getline(label_stream, temp_line)) {
    if (!rtrim(temp_line).empty()) {
      total_lines++;
    }
  }
  label_stream.clear();                  // 清除EOF标志
  label_stream.seekg(0, std::ios::beg);  // 回到文件开头

  printf("Total images to process: %d\n", total_lines);

  // 准备输出文件
  std::ofstream output_stream(output_file);
  if (!output_stream.is_open()) {
    printf("Failed to open output file: %s\n", output_file.c_str());
    label_stream.close();
    return;
  }

  std::string line;
  int processed_count = 0;
  std::string result_content;

  while (std::getline(label_stream, line)) {
    // 去除行尾空格
    line = rtrim(line);
    if (line.empty()) {
      continue;
    }

    // 构建完整图片路径
    std::string image_path = base_path + line;

    // 读取图片
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    if (!image) {
      printf("Failed to read image: %s\n", image_path.c_str());
      continue;
    }

    // 提取特征
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    std::vector<std::shared_ptr<ModelOutputInfo>> feature_outputs;

    feature_model_->inference(input_images, feature_outputs);

    // 使用formatFeatureOutput格式化输出
    std::shared_ptr<ModelFeatureInfo> feature_info =
        std::static_pointer_cast<ModelFeatureInfo>(feature_outputs[0]);

    std::string line_result = formatFeatureOutput(line, feature_info);

    // 添加到结果内容
    result_content += line_result;
    processed_count++;

    // 显示进度条
    showProgressBar(processed_count, total_lines);
  }

  writeResult(output_file, result_content);

  label_stream.close();

  printf(
      "\nFeature extraction completed. Processed %d images. Results saved to "
      "%s\n",
      processed_count, output_file.c_str());
}

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <model_id_name> <model_dir> <label_file> <output_file>\n",
           argv[0]);
    printf(
        "surpported_model_list:\nFEATURE_CVIFACE,FEATURE_BMFACE_R34,"
        "FEATURE_BMFACE_R50");
    return -1;
  }

  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string label_file = argv[3];
  std::string output_file = argv[4];

  // 创建模型工厂并获取特征提取模型

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> feature_model;
  struct stat path_stat;
  if (stat(model_dir.c_str(), &path_stat) == 0) {
    if (S_ISDIR(path_stat.st_mode)) {  // model_dir是文件夹：原有getModel调用
      model_factory.setModelDir(model_dir);
      feature_model = model_factory.getModel(model_id_name);
    } else if (S_ISREG(
                   path_stat.st_mode)) {  // model_dir是绝对路径：新getModel调用
      feature_model = model_factory.getModel(model_id_name, model_dir);
    } else {
      printf("Error: MODEL_DIR is neither dir nor file\n");
      return -1;
    }
  } else {
    printf("Error: Cannot access MODEL_DIR: %s\n", model_dir.c_str());
    return -1;
  }

  if (!feature_model) {
    printf("Failed to create feature extraction model\n");
    return -1;
  }

  // 创建评估器并运行评估
  FaceFeatureEvaluator evaluator(feature_model);
  std::vector<std::string> eval_files = {label_file};
  evaluator.evaluate(eval_files, output_file);

  return 0;
}