#include <sys/stat.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image/base_image.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
#include "utils/tokenizer_bpe.hpp"

static std::vector<std::string> read_non_empty_lines(
    const std::string& file_path) {
  std::vector<std::string> lines;
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    printf("Failed to open file: %s\n", file_path.c_str());
    return lines;
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  infile.close();
  return lines;
}

// 检查模型ID是否在允许的列表中
int is_valid_model_id(const char* id, const char* const valid_ids[],
                      int count) {
  for (int i = 0; i < count; i++) {
    if (strcmp(id, valid_ids[i]) == 0) {
      return 1;  // 有效ID
    }
  }
  return 0;  // 无效ID
}

bool file_exists(const std::string& path) {
  struct stat buf;
  return (stat(path.c_str(), &buf) == 0 && S_ISREG(buf.st_mode));
}
// 打印可用的模型ID
void print_valid_ids(const char* title, const char* const valid_ids[],
                     int count) {
  printf("%s 可用选项: ", title);
  for (int i = 0; i < count; i++) {
    printf("%s", valid_ids[i]);
    if (i < count - 1) {
      printf(", ");
    }
  }
  printf("\n");
}

static int init_video_writer(cv::VideoWriter& writer,
                             const std::string& output_path, double fps,
                             int width, int height) {
  if (fps <= 0.0) {
    fps = 25.0;
  }
  cv::Size size(width, height);

  // 优先使用 mp4v，失败则尝试其它编码
  const int fourcc_candidates[] = {
      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
      cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
  };
  for (int i = 0; i < (int)(sizeof(fourcc_candidates) / sizeof(int)); i++) {
    writer.open(output_path, fourcc_candidates[i], fps, size, true);
    if (writer.isOpened()) {
      return 0;
    }
  }
  return -1;
}

static void draw_label(cv::Mat& frame, const std::string& text) {
  const int x = 20;
  const int y = 40;
  const double font_scale = 0.9;
  const int thickness = 2;
  const int font_face = cv::FONT_HERSHEY_SIMPLEX;
  int baseline = 0;
  cv::Size text_size =
      cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

  // 背景矩形（黑底白字）
  cv::Rect bg(x - 10, y - text_size.height - 10, text_size.width + 20,
              text_size.height + baseline + 20);
  bg &= cv::Rect(0, 0, frame.cols, frame.rows);
  cv::rectangle(frame, bg, cv::Scalar(0, 0, 0), cv::FILLED);

  cv::putText(frame, text, cv::Point(x, y), font_face, font_scale,
              cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

int main(int argc, char** argv) {
  // 定义支持的图像模型ID
  const char* valid_img_models[] = {"FEATURE_CLIP_IMG",
                                    "FEATURE_MOBILECLIP2_IMG"};
  const int img_model_count =
      sizeof(valid_img_models) / sizeof(valid_img_models[0]);

  // 定义支持的文本模型ID
  const char* valid_text_models[] = {
      "FEATURE_CLIP_TEXT",
      "FEATURE_MOBILECLIP2_TEXT",
  };
  const int text_model_count =
      sizeof(valid_text_models) / sizeof(valid_text_models[0]);

  // 6 参数：model_dir 模式
  //   eval_clip_pipeline <model_dir> <img_model_id> <text_model_id>
  //                     <video_path.mp4> <text_dir> <output_video.mp4>
  //
  // 7 参数：绝对路径模式
  //   eval_clip_pipeline <img_model_id> <text_model_id>
  //                     <img_model_path> <text_model_path>
  //                     <video_path.mp4> <text_dir> <output_video.mp4>
  bool is_abspath_mode = (argc == 8);
  if (!(argc == 7 || argc == 8)) {
    printf("Usage (7参数 / model_dir 模式):\n");
    printf(
        "  %s <model_dir> <img_model_id> <text_model_id> "
        "<video_path.mp4> <text_dir> <output_video.mp4>\n",
        argv[0]);
    printf("Usage (8参数 / 绝对路径模式):\n");
    printf(
        "  %s <img_model_id> <text_model_id> "
        "<img_model_path> <text_model_path> "
        "<video_path.mp4> <text_dir> <output_video.mp4>\n",
        argv[0]);
    return -1;
  }

  std::string img_model_id_name, text_model_id_name;
  std::string model_dir, img_model_path, text_model_path;
  std::string video_path, txt_dir, output_video_path;

  if (!is_abspath_mode) {
    model_dir = argv[1];
    img_model_id_name = argv[2];
    text_model_id_name = argv[3];
    video_path = argv[4];
    txt_dir = argv[5];
    output_video_path = argv[6];
  } else {
    img_model_id_name = argv[1];
    text_model_id_name = argv[2];
    img_model_path = argv[3];
    text_model_path = argv[4];
    video_path = argv[5];
    txt_dir = argv[6];
    output_video_path = argv[7];
  }

  // 检查图像模型ID是否有效
  if (!is_valid_model_id(img_model_id_name.c_str(), valid_img_models,
                         img_model_count)) {
    printf("错误: 无效的图像模型ID: %s\n", img_model_id_name.c_str());
    print_valid_ids("可用的图像模型ID", valid_img_models, img_model_count);
    return -1;
  }

  // 检查文本模型ID是否有效
  if (!is_valid_model_id(text_model_id_name.c_str(), valid_text_models,
                         text_model_count)) {
    printf("错误: 无效的文本模型ID: %s\n", text_model_id_name.c_str());
    print_valid_ids("可用的文本模型ID", valid_text_models, text_model_count);
    return -1;
  }

  if (is_abspath_mode) {
    if (!file_exists(img_model_path)) {
      printf("错误: 图像模型文件不存在: %s\n", img_model_path.c_str());
      return -1;
    }
    if (!file_exists(text_model_path)) {
      printf("错误: 文本模型文件不存在: %s\n", text_model_path.c_str());
      return -1;
    }
  }

  if (!file_exists(video_path)) {
    printf("错误: 视频文件不存在: %s\n", video_path.c_str());
    return -1;
  }
  std::string encoder_file = txt_dir + "/encoder.txt";
  std::string bpe_file = txt_dir + "/vocab.txt";
  std::string input_file = txt_dir + "/input.txt";

  // 1. 读取 input.txt 文本行（用于可视化显示）
  std::vector<std::string> text_lines = read_non_empty_lines(input_file);
  if (text_lines.empty()) {
    printf("Warning: input.txt is empty, overlay will only show index/prob\n");
  }

  // 2. 加载模型工厂和模型（只加载一次）
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> model_clip_image;
  std::shared_ptr<BaseModel> model_clip_text;
  if (!is_abspath_mode) {
    model_factory.setModelDir(model_dir);
    model_clip_image = model_factory.getModel(img_model_id_name);
    model_clip_text = model_factory.getModel(text_model_id_name);
  } else {
    model_clip_image =
        model_factory.getModel(img_model_id_name, img_model_path);
    model_clip_text =
        model_factory.getModel(text_model_id_name, text_model_path);
  }

  if (!model_clip_image) {
    printf("Failed to load clip image model\n");
    return -1;
  }

  if (!model_clip_text) {
    printf("Failed to load clip text model\n");
    return -1;
  }

  // 3. 文本特征处理（只做一次）
  std::vector<std::vector<int32_t>> tokens;
  BytePairEncoder bpe(encoder_file, bpe_file);
  int result = bpe.tokenizerBPE(input_file, tokens);
  if (result != 0) {
    printf("Failed to tokenize text file\n");
    return -1;
  }
  if (!text_lines.empty() && tokens.size() != text_lines.size()) {
    printf(
        "Warning: tokens.size(%zu) != text_lines.size(%zu), overlay text may "
        "mismatch\n",
        tokens.size(), text_lines.size());
  }

  std::vector<std::shared_ptr<BaseImage>> input_texts;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::shared_ptr<BaseImage> text = ImageFactory::createImage(
        77, 1, ImageFormat::GRAY, TDLDataType::INT32, true);
    uint8_t* txt_buffer = text->getVirtualAddress()[0];
    memcpy(txt_buffer, tokens[i].data(), 77 * sizeof(int32_t));
    input_texts.push_back(text);
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_txt;
  model_clip_text->inference(input_texts, out_txt);

  if (out_txt.empty()) {
    printf("No text features extracted\n");
    return -1;
  }

  std::vector<std::vector<float>> text_features;
  for (size_t i = 0; i < out_txt.size(); i++) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_txt[i]);
    std::vector<float> feature_vec(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      feature_vec[j] = feature_ptr[j];
    }
    CommonUtils::normalize(feature_vec);
    text_features.push_back(feature_vec);
  }

  // 4. 读取 mp4 逐帧推理，叠字并保存视频
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    printf("Failed to open video: %s\n", video_path.c_str());
    return -1;
  }
  int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(cv::CAP_PROP_FPS);

  cv::VideoWriter writer;
  if (init_video_writer(writer, output_video_path, fps, width, height) != 0) {
    printf("Failed to open VideoWriter: %s\n", output_video_path.c_str());
    return -1;
  }

  printf("Video info: %dx%d fps=%.3f\n", width, height, fps);

  cv::Mat frame;
  uint32_t frame_id = 0;
  while (cap.read(frame)) {
    frame_id++;

    std::shared_ptr<BaseImage> image = ImageFactory::convertFromMat(frame);
    if (!image) {
      printf("convertFromMat failed at frame %u\n", frame_id);
      writer.write(frame);
      continue;
    }

    std::vector<std::shared_ptr<BaseImage>> input_image = {image};
    std::vector<std::shared_ptr<ModelOutputInfo>> out_img;
    model_clip_image->inference(input_image, out_img);

    if (out_img.empty() ||
        out_img[0]->getType() != ModelOutputType::FEATURE_EMBEDDING) {
      printf("No image features extracted at frame %u\n", frame_id);
      writer.write(frame);
      continue;
    }

    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_img[0]);
    std::vector<float> image_feature(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      image_feature[j] = feature_ptr[j];
    }
    CommonUtils::normalize(image_feature);

    // 计算相似度
    std::vector<float> logits;
    logits.reserve(text_features.size());
    for (size_t j = 0; j < text_features.size(); ++j) {
      float sim = CommonUtils::dot_product(image_feature, text_features[j]);
      logits.push_back(sim * 100.0f);
    }
    std::vector<float> probs = CommonUtils::softmax(logits);

    // 找最大概率对应的文本索引
    size_t max_idx = 0;
    float max_prob = probs.empty() ? 0.0f : probs[0];
    for (size_t j = 1; j < probs.size(); ++j) {
      if (probs[j] > max_prob) {
        max_prob = probs[j];
        max_idx = j;
      }
    }
    std::cout << "max_idx: " << max_idx;
    std::cout << "max_prob: " << max_prob << std::endl;
    std::ostringstream oss;
    if (!text_lines.empty() && max_idx < text_lines.size()) {
      oss << "top1: [" << max_idx << "] " << text_lines[max_idx]
          << "  p=" << std::fixed << std::setprecision(4) << max_prob;
    } else {
      oss << "top1_idx=" << max_idx << "  p=" << std::fixed
          << std::setprecision(4) << max_prob;
    }

    draw_label(frame, oss.str());

    writer.write(frame);

    if (frame_id % 30 == 0) {
      printf("processed frame_id=%u\n", frame_id);
    }
  }
  cap.release();
  writer.release();

  return 0;
}