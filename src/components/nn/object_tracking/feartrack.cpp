#include "feartrack.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
inline void parse_score_data(T* p_score_ptr, int score_size, float qscale,
                             float* max_score, int* max_i, int* max_j) {
  float max_avg_score = -1;
  int best_i = -1, best_j = -1;

  // 遍历score map (16x16)
  for (int i = 2; i < score_size - 2; i++) {
    for (int j = 2; j < score_size - 2; j++) {
      float center_score =
          static_cast<float>(p_score_ptr[i * score_size + j]) * qscale;
      if (center_score > 0.2f) {
        // 计算5x5区域的平均分数
        float avg_score = 0;
        for (int di = 0; di < 5; di++) {
          for (int dj = 0; dj < 5; dj++) {
            avg_score +=
                static_cast<float>(
                    p_score_ptr[(i - 2 + di) * score_size + (j - 2 + dj)]) *
                qscale;
          }
        }
        avg_score /= 25.0f;

        if (avg_score > max_avg_score) {
          max_avg_score = avg_score;
          best_i = i;
          best_j = j;
        }
      }
    }
  }

  *max_score = max_avg_score;
  *max_i = best_i;
  *max_j = best_j;
}

template <typename T>
inline void parse_regression_data(T* p_reg_ptr, int score_size, int i, int j,
                                  float qscale, float* x1, float* y1, float* x2,
                                  float* y2,
                                  const std::vector<std::vector<int>>& grid_x,
                                  const std::vector<std::vector<int>>& grid_y) {
  // 直接从regression map中获取四个通道的值
  float reg_x1 =
      static_cast<float>(
          p_reg_ptr[0 * score_size * score_size + i * score_size + j]) *
      qscale;
  float reg_y1 =
      static_cast<float>(
          p_reg_ptr[1 * score_size * score_size + i * score_size + j]) *
      qscale;
  float reg_x2 =
      static_cast<float>(
          p_reg_ptr[2 * score_size * score_size + i * score_size + j]) *
      qscale;
  float reg_y2 =
      static_cast<float>(
          p_reg_ptr[3 * score_size * score_size + i * score_size + j]) *
      qscale;

  *x1 = grid_x[i][j] - std::exp(reg_x1);
  *y1 = grid_y[i][j] - std::exp(reg_y1);
  *x2 = grid_x[i][j] + std::exp(reg_x2);
  *y2 = grid_y[i][j] + std::exp(reg_y2);
}

FearTrack::FearTrack() {
  net_param_.model_config.mean = {123.675, 116.28, 103.53};
  net_param_.model_config.std = {58.395, 57.12, 57.375};
  net_param_.model_config.rgb_order = "rgb";
  makeGrid();
}

FearTrack::~FearTrack() {}

int32_t FearTrack::onModelOpened() {
  // 获取输入输出层信息
  const auto& input_layers = net_->getInputNames();
  const auto& output_layers = net_->getOutputNames();

  if (input_layers.size() != 2 || output_layers.size() != 2) {
    LOGE("模型输入输出层数量不符合预期，输入层：%zu，输出层：%zu",
         input_layers.size(), output_layers.size());
    return -1;
  }
  return 0;
}

void FearTrack::makeGrid() {
  // 初始化网格坐标数组
  grid_x_.resize(score_size_, std::vector<int>(score_size_));
  grid_y_.resize(score_size_, std::vector<int>(score_size_));

  // 生成网格坐标
  for (int y = 0; y < score_size_; ++y) {
    for (int x = 0; x < score_size_; ++x) {
      grid_x_[y][x] = x * total_stride_;
      grid_y_[y][x] = y * total_stride_;
    }
  }
}

int32_t FearTrack::outputParse(
    const std::vector<std::shared_ptr<BaseImage>>& images,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) {
  return 0;
}

int32_t FearTrack::outputParse(
    const std::vector<std::vector<std::shared_ptr<BaseImage>>>& images,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) {
  // 获取回归和分类输出层名称
  std::string regression_output_name = net_->getOutputNames()[0];
  std::string score_output_name = net_->getOutputNames()[1];

  // 获取回归和分类输出张量
  std::shared_ptr<BaseTensor> regression_tensor =
      net_->getOutputTensor(regression_output_name);
  std::shared_ptr<BaseTensor> score_tensor =
      net_->getOutputTensor(score_output_name);

  // 获取张量信息
  TensorInfo regression_info = net_->getTensorInfo(regression_output_name);
  TensorInfo score_info = net_->getTensorInfo(score_output_name);

  // 遍历批次
  for (uint32_t b = 0; b < images.size(); b++) {
    uint32_t search_image_width = images[b][1]->getWidth();
    uint32_t search_image_height = images[b][1]->getHeight();

    // 创建输出结构
    std::shared_ptr<ModelBoxInfo> track_result =
        std::make_shared<ModelBoxInfo>();
    track_result->image_width = search_image_width;
    track_result->image_height = search_image_height;

    // 找到最高得分位置
    float max_score = -1;
    int max_i = -1, max_j = -1;

    // 根据数据类型处理score数据
    if (score_info.data_type == TDLDataType::INT8) {
      parse_score_data<int8_t>(score_tensor->getBatchPtr<int8_t>(b),
                               score_size_, score_info.qscale, &max_score,
                               &max_i, &max_j);
    } else if (score_info.data_type == TDLDataType::UINT8) {
      parse_score_data<uint8_t>(score_tensor->getBatchPtr<uint8_t>(b),
                                score_size_, score_info.qscale, &max_score,
                                &max_i, &max_j);
    } else if (score_info.data_type == TDLDataType::FP32) {
      parse_score_data<float>(score_tensor->getBatchPtr<float>(b), score_size_,
                              1.0f, &max_score, &max_i, &max_j);
    } else {
      LOGE("不支持的数据类型:%d\n", static_cast<int>(score_info.data_type));
      return -1;
    }

    if (max_i >= 0 && max_j >= 0) {
      // 解析边界框
      float x1, y1, x2, y2;

      // 根据数据类型处理regression数据
      if (regression_info.data_type == TDLDataType::INT8) {
        parse_regression_data<int8_t>(regression_tensor->getBatchPtr<int8_t>(b),
                                      score_size_, max_i, max_j,
                                      regression_info.qscale, &x1, &y1, &x2,
                                      &y2, grid_x_, grid_y_);
      } else if (regression_info.data_type == TDLDataType::UINT8) {
        parse_regression_data<uint8_t>(
            regression_tensor->getBatchPtr<uint8_t>(b), score_size_, max_i,
            max_j, regression_info.qscale, &x1, &y1, &x2, &y2, grid_x_,
            grid_y_);
      } else if (regression_info.data_type == TDLDataType::FP32) {
        parse_regression_data<float>(regression_tensor->getBatchPtr<float>(b),
                                     score_size_, max_i, max_j, 1.0f, &x1, &y1,
                                     &x2, &y2, grid_x_, grid_y_);
      } else {
        LOGE("不支持的数据类型:%d\n",
             static_cast<int>(regression_info.data_type));
        return -1;
      }
      ObjectBoxInfo bbox;
      bbox.score = max_score;
      bbox.x1 = x1;
      bbox.y1 = y1;
      bbox.x2 = x2;
      bbox.y2 = y2;

      // 添加到结果中
      track_result->bboxes.push_back(bbox);

      LOGI("跟踪结果：[%f, %f, %f, %f]，得分：%f", bbox.x1, bbox.y1, bbox.x2,
           bbox.y2, bbox.score);
    } else {
      LOGI("未找到有效的跟踪目标");
    }

    out_datas.push_back(track_result);
  }

  return 0;
}
