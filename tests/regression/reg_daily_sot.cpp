#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class SotTestSuite : public CVI_TDLModelTestSuite {
 public:
  SotTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~SotTestSuite() = default;

  std::shared_ptr<BaseModel> sot_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id = std::string(m_json_object["model_id"]);
    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();

    sot_ = TDLModelFactory::getInstance().getModel(
        model_id, model_path);  // One model id may correspond to multiple
                                // models with different sizes
    ASSERT_NE(sot_, nullptr);
    sot_->setModelThreshold(m_json_object["model_score_threshold"]);
  }

  nlohmann::ordered_json convertSotResult(const ObjectBoxInfo &box_info) {
    nlohmann::ordered_json result;  // is a list,contains bbox,conf,class_id
    nlohmann::ordered_json item;
    item["bbox"] = {box_info.x1, box_info.y1, box_info.x2, box_info.y2};
    item["score"] = box_info.score;
    item["class_id"] = box_info.class_id;
    result.push_back(item);
    return result;
  }

  virtual void TearDown() {}
};

TEST_F(SotTestSuite, accuracy) {
  const float reg_nms_threshold = m_json_object["reg_nms_threshold"];
  const float reg_score_diff_threshold =
      m_json_object["reg_score_diff_threshold"];

  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  TestFlag test_flag = CVI_TDLTestContext::getInstance().getTestFlag();
  nlohmann::ordered_json results;
  LOGIP("test_flag: %d", static_cast<int>(test_flag));
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    LOGIP("checkToGetProcessResult failed");
    return;
  }
  size_t sample_num = results.size();
  LOGIP("regression sample num: %d", sample_num);

  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_SOT);
  tracker->setModel(sot_);
  std::shared_ptr<BaseImage> image;
  std::vector<std::vector<float>> gt_dets;
  std::vector<std::vector<float>> pred_dets;
  int frame_id = 0;
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    gt_dets.clear();
    pred_dets.clear();
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    image = ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);
    ASSERT_NE(image, nullptr);
    if (frame_id == 0) {
      ObjectBoxInfo init_bbox;
      init_bbox.x1 = 387.0;
      init_bbox.y1 = 328.0;
      init_bbox.x2 = 556.0;
      init_bbox.y2 = 655.0;
      init_bbox.score = 1.0;
      init_bbox.class_id = 0;
      tracker->initialize(image, {}, init_bbox, 0);
    } else {
      TrackerInfo tracker_info;
      tracker->track(image, frame_id, tracker_info);

      if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
        LOGI("generate function res,image_path: %s\n", image_path.c_str());
        nlohmann::ordered_json result =
            convertSotResult(tracker_info.box_info_);
        iter.value() = result;
        continue;
      }
      nlohmann::ordered_json expected_det = iter.value()[0];
      gt_dets.push_back({expected_det["bbox"][0], expected_det["bbox"][1],
                         expected_det["bbox"][2], expected_det["bbox"][3],
                         expected_det["score"], expected_det["class_id"]});
      pred_dets.push_back({tracker_info.box_info_.x1, tracker_info.box_info_.y1,
                           tracker_info.box_info_.x2, tracker_info.box_info_.y2,
                           tracker_info.box_info_.score,
                           float(tracker_info.box_info_.class_id)});
      EXPECT_TRUE(matchObjects(gt_dets, pred_dets, reg_nms_threshold,
                               reg_score_diff_threshold));
    }
    frame_id++;
  }

  if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}

typedef struct {
  TrackerInfo tracker_info;
  std::shared_ptr<BaseImage> image;
  int epochs;
  std::shared_ptr<Tracker> tracker;
  int frame_id;
} RUN_TDL_THREAD_CPU_ARGS_SOT;
void *run_tdl_thread_cpu_sot(void *args) {
  RUN_TDL_THREAD_CPU_ARGS_SOT *pstArgs = (RUN_TDL_THREAD_CPU_ARGS_SOT *)args;
  double fps_period = 66.67;
  prctl(PR_SET_NAME, "inference_load", 0, 0, 0);
  for (int e = 0; e < pstArgs->epochs; ++e) {
    auto t0 = std::chrono::steady_clock::now();
    pstArgs->tracker->track(pstArgs->image, pstArgs->frame_id,
                            pstArgs->tracker_info);
    auto t1 = std::chrono::steady_clock::now();
    double time_consume =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            t1 - t0)
            .count();
    if (time_consume <= fps_period) {
      int sleep_time = fps_period - static_cast<int>(time_consume);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
  }
  return nullptr;
}

TEST_F(SotTestSuite, performance) {
  const int epochs = 105;
  const auto sample_period = std::chrono::milliseconds(100);
  const auto sample_period_cpu = std::chrono::milliseconds(800);
  const auto enable_period = std::chrono::milliseconds(200);
  std::string model_path = m_model_dir.string() + "/" + gen_model_dir() + "/" +
                           m_json_object["model_name"].get<std::string>() +
                           gen_model_suffix();

  float infer_time_ms;
  if (get_model_info(model_path, infer_time_ms) != 0) {
    LOGE("get model info failed");
    return;
  }

  const float reg_nms_threshold = m_json_object["reg_nms_threshold"];
  const float reg_score_diff_threshold =
      m_json_object["reg_score_diff_threshold"];

  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  TestFlag test_flag = CVI_TDLTestContext::getInstance().getTestFlag();
  nlohmann::ordered_json results;
  LOGIP("test_flag: %d", static_cast<int>(test_flag));
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    LOGIP("checkToGetProcessResult failed");
    return;
  }

  size_t sample_num = results.size();
  LOGIP("regression sample num: %d", sample_num);

  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_SOT);
  tracker->setModel(sot_);
  std::shared_ptr<BaseImage> image;
  std::vector<std::vector<float>> gt_dets;
  std::vector<std::vector<float>> pred_dets;
  auto iter = results.begin();
  gt_dets.clear();
  pred_dets.clear();
  std::string image_path =
      (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
  image = ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);
  ASSERT_NE(image, nullptr);
  if (!image) {
    LOGE("performance: failed to read image %s", image_path.c_str());
  }

  std::vector<double> tpu_samples;
  std::mutex tpu_mu;
  std::vector<double> cpu_samples;
  std::mutex cpu_mu;
  std::atomic<bool> sampling{false};
  pid_t tid;

  std::string tpu_usage_path;
  if (!confirm_path(tpu_usage_path)) {
    std::cerr << "Failed to confirm TPU usage path: " << tpu_usage_path
              << std::endl;
    tpu_usage_path.clear();
  }
  auto tpu_sampler = [&]() {
    enable_tpu_usage(tpu_usage_path);
    while (!sampling.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    while (sampling.load()) {
      read_tpu_usage(tpu_usage_path, tpu_samples, tpu_mu);
      std::this_thread::sleep_for(sample_period);
    }
  };
  auto cpu_sampler = [&]() {
    while (!sampling.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    while (sampling.load()) {
      std::this_thread::sleep_for(sample_period_cpu);
      read_cpu_line(cpu_samples, cpu_mu);
    }
  };

  TrackerInfo tracker_info;
  ObjectBoxInfo init_bbox;
  int frame_id = 0;
  init_bbox.x1 = 387.0;
  init_bbox.y1 = 328.0;
  init_bbox.x2 = 556.0;
  init_bbox.y2 = 655.0;
  init_bbox.score = 1.0;
  init_bbox.class_id = 0;
  tracker->initialize(image, {}, init_bbox, 0);

  std::thread th_cpu(cpu_sampler);
  RUN_TDL_THREAD_CPU_ARGS_SOT tdl_args = {.tracker_info = tracker_info,
                                          .image = image,
                                          .epochs = epochs,
                                          .tracker = tracker};
  {
    ScopedSampler_cpu sampler(sampling, th_cpu);
    pthread_t inferenceTDLThread;
    pthread_create(&inferenceTDLThread, nullptr, run_tdl_thread_cpu_sot,
                   &tdl_args);
    pid_t tid = get_tid_by_name("inference_load");
    sampling = true;
    pthread_join(inferenceTDLThread, NULL);
  }

  std::thread th_tpu(tpu_sampler);
  std::this_thread::sleep_for(enable_period);
  int count = (int)(3000.0f / infer_time_ms);

  RUN_TDL_THREAD_TPU_ARG_S tdl_args_tpu = {.model_path = model_path,
                                           .count = count};
  {
    ScopedSampler_tpu sampler(sampling, th_tpu);
    pthread_t stTDLThread;
    pthread_create(&stTDLThread, nullptr, run_tdl_thread_tpu, &tdl_args_tpu);
    pthread_join(stTDLThread, NULL);
  }
  load_show(cpu_samples, tpu_samples);
}  // end of TEST_F
}  // namespace unitest
}  // namespace cvitdl
