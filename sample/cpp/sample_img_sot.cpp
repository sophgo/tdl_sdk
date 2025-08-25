#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <experimental/filesystem>
#include <iostream>

#include "image/base_image.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"

namespace fs = std::experimental::filesystem;

// 全局状态枚举
enum class SystemStatus { DETECTION = 0, WAITING_INPUT = 1, TRACKING = 2 };

// 全局变量
SystemStatus g_status = SystemStatus::DETECTION;  // 初始状态设置为检测
std::chrono::steady_clock::time_point g_lost_start_time;  // 目标丢失的开始时刻
bool g_lost_timer_started = false;   // 目标丢失计时器是否已启动
const int LOST_TIMEOUT_SECONDS = 5;  // 目标丢失超时时间

static inline bool ends_with_any(const std::string &name,
                                 const std::vector<std::string> &exts) {
  for (const auto &ext : exts) {
    if (name.size() >= ext.size()) {
      std::string tail = name.substr(name.size() - ext.size());
      std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);
      if (tail == ext) return true;
    }
  }
  return false;
}

static std::vector<std::string> collect_images_sorted(
    const std::string &image_dir) {
  std::vector<std::string> files;
  const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp"};

  for (auto &p : fs::directory_iterator(image_dir)) {
    if (fs::is_regular_file(p.path())) {
      auto name = p.path().filename().string();
      if (ends_with_any(name, exts)) {
        files.push_back(p.path().string());
      }
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

// 设置终端为非阻塞模式
void setNonBlockingInput() {
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
}

// 恢复终端设置
void restoreTerminal() {
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  oldt.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, 0);
}

// 检查键盘输入
char checkKeyInput() {
  char ch = 0;
  if (read(STDIN_FILENO, &ch, 1) > 0) {
    return ch;
  }
  return 0;
}

// 检测状态处理函数
void processDetectionState(std::shared_ptr<BaseModel> &det_model,
                           std::shared_ptr<BaseImage> &image,
                           uint64_t frame_id) {
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  det_model->inference(input_images, out_datas);
  std::shared_ptr<ModelBoxInfo> det_result =
      std::static_pointer_cast<ModelBoxInfo>(out_datas[0]);

  std::cout << "frame id: " << frame_id << std::endl;

  if (det_result->bboxes.empty()) {
    std::cout << "there is no bbox." << std::endl;
  } else {
    for (size_t i = 0; i < det_result->bboxes.size(); ++i) {
      const auto &bbox = det_result->bboxes[i];
      std::cout << "bbox" << (i + 1) << ": " << bbox.x1 << "," << bbox.y1 << ","
                << bbox.x2 << "," << bbox.y2 << std::endl;
    }
  }
}

// 等待输入状态处理函数
ObjectBoxInfo processWaitingInputState(std::shared_ptr<BaseModel> &det_model,
                                       std::shared_ptr<BaseImage> &image,
                                       std::shared_ptr<Tracker> &tracker) {
  std::cout << "请输入一个bbox(x1,y1,x2,y2)或者一个点(x,y): " << std::endl;
  std::string input;
  std::getline(std::cin, input);

  std::istringstream iss(input);
  std::vector<float> values;
  std::string token;

  while (std::getline(iss, token, ',')) {
    try {
      values.push_back(std::stof(token));
    } catch (const std::exception &e) {
      std::cout << "输入格式错误，请重新输入" << std::endl;
      return processWaitingInputState(det_model, image, tracker);
    }
  }

  ObjectBoxInfo init_bbox;

  if (values.size() == 2) {
    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    det_model->inference(input_images, out_datas);
    std::shared_ptr<ModelBoxInfo> det_result =
        std::static_pointer_cast<ModelBoxInfo>(out_datas[0]);

    float x = values[0];
    float y = values[1];

    // 初始化跟踪器
    int ret = tracker->initialize(image, det_result->bboxes, x, y);
    if (ret != 0) {
      return processWaitingInputState(det_model, image, tracker);
    }

  } else if (values.size() == 4) {
    // 输入的是bbox(x1,y1,x2,y2)
    init_bbox.x1 = values[0];
    init_bbox.y1 = values[1];
    init_bbox.x2 = values[2];
    init_bbox.y2 = values[3];
    init_bbox.score = 1.0f;

    // 获取检测结果用于初始化
    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    det_model->inference(input_images, out_datas);
    std::shared_ptr<ModelBoxInfo> det_result =
        std::static_pointer_cast<ModelBoxInfo>(out_datas[0]);

    // 初始化跟踪器
    int ret = tracker->initialize(image, det_result->bboxes, init_bbox);
  } else {
    std::cout << "输入格式错误，请重新输入" << std::endl;
    return processWaitingInputState(det_model, image, tracker);
  }

  // 切换到跟踪状态
  g_status = SystemStatus::TRACKING;
  g_lost_timer_started = false;

  std::cout << "跟踪器初始化成功，切换到跟踪状态" << std::endl;
  return init_bbox;
}

// 跟踪状态处理函数
void processTrackingState(std::shared_ptr<Tracker> &tracker,
                          std::shared_ptr<BaseImage> &image,
                          uint64_t frame_id) {
  TrackerInfo tracker_info;
  tracker->track(image, frame_id, tracker_info);

  std::cout << "frame id: " << frame_id << std::endl;

  if (tracker_info.status_ != TrackStatus::LOST) {
    std::cout << "bbox: " << tracker_info.box_info_.x1 << ","
              << tracker_info.box_info_.y1 << "," << tracker_info.box_info_.x2
              << "," << tracker_info.box_info_.y2 << std::endl;

    // 重置丢失计时器
    g_lost_timer_started = false;
  } else {
    std::cout << "tracking the object failed." << std::endl;

    // 开始或继续丢失计时
    if (!g_lost_timer_started) {
      g_lost_start_time = std::chrono::steady_clock::now();
      g_lost_timer_started = true;
    } else {
      auto current_time = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                         current_time - g_lost_start_time)
                         .count();

      if (elapsed >= LOST_TIMEOUT_SECONDS) {
        std::cout << "目标丢失超过5秒，切换到检测状态" << std::endl;
        g_status = SystemStatus::DETECTION;
        g_lost_timer_started = false;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <sot_model_id> <det_model_id> <image_folder>\n";
    return 1;
  }

  // 解析命令行参数
  std::string model_dir = argv[1];
  ModelType sot_model_id = modelTypeFromString(argv[2]);
  ModelType det_model_id = modelTypeFromString(argv[3]);
  std::string image_folder = argv[4];

  // 收集图像文件
  auto image_files = collect_images_sorted(image_folder);
  std::cout << "找到 " << image_files.size() << " 张图像" << std::endl;

  // 获取跟踪模型和检测模型实例
  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> sot_model = model_factory.getModel(sot_model_id);
  std::shared_ptr<BaseModel> det_model = model_factory.getModel(det_model_id);

  // 创建跟踪器
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_SOT);
  tracker->setModel(sot_model);

  // 设置非阻塞输入
  setNonBlockingInput();

  std::cout << "开始处理图像序列，按 'i' 切换到等待输入状态..." << std::endl;

  // 处理每一帧图像
  for (size_t idx = 0; idx < image_files.size(); ++idx) {
    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_files[idx]);
    if (!frame) {
      std::cerr << "错误: 无法读取图像 " << image_files[idx] << std::endl;
      continue;
    }

    // 检查键盘输入
    char key = checkKeyInput();
    if (key == 'i' || key == 'I') {
      restoreTerminal();  // 临时恢复终端设置以便输入
      g_status = SystemStatus::WAITING_INPUT;
      processWaitingInputState(det_model, frame, tracker);
      setNonBlockingInput();  // 重新设置非阻塞输入
      continue;
    }

    // 根据当前状态处理
    auto start_time = std::chrono::steady_clock::now();
    switch (g_status) {
      case SystemStatus::DETECTION:
        processDetectionState(det_model, frame, idx);
        break;

      case SystemStatus::TRACKING:
        processTrackingState(tracker, frame, idx);
        break;
    }
    auto end_time = std::chrono::steady_clock::now();
    auto prosess_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time)
                            .count();
    int usleep_time = std::max(30 - int(prosess_time), 0) * 1000;
    usleep(usleep_time);
  }

  // 恢复终端设置
  restoreTerminal();

  std::cout << "\n处理完成，共处理 " << image_files.size() << " 帧图像"
            << std::endl;

  return 0;
}