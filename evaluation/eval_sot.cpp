#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"
#include "utils/tdl_log.hpp"

using std::string;
namespace fs = std::experimental::filesystem;

struct EvalResult {
  std::string seq_name;
  double pr{0.0};
  double sr{0.0};
  int occlusion_frames{0};
  int fn{0};
  int fp{0};
};

static inline bool iequals(const std::string &a, const std::string &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(a[i]) != std::tolower(b[i])) return false;
  }
  return true;
}

static inline bool ends_with_any(const std::string &name,
                                 const std::vector<std::string> &exts) {
  for (const auto &ext : exts) {
    if (name.size() >= ext.size()) {
      std::string tail = name.substr(name.size() - ext.size());
      if (iequals(tail, ext)) return true;
    }
  }
  return false;
}

static inline bool parse_float(const std::string &s, float &out) {
  // 支持 "nan" / "NaN" / "NAN"
  if (s.size() >= 3) {
    std::string low;
    low.resize(s.size());
    std::transform(s.begin(), s.end(), low.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (low == "nan") {
      out = std::numeric_limits<float>::quiet_NaN();
      return true;
    }
  }
  char *endp = nullptr;
  const char *cstr = s.c_str();
  errno = 0;
  float v = std::strtof(cstr, &endp);
  if (endp == cstr || errno == ERANGE) {
    return false;
  }
  out = v;
  return true;
}

static inline bool is_nan(float v) { return std::isnan(v); }

static inline bool bbox_is_nan(const std::vector<float> &bbox) {
  // bbox: x,y,w,h
  if (bbox.size() < 4) return true;
  return is_nan(bbox[0]) || is_nan(bbox[1]) || is_nan(bbox[2]) ||
         is_nan(bbox[3]);
}

static double iou_xywh(const std::vector<float> &a,
                       const std::vector<float> &b) {
  // 处理 NaN
  if (bbox_is_nan(a) || bbox_is_nan(b)) return 0.0;
  double x1 = a[0], y1 = a[1], w1 = a[2], h1 = a[3];
  double x2 = b[0], y2 = b[1], w2 = b[2], h2 = b[3];

  double left = std::max(x1, x2);
  double top = std::max(y1, y2);
  double right = std::min(x1 + w1, x2 + w2);
  double bottom = std::min(y1 + h1, y2 + h2);

  if (right <= left || bottom <= top) return 0.0;

  double inter = (right - left) * (bottom - top);
  double area1 = w1 * h1;
  double area2 = w2 * h2;
  double uni = area1 + area2 - inter;
  if (uni <= 0.0) return 0.0;
  return inter / uni;
}

static inline double center_distance_le_20(const std::vector<float> &gtb,
                                           const std::vector<float> &pb) {
  if (bbox_is_nan(gtb) || bbox_is_nan(pb)) return false;
  double gx = gtb[0] + gtb[2] / 2.0;
  double gy = gtb[1] + gtb[3] / 2.0;
  double px = pb[0] + pb[2] / 2.0;
  double py = pb[1] + pb[3] / 2.0;
  double dx = gx - px, dy = gy - py;
  double dist = std::sqrt(dx * dx + dy * dy);
  return dist <= 20.0;
}

static bool read_annotations_xywh(const std::string &anno_file,
                                  std::vector<std::vector<float>> &annos) {
  std::ifstream ifs(anno_file);
  if (!ifs.is_open()) return false;
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<float> bbox(4, std::numeric_limits<float>::quiet_NaN());
    std::stringstream ss(line);
    std::string token;
    int idx = 0;
    while (std::getline(ss, token, ',') && idx < 4) {
      float v;
      if (!parse_float(token, v)) {
        // 解析失败，保持 NaN
      } else {
        bbox[idx] = v;
      }
      ++idx;
    }
    // 不足4项用 NaN 填充
    annos.push_back(bbox);
  }
  return true;
}

static std::vector<std::string> collect_images_sorted(
    const std::string &seq_dir) {
  std::vector<std::string> files;
  const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp"};
  for (auto &p : fs::directory_iterator(seq_dir)) {
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

static std::vector<fs::path> list_sequence_dirs(const std::string &image_dir) {
  std::vector<fs::path> seqs;
  for (auto &p : fs::directory_iterator(image_dir)) {
    if (fs::is_directory(p.path())) {
      seqs.push_back(p.path());
    }
  }
  std::sort(seqs.begin(), seqs.end());
  return seqs;
}

static std::string join_path(const std::string &a, const std::string &b) {
  fs::path pa(a);
  pa /= b;
  return pa.string();
}

int main(int argc, char **argv) {
  // 参数：model_dir model_id image_dir txt_dir
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <image_dir> <txt_dir> <save_dir>\n";
    return 1;
  }
  std::string model_dir = argv[1];
  std::string image_root = argv[2];
  std::string txt_root = argv[3];
  std::string save_dir = argv[4];
  ModelType sot_model_type = ModelType::TRACKING_FEARTRACK;

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> sot_model = model_factory.getModel(sot_model_type);

  auto seq_dirs = list_sequence_dirs(image_root);
  if (seq_dirs.empty()) {
    std::cerr << "No sequence folders found in: " << image_root << "\n";
    return 1;
  }

  std::vector<EvalResult> results;
  double total_pr = 0.0, total_sr = 0.0;
  int total_fp = 0, total_fn = 0;
  const std::vector<double> sr_thresholds = [] {
    std::vector<double> v;
    for (int i = 0; i <= 20; ++i) v.push_back(i * 0.05);
    return v;
  }();

  int seq_idx = 0;
  for (const auto &seq_path : seq_dirs) {
    ++seq_idx;
    std::string seq_name = seq_path.filename().string();
    std::cout << "######################## 开始处理序列：" << seq_name
              << " 进度：" << seq_idx << "/" << seq_dirs.size()
              << " ########################\n";

    std::string anno_file = join_path(txt_root, seq_name + ".txt");
    std::vector<std::vector<float>> annotations;
    if (!read_annotations_xywh(anno_file, annotations)) {
      std::cout << "找不到或无法读取序列 " << seq_name << " 的标注文件，跳过\n";
      continue;
    }

    auto image_files = collect_images_sorted(seq_path.string());
    if (image_files.empty()) {
      std::cout << "序列 " << seq_name << " 中没有找到图像文件，跳过\n";
      continue;
    }

    if (annotations.size() != image_files.size()) {
      std::cout << "序列 " << seq_name << " 的图像数量 (" << image_files.size()
                << ") 与标注数量 (" << annotations.size() << ") 不匹配，跳过\n";
      continue;
    }

    // 第一帧
    std::shared_ptr<BaseImage> first_img =
        ImageFactory::readImage(image_files[0]);
    if (!first_img) {
      std::cout << "序列 " << seq_name << " 第一帧读取失败，跳过\n";
      continue;
    }

    auto init_bbox = annotations[0];  // x,y,w,h
    if (bbox_is_nan(init_bbox)) {
      std::cout << "序列 " << seq_name << " 的初始边界框包含NaN值，跳过\n";
      continue;
    }

    // 转成 x1,y1,x2,y2
    ObjectBoxInfo init_box;
    init_box.x1 = init_bbox[0];
    init_box.y1 = init_bbox[1];
    init_box.x2 = init_bbox[0] + init_bbox[2];
    init_box.y2 = init_bbox[1] + init_bbox[3];
    init_box.score = 1.0f;

    // 初始化跟踪器
    std::shared_ptr<Tracker> tracker =
        TrackerFactory::createTracker(TrackerType::TDL_SOT);
    tracker->setModel(sot_model);
    if (tracker->initialize(first_img, {}, init_box, 0) != 0) {
      std::cout << "序列 " << seq_name << " 跟踪器初始化失败，跳过\n";
      continue;
    }

    // 指标统计
    int precision_correct = 0;
    std::vector<double> iou_values;
    iou_values.reserve(image_files.size() > 0 ? image_files.size() - 1 : 0);
    int seq_fp = 0, seq_fn = 0;
    int occlusion_frame_count = 0;

    // 进度条准备
    size_t total_frames = image_files.size() > 1 ? (image_files.size() - 1) : 0;
    auto print_progress = [&](size_t done) {
      if (total_frames == 0) return;
      double progress =
          static_cast<double>(done) / static_cast<double>(total_frames);
      int barWidth = 30;
      int pos = static_cast<int>(barWidth * progress);
      std::cout << "\r进度: [";
      for (int i = 0; i < barWidth; ++i) std::cout << (i < pos ? '#' : '.');
      std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0)
                << "% (" << done << "/" << total_frames << ")";
      std::cout.flush();
    };

    std::string save_path = join_path(save_dir, seq_name + ".txt");
    std::ofstream saveofs(save_path);
    if (!saveofs.is_open()) {
      std::cout << "Failed to open " << save_path << " for writing\n";
      continue;
    }
    saveofs << std::fixed << std::setprecision(2);

    saveofs << init_bbox[0] << " " << init_bbox[1] << " " << init_bbox[2] << " "
            << init_bbox[3] << "\n";

    // 逐帧
    for (size_t idx = 1; idx < image_files.size(); ++idx) {
      // 更新进度条
      print_progress(idx - 1);
      std::shared_ptr<BaseImage> frame =
          ImageFactory::readImage(image_files[idx]);

      if (!frame) {
        // std::cout << "序列 " << seq_name << " 的第 " << idx
        //        << " 帧读取失败，跳过此帧\n";
        continue;
      }

      // GT
      auto gt_bbox = annotations[idx];  // x,y,w,h
      bool gt_is_nan = bbox_is_nan(gt_bbox);
      if (gt_is_nan) occlusion_frame_count++;

      // 跟踪
      TrackerInfo tracker_info;
      int tr = tracker->track(frame, idx, tracker_info);
      if (tr != 0) {
        // std::cout << "序列 " << seq_name << " 的第 " << idx
        //        << " 帧跟踪失败，跳过此帧\n";
        continue;
      }

      bool pred_is_nan = (tracker_info.status_ == TrackStatus::LOST);

      // 将预测转换为 x,y,w,h
      std::vector<float> pred_xywh(4, std::numeric_limits<float>::quiet_NaN());
      if (!pred_is_nan) {
        float px = tracker_info.box_info_.x1;
        float py = tracker_info.box_info_.y1;
        float pw = tracker_info.box_info_.x2 - tracker_info.box_info_.x1;
        float ph = tracker_info.box_info_.y2 - tracker_info.box_info_.y1;
        pred_xywh = {px, py, pw, ph};
        saveofs << px << "," << py << "," << pw << "," << ph << "\n";
      } else {
        saveofs << "nan,nan,nan,nan\n";
      }
    }

    if (total_frames > 0) {
      print_progress(total_frames);
      std::cout << std::endl;
    }
  }

  return 0;
}