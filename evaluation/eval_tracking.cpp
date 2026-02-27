#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>

#include "evaluator.hpp"
#include "model/base_model.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"

#include <sstream>

static void draw_track_results_from_str(cv::Mat &frame,
                                        const std::string &str_content,
                                        int img_width, int img_height) {
  std::istringstream iss(str_content);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty()) continue;
    std::istringstream ls(line);
    int object_type = 0;
    float ctx = 0.0f;
    float cty = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    int track_id = -1;
    float score = 0.0f;
    if (!(ls >> object_type >> ctx >> cty >> w >> h >> track_id >> score)) {
      continue;
    }

    float x1f = (ctx - w * 0.5f) * img_width;
    float y1f = (cty - h * 0.5f) * img_height;
    float x2f = (ctx + w * 0.5f) * img_width;
    float y2f = (cty + h * 0.5f) * img_height;

    int x1 = std::max(0, std::min(img_width - 1, static_cast<int>(x1f)));
    int y1 = std::max(0, std::min(img_height - 1, static_cast<int>(y1f)));
    int x2 = std::max(0, std::min(img_width - 1, static_cast<int>(x2f)));
    int y2 = std::max(0, std::min(img_height - 1, static_cast<int>(y2f)));

    if (x2 <= x1 || y2 <= y1) continue;

    cv::Scalar color(0, 255, 0);
    if (object_type == static_cast<int>(TDLObjectType::OBJECT_TYPE_FACE)) {
      color = cv::Scalar(0, 0, 255);
    }

    cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1), color, 2);
    cv::putText(frame, std::to_string(track_id),
                cv::Point(x1, std::max(0, y1 - 3)), cv::FONT_HERSHEY_SIMPLEX,
                0.7, color, 2);
  }
}

static bool make_dir(const std::string &path, mode_t mode = 0755) {
  if (path.empty()) return false;
  if (mkdir(path.c_str(), mode) == 0) {
    return true;
  }
  if (errno == EEXIST) {
    return true;
  }
  return false;
}

class TrackingEvaluator : public Evaluator {
 public:
  TrackingEvaluator(std::vector<std::shared_ptr<BaseModel>> det_models,
                    std::shared_ptr<Tracker> tracker);
  TrackingEvaluator(std::vector<std::shared_ptr<BaseModel>> det_models,
                    std::shared_ptr<Tracker> tracker,
                    const std::string &output_video_path);
  ~TrackingEvaluator();

  void evaluate(const std::vector<std::string> &eval_files,
                const std::string &output_dir);

 private:
  void evaluate_video(const std::string &video_file,
                      const std::string &output_dir);
  std::vector<std::shared_ptr<BaseModel>> det_models_;
  std::shared_ptr<Tracker> tracker_;
  std::string output_video_path_;
};
TrackingEvaluator::TrackingEvaluator(
    std::vector<std::shared_ptr<BaseModel>> det_models,
    std::shared_ptr<Tracker> tracker)
    : det_models_(det_models), tracker_(tracker) {}

TrackingEvaluator::TrackingEvaluator(
    std::vector<std::shared_ptr<BaseModel>> det_models,
    std::shared_ptr<Tracker> tracker, const std::string &output_video_path)
    : det_models_(det_models),
      tracker_(tracker),
      output_video_path_(output_video_path) {}

TrackingEvaluator::~TrackingEvaluator() {}
void TrackingEvaluator::evaluate(const std::vector<std::string> &eval_files,
                                 const std::string &output_dir) {
  for (auto &video_file : eval_files) {
    evaluate_video(video_file, output_dir);
  }
}
void TrackingEvaluator::evaluate_video(const std::string &video_file,
                                       const std::string &output_dir) {
  printf("Evaluating video: %s\n", video_file.c_str());
  cv::VideoCapture cap(video_file);
  if (!cap.isOpened()) {
    printf("Failed to open video: %s\n", video_file.c_str());
    return;
  }
  cv::Mat frame;
  uint32_t frame_id = 0;
  char sz_frame_name[1024];
  img_width_ = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  img_height_ = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  tracker_->setImgSize(img_width_, img_height_);

  cv::VideoWriter writer;
  bool save_video = !output_video_path_.empty();
  int out_w = img_width_ / 2;
  int out_h = img_height_ / 2;
  if (out_w <= 0) out_w = img_width_;
  if (out_h <= 0) out_h = img_height_;
  if (save_video) {
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25;
    writer.open(output_video_path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(out_w, out_h));
    if (!writer.isOpened()) {
      printf("Failed to open output video: %s\n", output_video_path_.c_str());
      save_video = false;
    }
  }

  while (cap.read(frame)) {
    sprintf(sz_frame_name, "%s/%08d.txt", output_dir.c_str(), frame_id);
    std::shared_ptr<BaseImage> image = ImageFactory::convertFromMat(frame);
    std::vector<ObjectBoxInfo> det_results;
    std::vector<TrackerInfo> track_results;
    std::vector<std::shared_ptr<BaseImage>> det_images{image};
    for (auto &det_model : det_models_) {
      std::vector<std::shared_ptr<ModelOutputInfo>> model_output_infos;
      det_model->inference(det_images, model_output_infos);
      if (model_output_infos[0]->getType() ==
          ModelOutputType::OBJECT_DETECTION) {
        auto object_detection_info =
            std::static_pointer_cast<ModelBoxInfo>(model_output_infos[0]);
        for (auto &box_info : object_detection_info->bboxes) {
          printf("box_info: %d,%d, %.2f, %.2f, %.2f, %.2f, %.2f\n",
                 box_info.class_id, int(box_info.object_type), box_info.score,
                 box_info.x1, box_info.y1, box_info.x2, box_info.y2);
        }
        det_results.insert(det_results.end(),
                           object_detection_info->bboxes.begin(),
                           object_detection_info->bboxes.end());
      } else if (model_output_infos[0]->getType() ==
                 ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
        auto object_detection_with_landmarks_info =
            std::static_pointer_cast<ModelBoxLandmarkInfo>(
                model_output_infos[0]);
        for (auto &box_landmark_info :
             object_detection_with_landmarks_info->box_landmarks) {
          ObjectBoxInfo box_info(box_landmark_info.class_id,
                                 box_landmark_info.score, box_landmark_info.x1,
                                 box_landmark_info.y1, box_landmark_info.x2,
                                 box_landmark_info.y2);
          box_info.object_type = box_landmark_info.object_type;
          det_results.push_back(box_info);
        }
      } else {
        printf("model_output_infos[0]->getType() not supported\n");
      }
    }
    tracker_->track(det_results, frame_id, track_results);
    std::string str_content = packOutput(track_results, false);
    if (!str_content.empty()) {
      writeResult(sz_frame_name, str_content);
    }
    if (save_video) {
      cv::Mat vis = frame.clone();
      if (!str_content.empty()) {
        draw_track_results_from_str(vis, str_content, img_width_, img_height_);
      }
      cv::Mat vis_resize;
      cv::resize(vis, vis_resize, cv::Size(out_w, out_h));
      writer.write(vis_resize);
    }
    frame_id++;
    printf("frame_id: %d\n", frame_id);
  }
  cap.release();
  if (save_video) {
    writer.release();
  }
  printf("end of video: %s\n", video_file.c_str());
}

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5 && argc != 6) {
    printf("Usage 1 : %s <model_dir> <video_file> <output_dir>\n", argv[0]);
    printf(
        "Usage 1+ : %s <model_dir> <video_file> <output_dir> <output_video>\n",
        argv[0]);
    printf(
        "Usage 2 : %s <face_model_path> <person_model_path> <video_file> "
        "<output_dir>\n",
        argv[0]);
    printf(
        "Usage 2+ : %s <face_model_path> <person_model_path> <video_file> "
        "<output_dir> <output_video>\n",
        argv[0]);
    return -1;
  }
  std::string model_dir, face_model_path, person_model_path;
  std::string video_file, output_dir;
  std::string output_video_path;
  auto ends_with = [](const std::string &s, const std::string &suffix) -> bool {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  std::string first_arg = argv[1];
  bool is_folder_mode =
      !(ends_with(first_arg, ".cvimodel") || ends_with(first_arg, ".bmodel"));
  if (is_folder_mode) {
    // 只传一个文件夹
    model_dir = argv[1];
    video_file = argv[2];
    output_dir = argv[3];
    if (argc == 5) {
      output_video_path = argv[4];
    }
  } else {
    // 传两个模型的绝对路径
    face_model_path = argv[1];
    person_model_path = argv[2];
    video_file = argv[3];
    output_dir = argv[4];
    if (argc == 6) {
      output_video_path = argv[5];
    }
  }

  if (!make_dir(output_dir, 0777)) {
    printf("Failed to create output_dir: %s\n", output_dir.c_str());
    return -1;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();

  std::shared_ptr<BaseModel> face_model;
  std::shared_ptr<BaseModel> person_model;
  struct stat path_stat;

  if (is_folder_mode) {
    model_factory.setModelDir(model_dir);
    face_model = model_factory.getModel(ModelType::SCRFD_DET_FACE);
    if (face_model == nullptr) {
      printf("Failed to get face model\n");
      return -1;
    }
    person_model = model_factory.getModel(ModelType::MBV2_DET_PERSON);
    if (person_model == nullptr) {
      printf("Failed to get person model\n");
      return -1;
    }
  } else {
    if (stat(face_model_path.c_str(), &path_stat) != 0 ||
        !S_ISREG(path_stat.st_mode)) {
      printf("Error: face model path is not a valid file: %s\n",
             face_model_path.c_str());
      return -1;
    }
    if (stat(person_model_path.c_str(), &path_stat) != 0 ||
        !S_ISREG(path_stat.st_mode)) {
      printf("Error: person model path is not a valid file: %s\n",
             person_model_path.c_str());
      return -1;
    }
    // 直接通过绝对路径加载两个模型
    face_model =
        model_factory.getModel(ModelType::SCRFD_DET_FACE, face_model_path);
    if (!face_model) {
      printf("Failed to load face model from path\n");
      return -1;
    }
    person_model =
        model_factory.getModel(ModelType::MBV2_DET_PERSON, person_model_path);
    if (!person_model) {
      printf("Failed to load person model from path\n");
      return -1;
    }
  }

  std::vector<std::string> video_files = {video_file};
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  std::map<TDLObjectType, TDLObjectType> object_pair_config;
  object_pair_config[TDLObjectType::OBJECT_TYPE_FACE] =
      TDLObjectType::OBJECT_TYPE_PERSON;
  tracker->setPairConfig(object_pair_config);
  if (tracker == nullptr) {
    printf("Failed to get tracker\n");
    return -1;
  }
  std::vector<std::shared_ptr<BaseModel>> det_models{face_model, person_model};
  TrackingEvaluator evaluator(det_models, tracker, output_video_path);
  evaluator.evaluate(video_files, output_dir);
}
