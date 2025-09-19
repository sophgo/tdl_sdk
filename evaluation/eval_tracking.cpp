#include <opencv2/opencv.hpp>

#include "evaluator.hpp"
#include "model/base_model.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"
class TrackingEvaluator : public Evaluator {
 public:
  TrackingEvaluator(std::vector<std::shared_ptr<BaseModel>> det_models,
                    std::shared_ptr<Tracker> tracker);
  ~TrackingEvaluator();

  void evaluate(const std::vector<std::string> &eval_files,
                const std::string &output_dir);

 private:
  void evaluate_video(const std::string &video_file,
                      const std::string &output_dir);
  std::vector<std::shared_ptr<BaseModel>> det_models_;
  std::shared_ptr<Tracker> tracker_;
};
TrackingEvaluator::TrackingEvaluator(
    std::vector<std::shared_ptr<BaseModel>> det_models,
    std::shared_ptr<Tracker> tracker)
    : det_models_(det_models), tracker_(tracker) {}

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
    std::string str_content = packOutput(track_results);
    writeResult(sz_frame_name, str_content);
    frame_id++;
    printf("frame_id: %d\n", frame_id);
  }
  cap.release();
  printf("end of video: %s\n", video_file.c_str());
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_dir> <video_file> <output_dir>\n", argv[0]);
    return -1;
  }
  std::string model_dir = argv[1];
  std::string video_file = argv[2];
  std::string output_dir = argv[3];

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::vector<std::string> video_files = {video_file};

  std::shared_ptr<BaseModel> face_model =
      model_factory.getModel(ModelType::SCRFD_DET_FACE);
  if (face_model == nullptr) {
    printf("Failed to get face model\n");
    return -1;
  }
  std::shared_ptr<BaseModel> person_model =
      model_factory.getModel(ModelType::MBV2_DET_PERSON);
  if (person_model == nullptr) {
    printf("Failed to get person model\n");
    return -1;
  }
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
  TrackingEvaluator evaluator(det_models, tracker);
  evaluator.evaluate(video_files, output_dir);
}
