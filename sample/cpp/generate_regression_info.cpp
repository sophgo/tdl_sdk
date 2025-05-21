#include <dirent.h>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
namespace fs = std::experimental::filesystem;

void constructModelIdMapping(
    std::map<std::string, ModelType> &model_id_mapping) {
  model_id_mapping["MBV2_DET_PERSON_256_448"] =
      ModelType::MBV2_DET_PERSON_256_448;
  model_id_mapping["YOLOV8N_DET_HAND"] = ModelType::YOLOV8N_DET_HAND;
  model_id_mapping["YOLOV8N_DET_PET_PERSON"] =
      ModelType::YOLOV8N_DET_PET_PERSON;
  model_id_mapping["YOLOV8N_DET_PERSON_VEHICLE"] =
      ModelType::YOLOV8N_DET_PERSON_VEHICLE;
  model_id_mapping["YOLOV8N_DET_HAND_FACE_PERSON"] =
      ModelType::YOLOV8N_DET_HAND_FACE_PERSON;
  model_id_mapping["YOLOV8N_DET_HEAD_PERSON"] =
      ModelType::YOLOV8N_DET_HEAD_PERSON;
  model_id_mapping["YOLOV8N_DET_HEAD_HARDHAT"] =
      ModelType::YOLOV8N_DET_HEAD_HARDHAT;
  model_id_mapping["YOLOV8N_DET_FIRE_SMOKE"] =
      ModelType::YOLOV8N_DET_FIRE_SMOKE;
  model_id_mapping["YOLOV8N_DET_FIRE"] = ModelType::YOLOV8N_DET_FIRE;
  model_id_mapping["YOLOV8N_DET_HEAD_SHOULDER"] =
      ModelType::YOLOV8N_DET_HEAD_SHOULDER;
  model_id_mapping["YOLOV8N_DET_LICENSE_PLATE"] =
      ModelType::YOLOV8N_DET_LICENSE_PLATE;
  model_id_mapping["YOLOV8N_DET_TRAFFIC_LIGHT"] =
      ModelType::YOLOV8N_DET_TRAFFIC_LIGHT;
  model_id_mapping["YOLOV8N_DET_MONITOR_PERSON"] =
      ModelType::YOLOV8N_DET_MONITOR_PERSON;
  model_id_mapping["SCRFD_DET_FACE"] = ModelType::SCRFD_DET_FACE;
  model_id_mapping["CLS_RGBLIVENESS"] = ModelType::CLS_RGBLIVENESS;
  model_id_mapping["CLS_SOUND_BABAY_CRY"] = ModelType::CLS_SOUND_BABAY_CRY;
  model_id_mapping["CLS_SOUND_COMMAND"] = ModelType::CLS_SOUND_COMMAND;
  model_id_mapping["KEYPOINT_HAND"] = ModelType::KEYPOINT_HAND;
  model_id_mapping["KEYPOINT_LICENSE_PLATE"] =
      ModelType::KEYPOINT_LICENSE_PLATE;
  model_id_mapping["KEYPOINT_YOLOV8POSE_PERSON17"] =
      ModelType::KEYPOINT_YOLOV8POSE_PERSON17;
  model_id_mapping["KEYPOINT_SIMCC_PERSON17"] =
      ModelType::KEYPOINT_SIMCC_PERSON17;
}

std::string getTxtName(std::string &dst_root, std::string &img_name) {
  if (!fs::exists(dst_root)) {
    fs::create_directories(dst_root);
  }
  if (dst_root.back() != '/') {
    dst_root += '/';
  }
  std::string txt_name = dst_root + img_name + ".txt";
  return txt_name;
}

void saveDetectionResults(std::string &dst_root, std::string &img_name,
                          const std::shared_ptr<ModelOutputInfo> &out_data) {
  std::string txt_name = getTxtName(dst_root, img_name);
  std::ofstream outfile(txt_name);

  if (out_data->getType() == ModelOutputType::OBJECT_DETECTION) {
    std::shared_ptr<ModelBoxInfo> obj_meta =
        std::static_pointer_cast<ModelBoxInfo>(out_data);
    for (const auto &bbox : obj_meta->bboxes) {
      outfile << std::fixed << std::setprecision(2) << bbox.x1 << " " << bbox.y1
              << " " << bbox.x2 << " " << bbox.y2 << " ";
      outfile << bbox.class_id << " " << std::fixed << std::setprecision(2)
              << bbox.score << std::endl;
    }
  } else if (out_data->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
    for (const auto &box_landmark : obj_meta->box_landmarks) {
      outfile << std::fixed << std::setprecision(2) << box_landmark.x1 << " "
              << box_landmark.y1 << " " << box_landmark.x2 << " "
              << box_landmark.y2 << " " << box_landmark.score << std::endl;
    }
  } else {
    std::cout << "Unsupported output type: "
              << static_cast<int>(out_data->getType()) << std::endl;
  }
  outfile.close();
  std::cout << "write file " << txt_name << " done" << std::endl;
}

void saveClassificationResults(
    std::string &dst_root, std::string &img_name,
    const std::shared_ptr<ModelOutputInfo> &out_data) {
  std::string txt_name = getTxtName(dst_root, img_name);

  std::ofstream outfile(txt_name);
  std::shared_ptr<ModelClassificationInfo> cls_meta =
      std::static_pointer_cast<ModelClassificationInfo>(out_data);
  outfile << cls_meta->topk_class_ids[0] << " " << std::fixed
          << std::setprecision(2) << cls_meta->topk_scores[0] << std::endl;

  outfile.close();
  std::cout << "write file " << txt_name << " done" << std::endl;
}

void saveKeypointResults(std::string &dst_root, std::string &img_name,
                         const std::shared_ptr<ModelOutputInfo> &out_data) {
  std::string txt_name = getTxtName(dst_root, img_name);
  std::ofstream outfile(txt_name);

  std::vector<float> landmarks_x;
  std::vector<float> landmarks_y;
  std::vector<float> landmarks_score;

  if (out_data->getType() == ModelOutputType::OBJECT_LANDMARKS) {
    std::shared_ptr<ModelLandmarksInfo> keypoint_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_data);
    landmarks_x = keypoint_meta->landmarks_x;
    landmarks_y = keypoint_meta->landmarks_y;
    landmarks_score = keypoint_meta->landmarks_score;
  } else if (out_data->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
    ObjectBoxLandmarkInfo keypoint_meta = obj_meta->box_landmarks[0];
    landmarks_x = keypoint_meta.landmarks_x;
    landmarks_y = keypoint_meta.landmarks_y;
    landmarks_score = keypoint_meta.landmarks_score;

  } else {
    std::cout << "Unsupported output type: "
              << static_cast<int>(out_data->getType()) << std::endl;
  }

  size_t num_keypoints = landmarks_x.size();
  size_t num_landmarks_score = landmarks_score.size();

  if (num_keypoints != num_landmarks_score) {
    for (int k = 0; k < num_keypoints; k++) {
      outfile << std::fixed << std::setprecision(4) << landmarks_x[k] << " "
              << landmarks_y[k] << std::endl;
    }
  } else {
    for (int k = 0; k < num_keypoints; k++) {
      outfile << std::fixed << std::setprecision(4) << landmarks_x[k] << " "
              << landmarks_y[k] << " " << landmarks_score[k] << std::endl;
    }
  }

  outfile.close();
  std::cout << "write file " << txt_name << " done" << std::endl;
}

std::vector<std::shared_ptr<BaseImage>> getInputDatas(std::string &image_path) {
  std::vector<std::shared_ptr<BaseImage>> input_datas;
  if (image_path.size() >= 4 &&
      image_path.substr(image_path.size() - 4) != ".bin") {
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    input_datas = {image};
  } else {
    std::vector<uint8_t> buffer;
    if (!CommonUtils::readBinaryFile(image_path, buffer)) {
      printf("read file failed\n");
      throw std::runtime_error("Failed to read binary file: " + image_path);
    }
    std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
        buffer.size(), 1, ImageFormat::GRAY, TDLDataType::UINT8, true);

    uint8_t *data_buffer = bin_data->getVirtualAddress()[0];
    memcpy(data_buffer, buffer.data(), buffer.size() * sizeof(uint8_t));
    input_datas = {bin_data};
  }
  return input_datas;
};

int main(int argc, char **argv) {
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_dir = argv[3];
  std::string dst_root = argv[4];
  float model_threshold;
  if (argc == 6) {
    model_threshold = atof(argv[5]);
  } else {
    model_threshold = 0.5;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.setModelDir(model_dir);
  model_factory.loadModelConfig();
  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id_name);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }
  model->setModelThreshold(model_threshold);
  ModelType model_id = modelTypeFromString(model_id_name);

  for (const auto &entry : fs::directory_iterator(image_dir)) {
    std::string image_path = entry.path().string();
    std::vector<std::shared_ptr<BaseImage>> input_datas;
    input_datas = getInputDatas(image_path);

    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    model->inference(input_datas, out_datas);

    std::string img_name = fs::path(image_path).stem().string();
    ModelOutputType out_type = out_datas[0]->getType();
    if (out_type == ModelOutputType::OBJECT_DETECTION) {
      saveDetectionResults(dst_root, img_name, out_datas[0]);
    } else if (out_type == ModelOutputType::CLASSIFICATION) {
      saveClassificationResults(dst_root, img_name, out_datas[0]);
    } else if (out_type == ModelOutputType::OBJECT_LANDMARKS) {
      saveKeypointResults(dst_root, img_name, out_datas[0]);
    } else if (out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
      if (model_id == ModelType::KEYPOINT_YOLOV8POSE_PERSON17) {
        saveKeypointResults(dst_root, img_name, out_datas[0]);
      } else if (model_id == ModelType::SCRFD_DET_FACE) {
        saveDetectionResults(dst_root, img_name, out_datas[0]);
      } else {
        std::cout << "Unsupported model_id: " << static_cast<int>(model_id)
                  << std::endl;
      }
    } else {
      std::cout << "Unsupported output type: " << static_cast<int>(out_type)
                << std::endl;
    }
  }
  return 0;
}