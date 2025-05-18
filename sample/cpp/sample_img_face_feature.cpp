
#include "tdl_model_factory.hpp"

template <typename T>
void embeddingToVec(void *embedding,
                    size_t num,
                    std::vector<float> &feature_vec) {
  T *feature = reinterpret_cast<T *>(embedding);
  for (size_t i = 0; i < num; ++i) {
    feature_vec[i] = (float)feature[i];
  }
  return;
}

std::shared_ptr<BaseImage> face_crop_align(
    std::shared_ptr<BaseImage> image,
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta) {
  float landmarks[10];
  for (int i = 0; i < 5; i++) {
    landmarks[2 * i] = landmarks_meta->landmarks_x[i];
    landmarks[2 * i + 1] = landmarks_meta->landmarks_y[i];
  }

  std::shared_ptr<BaseImage> face_crop =
      ImageFactory::alignFace(image, landmarks, nullptr, 5, nullptr);
  if (!face_crop) {
    printf("Failed to align face\n");
    return nullptr;
  }
  return face_crop;
}

std::vector<
    std::pair<std::shared_ptr<BaseImage>, std::shared_ptr<ModelLandmarksInfo>>>
extract_crop_face_landmark(
    std::shared_ptr<BaseModel> model_fl,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<ModelOutputInfo>> face_metas) {
  std::vector<std::pair<std::shared_ptr<BaseImage>,
                        std::shared_ptr<ModelLandmarksInfo>>>
      face_crops_landmark;
  std::shared_ptr<BasePreprocessor> preprocessor = model_fl->getPreprocessor();

  std::vector<std::shared_ptr<BaseImage>> face_crops;
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;

  char sz_img_name[128];
  for (size_t i = 0; i < images.size(); i++) {
    std::shared_ptr<ModelBoxLandmarkInfo> face_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(face_metas[i]);

    int x1 = face_meta->box_landmarks[0].x1;
    int y1 = face_meta->box_landmarks[0].y1;
    int x2 = face_meta->box_landmarks[0].x2;
    int y2 = face_meta->box_landmarks[0].y2;

    int box_width = x2 - x1;
    int box_height = y2 - y1;
    int crop_size = int(std::max(box_width, box_height) * 1.2);
    int crop_x1 = x1 - (crop_size - box_width) / 2;
    int crop_y1 = y1 - (crop_size - box_height) / 2;
    int crop_x2 = x2 + (crop_size - box_width) / 2;
    int crop_y2 = y2 + (crop_size - box_height) / 2;
    crop_x1 = std::max(crop_x1, 0);
    crop_y1 = std::max(crop_y1, 0);
    crop_x2 = std::min(crop_x2, (int)images[i]->getWidth());
    crop_y2 = std::min(crop_y2, (int)images[i]->getHeight());

    std::shared_ptr<BaseImage> face_crop = preprocessor->crop(
        images[i], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
    std::cout << "crop_x1: " << crop_x1 << " crop_y1: " << crop_y1
              << " crop_x2: " << crop_x2 << " crop_y2: " << crop_y2
              << ",face_crop_addr: " << face_crop << std::endl;
    sprintf(sz_img_name, "face_crop_%d.jpg", i);
    ImageFactory::writeImage(sz_img_name, face_crop);
    face_crops.push_back(face_crop);
  }
  model_fl->inference(face_crops, out_datas);
  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_datas[i]);
    face_crops_landmark.push_back(
        std::make_pair(face_crops[i], landmarks_meta));
  }
  return face_crops_landmark;
}

void visualize_face_crop(
    std::vector<std::pair<std::shared_ptr<BaseImage>,
                          std::shared_ptr<ModelLandmarksInfo>>>
        face_crops_landmark) {
  char sz_img_name[128];
  for (size_t i = 0; i < face_crops_landmark.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        face_crops_landmark[i].second;
    std::shared_ptr<BaseImage> face_crop = face_crops_landmark[i].first;

    cv::Mat mat;
    bool is_rgb;
    int32_t ret = ImageFactory::convertToMat(face_crop, mat, is_rgb);
    if (ret != 0) {
      printf("Failed to convert to mat\n");
      return;
    }
    for (int j = 0; j < landmarks_meta->landmarks_x.size(); j++) {
      cv::circle(mat,
                 cv::Point(landmarks_meta->landmarks_x[j],
                           landmarks_meta->landmarks_y[j]),
                 2, cv::Scalar(0, 0, 255), -1);
    }
    sprintf(sz_img_name, "face_crop_landmark_%d.jpg", i);
    cv::imwrite(sz_img_name, mat);
  }
}

void construct_model_id_mapping(
    std::map<std::string, ModelType> &model_id_mapping) {
  model_id_mapping["RESNET_FEATURE_BMFACE_R34"] =
      ModelType::RESNET_FEATURE_BMFACE_R34;
  model_id_mapping["RESNET_FEATURE_BMFACE_R50"] =
      ModelType::RESNET_FEATURE_BMFACE_R50;

  model_id_mapping["RECOGNITION_CVIFACE"] = ModelType::RECOGNITION_CVIFACE;
}

void set_preprocess_parameters(std::shared_ptr<BaseModel> model) {
  PreprocessParams pre_params;

  model->getPreprocessParameters(pre_params);

  pre_params.mean[0] = 0.99609375;
  pre_params.mean[1] = 0.99609375;
  pre_params.mean[2] = 0.99609375;

  pre_params.scale[0] = 0.0078125;
  pre_params.scale[1] = 0.0078125;
  pre_params.scale[2] = 0.0078125;

  model->setPreprocessParameters(pre_params);
}

int main(int argc, char **argv) {
  std::map<std::string, ModelType> model_id_mapping;
  construct_model_id_mapping(model_id_mapping);

  if (argc != 5) {
    printf(
        "Usage: %s <feature_extraction_model_id_name> <model_dir> "
        "<image_path1> <image_path2>\n",
        argv[0]);
    printf("feature_extraction_model_id_name:\n");
    for (auto &item : model_id_mapping) {
      printf("%s\n", item.first.c_str());
    }
    return -1;
  }
  std::string model_dir = argv[2];
  std::string image_path1 = argv[3];
  std::string image_path2 = argv[4];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path1);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }
  std::shared_ptr<BaseImage> image2 = ImageFactory::readImage(image_path2);
  if (!image2) {
    printf("Failed to create image2\n");
    return -1;
  }
  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_fd =
      model_factory.getModel(ModelType::SCRFD_DET_FACE);
  if (!model_fd) {
    printf("Failed to create model_fd\n");
    return -1;
  }
  std::shared_ptr<BaseModel> model_fl =
      model_factory.getModel(ModelType::KEYPOINT_FACE_V2);
  if (!model_fl) {
    printf("Failed to create model_fl\n");
    return -1;
  }

  std::string model_id_name = argv[1];

  std::shared_ptr<BaseModel> model_fe = model_factory.getModel(model_id_name);
  if (!model_fe) {
    printf("Failed to create model_fe\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_fd, out_fl, out_fe;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1, image2};
  model_fd->inference(input_images, out_fd);

  std::vector<std::pair<std::shared_ptr<BaseImage>,
                        std::shared_ptr<ModelLandmarksInfo>>>
      face_crops_landmark =
          extract_crop_face_landmark(model_fl, input_images, out_fd);

  std::vector<std::shared_ptr<BaseImage>> face_aligns;

  char sz_img_name[128];
  for (size_t i = 0; i < face_crops_landmark.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        face_crops_landmark[i].second;
    out_fl.push_back(landmarks_meta);
    std::shared_ptr<BaseImage> face_crop = face_crops_landmark[i].first;
    std::shared_ptr<BaseImage> face_align =
        face_crop_align(face_crop, landmarks_meta);
    face_aligns.push_back(face_align);
    sprintf(sz_img_name, "face_align_%d.jpg", i);
    ImageFactory::writeImage(sz_img_name, face_align);
  }

  visualize_face_crop(face_crops_landmark);
  if (face_aligns.size() != 2) {
    printf("face_aligns size is not 2\n");
    return -1;
  }

  // extract algined face

  model_fe->inference(face_aligns, out_fe);
  std::vector<std::vector<float>> features;
  for (size_t i = 0; i < out_fe.size(); i++) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_fe[i]);
    printf("feature size: %d\n", feature_meta->embedding_num);
    std::vector<float> feature_vec(feature_meta->embedding_num);

    printf("feature_meta->embedding_type: %d\n", feature_meta->embedding_type);
    switch (feature_meta->embedding_type) {
      case TDLDataType::INT8:
        embeddingToVec<int8_t>(feature_meta->embedding,
                               feature_meta->embedding_num, feature_vec);
        break;
      case TDLDataType::UINT8:
        embeddingToVec<uint8_t>(feature_meta->embedding,
                                feature_meta->embedding_num, feature_vec);
        break;
      case TDLDataType::FP32:
        embeddingToVec<float>(feature_meta->embedding,
                              feature_meta->embedding_num, feature_vec);
        break;
      default:
        assert(false && "Unsupported embedding_type");
    }

    features.push_back(feature_vec);
  }
  float sim = 0;
  float norm1 = 0;
  float norm2 = 0;
  for (size_t i = 0; i < features[0].size(); i++) {
    sim += features[0][i] * features[1][i];
    norm1 += features[0][i] * features[0][i];
    norm2 += features[1][i] * features[1][i];
  }
  norm1 = sqrt(norm1);
  norm2 = sqrt(norm2);
  printf("sim: %f\n", sim / (norm1 * norm2));

  return 0;
}
