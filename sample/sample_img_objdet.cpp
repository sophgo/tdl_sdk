
#include "tdl_model_factory.hpp"

void visualize_object_detection(std::shared_ptr<BaseImage> image,
                                std::shared_ptr<ModelBoxInfo> obj_meta,
                                const std::string &str_img_name) {
  cv::Mat mat;
  bool is_rgb;
  int32_t ret = ImageFactory::convertToMat(image, mat, is_rgb);
  if (ret != 0) {
    std::cout << "Failed to convert to mat" << std::endl;
    return;
  }

  if (is_rgb) {
    std::cout << "convert to bgr" << std::endl;
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  }
  std::cout << "object_meta->bboxes.size(): " << obj_meta->bboxes.size()
            << std::endl;
  for (size_t i = 0; i < obj_meta->bboxes.size(); i++) {
    cv::rectangle(
        mat,
        cv::Rect(int(obj_meta->bboxes[i].x1), int(obj_meta->bboxes[i].y1),
                 int(obj_meta->bboxes[i].x2 - obj_meta->bboxes[i].x1),
                 int(obj_meta->bboxes[i].y2 - obj_meta->bboxes[i].y1)),
        cv::Scalar(0, 0, 255), 2);
    char sz_text[128];
    sprintf(sz_text, "%d,%.2f", obj_meta->bboxes[i].class_id,
            obj_meta->bboxes[i].score);
    cv::putText(
        mat, sz_text,
        cv::Point(int(obj_meta->bboxes[i].x1), int(obj_meta->bboxes[i].y1)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
  }
  std::cout << "save image to " << str_img_name << std::endl;

  cv::imwrite(str_img_name, mat);
}

void construct_model_id_mapping(
    std::map<std::string, TDL_MODEL_TYPE> &model_id_mapping) {
  model_id_mapping["TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV10"] =
      TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV10;
  model_id_mapping["TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_HARDHAT"] =
      TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_HARDHAT;
  model_id_mapping["TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE"] =
      TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE;
  model_id_mapping["TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN"] =
      TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN;
}

int main(int argc, char **argv) {
  std::map<std::string, TDL_MODEL_TYPE> model_id_mapping;
  construct_model_id_mapping(model_id_mapping);

  if (argc != 4 && argc != 5) {
    printf(
        "Usage: %s <model_id_name> <model_path> <image_path> "
        "<model_threshold>\n",
        argv[0]);
    printf("Usage: %s <model_id_name> <model_path> <image_path>\n", argv[0]);
    printf("model_id_name:\n");
    for (auto &item : model_id_mapping) {
      printf("%s\n", item.first.c_str());
    }
    return -1;
  }
  float model_threshold = 0.5;
  if (argc == 5) {
    model_threshold = atof(argv[4]);
  }
  std::string model_id_name = argv[1];
  if (model_id_mapping.find(model_id_name) == model_id_mapping.end()) {
    printf("model_id_name not found\n");
    return -1;
  }
  TDL_MODEL_TYPE model_id = model_id_mapping[model_id_name];
  std::string model_path = argv[2];
  std::string image_path = argv[3];

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }
  TDLModelFactory model_factory;
  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }
  model->setModelThreshold(model_threshold);
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    if (out_datas[i]->getType() != ModelOutputType::OBJECT_DETECTION) {
      printf("out_datas[%d] is not ModelOutputType::OBJECT_DETECTION\n", i);
      continue;
    }
    std::shared_ptr<ModelBoxInfo> obj_meta =
        std::static_pointer_cast<ModelBoxInfo>(out_datas[i]);
    for (int i = 0; i < obj_meta->bboxes.size(); i++) {
      std::cout << "obj_meta_index: " << i << "  "
                << "class: " << obj_meta->bboxes[i].class_id << "  "
                << "score: " << obj_meta->bboxes[i].score << "  "
                << "bbox: " << obj_meta->bboxes[i].x1 << " "
                << obj_meta->bboxes[i].y1 << " " << obj_meta->bboxes[i].x2
                << " " << obj_meta->bboxes[i].y2 << std::endl;
    }
    std::string str_img_name = "object_detection_" + std::to_string(i) + ".jpg";
    visualize_object_detection(image, obj_meta, str_img_name);
  }

  return 0;
}
