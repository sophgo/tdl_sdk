
#include "tdl_model_factory.hpp"

void visualize_lane_detection(
    std::shared_ptr<BaseImage> image,
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta,
    const std::string &save_path) {
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

  for (int i = 0; i < obj_meta->box_landmarks.size(); i++) {
    int x0 = (int)obj_meta->box_landmarks[i].landmarks_x[0];
    int y0 = (int)obj_meta->box_landmarks[i].landmarks_y[0];
    int x1 = (int)obj_meta->box_landmarks[i].landmarks_x[1];
    int y1 = (int)obj_meta->box_landmarks[i].landmarks_y[1];

    cv::line(mat, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 3);
  }

  cv::imwrite(save_path, mat);

}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_path> <image_path> \n", argv[0]);
    return -1;
  }
  std::string model_path = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  TDLModelFactory model_factory;

  std::shared_ptr<BaseModel> model_od =
      model_factory.getModel(ModelType::LANE_DETECTION_LSTR, model_path);
  if (!model_od) {
    printf("Failed to create model_od\n");
    return -1;
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_od->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_datas[i]);

    uint32_t image_width = input_images[i]->getWidth();
    uint32_t image_height = input_images[i]->getHeight();
  
    if (obj_meta->box_landmarks.size() == 0) {
      printf("No object detected\n");
    } else {
      for (size_t j = 0; j < obj_meta->box_landmarks.size(); j++) {
        printf("lane %d\n",j);
        for (int k = 0; k < 2; k++) {
          printf("%d: %f %f\n", k, obj_meta->box_landmarks[j].landmarks_x[k],
                 obj_meta->box_landmarks[j].landmarks_y[k]);
        }
      }
    }
    visualize_lane_detection(image1, obj_meta,"lstr_lane_detection.jpg");
  }

  return 0;
}

