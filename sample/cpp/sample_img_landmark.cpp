
#include "tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_dir> <image_path>\n", argv[0]);
    return -1;
  }
  std::string model_dir = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }
  TDLModelFactory model_factory(model_dir);
  std::shared_ptr<BaseModel> model =
      model_factory.getModel(ModelType::IMG_KEYPOINT_FACE_V2);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  std::vector<std::vector<float>> landmarks;
  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_datas[i]);
    std::vector<float> landmark;
    printf("face_meta size: %d,score:%f,blurness:%f\n",
           landmarks_meta->landmarks_x.size(),
           landmarks_meta
               ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE],
           landmarks_meta->attributes
               [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER]);
    for (size_t j = 0; j < landmarks_meta->landmarks_x.size(); j++) {
      printf("face_meta pts: %f, %f\n", landmarks_meta->landmarks_x[j],
             landmarks_meta->landmarks_y[j]);
      landmark.push_back(landmarks_meta->landmarks_x[j]);
      landmark.push_back(landmarks_meta->landmarks_y[j]);
    }
    landmarks.push_back(landmark);
  }

  cv::Mat mat_img;
  bool is_rgb = false;
  int32_t ret = ImageFactory::convertToMat(image, mat_img, is_rgb);
  if (ret != 0) {
    printf("Failed to convert to mat\n");
    return -1;
  }
  if (is_rgb) {
    cv::cvtColor(mat_img, mat_img, cv::COLOR_RGB2BGR);
  }
  for (size_t i = 0; i < landmarks.size(); i++) {
    for (size_t j = 0; j < landmarks[i].size(); j += 2) {
      printf("landmark: %f, %f\n", landmarks[i][j], landmarks[i][j + 1]);
      cv::circle(mat_img, cv::Point(landmarks[i][j], landmarks[i][j + 1]), 2,
                 cv::Scalar(0, 0, 255), -1);
    }
  }
  cv::imwrite("landmark.jpg", mat_img);

  return 0;
}