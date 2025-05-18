
#include "tdl_model_factory.hpp"

void visualize_face_detection(std::shared_ptr<BaseImage> image,
                              std::shared_ptr<ModelBoxLandmarkInfo> face_meta,
                              const std::string &str_img_name) {
  cv::Mat mat;
  bool is_rgb;
  int32_t ret = ImageFactory::convertToMat(image, mat, is_rgb);
  if (ret != 0) {
    printf("Failed to convert to mat\n");
    return;
  }
  if (is_rgb) {
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  }
  for (size_t i = 0; i < face_meta->box_landmarks.size(); i++) {
    cv::rectangle(mat,
                  cv::Rect(int(face_meta->box_landmarks[i].x1),
                           int(face_meta->box_landmarks[i].y1),
                           int(face_meta->box_landmarks[i].x2 -
                               face_meta->box_landmarks[i].x1),
                           int(face_meta->box_landmarks[i].y2 -
                               face_meta->box_landmarks[i].y1)),
                  cv::Scalar(0, 0, 255), 2);
    for (int j = 0; j < face_meta->box_landmarks[i].landmarks_x.size(); j++) {
      cv::circle(mat,
                 cv::Point(int(face_meta->box_landmarks[i].landmarks_x[j]),
                           int(face_meta->box_landmarks[i].landmarks_y[j])),
                 3, cv::Scalar(0, 0, 255), -1);
    }
  }
  cv::imwrite(str_img_name, mat);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_dir> <image_path> \n", argv[0]);
    return -1;
  }
  std::string model_dir = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  } else {
    printf("image readed,width:%d,height:%d\n", image1->getWidth(),
           image1->getHeight());
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

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_fd->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelBoxLandmarkInfo> face_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_datas[i]);

    if (face_meta->box_landmarks.size() == 0) {
      printf("No face detected\n");

    } else {
      for (size_t j = 0; j < face_meta->box_landmarks.size(); j++) {
        printf("face_%d,box= [%f, %f, %f, %f],score= %f\n", j,
               face_meta->box_landmarks[j].x1, face_meta->box_landmarks[j].y1,
               face_meta->box_landmarks[j].x2, face_meta->box_landmarks[j].y2,
               face_meta->box_landmarks[j].score);
      }
    }
    char sz_img_name[128];
    sprintf(sz_img_name, "face_detection_%d.jpg", i);
    visualize_face_detection(image1, face_meta, sz_img_name);
  }

  return 0;
}
