#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "face/cvtdl_face_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"

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
      model_factory.getModel(TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  std::vector<std::vector<float>> landmarks;
  for (size_t i = 0; i < out_datas.size(); i++) {
    std::vector<float> landmark;
    cvtdl_face_info_t *face_meta = (cvtdl_face_info_t *)out_datas[i];
    printf("face_meta size: %d,score:%f,blurness:%f\n", face_meta->pts.size,
           face_meta->pts.score, face_meta->blurness);
    for (size_t j = 0; j < face_meta->pts.size; j++) {
      printf("face_meta pts: %f, %f\n", face_meta->pts.x[j],
             face_meta->pts.y[j]);
      landmark.push_back(face_meta->pts.x[j]);
      landmark.push_back(face_meta->pts.y[j]);
    }
    landmarks.push_back(landmark);
  }

  model_factory.releaseOutput(TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2,
                              out_datas);

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