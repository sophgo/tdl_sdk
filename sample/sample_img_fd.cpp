#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "face/cvtdl_face_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"

void visualize_face_detection(std::shared_ptr<BaseImage> image,
                              cvtdl_face_t *face_meta,
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
  for (size_t i = 0; i < face_meta->size; i++) {
    cvtdl_face_info_t *face_info = &face_meta->info[i];
    cv::rectangle(mat,
                  cv::Rect(int(face_info->bbox.x1), int(face_info->bbox.y1),
                           int(face_info->bbox.x2 - face_info->bbox.x1),
                           int(face_info->bbox.y2 - face_info->bbox.y1)),
                  cv::Scalar(0, 0, 255), 2);
    for (int j = 0; j < face_info->pts.size; j++) {
      cv::circle(mat,
                 cv::Point(int(face_info->pts.x[j]), int(face_info->pts.y[j])),
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
  }

  TDLModelFactory model_factory(model_dir);
  std::shared_ptr<BaseModel> model_fd =
      model_factory.getModel(TDL_MODEL_TYPE_FACE_DETECTION_SCRFD);
  if (!model_fd) {
    printf("Failed to create model_fd\n");
    return -1;
  }

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_fd->inference(input_images, out_datas);

  std::vector<cvtdl_face_t *> face_metas;
  for (size_t i = 0; i < out_datas.size(); i++) {
    cvtdl_face_t *face_meta = (cvtdl_face_t *)out_datas[i];

    if (face_meta->size == 0) {
      printf("No face detected\n");

    } else {
      for (size_t j = 0; j < face_meta->size; j++) {
        cvtdl_face_info_t *face_info = &face_meta->info[j];
        printf("face_%d,box= [%f, %f, %f, %f],score= %f\n", j,
               face_info->bbox.x1, face_info->bbox.y1, face_info->bbox.x2,
               face_info->bbox.y2, face_info->bbox.score);
      }
    }
    face_metas.push_back(face_meta);
  }

  visualize_face_detection(image1, face_metas[0], "face_detection.jpg");
  model_factory.releaseOutput(TDL_MODEL_TYPE_FACE_DETECTION_SCRFD, out_datas);
  return 0;
}
