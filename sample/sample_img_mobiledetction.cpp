#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "face/cvtdl_face_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"

void visualize_obj_detection(std::shared_ptr<BaseImage> image,
                              cvtdl_object_t *obj_meta,
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
  for (size_t i = 0; i < obj_meta->size; i++) {
    cvtdl_object_info_t *obj_info = &obj_meta->info[i];
    cv::rectangle(mat,
                  cv::Rect(int(obj_info->bbox.x1), int(obj_info->bbox.y1),
                           int(obj_info->bbox.x2 - obj_info->bbox.x1),
                           int(obj_info->bbox.y2 - obj_info->bbox.y1)),
                  cv::Scalar(0, 0, 255), 2);
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
  std::string model_path = model_dir + "/mobiledetv2-pedestrian-d0-448_cv186x.bmodel";
  TDLModelFactory model_factory(model_dir);
  std::shared_ptr<BaseModel> model_od =
      model_factory.getModel(TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN, model_path);
  if (!model_od) {
    printf("Failed to create model_od\n");
    return -1;
  }

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_od->inference(input_images, out_datas);

  std::vector<cvtdl_object_t *> obj_metas;
  for (size_t i = 0; i < out_datas.size(); i++) {
    cvtdl_object_t *obj_meta = (cvtdl_object_t *)out_datas[i];

    if (obj_meta->size == 0) {
      printf("No object detected\n");

    } else {
      for (size_t j = 0; j < obj_meta->size; j++) {
        cvtdl_object_info_t *obj_info = &obj_meta->info[j];
        printf("obj_%d,box= [%f, %f, %f, %f],score= %f\n", j,
               obj_info->bbox.x1, obj_info->bbox.y1, obj_info->bbox.x2,
               obj_info->bbox.y2, obj_info->bbox.score);
      }
    }
    obj_metas.push_back(obj_meta);
  }

  visualize_obj_detection(image1, obj_metas[0], "obj_detection.jpg");
  model_factory.releaseOutput(TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN, out_datas);
  return 0;
}
