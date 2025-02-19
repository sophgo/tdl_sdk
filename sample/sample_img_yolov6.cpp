#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "face/cvtdl_face_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_path> <image_path>\n", argv[0]);
    return -1;
  }
  std::string model_path = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }
  TDLModelFactory model_factory;
  TDL_MODEL_TYPE model_id = TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV6;

  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  cv::Mat src_mat = cv::imread(image_path);
  cv::imwrite("src_mat_in.jpg", src_mat);
  cv::Mat temp_resized = cv::Mat::zeros(640, 640, CV_8UC3);
  cv::Rect roi(0, 140, 640, 360);
  cv::resize(src_mat, temp_resized, cv::Size(640, 640), 0, 0,
             cv::INTER_NEAREST);
  cv::imwrite("temp_resized_out.jpg", temp_resized);

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    cvtdl_object_t *obj_meta = (cvtdl_object_t *)out_datas[i];
    for (int i = 0; i < obj_meta->size; i++) {
      std::cout << "obj_meta_index: " << i << "  "
                << "class: " << obj_meta->info[i].classes << "  "
                << "score: " << obj_meta->info[i].bbox.score << "  "
                << "bbox: " << obj_meta->info[i].bbox.x1 << " "
                << obj_meta->info[i].bbox.y1 << " " << obj_meta->info[i].bbox.x2
                << " " << obj_meta->info[i].bbox.y2 << std::endl;
    }
  }
  model_factory.releaseOutput(model_id, out_datas);
  return 0;
}