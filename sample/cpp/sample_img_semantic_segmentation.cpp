
#include "tdl_model_factory.hpp"

void visualize_mask(std::shared_ptr<ModelSegmentationInfo> seg_meta,
                    const std::string &str_img_name) {
  uint32_t output_width = seg_meta->output_width;
  uint32_t output_height = seg_meta->output_height;

  cv::Mat src(output_height, output_width, CV_8UC1, seg_meta->class_id,
              output_width * sizeof(uint8_t));
  cv::Mat dst;
  src.convertTo(dst, CV_8U, 50);  // x 50 to visualize
  cv::imwrite(str_img_name, dst);
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

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_seg =
      model_factory.getModel(ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE);
  if (!model_seg) {
    printf("Failed to create model_seg\n");
    return -1;
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_seg->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelSegmentationInfo> seg_meta =
        std::static_pointer_cast<ModelSegmentationInfo>(out_datas[i]);

    visualize_mask(seg_meta, "topfoemer_seg_mask.png");

    for (uint32_t x = 0; x < seg_meta->output_height; ++x) {
      for (uint32_t y = 0; y < seg_meta->output_width; ++y) {
        printf("%d ", (int)seg_meta->class_id[x * seg_meta->output_width + y]);
      }
      printf("\n");
    }
  }

  return 0;
}
