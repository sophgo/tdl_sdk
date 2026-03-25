#include "tdl_model_factory.hpp"

void visualize_motion_detection(
    std::shared_ptr<BaseImage> image,
    std::shared_ptr<ModelBoxSegmentationInfo> box_meta,
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
  for (size_t i = 0; i < box_meta->box_seg.size(); i++) {
    cv::rectangle(
        mat,
        cv::Rect(int(box_meta->box_seg[i].x1), int(box_meta->box_seg[i].y1),
                 int(box_meta->box_seg[i].x2 - box_meta->box_seg[i].x1),
                 int(box_meta->box_seg[i].y2 - box_meta->box_seg[i].y1)),
        cv::Scalar(255, 255, 255), 2);
  }
  cv::imwrite(str_img_name, mat);
}

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    printf("Usage: %s <model_dir> <image1_path> <image2_path> [image3_path]\n",
           argv[0]);
    return -1;
  }
  std::string model_dir = argv[1];
  int num_images = argc - 2;
  std::vector<std::string> image_paths;
  for (int i = 0; i < num_images; ++i) {
    image_paths.push_back(argv[2 + i]);
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_seg =
      model_factory.getModel(ModelType::TOPFORMER_SEG_MOTION);
  if (!model_seg) {
    printf("Failed to create model_seg\n");
    return -1;
  }

  int count = 1;  // 34 to calculate time

  for (int b = 0; b < count; ++b) {
    for (int i = 0; i < num_images; ++i) {
      std::shared_ptr<BaseImage> image =
          ImageFactory::readImage(image_paths[i], ImageFormat::GRAY);
      if (!image) {
        printf("Failed to create image%d\n", i + 1);
        return -1;
      } else {
        printf("image%d readed,width:%d,height:%d\n", i + 1, image->getWidth(),
               image->getHeight());
      }

      std::shared_ptr<ModelOutputInfo> out_data =
          std::make_shared<ModelBoxSegmentationInfo>();
      model_seg->inference(image, out_data);

      if (out_data != nullptr) {
        std::shared_ptr<ModelBoxSegmentationInfo> box_meta =
            std::static_pointer_cast<ModelBoxSegmentationInfo>(out_data);

        if (box_meta->box_seg.size() == 0) {
          printf("frame %d: No motion detected\n", i + 1);
        } else {
          for (size_t j = 0; j < box_meta->box_seg.size(); j++) {
            printf("frame %d: box_%ld= [%f, %f, %f, %f]\n", i + 1, j,
                   box_meta->box_seg[j].x1, box_meta->box_seg[j].y1,
                   box_meta->box_seg[j].x2, box_meta->box_seg[j].y2);
          }
        }
        char sz_img_name[128];
        sprintf(sz_img_name, "seg_motion_%d.jpg", i + 1);
        if (count == 1) {
          visualize_motion_detection(image, box_meta, sz_img_name);
        }
      } else {
        printf("out_datas.size() is 0, frame %d\n", i + 1);
      }
    }
  }

  return 0;
}
