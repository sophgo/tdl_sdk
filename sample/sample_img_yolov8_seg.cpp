
#include "tdl_model_factory.hpp"

void visualize_maskOutlinePoint(
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta, uint32_t image_height,
    uint32_t image_width) {
  int proto_h = obj_meta->mask_height;
  int proto_w = obj_meta->mask_width;

  for (uint32_t i = 0; i < obj_meta->box_seg.size(); i++) {
    cv::Mat src(proto_h, proto_w, CV_8UC1, obj_meta->box_seg[i].mask,
                proto_w * sizeof(uint8_t));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // search for contours
    cv::findContours(src, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
    // find the longest contour
    int longest_index = -1;
    size_t max_length = 0;
    for (size_t i = 0; i < contours.size(); i++) {
      if (contours[i].size() > max_length) {
        max_length = contours[i].size();
        longest_index = i;
      }
    }
    if (longest_index >= 0 && max_length >= 1) {
      float ratio_height = (proto_h / static_cast<float>(image_height));
      float ratio_width = (proto_w / static_cast<float>(image_width));
      int source_y_offset, source_x_offset;
      if (ratio_height > ratio_width) {
        source_x_offset = 0;
        source_y_offset = (proto_h - image_height * ratio_width) / 2;
      } else {
        source_x_offset = (proto_w - image_width * ratio_height) / 2;
        source_y_offset = 0;
      }
      int source_region_height = proto_h - 2 * source_y_offset;
      int source_region_width = proto_w - 2 * source_x_offset;
      // calculate scaling factor
      float height_scale = static_cast<float>(image_height) /
                           static_cast<float>(source_region_height);
      float width_scale = static_cast<float>(image_width) /
                          static_cast<float>(source_region_width);
      obj_meta->box_seg[i].mask_point_size = max_length;
      obj_meta->box_seg[i].mask_point =
          new float[2 * max_length * sizeof(float)];
      size_t j = 0;
      for (const auto &point : contours[longest_index]) {
        obj_meta->box_seg[i].mask_point[2 * j] =
            (point.x - source_x_offset) * width_scale;
        obj_meta->box_seg[i].mask_point[2 * j + 1] =
            (point.y - source_y_offset) * height_scale;
        j++;
      }
    }
  }
}

void visualize_object_detection(
    std::shared_ptr<BaseImage> image,
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta,
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

  for (size_t i = 0; i < obj_meta->box_seg.size(); i++) {
    cv::rectangle(
        mat,
        cv::Rect(int(obj_meta->box_seg[i].x1), int(obj_meta->box_seg[i].y1),
                 int(obj_meta->box_seg[i].x2 - obj_meta->box_seg[i].x1),
                 int(obj_meta->box_seg[i].y2 - obj_meta->box_seg[i].y1)),
        cv::Scalar(0, 0, 255), 2);
    char sz_text[128];
    sprintf(sz_text, "%d,%.2f", obj_meta->box_seg[i].class_id,
            obj_meta->box_seg[i].score);
    cv::putText(
        mat, sz_text,
        cv::Point(int(obj_meta->box_seg[i].x1), int(obj_meta->box_seg[i].y1)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
  }
  std::cout << "save image to " << str_img_name << std::endl;

  cv::imwrite(str_img_name, mat);
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

  std::shared_ptr<BaseModel> model_od = model_factory.getModel(
      TDL_MODEL_TYPE_INSTANCE_SEGMENTATION_YOLOV8, model_path);
  if (!model_od) {
    printf("Failed to create model_od\n");
    return -1;
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_od->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelBoxSegmentationInfo> obj_meta =
        std::static_pointer_cast<ModelBoxSegmentationInfo>(out_datas[i]);

    uint32_t image_width = input_images[i]->getWidth();
    uint32_t image_height = input_images[i]->getHeight();

    printf("Sample Image dimensions - height: %d, width: %d\n", image_height,
           image_width);
    visualize_maskOutlinePoint(obj_meta, image_height, image_width);
    if (obj_meta->box_seg.size() == 0) {
      printf("No object detected\n");
    } else {
      cv::Mat image = cv::imread(argv[2]);
      for (size_t j = 0; j < obj_meta->box_seg.size(); j++) {
        std::cout << "obj_meta_index: " << j << "  "
                  << "class: " << obj_meta->box_seg[j].class_id << "  "
                  << "score: " << obj_meta->box_seg[j].score << "  "
                  << "bbox: " << obj_meta->box_seg[j].x1 << " "
                  << obj_meta->box_seg[j].y1 << " " << obj_meta->box_seg[j].x2
                  << " " << obj_meta->box_seg[j].y2 << std::endl;
        printf("points=[");
        std::vector<cv::Point> points;
        for (uint32_t k = 0; k < obj_meta->box_seg[j].mask_point_size; k++) {
          printf("(%f,%f),", obj_meta->box_seg[j].mask_point[2 * k],
                 obj_meta->box_seg[j].mask_point[2 * k + 1]);
          points.push_back(cv::Point(
              static_cast<int>(obj_meta->box_seg[j].mask_point[2 * k]),
              static_cast<int>(obj_meta->box_seg[j].mask_point[2 * k + 1])));
        }
        printf("]\n");

        if (points.size() > 1) {
          cv::polylines(image, points, true, cv::Scalar(0, 255, 0), 2,
                        cv::LINE_AA);  // 绿色线条
        }
      }

      cv::imwrite("./yolov8_segmentation.jpg", image);
    }
    std::string str_img_name = "object_detection_" + std::to_string(i) + ".jpg";
    visualize_object_detection(image1, obj_meta, "yolov8_seg.jpg");
  }

  return 0;
}
