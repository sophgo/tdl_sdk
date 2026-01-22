#include <iostream>
#include <opencv2/opencv.hpp>
#include "common/common_types.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
#include "utils/detection_helper.hpp"

void printPreprocessParams(const PreprocessParams& p) {
  std::cout << "PreprocessParams:\n";
  std::cout << "dst_image_format: " << static_cast<int>(p.dst_image_format)
            << "\n";
  std::cout << "dst_pixdata_type: " << static_cast<int>(p.dst_pixdata_type)
            << "\n";
  std::cout << "  dst_width:        " << p.dst_width << "\n";
  std::cout << "  dst_height:       " << p.dst_height << "\n";
  std::cout << "  crop_x:           " << p.crop_x << "\n";
  std::cout << "  crop_y:           " << p.crop_y << "\n";
  std::cout << "  crop_width:       " << p.crop_width << "\n";
  std::cout << "  crop_height:      " << p.crop_height << "\n";
  std::cout << "  mean:             [" << p.mean[0] << ", " << p.mean[1] << ", "
            << p.mean[2] << "]\n";
  std::cout << "  scale:            [" << p.scale[0] << ", " << p.scale[1]
            << ", " << p.scale[2] << "]\n";
  std::cout << "  keep_aspect_ratio:"
            << (p.keep_aspect_ratio ? " true" : " false") << "\n";
}
int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s <moded_id> <model_dir> <image_path>\n", argv[0]);
    return -1;
  }
  std::string moded_id = argv[1];
  std::string model_dir = argv[2];

  std::string strf = argv[3];

  int frame_size = 0;

  std::vector<uint8_t> buffer;
  if (!CommonUtils::readBinaryFile(strf, buffer)) {
    printf("read file failed\n");
    return -1;
  }
  frame_size = buffer.size();
  printf("frame_size:%d\n", frame_size);

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  // 方式2：使用 skip_input_alloc = true，在 addInput 时不分配输入内存
  // 1. 获取未打开的模型实例
  std::shared_ptr<BaseModel> model_od =
      model_factory.getModelWithoutOpen(moded_id);
  if (model_od == nullptr) {
    printf("Failed to get model\n");
    return -1;
  }

  // 2. 获取 NetParam 并设置 skip_input_alloc = true
  NetParam& net_param = model_od->getNetParam();
  net_param.skip_input_alloc = true;  // 不在 addInput 时分配内存
  printf("skip_input_alloc set to true\n");

  // 3. 打开模型（此时 addInput 不会分配输入 tensor 内存）
  int ret = model_od->modelOpen();
  if (ret != 0) {
    printf("Failed to open model\n");
    return -1;
  }

  PreprocessParams preprocess_params;
  ret = model_od->getPreprocessParameters(preprocess_params);
  if (ret != 0) {
    printf("Failed to get preprocess parameters\n");
    return -1;
  }

  // 创建 TENSOR_FRAME 类型的图像，会申请内存
  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      preprocess_params.dst_width, preprocess_params.dst_height,
      ImageFormat::RGB_PLANAR, preprocess_params.dst_pixdata_type, true,
      InferencePlatform::UNKOWN, ImageType::TENSOR_FRAME);

  // 将 image 的内存设置给输入 tensor
  // 由于 skip_input_alloc = true，tensor 没有自己的内存，直接使用 image 的内存
  // ret = model_od->setInputTensorFromImage("", bin_data);
  if (ret != 0) {
    printf("Failed to set input tensor from image\n");
    return -1;
  }
  printf(
      "Successfully set input tensor from image, no duplicate memory "
      "allocation\n");

  std::vector<uint8_t*> dst_ptrs = bin_data->getVirtualAddress();
  std::vector<uint32_t> strides = bin_data->getStrides();
  uint32_t plane_num = bin_data->getPlaneNum();
  uint32_t w = bin_data->getWidth();
  uint32_t h = bin_data->getHeight();
  uint32_t element_bytes =
      CommonUtils::getDataTypeSize(bin_data->getPixDataType());
  uint32_t plane_size = w * h * element_bytes;

  // buffer 中的数据是连续存储的（RGB_PLANAR 格式，3个 plane 连续）
  const uint8_t* src_buffer = buffer.data();
  uint32_t src_offset = 0;

  for (uint32_t i = 0; i < plane_num; i++) {
    uint8_t* dst_ptr = dst_ptrs[i];
    uint32_t img_stride_i = strides[i];

    if (img_stride_i == w * element_bytes) {
      // stride 匹配，直接拷贝整个 plane
      memcpy(dst_ptr, src_buffer + src_offset, plane_size);
      src_offset += plane_size;
    } else {
      // stride 不匹配，需要逐行拷贝
      for (uint32_t j = 0; j < h; j++) {
        uint8_t* dst_row_ptr = dst_ptr + j * img_stride_i;
        const uint8_t* src_row_ptr =
            src_buffer + src_offset + j * w * element_bytes;
        memcpy(dst_row_ptr, src_row_ptr, w * element_bytes);
      }
      src_offset += plane_size;
    }
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {bin_data};
  model_od->inference(input_images, out_datas);

  int input_width = 329;
  int input_height = 494;
  std::vector<float> rescale_params = DetectionHelper::getRescaleConfig(
      preprocess_params, input_width, input_height);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(out_datas[i]);
    uint32_t image_width = input_images[i]->getWidth();
    uint32_t image_height = input_images[i]->getHeight();
    if (obj_meta->box_landmarks.size() == 0) {
      printf("No object detected\n");
    } else {
      for (size_t j = 0; j < obj_meta->box_landmarks.size(); j++) {
        DetectionHelper::rescaleBbox(obj_meta->box_landmarks[j],
                                     rescale_params);
        std::cout << "obj_meta_index: " << j << "  "
                  << "class: " << obj_meta->box_landmarks[j].class_id << "  "
                  << "score: " << obj_meta->box_landmarks[j].score << "  "
                  << "bbox: " << obj_meta->box_landmarks[j].x1 << " "
                  << obj_meta->box_landmarks[j].y1 << " "
                  << obj_meta->box_landmarks[j].x2 << " "
                  << obj_meta->box_landmarks[j].y2 << std::endl;
        for (int k = 0; k < obj_meta->box_landmarks[j].landmarks_score.size();
             k++) {
          printf("%d: %f %f %f\n", k, obj_meta->box_landmarks[j].landmarks_x[k],
                 obj_meta->box_landmarks[j].landmarks_y[k],
                 obj_meta->box_landmarks[j].landmarks_score[k]);
        }
      }
    }
    // visualize_keypoints_detection(image1, out_datas[i], 0.5,
    //                               "yolov8_keypoints.jpg");
  }

  return 0;
}