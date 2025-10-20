
#include <sys/stat.h>
#include <unistd.h>
#include "tdl_model_factory.hpp"

void visualize_object_detection(std::shared_ptr<BaseImage> image,
                                std::shared_ptr<ModelBoxInfo> obj_meta,
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
  std::cout << "object_meta->bboxes.size(): " << obj_meta->bboxes.size()
            << std::endl;
  for (size_t i = 0; i < obj_meta->bboxes.size(); i++) {
    cv::Rect rect(int(obj_meta->bboxes[i].x1), int(obj_meta->bboxes[i].y1),
                  int(obj_meta->bboxes[i].x2 - obj_meta->bboxes[i].x1),
                  int(obj_meta->bboxes[i].y2 - obj_meta->bboxes[i].y1));
    cv::rectangle(mat, rect, cv::Scalar(0, 0, 255), 2);
    char sz_text[128];
    sprintf(sz_text, "%d:%.2f", obj_meta->bboxes[i].class_id,
            obj_meta->bboxes[i].score);
    int center_x = (rect.x + rect.x + rect.width) / 2;
    int center_y = (rect.y + rect.y + rect.height) / 2;
    cv::putText(mat, sz_text, cv::Point(center_x, center_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
  }
  std::cout << "save image to " << str_img_name << std::endl;

  cv::imwrite(str_img_name, mat);
}

bool path_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool is_directory(const std::string &path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) {
    return false;
  }
  return S_ISDIR(buffer.st_mode);
}

bool is_regular_file(const std::string &path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) {
    return false;
  }
  return S_ISREG(buffer.st_mode);
}

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5 && argc != 6) {
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_path> "
        "<model_threshold>\n",
        argv[0]);
    printf(
        "Usage: %s <model_id_name> <model_dir> <image_path> "
        "<model_threshold> <use_runtime_memory>\n",
        argv[0]);
    printf("Usage: %s <model_id_name> <model_dir> <image_path>\n", argv[0]);
    printf("model_id_name:\n");
    for (auto &item : kAllModelTypes) {
      printf("%s\n", modelTypeToString(item).c_str());
    }
    return -1;
  }
  float model_threshold = 0.5;
  if (argc == 5) {
    model_threshold = atof(argv[4]);
  }
  int use_runtime_memory = 0;
  if (argc == 6) {
    use_runtime_memory = std::atoi(argv[5]);
  }
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_path = argv[3];

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }
  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model = nullptr;
  std::vector<std::unique_ptr<MemoryBlock>> mem_blocks;
  std::shared_ptr<BaseMemoryPool> pool = MemoryPoolFactory::createMemoryPool();
  if (use_runtime_memory) {
    std::vector<uint64_t> mem_addrs;
    std::vector<uint32_t> mem_sizes;
    std::string model_path =
        model_factory.getModelPath(modelTypeFromString(model_id_name));
    NetFactory::getModelMemInfo(model_path, mem_addrs, mem_sizes);

    mem_addrs.clear();
    for (uint32_t i = 0; i < mem_sizes.size(); i++) {
      if (mem_sizes[i] == 0) {
        mem_addrs.push_back(0);
        continue;
      }
      std::unique_ptr<MemoryBlock> mem_block = pool->allocate(mem_sizes[i]);
      mem_addrs.push_back(mem_block->physicalAddress);
      mem_blocks.push_back(std::move(mem_block));
    }
    model = TDLModelFactory::getInstance().getModel(
        modelTypeFromString(model_id_name), model_path, mem_addrs, mem_sizes);
  } else {
    model = model_factory.getModel(modelTypeFromString(model_id_name));
  }

  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }
  model->setModelThreshold(model_threshold);
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);
  // for (int i = 0; i < 1001; i++) {
  //   out_datas.clear();
  //   model->inference(input_images, out_datas);
  // }

  std::cout << "out_datas.size: " << out_datas.size() << std::endl;

  for (size_t i = 0; i < out_datas.size(); i++) {
    if (out_datas[i]->getType() != ModelOutputType::OBJECT_DETECTION) {
      printf("out_datas[%ld] is not ModelOutputType::OBJECT_DETECTION\n", i);
      continue;
    }
    std::shared_ptr<ModelBoxInfo> obj_meta =
        std::static_pointer_cast<ModelBoxInfo>(out_datas[i]);
    for (size_t i = 0; i < obj_meta->bboxes.size(); i++) {
      std::cout << "obj_meta_index: " << i << "  "
                << "class: " << obj_meta->bboxes[i].class_id << "  "
                << "score: " << obj_meta->bboxes[i].score << "  "
                << "bbox: " << obj_meta->bboxes[i].x1 << " "
                << obj_meta->bboxes[i].y1 << " " << obj_meta->bboxes[i].x2
                << " " << obj_meta->bboxes[i].y2 << std::endl;
    }
    std::string str_img_name = "object_detection_" + std::to_string(i) + ".jpg";
    visualize_object_detection(image, obj_meta, str_img_name);
  }

  if (use_runtime_memory) {
    for (uint32_t i = 0; i < mem_blocks.size(); i++) {
      pool->release(mem_blocks[i]);
    }
  }

  return 0;
}
