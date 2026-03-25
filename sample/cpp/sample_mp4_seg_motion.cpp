#include <algorithm>
#include <cstring>
#include <experimental/filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "opencv2/opencv.hpp"

#include "tdl_model_factory.hpp"

namespace fs = std::experimental::filesystem;

static bool create_directories(const std::string& path) {
  return fs::create_directories(path);
}

static void draw_seg_motion(
    cv::Mat& mat, const std::shared_ptr<ModelBoxSegmentationInfo>& meta) {
  if (!meta) return;
  for (size_t i = 0; i < meta->box_seg.size(); i++) {
    cv::rectangle(mat,
                  cv::Rect(int(meta->box_seg[i].x1), int(meta->box_seg[i].y1),
                           int(meta->box_seg[i].x2 - meta->box_seg[i].x1),
                           int(meta->box_seg[i].y2 - meta->box_seg[i].y1)),
                  cv::Scalar(255, 255, 255), 2);
  }
}

static void process_video(const std::string& input_file,
                          const std::string& output_file,
                          const std::shared_ptr<BaseModel>& model) {
  std::cout << "Processing: " << input_file << " -> " << output_file
            << std::endl;

  cv::VideoCapture cap(input_file);
  if (!cap.isOpened()) {
    std::cerr << "Error opening video file: " << input_file << std::endl;
    return;
  }

  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  bool use_video_writer = true;
  cv::VideoWriter writer;
  try {
    writer.open(output_file, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                cv::Size(width, height));
    if (!writer.isOpened()) {
      std::cerr << "Warning: Failed to create video writer for " << output_file
                << ". Falling back to saving images." << std::endl;
      use_video_writer = false;
    }
  } catch (...) {
    std::cerr << "Exception creating video writer. Falling back to images."
              << std::endl;
    use_video_writer = false;
  }

  std::string output_frames_dir;
  if (!use_video_writer) {
    output_frames_dir = output_file + "_frames";
    create_directories(output_frames_dir);
    std::cout << "Output will be saved to directory: " << output_frames_dir
              << std::endl;
  }

  cv::Mat frame;
  int frame_count = 0;
  while (cap.read(frame)) {
    if (frame.empty()) break;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::shared_ptr<BaseImage> image = ImageFactory::createImage(
        gray.cols, gray.rows, ImageFormat::GRAY, TDLDataType::UINT8, false);
    if (image) {
      if (image->allocateMemory() != 0) {
        image = nullptr;
      }
    }
    if (image) {
      uint32_t stride = image->getStrides()[0];
      uint8_t* ptr_dst = image->getVirtualAddress()[0];
      uint8_t* ptr_src = gray.data;
      for (int r = 0; r < gray.rows; r++) {
        uint8_t* dst = ptr_dst + r * stride;
        std::memcpy(dst, ptr_src + r * gray.step[0], gray.cols);
      }
      image->flushCache();

      std::shared_ptr<ModelOutputInfo> out_data =
          std::make_shared<ModelBoxSegmentationInfo>();
      model->inference(image, out_data);

      if (out_data != nullptr &&
          out_data->getType() ==
              ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
        std::shared_ptr<ModelBoxSegmentationInfo> seg_meta =
            std::static_pointer_cast<ModelBoxSegmentationInfo>(out_data);
        draw_seg_motion(frame, seg_meta);
      }
    }

    if (use_video_writer) {
      writer.write(frame);
    } else {
      char frame_name[256];
      snprintf(frame_name, sizeof(frame_name), "%s/%06d.jpg",
               output_frames_dir.c_str(), frame_count);
      cv::imwrite(frame_name, frame);
    }

    frame_count++;
    if (frame_count % 30 == 0) {
      std::cout << "Processed " << frame_count << "/" << total_frames
                << " frames\r" << std::flush;
    }
  }
  std::cout << "Processed " << frame_count << "/" << total_frames
            << " frames. Done." << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <model_dir> <input_dir> <output_dir>"
              << std::endl;
    return -1;
  }

  std::string model_dir = argv[1];
  std::string input_dir = argv[2];
  std::string output_dir = argv[3];

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_seg =
      model_factory.getModel(ModelType::TOPFORMER_SEG_MOTION);
  if (!model_seg) {
    std::cerr << "Failed to load TOPFORMER_SEG_MOTION model from " << model_dir
              << std::endl;
    return -1;
  }

  try {
    if (!fs::exists(input_dir)) {
      std::cerr << "Input directory does not exist: " << input_dir << std::endl;
      return -1;
    }

    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
      if (fs::is_regular_file(entry) && entry.path().extension() == ".mp4") {
        fs::path entry_path = entry.path();
        fs::path base_path = input_dir;

        std::string full_path = entry_path.string();
        std::string base_str = base_path.string();
        if (!base_str.empty() && base_str.back() != '/') base_str += "/";

        std::string rel_path_str;
        if (full_path.find(base_str) == 0) {
          rel_path_str = full_path.substr(base_str.length());
        } else {
          rel_path_str = entry_path.filename().string();
        }

        std::string out_path = (fs::path(output_dir) / rel_path_str).string();
        std::string out_parent = fs::path(out_path).parent_path().string();
        create_directories(out_parent);

        process_video(entry.path().string(), out_path, model_seg);
      }
    }
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Filesystem error: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
