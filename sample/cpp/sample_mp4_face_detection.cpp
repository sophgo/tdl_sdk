#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>

#include "opencv2/opencv.hpp"

#include "tdl_model_factory.hpp"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

// Helper to create directory recursively
bool create_directories(const std::string& path) {
  return fs::create_directories(path);
}

// Face visualization and drawing on frame
void draw_face_detection(cv::Mat& mat,
                         std::shared_ptr<ModelBoxLandmarkInfo> face_meta) {
  if (!face_meta) return;

  for (size_t i = 0; i < face_meta->box_landmarks.size(); i++) {
    // Draw bounding box
    cv::rectangle(mat,
                  cv::Rect(int(face_meta->box_landmarks[i].x1),
                           int(face_meta->box_landmarks[i].y1),
                           int(face_meta->box_landmarks[i].x2 -
                               face_meta->box_landmarks[i].x1),
                           int(face_meta->box_landmarks[i].y2 -
                               face_meta->box_landmarks[i].y1)),
                  cv::Scalar(0, 0, 255), 2);

    // Draw landmarks
    for (size_t j = 0; j < face_meta->box_landmarks[i].landmarks_x.size();
         j++) {
      cv::circle(mat,
                 cv::Point(int(face_meta->box_landmarks[i].landmarks_x[j]),
                           int(face_meta->box_landmarks[i].landmarks_y[j])),
                 2, cv::Scalar(0, 255, 0), -1);
    }

    // Draw score
    // std::string score_str =
    // std::to_string(face_meta->box_landmarks[i].score).substr(0, 4);
    // cv::putText(mat, score_str,
    //             cv::Point(int(face_meta->box_landmarks[i].x1),
    //             int(face_meta->box_landmarks[i].y1) - 5),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }
}

void process_video(const std::string& input_file,
                   const std::string& output_file,
                   std::shared_ptr<BaseModel> model) {
  std::cout << "Processing: " << input_file << " -> " << output_file
            << std::endl;

  // 使用完整的OpenCV命名空间
  cv::VideoCapture cap(input_file);
  if (!cap.isOpened()) {
    std::cerr << "Error opening video file: " << input_file << std::endl;
    return;
  }

  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  // Try to create video writer
  bool use_video_writer = true;
  cv::VideoWriter writer;

  // Check if we can write video (fallback to images if not supported or fails)
  try {
    writer.open(output_file, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                cv::Size(1920, 1080));
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

  // If fallback, create directory for frames
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

    // Convert cv::Mat to BaseImage for inference
    std::shared_ptr<BaseImage> image =
        ImageFactory::convertFromMat(frame, false);

    if (image) {
      std::vector<std::shared_ptr<BaseImage>> input_images = {image};
      std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;

      model->inference(input_images, out_datas);

      if (!out_datas.empty()) {
        std::shared_ptr<ModelBoxLandmarkInfo> face_meta =
            std::static_pointer_cast<ModelBoxLandmarkInfo>(out_datas[0]);
        draw_face_detection(frame, face_meta);
      }
    }

    if (use_video_writer) {
      cv::Mat resize_frame;
      // 将帧缩放到1920x1080
      cv::resize(frame, resize_frame, cv::Size(1920, 1080));
      writer.write(resize_frame);
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

  // Initialize Model
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_fd =
      model_factory.getModel(ModelType::SCRFD_DET_FACE);

  model_fd->setModelThreshold(0.3);

  if (!model_fd) {
    std::cerr << "Failed to load SCRFD_DET_FACE model from " << model_dir
              << std::endl;
    return -1;
  }

  // Traverse directory
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

        if (base_str.back() != '/') base_str += "/";

        std::string rel_path_str;
        if (full_path.find(base_str) == 0) {
          rel_path_str = full_path.substr(base_str.length());
        } else {
          rel_path_str = entry_path.filename().string();
        }

        std::string out_path = (fs::path(output_dir) / rel_path_str).string();
        std::string out_parent = fs::path(out_path).parent_path().string();

        if (!create_directories(out_parent)) {
          // Ignore
        }

        process_video(entry.path().string(), out_path, model_fd);
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
