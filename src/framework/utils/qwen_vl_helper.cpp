#include "utils/qwen_vl_helper.hpp"

// -----------------------------------------------------------------------
// 常量定义
// -----------------------------------------------------------------------
const int IMAGE_FACTOR = 28;
const int MIN_PIXELS = 4 * 28 * 28;
const int MAX_PIXELS = 16384 * 28 * 28;
const int MAX_RATIO = 200;

const int VIDEO_MIN_PIXELS = 128 * 28 * 28;
const int VIDEO_MAX_PIXELS = 768 * 28 * 28;
const int FRAME_FACTOR = 2;
const double FPS = 2.0;
const int FPS_MIN_FRAMES = 4;
const int FPS_MAX_FRAMES = 768;
const int VIDEO_TOTAL_PIXELS = int(128000 * 28 * 28 * 0.9);

int round_by_factor(int number, int factor) {
  return int(std::round(double(number) / factor)) * factor;
}

int ceil_by_factor(int number, int factor) {
  return int(std::ceil(double(number) / factor)) * factor;
}

int floor_by_factor(int number, int factor) {
  return int(std::floor(double(number) / factor)) * factor;
}

cv::Size smart_resize(int height, int width, int factor, int min_pixels,
                      int max_pixels) {
  if (double(std::max(height, width)) / std::min(height, width) > MAX_RATIO) {
    throw std::runtime_error("absolute aspect ratio exceeds limit");
  }
  int h_bar = std::max(factor, round_by_factor(height, factor));
  int w_bar = std::max(factor, round_by_factor(width, factor));
  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt(double(height * width) / max_pixels);
    h_bar = floor_by_factor(int(height / beta), factor);
    w_bar = floor_by_factor(int(width / beta), factor);
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(double(min_pixels) / (height * width));
    h_bar = ceil_by_factor(int(height * beta), factor);
    w_bar = ceil_by_factor(int(width * beta), factor);
  }
  return cv::Size(w_bar, h_bar);
}

int smart_nframes(int total_frames, double video_fps, int desired_nframes,
                  double desired_fps, int min_frames = FPS_MIN_FRAMES,
                  int max_frames = FPS_MAX_FRAMES) {
  int nframes = 0;
  if (desired_nframes > 0) {
    nframes = round_by_factor(desired_nframes, FRAME_FACTOR);
  } else {
    nframes = int(total_frames / video_fps * desired_fps);
    nframes = std::min(std::max(nframes, min_frames),
                       std::min(max_frames, total_frames));
    nframes = floor_by_factor(nframes, FRAME_FACTOR);
  }
  if (nframes < FRAME_FACTOR || nframes > total_frames) {
    throw std::runtime_error("nframes is out of valid range");
  }
  return nframes;
}

QwenVLHelper::QwenVLHelper() {}

QwenVLHelper::~QwenVLHelper() {}

std::vector<cv::Mat> QwenVLHelper::fetchImage(
    const std::string &image_path, const std::map<std::string, int> &args) {
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    throw std::runtime_error("cannot open image: " + image_path);
  }
  int size_factor = IMAGE_FACTOR;
  int min_pixels = MIN_PIXELS;
  int max_pixels = MAX_PIXELS;
  int resized_height = img.rows;
  int resized_width = img.cols;
  if (args.find("resized_height") != args.end() &&
      args.find("resized_width") != args.end()) {
    resized_height = args.at("resized_height");
    resized_width = args.at("resized_width");
  } else if (args.find("min_pixels") != args.end() &&
             args.find("max_pixels") != args.end()) {
    min_pixels = args.at("min_pixels");
    max_pixels = args.at("max_pixels");
    cv::Size new_size =
        smart_resize(img.rows, img.cols, size_factor, min_pixels, max_pixels);
    resized_height = new_size.height;
    resized_width = new_size.width;
  }
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resized_width, resized_height), 0, 0,
             cv::INTER_CUBIC);
  std::vector<cv::Mat> bgr_frames;
  cv::split(resized, bgr_frames);
  return bgr_frames;
}
std::map<std::string, float> QwenVLHelper::testFetchVideoTs(
    const std::string &video_path, double desired_fps, int desired_nframes,
    int max_video_sec) {
  int size_factor = IMAGE_FACTOR;
  int min_pixels = VIDEO_MIN_PIXELS;
  int total_pixels = VIDEO_TOTAL_PIXELS;
  if (desired_nframes > 0 && desired_fps > 0) {
    printf(
        "desired_nframes:%d,desired_fps:%f,only one of desired_nframes and "
        "desired_fps should be set\n",
        desired_nframes, desired_fps);
    return {};
  }
  double tick_video_open = cv::getTickCount();
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("cannot open video: " + video_path);
  }
  double tick_video_open_end = cv::getTickCount();

  int total_frames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int img_width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int img_height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double video_fps = cap.get(cv::CAP_PROP_FPS);

  int nframes =
      smart_nframes(total_frames, video_fps, desired_nframes, desired_fps);

  printf(
      "total_frames: %d, video_fps: %f, img_width: %d, img_height: "
      "%d,desired_fps:%f,desired_nframes:%d,sample_nframes:%d\n",
      total_frames, video_fps, img_width, img_height, desired_fps,
      desired_nframes, nframes);

  std::vector<int> indices;
  for (int i = 0; i < nframes; i++) {
    int idx = int(std::round(double(i) / (nframes - 1) * (total_frames - 1)));
    indices.push_back(idx);
  }

  int max_process_frames = max_video_sec * video_fps;
  std::vector<std::vector<cv::Mat>> frames;  // std::vector<cv::Mat> = <b,g,r>
  int current_frame = 0;
  int target_idx = 0;
  cv::Mat frame;

  int last_frame = indices[indices.size() - 1];

  double tick_process_total = 0;
  double tick_read_start = cv::getTickCount();
  while (cap.read(frame)) {
    if (current_frame == indices[target_idx]) {
      double tick_process = cv::getTickCount();
      cv::Size new_size =
          smart_resize(frame.rows, frame.cols, size_factor, min_pixels,
                       total_pixels / std::max(nframes, 1));
      cv::Mat resized;
      cv::resize(frame, resized, new_size, 0, 0, cv::INTER_CUBIC);
      std::vector<cv::Mat> bgr_frames;
      cv::split(resized, bgr_frames);
      frames.push_back(bgr_frames);
      tick_process_total += cv::getTickCount() - tick_process;
      target_idx++;

      if (target_idx >= static_cast<int>(indices.size())) break;
      if (desired_nframes > 0 && current_frame >= desired_nframes) break;
    }
    current_frame++;
    if (current_frame > last_frame) {
      break;
    }
    if (max_process_frames > 0 && current_frame >= max_process_frames) {
      break;
    }
  }
  double tick_read_end = cv::getTickCount();
  cap.release();
  double tick_video_close = cv::getTickCount();
  double duration_video_open =
      (tick_video_open_end - tick_video_open) * 1000 / cv::getTickFrequency();
  double duration_read =
      (tick_read_end - tick_read_start) * 1000 / cv::getTickFrequency();
  double duration_process = tick_process_total * 1000 / cv::getTickFrequency();
  double duration_video_close =
      (tick_video_close - tick_read_end) * 1000 / cv::getTickFrequency();

  std::map<std::string, float> result;
  result["duration_video_open"] = duration_video_open;
  result["duration_read"] = duration_read;
  result["duration_process"] = duration_process;
  result["duration_video_close"] = duration_video_close;
  result["read_frames"] = current_frame;
  result["process_frames"] = target_idx;
  return result;
}

std::vector<std::vector<cv::Mat>> QwenVLHelper::fetchVideo(
    const std::string &video_path, double desired_fps, int desired_nframes,
    int max_video_sec) {
  int size_factor = IMAGE_FACTOR;
  int min_pixels = VIDEO_MIN_PIXELS;
  int total_pixels = VIDEO_TOTAL_PIXELS;
  if (desired_nframes > 0 && desired_fps > 0) {
    printf(
        "desired_nframes:%d,desired_fps:%f,only one of desired_nframes and "
        "desired_fps should be set\n",
        desired_nframes, desired_fps);
    return {};
  }
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("cannot open video: " + video_path);
  }
  int total_frames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int img_width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int img_height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double video_fps = cap.get(cv::CAP_PROP_FPS);

  int nframes =
      smart_nframes(total_frames, video_fps, desired_nframes, desired_fps);

  printf(
      "total_frames: %d, video_fps: %f, img_width: %d, img_height: "
      "%d,desired_fps:%f,desired_nframes:%d,sample_nframes:%d\n",
      total_frames, video_fps, img_width, img_height, desired_fps,
      desired_nframes, nframes);

  std::vector<int> indices;
  for (int i = 0; i < nframes; i++) {
    int idx = int(std::round(double(i) / (nframes - 1) * (total_frames - 1)));
    indices.push_back(idx);
  }

  int max_process_frames = max_video_sec * video_fps;
  std::vector<std::vector<cv::Mat>> frames;  // std::vector<cv::Mat> = <b,g,r>
  int current_frame = 0;
  int target_idx = 0;
  cv::Mat frame;

  int last_frame = indices[indices.size() - 1];

  double tick_process_total = 0;
  double tick_read_start = cv::getTickCount();
  while (cap.read(frame)) {
    if (current_frame == indices[target_idx]) {
      double tick_process = cv::getTickCount();
      cv::Size new_size =
          smart_resize(frame.rows, frame.cols, size_factor, min_pixels,
                       total_pixels / std::max(nframes, 1));
      cv::Mat resized;
      cv::resize(frame, resized, new_size, 0, 0, cv::INTER_CUBIC);
      std::vector<cv::Mat> bgr_frames;
      cv::split(resized, bgr_frames);
      frames.push_back(bgr_frames);
      tick_process_total += cv::getTickCount() - tick_process;
      target_idx++;

      if (target_idx >= static_cast<int>(indices.size())) break;
      if (desired_nframes > 0 && current_frame >= desired_nframes) break;
    }
    current_frame++;
    if (current_frame > last_frame) {
      break;
    }
    if (max_process_frames > 0 && current_frame >= max_process_frames) {
      break;
    }
  }

  cap.release();
  return frames;
}
