#include "ocr_recognition.hpp"
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem.h"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core_utils.hpp"
#include "cvi_sys.h"

#include "core/utils/vpss_helper.h"

#include <fstream>
#include <iostream>
#include <sstream>
#ifdef ENABLE_CVI_TDL_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif
#define R_SCALE (0.003922)
#define G_SCALE (0.003922)
#define B_SCALE (0.003922)
#define R_MEAN (0)
#define G_MEAN (0)
#define B_MEAN (0)

namespace cvitdl {

OCRRecognition::OCRRecognition() : Core(CVI_MEM_DEVICE) {}

OCRRecognition::~OCRRecognition() {}

int OCRRecognition::setupInputPreprocess(std::vector<InputPreprecessSetup>* data) {
  if (data->size() != 1) {
    LOGE("OCRRecognition only has 1 input.\n");
    return CVI_TDL_ERR_INVALID_ARGS;
  }
  (*data)[0].factor[0] = R_SCALE;
  (*data)[0].factor[1] = G_SCALE;
  (*data)[0].factor[2] = B_SCALE;
  (*data)[0].mean[0] = R_MEAN;
  (*data)[0].mean[1] = G_MEAN;
  (*data)[0].mean[2] = B_MEAN;
  (*data)[0].use_quantize_scale = true;
  return CVI_TDL_SUCCESS;
}

void draw_and_save_picture(VIDEO_FRAME_INFO_S* bg, const std::string& save_path) {
  CVI_U8* r_plane = (CVI_U8*)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);
  CVI_U8* g_plane = (CVI_U8*)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[1], bg->stVFrame.u32Length[1]);
  CVI_U8* b_plane = (CVI_U8*)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[2], bg->stVFrame.u32Length[2]);

  cv::Mat r_mat(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC1, r_plane);
  cv::Mat g_mat(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC1, g_plane);
  cv::Mat b_mat(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC1, b_plane);

  std::vector<cv::Mat> channels = {b_mat, g_mat, r_mat};
  cv::Mat img_rgb;
  cv::merge(channels, img_rgb);

  cv::imwrite(save_path, img_rgb);

  CVI_SYS_Munmap(r_plane, bg->stVFrame.u32Length[0]);
  CVI_SYS_Munmap(g_plane, bg->stVFrame.u32Length[1]);
  CVI_SYS_Munmap(b_plane, bg->stVFrame.u32Length[2]);
}

int argmax(float* start, float* end) {
  float* max_iter = std::max_element(start, end);
  int max_idx = max_iter - start;
  return max_idx;
}

void read_char(std::vector<std::string>& chars) {
  std::string filePath = "/tmp/zhiling/180x/ppocr_keys_v1.txt";
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open character map file.");
  }

  chars.clear();
  std::string line;
  while (std::getline(file, line)) {
    chars.push_back(line);
  }
  std::string request = "Finish read txt!!!";
  std::cout << request << std::endl;
}

std::pair<std::string, float> OCRRecognition::decode(const std::vector<int>& text_index,
                                                     const std::vector<float>& text_prob,
                                                     std::vector<std::string>& chars,
                                                     bool is_remove_duplicate) {
  std::string text;
  std::vector<float> conf_list;

  for (size_t idx = 0; idx < text_index.size(); ++idx) {
    if (is_remove_duplicate && idx > 0 && text_index[idx - 1] == text_index[idx]) {
      continue;
    }
    text += chars[text_index[idx]];
    conf_list.push_back(text_prob[idx]);
  }

  float average_conf =
      conf_list.empty()
          ? 0.0f
          : std::accumulate(conf_list.begin(), conf_list.end(), 0.0f) / conf_list.size();
  return {text, average_conf};
}

void OCRRecognition::greedy_decode(float* prebs, std::vector<std::string>& chars) {
  CVI_SHAPE output_shape = getOutputShape(0);
  int outShapeC = output_shape.dim[1];
  int outHeight = output_shape.dim[2];
  int outWidth = output_shape.dim[3];

  std::vector<int> argmax_results(outShapeC, 0);
  std::vector<float> confidences(outShapeC, 0.0f);

  for (int t = 0; t < outShapeC; ++t) {
    float* start = prebs + t * outWidth * outHeight;
    float* end = prebs + (t + 1) * outWidth * outHeight;
    int argmax_idx = argmax(start, end);
    argmax_results[t] = argmax_idx;
    confidences[t] = *(start + argmax_idx);
  }
  auto result = decode(argmax_results, confidences, chars, true);
  std::cout << "Decoded Text: " << result.first << ", Average Confidence: " << result.second
            << std::endl;
}

void saveToCSV(float** data, int numRows, int numCols, const std::string& filename) {
  std::ofstream outputFile(filename);
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      outputFile << data[i][j];

      if (j != numCols - 1) {
        outputFile << ",";
      }
    }
    outputFile << std::endl;
  }
  outputFile.close();
}

int OCRRecognition::inference(VIDEO_FRAME_INFO_S* frame, cvtdl_object_t* obj_meta) {
  if (obj_meta->size == 0) {
    return CVI_TDL_SUCCESS;
  }

  std::vector<std::string> myCharacters;
  read_char(myCharacters);

  for (uint32_t i = 0; i < obj_meta->size; ++i) {
    cvtdl_object_info_t obj_info = info_extern_crop_resize_img(
        frame->stVFrame.u32Width, frame->stVFrame.u32Height, &(obj_meta->info[i]));
    VIDEO_FRAME_INFO_S* cropped_frame = new VIDEO_FRAME_INFO_S;
    memset(cropped_frame, 0, sizeof(VIDEO_FRAME_INFO_S));
    CVI_SHAPE shape = getInputShape(0);
    int height = shape.dim[2];
    int width = shape.dim[3];
    vpssCropImage(frame, cropped_frame, obj_meta->info[i].bbox, width, height,
                  PIXEL_FORMAT_RGB_888_PLANAR);

    std::vector<VIDEO_FRAME_INFO_S*> frames = {cropped_frame};
    int ret = run(frames);
    if (ret != CVI_TDL_SUCCESS) {
      mp_vpss_inst->releaseFrame(cropped_frame, 0);
      delete cropped_frame;
      return ret;
    }

    float* out = getOutputRawPtr<float>(0);
    greedy_decode(out, myCharacters);
    mp_vpss_inst->releaseFrame(cropped_frame, 0);
    delete cropped_frame;
  }
  return CVI_TDL_SUCCESS;
}

}  // namespace cvitdl