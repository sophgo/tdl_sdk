#include "liveness.hpp"
#include "template_matching.hpp"

#include "core/cviai_types_free.h"
#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define RESIZE_SIZE 112
#define LIVENESS_SCALE (1 / 255.0)
#define LIVENESS_N 1
#define LIVENESS_C 6
#define LIVENESS_WIDTH 32
#define LIVENESS_HEIGHT 32
#define CROP_NUM 9
#define MIN_FACE_WIDTH 25
#define MIN_FACE_HEIGHT 25
#define OUTPUT_NAME "fc2_dequant"

using namespace std;

namespace cviai {

static vector<vector<cv::Mat>> image_preprocess(VIDEO_FRAME_INFO_S *frame,
                                                VIDEO_FRAME_INFO_S *sink_buffer, cvai_face_t *meta,
                                                cvai_liveness_ir_position_e ir_pos) {
  cv::Mat rgb_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3);
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)frame->stVFrame.pu8VirAddr[0];
  for (int i = 0; i < rgb_frame.rows; i++) {
    memcpy(rgb_frame.ptr(i, 0), va_rgb + frame->stVFrame.u32Stride[0] * i, rgb_frame.cols * 3);
  }
  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  frame->stVFrame.pu8VirAddr[0] = NULL;

  if (rgb_frame.data == nullptr) {
    printf("src Image is empty!\n");
    return vector<vector<cv::Mat>>{};
  }

  cv::Mat ir_frame(sink_buffer->stVFrame.u32Height, sink_buffer->stVFrame.u32Width, CV_8UC3);
  sink_buffer->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(sink_buffer->stVFrame.u64PhyAddr[0],
                                                               sink_buffer->stVFrame.u32Length[0]);
  va_rgb = (char *)sink_buffer->stVFrame.pu8VirAddr[0];
  for (int i = 0; i < ir_frame.rows; i++) {
    memcpy(ir_frame.ptr(i, 0), va_rgb + sink_buffer->stVFrame.u32Stride[0] * i, ir_frame.cols * 3);
  }
  CVI_SYS_Munmap((void *)sink_buffer->stVFrame.pu8VirAddr[0], sink_buffer->stVFrame.u32Length[0]);

  if (ir_frame.data == nullptr) {
    printf("sink Image is empty!\n");
    return vector<vector<cv::Mat>>{};
  }

  vector<vector<cv::Mat>> input_mat(meta->size, vector<cv::Mat>());
  for (uint32_t i = 0; i < meta->size; i++) {
    cvai_face_info_t face_info =
        bbox_rescale(frame->stVFrame.u32Width, frame->stVFrame.u32Height, meta, i);
    cv::Rect box;
    box.x = face_info.bbox.x1;
    box.y = face_info.bbox.y1;
    box.width = face_info.bbox.x2 - box.x;
    box.height = face_info.bbox.y2 - box.y;
    CVI_AI_FreeCpp(&face_info);

    if (box.width <= MIN_FACE_WIDTH || box.height <= MIN_FACE_HEIGHT) continue;
    cv::Mat crop_rgb_frame = rgb_frame(box);
    cv::Mat crop_ir_frame = template_matching(crop_rgb_frame, ir_frame, box, ir_pos);

    cv::Mat color, ir;
    cv::resize(crop_rgb_frame, color, cv::Size(RESIZE_SIZE, RESIZE_SIZE));
    cv::resize(crop_ir_frame, ir, cv::Size(RESIZE_SIZE, RESIZE_SIZE));

    vector<cv::Mat> colors = TTA_9_cropps(color);
    vector<cv::Mat> irs = TTA_9_cropps(ir);

    vector<cv::Mat> input_v;
    for (size_t i = 0; i < colors.size(); i++) {
      cv::Mat temp;
      cv::merge(vector<cv::Mat>{colors[i], irs[i]}, temp);
      input_v.push_back(temp);
    }
    input_mat[i] = input_v;
  }

  return input_mat;
}

Liveness::Liveness(cvai_liveness_ir_position_e ir_position) {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->batch_size = 9;

  m_ir_pos = ir_position;
}

int Liveness::inference(VIDEO_FRAME_INFO_S *rgbFrame, VIDEO_FRAME_INFO_S *irFrame,
                        cvai_face_t *meta) {
  if (meta->size <= 0) {
    cout << "meta->size <= 0" << endl;
    return CVI_FAILURE;
  }

  vector<vector<cv::Mat>> input_mats = image_preprocess(rgbFrame, irFrame, meta, m_ir_pos);
  if (input_mats.empty()) {
    cout << "input_mat.empty" << endl;
    return CVI_FAILURE;
  }

  for (uint32_t i = 0; i < meta->size; i++) {
    float conf0 = 0.0;
    float conf1 = 0.0;

    vector<cv::Mat> input = input_mats[i];
    if (input.empty()) continue;

    prepareInputTensor(input);

    run(rgbFrame);

    CVI_TENSOR *out = CVI_NN_GetTensorByName(OUTPUT_NAME, mp_output_tensors, m_output_num);
    float *out_data = (float *)CVI_NN_TensorPtr(out);
    for (int j = 0; j < CROP_NUM; j++) {
      conf0 += out_data[j * 2];
      conf1 += out_data[(j * 2) + 1];
    }

    conf0 /= input.size();
    conf1 /= input.size();

    float max = std::max(conf0, conf1);
    float f0 = std::exp(conf0 - max);
    float f1 = std::exp(conf1 - max);
    float score = f1 / (f0 + f1);

    meta->info[i].liveness_score = score;
    // cout << "Face[" << i << "] liveness score: " << score << endl;
  }

  return CVI_SUCCESS;
}

void Liveness::prepareInputTensor(vector<cv::Mat> &input_mat) {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  float *input_ptr = (float *)CVI_NN_TensorPtr(input);

  for (int j = 0; j < CROP_NUM; j++) {
    cv::Mat tmpchannels[LIVENESS_C];
    cv::split(input_mat[j], tmpchannels);

    for (int c = 0; c < LIVENESS_C; ++c) {
      tmpchannels[c].convertTo(tmpchannels[c], CV_32F, LIVENESS_SCALE, 0);

      int size = tmpchannels[c].rows * tmpchannels[c].cols;
      for (int r = 0; r < tmpchannels[c].rows; ++r) {
        memcpy(input_ptr + size * c + tmpchannels[c].cols * r, tmpchannels[c].ptr(r, 0),
               tmpchannels[c].cols * sizeof(float));
      }
    }
    input_ptr += CVI_NN_TensorCount(input) / CROP_NUM;
  }
}

}  // namespace cviai