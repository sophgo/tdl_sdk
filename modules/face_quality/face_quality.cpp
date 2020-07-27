#include "face_quality.hpp"
#include "core_utils.hpp"
#include "cviai_types_free.h"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define SCALE_B (1.0 / (255.0 * 0.229))
#define SCALE_G (1.0 / (255.0 * 0.224))
#define SCALE_R (1.0 / (255.0 * 0.225))
#define MEAN_B -(0.485 / 0.229)
#define MEAN_G -(0.456 / 0.224)
#define MEAN_R -(0.406 / 0.225)
#define NAME_SCORE "score_Softmax"

#define PI 3.14159265358979f

namespace cviai {

static float saturate(const float &val, const float &minVal, const float &maxVal) {
  return std::min(std::max(val, minVal), maxVal);
}

static float cal_distance(const cv::Point &p1, const cv::Point &p2) {
  float x = p1.x - p2.x;
  float y = p1.y - p2.y;
  return sqrtf(x * x + y * y);
}

static float cal_angle(const cv::Point &pt1, const cv::Point &pt2) {
  return 360 - cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);
}

static float cal_slant(int ln, int lf, const float Rn, float theta) {
  float dz = 0;
  float slant = 0;
  const float m1 = ((float)ln * ln) / ((float)lf * lf);
  const float m2 = (cos(theta)) * (cos(theta));
  const float Rn_sq = Rn * Rn;

  if (m2 == 1) {
    dz = sqrt(Rn_sq / (m1 + Rn_sq));
  }
  if (m2 >= 0 && m2 < 1) {
    dz = sqrt(
        (Rn_sq - m1 - 2 * m2 * Rn_sq + sqrt(((m1 - Rn_sq) * (m1 - Rn_sq)) + 4 * m1 * m2 * Rn_sq)) /
        (2 * (1 - m2) * Rn_sq));
  }
  slant = acos(dz);
  return slant;
}

static int get_face_direction(cvai_pts_t face_pts, float &roll, float &pitch, float &yaw) {
  cv::Point leye = cv::Point(face_pts.x[0], face_pts.y[0]);
  cv::Point reye = cv::Point(face_pts.x[1], face_pts.y[1]);
  cv::Point noseTip = cv::Point(face_pts.x[2], face_pts.y[2]);
  cv::Point lmouth = cv::Point(face_pts.x[3], face_pts.y[3]);
  cv::Point rmouth = cv::Point(face_pts.x[4], face_pts.y[4]);
  cv::Point midEye = cv::Point((leye.x + reye.x) * 0.5, (leye.y + reye.y) * 0.5);
  cv::Point midMouth = cv::Point((lmouth.x + rmouth.x) * 0.5, (lmouth.y + rmouth.y) * 0.5);
  cv::Point noseBase = cv::Point((midMouth.x + midEye.x) * 0.5, (midMouth.y + midEye.y) * 0.5);

  float noseBase_noseTip_distance = cal_distance(noseTip, noseBase);
  float midEye_midMouth_distance = cal_distance(midEye, midMouth);
  float symm = cal_angle(noseBase, midEye);
  float tilt = cal_angle(noseBase, noseTip);
  float theta = (std::abs(tilt - symm)) * (PI / 180.0);
  float slant = cal_slant(noseBase_noseTip_distance, midEye_midMouth_distance, 0.5, theta);

  CvPoint3D32f normal;
  normal.x = sin(slant) * (cos((360 - tilt) * (PI / 180.0)));
  normal.y = sin(slant) * (sin((360 - tilt) * (PI / 180.0)));
  normal.z = -cos(slant);

  yaw = acos((std::abs(normal.z)) / (std::sqrt(normal.x * normal.x + normal.z * normal.z)));
  if (noseTip.x < noseBase.x) yaw = -yaw;
  yaw = saturate(yaw, -1.f, 1.f);

  pitch = acos(std::sqrt((normal.x * normal.x + normal.z * normal.z) /
                         (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)));
  if (noseTip.y > noseBase.y) pitch = -pitch;
  pitch = saturate(pitch, -1.f, 1.f);

  roll = cal_angle(leye, reye);
  if (roll > 180) roll = roll - 360;
  roll /= 90;
  roll = saturate(roll, -1.f, 1.f);

  return 0;
}

FaceQuality::FaceQuality() { mp_config = std::make_unique<ModelConfig>(); }

int FaceQuality::inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_BGR_888) {
    printf("Error: pixel format not match.\n");
    return CVI_FAILURE;
  }

  int img_width = frame->stVFrame.u32Width;
  int img_height = frame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_8UC3);
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)frame->stVFrame.pu8VirAddr[0];
  int dst_width = image.cols;
  int dst_height = image.rows;

  for (int i = 0; i < dst_height; i++) {
    memcpy(image.ptr(i, 0), va_rgb + frame->stVFrame.u32Stride[0] * i, dst_width * 3);
  }
  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);

  std::vector<float> mean = {MEAN_R, MEAN_G, MEAN_B};
  std::vector<float> scale = {SCALE_R, SCALE_G, SCALE_B};
  for (int i = 0; i < meta->size; i++) {
    cvai_face_info_t face_info =
        bbox_rescale(frame->stVFrame.u32Width, frame->stVFrame.u32Height, meta, i);
    cv::Mat crop_frame(input->shape.dim[2], input->shape.dim[3], image.type());
    face_align(image, crop_frame, face_info, input->shape.dim[3], input->shape.dim[2]);

    cv::Mat tmpchannels[3];
    cv::split(crop_frame, tmpchannels);

    for (int i = 0; i < 3; i++) {
      tmpchannels[i].convertTo(tmpchannels[i], CV_32F, scale[i], mean[i]);
      int size = tmpchannels[i].rows * tmpchannels[i].cols;
      for (int r = 0; r < tmpchannels[i].rows; ++r) {
        memcpy((float *)CVI_NN_TensorPtr(input) + size * i + tmpchannels[i].cols * r,
               tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
      }
    }

    run(frame);

    CVI_TENSOR *out = CVI_NN_GetTensorByName(NAME_SCORE, mp_output_tensors, m_output_num);
    float *score = (float *)CVI_NN_TensorPtr(out);
    meta->face_info[i].face_quality = score[1];

    float roll = 0, pitch = 0, yaw = 0;
    get_face_direction(face_info.face_pts, roll, pitch, yaw);
    if (abs(pitch) > 0.4 || abs(yaw) > 0.4) {
      meta->face_info[i].face_quality = 0;
    }

    CVI_AI_FreeCpp(&face_info);
  }

  return CVI_SUCCESS;
}

}  // namespace cviai
