#ifndef _CVI_FACE_UTILS_H_
#define _CVI_FACE_UTILS_H_

#include "core/face/cvai_face_types.h"
#include "cvi_comm_video.h"
#include "opencv2/opencv.hpp"

namespace cviai {
cvai_face_info_t bbox_rescale(const float width, const float height, const float new_width,
                              const float new_height, const cvai_face_info_t &face_info);
cvai_face_info_t bbox_rescale(const float new_width, const float new_height,
                              const cvai_face_t &face_meta, const int face_idx);
int face_align(const cv::Mat &image, cv::Mat &aligned, const cvai_face_info_t &face_info);
int face_align_gdc(const VIDEO_FRAME_INFO_S *inFrame, VIDEO_FRAME_INFO_S *outFrame,
                   const cvai_face_info_t &face_info);
}  // namespace cviai
#endif
