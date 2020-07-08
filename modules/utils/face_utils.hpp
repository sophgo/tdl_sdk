#ifndef _CVI_FACE_UTILS_H_
#define _CVI_FACE_UTILS_H_

#include "cvi_comm_video.h"
#include "face/cvi_face_types.h"
#include "opencv2/opencv.hpp"

namespace cviai {
cvi_face_info_t bbox_rescale(VIDEO_FRAME_INFO_S *frame, cvi_face_t *face_meta, int face_idx);
int face_align(const cv::Mat &image, cv::Mat &aligned, const cvi_face_info_t &face_info, int width,
               int height);
}  // namespace cviai
#endif
