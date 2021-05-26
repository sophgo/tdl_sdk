#include "md.hpp"
#include <opencv2/opencv.hpp>
#include "core/utils/vpss_helper.h"
#include "cviai_log.hpp"

CVI_S32 VideoFrameCopy2Image(IVE_HANDLE ive_handle, VIDEO_FRAME_INFO_S *src, IVE_IMAGE_S *dst) {
  IVE_IMAGE_S input_image;
  bool do_unmap = false;
  size_t image_size =
      src->stVFrame.u32Length[0] + src->stVFrame.u32Length[1] + src->stVFrame.u32Length[2];
  if (src->stVFrame.pu8VirAddr[0] == NULL) {
    src->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(src->stVFrame.u64PhyAddr[0], image_size);
    do_unmap = true;
  }

  CVI_S32 ret = CVI_IVE_VideoFrameInfo2Image(src, &input_image);
  if (ret != CVI_SUCCESS) {
    printf("CVI_IVE_VideoFrameInfo2Image fail %x\n", ret);
    return ret;
  }
  IVE_DMA_CTRL_S ctrl;
  ctrl.enMode = IVE_DMA_MODE_DIRECT_COPY;
  ret = CVI_IVE_DMA(ive_handle, &input_image, dst, &ctrl, false);

  if (do_unmap) {
    CVI_SYS_Munmap((void *)src->stVFrame.pu8VirAddr[0], image_size);
  }

  CVI_SYS_FreeI(ive_handle, &input_image);

  if (ret != CVI_SUCCESS) {
    printf("CVI_IVE_DMA fail %x\n", ret);
    return ret;
  }

  return CVI_SUCCESS;
}

MotionDetection::MotionDetection(IVE_HANDLE handle, VIDEO_FRAME_INFO_S *init_frame, uint32_t th,
                                 double _min_area)
    : ive_handle(handle),
      count(0),
      threshold(th),
      min_area(_min_area),
      im_width(init_frame->stVFrame.u32Width),
      im_height(init_frame->stVFrame.u32Height) {
  uint32_t voWidth = init_frame->stVFrame.u32Width;
  uint32_t voHeight = init_frame->stVFrame.u32Height;
  CREATE_VBFRAME_HELPER(&blk[0], &vbsrc[0], voWidth, voHeight, PIXEL_FORMAT_YUV_400);

  CREATE_VBFRAME_HELPER(&blk[1], &vbsrc[1], voWidth, voHeight, PIXEL_FORMAT_YUV_400);

  CVI_IVE_VideoFrameInfo2Image(&vbsrc[0], &src[0]);
  CVI_IVE_VideoFrameInfo2Image(&vbsrc[1], &src[1]);
  CVI_IVE_CreateImage(ive_handle, &tmp, IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(ive_handle, &andframe[0], IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(ive_handle, &andframe[1], IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(ive_handle, &bk_dst, IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);

  VideoFrameCopy2Image(handle, init_frame, &src[0]);
}

MotionDetection::~MotionDetection() {
  CVI_VB_ReleaseBlock(blk[0]);
  CVI_VB_ReleaseBlock(blk[1]);
  CVI_SYS_FreeI(ive_handle, &tmp);
  CVI_SYS_FreeI(ive_handle, &andframe[0]);
  CVI_SYS_FreeI(ive_handle, &andframe[1]);
  CVI_SYS_FreeI(ive_handle, &bk_dst);
}

CVI_S32 MotionDetection::update_background(VIDEO_FRAME_INFO_S *frame) {
  return VideoFrameCopy2Image(ive_handle, frame, &src[0]);
}

void MotionDetection::construct_bbox(std::vector<cv::Rect> dets, cvai_object_t *out) {
  out->size = dets.size();
  out->info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * out->size);
  out->height = im_height;
  out->width = im_width;
  out->rescale_type = RESCALE_RB;

  memset(out->info, 0, sizeof(cvai_object_info_t) * out->size);
  for (uint32_t i = 0; i < out->size; ++i) {
    out->info[i].bbox.x1 = dets[i].x;
    out->info[i].bbox.y1 = dets[i].y;
    out->info[i].bbox.x2 = dets[i].x + dets[i].width;
    out->info[i].bbox.y2 = dets[i].y + dets[i].height;
    out->info[i].bbox.score = 0;
    out->info[i].classes = -1;
    memset(out->info[i].name, 0, sizeof(out->info[i].name));
  }
}

CVI_S32 MotionDetection::detect(VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj_meta) {
  static int c = 0;

  CVI_S32 ret = VideoFrameCopy2Image(ive_handle, frame, &src[1]);
  if (ret != CVI_SUCCESS) {
    return ret;
  }

  if (count > 2) {
    // Sub - threshold - dilate
    IVE_SUB_CTRL_S iveSubCtrl;
    iveSubCtrl.enMode = IVE_SUB_MODE_ABS;
    ret = CVI_IVE_Sub(ive_handle, &src[1], &src[0], &tmp, &iveSubCtrl, 0);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_IVE_Sub fail %x\n", ret);
      return ret;
    }

    IVE_THRESH_CTRL_S iveTshCtrl;
    iveTshCtrl.enMode = IVE_THRESH_MODE_BINARY;
    iveTshCtrl.u8MinVal = 0;
    iveTshCtrl.u8MaxVal = 255;
    iveTshCtrl.u8LowThr = threshold;
    ret = CVI_IVE_Thresh(ive_handle, &tmp, &tmp, &iveTshCtrl, 0);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_IVE_Sub fail %x\n", ret);
      return ret;
    }

    IVE_DILATE_CTRL_S stDilateCtrl;
    CVI_U8 arr[] = {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
    memcpy(stDilateCtrl.au8Mask, arr, 25 * sizeof(CVI_U8));
    CVI_IVE_Dilate(ive_handle, &tmp, &bk_dst, &stDilateCtrl, 0);

    CVI_IVE_BufRequest(ive_handle, &bk_dst);

    VIDEO_FRAME_INFO_S bk_frame;
    CVI_IVE_Image2VideoFrameInfo(&bk_dst, &bk_frame, false);

    int img_width = bk_frame.stVFrame.u32Width;
    int img_height = bk_frame.stVFrame.u32Height;

    bool do_unmap = false;
    if (bk_frame.stVFrame.pu8VirAddr[0] == NULL) {
      bk_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(bk_frame.stVFrame.u64PhyAddr[0],
                                                                    bk_frame.stVFrame.u32Length[0]);
      do_unmap = true;
    }

    cv::Mat image(img_height, img_width, CV_8UC1, bk_frame.stVFrame.pu8VirAddr[0],
                  bk_frame.stVFrame.u32Stride[0]);

    c++;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> bboxes;
    for (auto c : contours) {
      double area = cv::contourArea(c);
      if (area > min_area) {
        bboxes.push_back(cv::boundingRect(c));
      }
    }

    construct_bbox(bboxes, obj_meta);
    if (do_unmap) {
      CVI_SYS_Munmap((void *)bk_frame.stVFrame.pu8VirAddr[0], bk_frame.stVFrame.u32Length[0]);
      bk_frame.stVFrame.pu8VirAddr[0] = NULL;
      bk_frame.stVFrame.pu8VirAddr[1] = NULL;
      bk_frame.stVFrame.pu8VirAddr[2] = NULL;
    }
  } else {
    obj_meta->size = 0;
    obj_meta->info = nullptr;
    obj_meta->height = im_height;
    obj_meta->width = im_width;
    obj_meta->rescale_type = RESCALE_RB;
  }

  // update_interval = 10;
  // if (count == 0) {
  //   IVE_DMA_CTRL_S ctrl;
  //   ctrl.enMode = IVE_DMA_MODE_DIRECT_COPY;
  //   ret = CVI_IVE_DMA(ive_handle, &src[1], &src[0], &ctrl, false);
  // } else if (update_interval > 0 && ((count % (uint32_t)update_interval) == 0)) {
  //   float ax = 0.95;
  //   float by = 0.05;
  //   CVI_IVE_BufRequest(this->ive_handle, &src[0]);
  //   CVI_IVE_BufRequest(this->ive_handle, &src[1]);
  //   CVI_U16 strideWidth = src[1].u16Stride[0];
  //   for (int c = 0; c < 1; c++) {
  //     for (uint32_t i = 0; i < im_height; i++) {
  //       for (uint32_t j = 0; j < im_width; j++) {
  //         src[0].pu8VirAddr[c][i * strideWidth + j] = (int)(ax * src[0].pu8VirAddr[c][i *
  //         strideWidth + j] + by * (src[1].pu8VirAddr[c][i * strideWidth + j]));
  //       }
  //     }
  //   }
  //   CVI_IVE_BufFlush(ive_handle, &src[0]);
  // }

  count++;
  return CVI_SUCCESS;
}