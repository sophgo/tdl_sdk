#include "md.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include "core/core/cvai_errno.h"
#include "core/utils/vpss_helper.h"
#include "cviai_log.hpp"
#include "error_msg.hpp"
#include "vpss_engine.hpp"

static CVI_S32 VideoFrameCopy2Image(IVE_HANDLE ive_handle, VIDEO_FRAME_INFO_S *src,
                                    IVE_IMAGE_S *dst) {
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
    LOGE("CVI_IVE_VideoFrameInfo2Image fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }
  IVE_DMA_CTRL_S ctrl;
  ctrl.enMode = IVE_DMA_MODE_DIRECT_COPY;
  ret = CVI_IVE_DMA(ive_handle, &input_image, dst, &ctrl, false);

  if (do_unmap) {
    CVI_SYS_Munmap((void *)src->stVFrame.pu8VirAddr[0], image_size);
  }

  CVI_SYS_FreeI(ive_handle, &input_image);

  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_DMA fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  return CVIAI_SUCCESS;
}

MotionDetection::MotionDetection(IVE_HANDLE handle, uint32_t th, double _min_area, uint32_t timeout,
                                 cviai::VpssEngine *engine)
    : ive_handle(handle),
      count(0),
      threshold(th),
      min_area(_min_area),
      m_vpss_engine(engine),
      m_vpss_timeout(timeout) {}

CVI_S32 MotionDetection::init(VIDEO_FRAME_INFO_S *init_frame) {
  im_width = init_frame->stVFrame.u32Width;
  im_height = init_frame->stVFrame.u32Height;
  uint32_t voWidth = init_frame->stVFrame.u32Width;
  uint32_t voHeight = init_frame->stVFrame.u32Height;
  CVI_IVE_CreateImage(ive_handle, &tmp, IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(ive_handle, &bk_dst, IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(ive_handle, &src[0], IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(ive_handle, &src[1], IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);

  return copy_image(init_frame, &src[0]);
}

MotionDetection::~MotionDetection() {
  CVI_SYS_FreeI(ive_handle, &bk_dst);
  CVI_SYS_FreeI(ive_handle, &tmp);
  CVI_SYS_FreeI(ive_handle, &src[0]);
  CVI_SYS_FreeI(ive_handle, &src[1]);
}

CVI_S32 MotionDetection::vpss_process(VIDEO_FRAME_INFO_S *srcframe, VIDEO_FRAME_INFO_S *dstframe) {
  VPSS_CHN_ATTR_S chnAttr;
  VPSS_CHN_DEFAULT_HELPER(&chnAttr, srcframe->stVFrame.u32Width, srcframe->stVFrame.u32Height,
                          PIXEL_FORMAT_YUV_400, true);
  CVI_S32 ret = m_vpss_engine->sendFrame(srcframe, &chnAttr, 1);
  if (ret != CVI_SUCCESS) {
    LOGE("Failed to send vpss frame: %s\n", cviai::get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_SEND_FRAME;
  }
  ret = m_vpss_engine->getFrame(dstframe, 0, m_vpss_timeout);
  if (ret != CVI_SUCCESS) {
    LOGE("Failed to get vpss frame: %s\n", cviai::get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_SEND_FRAME;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 MotionDetection::update_background(VIDEO_FRAME_INFO_S *frame) {
  return copy_image(frame, &src[0]);
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

CVI_S32 MotionDetection::copy_image(VIDEO_FRAME_INFO_S *srcframe, IVE_IMAGE_S *dst) {
  std::shared_ptr<VIDEO_FRAME_INFO_S> frame;
  CVI_S32 ret = CVIAI_SUCCESS;
  if (srcframe->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_400) {
    frame =
        std::shared_ptr<VIDEO_FRAME_INFO_S>(new VIDEO_FRAME_INFO_S, [this](VIDEO_FRAME_INFO_S *f) {
          this->m_vpss_engine->releaseFrame(f, 0);
          delete f;
        });

    ret = vpss_process(srcframe, frame.get());
  } else {
    frame = std::shared_ptr<VIDEO_FRAME_INFO_S>(srcframe, [](VIDEO_FRAME_INFO_S *) {});
  }

  if (ret == CVIAI_SUCCESS) {
    return VideoFrameCopy2Image(ive_handle, frame.get(), dst);
  }
  return ret;
}

CVI_S32 MotionDetection::detect(VIDEO_FRAME_INFO_S *srcframe, cvai_object_t *obj_meta) {
  static int c = 0;

  CVI_S32 ret = copy_image(srcframe, &src[1]);
  if (ret != CVI_SUCCESS) {
    LOGE("Failed to copy frame to IVE image\n");
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  if (count > 2) {
    // Sub - threshold - dilate
    IVE_SUB_CTRL_S iveSubCtrl;
    iveSubCtrl.enMode = IVE_SUB_MODE_ABS;
    ret = CVI_IVE_Sub(ive_handle, &src[1], &src[0], &tmp, &iveSubCtrl, 0);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_IVE_Sub fail %x\n", ret);
      return CVIAI_ERR_MD_OPERATION_FAILED;
    }

    IVE_THRESH_CTRL_S iveTshCtrl;
    iveTshCtrl.enMode = IVE_THRESH_MODE_BINARY;
    iveTshCtrl.u8MinVal = 0;
    iveTshCtrl.u8MaxVal = 255;
    iveTshCtrl.u8LowThr = threshold;
    ret = CVI_IVE_Thresh(ive_handle, &tmp, &tmp, &iveTshCtrl, 0);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_IVE_Sub fail %x\n", ret);
      return CVIAI_ERR_MD_OPERATION_FAILED;
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
  return CVIAI_SUCCESS;
}