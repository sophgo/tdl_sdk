#include "md.hpp"
#include <memory>
#include "core/core/cvai_errno.h"
#include "core/utils/vpss_helper.h"
#include "cviai_log.hpp"
#include "error_msg.hpp"
#include "vpss_engine.hpp"

#ifdef ENABLE_CVIAI_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif

// #define DEBUG_MD

using namespace ive;

static CVI_S32 VideoFrameCopy2Image(IVE *ive_instance, VIDEO_FRAME_INFO_S *src, IVEImage *dst) {
  IVEImage input_image;
  bool do_unmap = false;
  size_t image_size =
      src->stVFrame.u32Length[0] + src->stVFrame.u32Length[1] + src->stVFrame.u32Length[2];
  if (src->stVFrame.pu8VirAddr[0] == NULL) {
    src->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(src->stVFrame.u64PhyAddr[0], image_size);
    do_unmap = true;
  }

  CVI_S32 ret = input_image.fromFrame(src);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_VideoFrameInfo2Image fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  ret = ive_instance->dma(&input_image, dst);

  if (do_unmap) {
    CVI_SYS_Munmap((void *)src->stVFrame.pu8VirAddr[0], image_size);
  }

  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_DMA fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  return CVIAI_SUCCESS;
}

MotionDetection::MotionDetection(IVE *_ive_instance, uint32_t th, double _min_area,
                                 uint32_t timeout, cviai::VpssEngine *engine)
    : ive_instance(_ive_instance),
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
  md_output.create(ive_instance, ImageType::U8C1, voWidth, voHeight);
  background_img.create(ive_instance, ImageType::U8C1, voWidth, voHeight);

  CVI_S32 ret = copy_image(init_frame, &background_img);

#ifdef DEBUG_MD
  LOGI("MD DEBUG: write: background.png\n");
  background_img.write("background.png");
#endif
  return ret;
}

MotionDetection::~MotionDetection() {
  md_output.free();
  background_img.free();
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
  return copy_image(frame, &background_img);
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

bool MotionDetection::overlap(const cv::Rect &bbox1, const cv::Rect &bbox2) {
  if (bbox1.x >= (bbox2.x + bbox2.width) || bbox2.x >= (bbox1.x + bbox1.width)) {
    return false;
  }

  if ((bbox1.y) >= (bbox2.y + bbox2.height) || (bbox2.y >= bbox1.y + bbox1.height)) {
    return false;
  }
  return true;
}

std::vector<uint32_t> MotionDetection::getAllOverlaps(const std::vector<cv::Rect> bboxes,
                                                      const cv::Rect &bounds, uint32_t index) {
  std::vector<uint32_t> overlaps;
  for (size_t i = 0; i < bboxes.size(); i++) {
    if (i != index) {
      if (overlap(bounds, bboxes[i])) {
        overlaps.push_back(i);
      }
    }
  }

  return overlaps;
}

void MotionDetection::mergebbox(std::vector<cv::Rect> &bboxes) {
  // go through the boxes and start merging
  uint32_t merge_margin = 20;

  // this is gonna take a long time
  bool finished = false;
  while (!finished) {
    // set end condition
    finished = true;

    // loop through boxes
    uint32_t index = 0;
    while (index < bboxes.size()) {
      cv::Rect curr_enlarged = bboxes[index];

      // enlarge bbox with margin
      curr_enlarged.x -= merge_margin;
      curr_enlarged.y -= merge_margin;
      curr_enlarged.width += (merge_margin * 2);
      curr_enlarged.height += (merge_margin * 2);

      // get matching boxes
      std::vector<uint32_t> overlaps = getAllOverlaps(bboxes, curr_enlarged, index);

      // check if empty
      if (overlaps.size() > 0) {
        overlaps.push_back(index);

        // convert to a contour
        std::vector<cv::Point> contour;
        for (auto ind : overlaps) {
          cv::Rect &rect = bboxes[ind];
          contour.push_back(rect.tl());
          contour.push_back(rect.br());
        }

// get bounding rect
#ifdef ENABLE_CVIAI_CV_UTILS
        cv::Rect merged = cviai::boundingRect(contour);
#else
        cv::Rect merged = cv::boundingRect(contour);
#endif

        // remove boxes from list
        std::sort(overlaps.begin(), overlaps.end(), std::greater<uint32_t>());
        for (auto remove_ind : overlaps) {
          bboxes.erase(bboxes.begin() + remove_ind);
        }

        bboxes.push_back(merged);

        // set flag
        finished = false;
        break;
      }

      // increment
      index += 1;
    }
  }
}

CVI_S32 MotionDetection::do_vpss_ifneeded(VIDEO_FRAME_INFO_S *srcframe,
                                          std::shared_ptr<VIDEO_FRAME_INFO_S> &frame) {
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
  return ret;
}

CVI_S32 MotionDetection::copy_image(VIDEO_FRAME_INFO_S *srcframe, ive::IVEImage *dst) {
  std::shared_ptr<VIDEO_FRAME_INFO_S> frame;
  CVI_S32 ret = do_vpss_ifneeded(srcframe, frame);

  if (ret == CVIAI_SUCCESS) {
    return VideoFrameCopy2Image(ive_instance, frame.get(), dst);
  }
  return ret;
}

CVI_S32 convert2Image(VIDEO_FRAME_INFO_S *srcframe, IVEImage *dst) {
  bool do_unmap_src = false;

  size_t image_size = srcframe->stVFrame.u32Length[0] + srcframe->stVFrame.u32Length[1] +
                      srcframe->stVFrame.u32Length[2];
  if (srcframe->stVFrame.pu8VirAddr[0] == NULL) {
    srcframe->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(srcframe->stVFrame.u64PhyAddr[0], image_size);
    do_unmap_src = true;
  }

  CVI_S32 ret = dst->fromFrame(srcframe);
  if (ret != CVI_SUCCESS) {
    LOGE("Convert frame to IVE_IMAGE_S fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  if (do_unmap_src) {
    CVI_SYS_Munmap((void *)srcframe->stVFrame.pu8VirAddr[0], image_size);
  }
  return CVIAI_SUCCESS;
}

CVI_S32 MotionDetection::detect(VIDEO_FRAME_INFO_S *srcframe, cvai_object_t *obj_meta) {
  static int c = 0;
  CVI_S32 ret = CVI_SUCCESS;
  if (count > 2) {
    IVEImage srcImg;
    std::shared_ptr<VIDEO_FRAME_INFO_S> processed_frame;
    ret = do_vpss_ifneeded(srcframe, processed_frame);
    if (ret != CVIAI_SUCCESS) {
      return ret;
    }

    ret = convert2Image(processed_frame.get(), &srcImg);
    if (ret != CVIAI_SUCCESS) {
      LOGE("failed to convert VIDEO_FRAME_INFO_S to IVE_IMAGE_S, ret=%d\n", ret);
      return ret;
    }
#ifdef DEBUG_MD
    LOGI("MD DEBUG: write: src.png\n");
    srcImg.write("src.png");
#endif
    // Sub - threshold - dilate
    ret = ive_instance->sub(&srcImg, &background_img, &md_output);
#ifdef DEBUG_MD
    LOGI("MD DEBUG: write: sub.png\n");
    md_output.write("sub.png");
#endif
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_IVE_Sub fail %x\n", ret);
      return CVIAI_ERR_MD_OPERATION_FAILED;
    }

    ret = ive_instance->thresh(&md_output, &md_output, ThreshMode::BINARY, threshold, 0, 0, 0, 255);
#ifdef DEBUG_MD
    LOGI("MD DEBUG: write: thresh.png\n");
    md_output.write("thresh.png");
#endif
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_IVE_Sub fail %x\n", ret);
      return CVIAI_ERR_MD_OPERATION_FAILED;
    }

    ret = ive_instance->dilate(&md_output, &md_output, {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
                                                        1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0});
    if (ret != CVI_SUCCESS) {
      LOGE("dilate fail %x\n", ret);
      return CVIAI_ERR_MD_OPERATION_FAILED;
    }
#ifdef DEBUG_MD
    LOGI("MD DEBUG: write: dialte.png\n");
    md_output.write("dilate.png");
#endif
    VIDEO_FRAME_INFO_S md_output_frame;
    md_output.bufRequest();
    md_output.toFrame(&md_output_frame);

    int img_width = md_output_frame.stVFrame.u32Width;
    int img_height = md_output_frame.stVFrame.u32Height;

    bool do_unmap = false;
    if (md_output_frame.stVFrame.pu8VirAddr[0] == NULL) {
      md_output_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
          md_output_frame.stVFrame.u64PhyAddr[0], md_output_frame.stVFrame.u32Length[0]);
      do_unmap = true;
    }
    cv::Mat image(img_height, img_width, CV_8UC1, md_output_frame.stVFrame.pu8VirAddr[0],
                  md_output_frame.stVFrame.u32Stride[0]);

    c++;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Rect> bboxes;

#ifdef ENABLE_CVIAI_CV_UTILS
    cviai::findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
#else
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
#endif

#ifdef ENABLE_CVIAI_CV_UTILS
    for (auto c : contours) {
      double area = cviai::contourArea(c);
      if (area > min_area) {
        bboxes.push_back(cviai::boundingRect(c));
      }
    }
#else
    for (auto c : contours) {
      double area = cv::contourArea(c);
      if (area > min_area) {
        bboxes.push_back(cv::boundingRect(c));
      }
    }
#endif

    mergebbox(bboxes);

    construct_bbox(bboxes, obj_meta);
    if (do_unmap) {
      CVI_SYS_Munmap((void *)md_output_frame.stVFrame.pu8VirAddr[0],
                     md_output_frame.stVFrame.u32Length[0]);
      md_output_frame.stVFrame.pu8VirAddr[0] = NULL;
      md_output_frame.stVFrame.pu8VirAddr[1] = NULL;
      md_output_frame.stVFrame.pu8VirAddr[2] = NULL;
    }

  } else {
    obj_meta->size = 0;
    obj_meta->info = nullptr;
    obj_meta->height = im_height;
    obj_meta->width = im_width;
    obj_meta->rescale_type = RESCALE_RB;
  }

  count++;
  return CVIAI_SUCCESS;
}