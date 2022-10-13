#include "md.hpp"
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "ccl.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai_log.hpp"
#include "error_msg.hpp"
#include "vpss_engine.hpp"

#ifndef NO_OPENCV
#include <opencv2/core.hpp>
#ifdef ENABLE_CVIAI_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif
#endif

// #define DEBUG_MD

using namespace ive;

static CVI_S32 VideoFrameCopy2Image(IVE *ive_instance, VIDEO_FRAME_INFO_S *src,
                                    IVEImage *tmp_ive_image, IVEImage *dst) {
  bool do_unmap = false;
  size_t image_size =
      src->stVFrame.u32Length[0] + src->stVFrame.u32Length[1] + src->stVFrame.u32Length[2];
  if (src->stVFrame.pu8VirAddr[0] == NULL) {
    src->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(src->stVFrame.u64PhyAddr[0], image_size);
    do_unmap = true;
  }
  CVI_S32 ret = CVI_SUCCESS;
  ret = tmp_ive_image->fromFrame(src);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_VideoFrameInfo2Image fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  ret = ive_instance->dma(tmp_ive_image, dst);

  if (do_unmap) {
    CVI_SYS_Munmap((void *)src->stVFrame.pu8VirAddr[0], image_size);
  }

  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_DMA fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  return CVIAI_SUCCESS;
}

MotionDetection::MotionDetection(IVE *_ive_instance, uint32_t timeout, cviai::VpssEngine *engine)
    : ive_instance(_ive_instance), m_vpss_engine(engine), m_vpss_timeout(timeout) {}

CVI_S32 MotionDetection::init(VIDEO_FRAME_INFO_S *init_frame) {
  CVI_S32 ret = construct_images(init_frame);

  if (ret == CVI_SUCCESS) {
    ret = copy_image(init_frame, &background_img);
  }
  p_ccl_instance = create_connect_instance();
#ifdef DEBUG_MD
  LOGI("MD DEBUG: write: background.yuv\n");
  background_img.write("background.yuv");
#endif
  return ret;
}

MotionDetection::~MotionDetection() {
  free_all();
  destroy_connected_component(p_ccl_instance);
}

void MotionDetection::free_all() {
  md_output.free();
  background_img.free();
}

CVI_S32 MotionDetection::construct_images(VIDEO_FRAME_INFO_S *init_frame) {
  im_width = init_frame->stVFrame.u32Width;
  im_height = init_frame->stVFrame.u32Height;
  uint32_t voWidth = init_frame->stVFrame.u32Width;
  uint32_t voHeight = init_frame->stVFrame.u32Height;
  CVI_S32 ret = CVIAI_SUCCESS;

  m_padding.left = ive_instance->getAlignedWidth(1);
  m_padding.right = 1;
  m_padding.top = 1;
  m_padding.bottom = 1;
#ifdef NO_OPENCV  // only phobos do not need padding because use custom ccl
  memset((void *)&m_padding, 0, sizeof(m_padding));
#endif
  // create image with padding (1, 1, 1, 1).
  uint32_t extend_aligned_width = voWidth + m_padding.left + m_padding.right;
  uint32_t extend_aligned_height = voHeight + m_padding.top + m_padding.bottom;
  ret = md_output.create(ive_instance, ImageType::U8C1, extend_aligned_width, extend_aligned_height,
                         true);
  if (ret != CVIAI_SUCCESS) {
    LOGE("Cannot create buffer image in MotionDetection, ret=0x%x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  ret = background_img.create(ive_instance, ImageType::U8C1, voWidth, voHeight);
  if (ret != CVIAI_SUCCESS) {
    LOGE("Cannot create buffer image in MotionDetection, ret=0x%x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }
  return CVIAI_SUCCESS;
}

CVI_S32 MotionDetection::vpss_process(VIDEO_FRAME_INFO_S *srcframe, VIDEO_FRAME_INFO_S *dstframe) {
  VPSS_CHN_ATTR_S chnAttr;
  std::cout << "to do preprocess ,w:" << srcframe->stVFrame.u32Width << std::endl;
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
  if (frame->stVFrame.u32Width != background_img.getWidth() ||
      frame->stVFrame.u32Height != background_img.getHeight()) {
    free_all();
    if (construct_images(frame) != CVIAI_SUCCESS) {
      return CVIAI_ERR_MD_OPERATION_FAILED;
    }
  }

  CVI_S32 ret = copy_image(frame, &background_img);

#ifdef DEBUG_MD
  LOGI("MD DEBUG: write: background.yuv\n");
  background_img.write("background.yuv");
#endif

  return ret;
}
#ifndef NO_OPENCV
void construct_bbox(std::vector<cv::Rect> dets, int im_width, int im_height, cvai_object_t *out) {
  CVI_AI_MemAllocInit(dets.size(), out);
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

bool overlap(const cv::Rect &bbox1, const cv::Rect &bbox2) {
  if (bbox1.x >= (bbox2.x + bbox2.width) || bbox2.x >= (bbox1.x + bbox1.width)) {
    return false;
  }

  if ((bbox1.y) >= (bbox2.y + bbox2.height) || (bbox2.y >= bbox1.y + bbox1.height)) {
    return false;
  }
  return true;
}

std::vector<uint32_t> getAllOverlaps(const std::vector<cv::Rect> bboxes, const cv::Rect &bounds,
                                     uint32_t index) {
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

void mergebbox(std::vector<cv::Rect> &bboxes) {
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
#endif

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
    return VideoFrameCopy2Image(ive_instance, frame.get(), &tmp_cpy_img_, dst);
  }
  return ret;
}

void dump_frame_img(const char *szimg, uint8_t *p_img, int width, int height) {
  std::cout << "to write subimg,w:" << width << ",height:" << height << ",addr:" << (void *)p_img
            << std::endl;
  FILE *fp = fopen(szimg, "wb");
  fwrite(&width, 4, 1, fp);
  fwrite(&height, 4, 1, fp);
  fwrite(p_img, width * height, 1, fp);
  fclose(fp);
}

CVI_S32 MotionDetection::detect(VIDEO_FRAME_INFO_S *srcframe, cvai_object_t *obj_meta,
                                uint8_t threshold, double min_area) {
  if (srcframe->stVFrame.u32Height != im_height || srcframe->stVFrame.u32Width != im_width) {
    LOGE("Height and width of frame isn't equal to background image in MotionDetection\n");
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }
  md_timer_.TicToc(0, "start");
  CVI_S32 ret = CVI_SUCCESS;
  std::shared_ptr<VIDEO_FRAME_INFO_S> processed_frame;
  ret = do_vpss_ifneeded(srcframe, processed_frame);
  if (ret != CVIAI_SUCCESS) {
    return ret;
  }

  bool do_unmap_src = false;
  size_t image_size = processed_frame->stVFrame.u32Length[0] +
                      processed_frame->stVFrame.u32Length[1] +
                      processed_frame->stVFrame.u32Length[2];
  if (processed_frame->stVFrame.pu8VirAddr[0] == NULL) {
    processed_frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(processed_frame->stVFrame.u64PhyAddr[0], image_size);
    do_unmap_src = true;
  }

  ret = tmp_src_img_.fromFrame(processed_frame.get());
  if (ret != CVI_SUCCESS) {
    LOGE("Convert frame to IVE_IMAGE_S fail %x\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }
  md_timer_.TicToc(1, "preprocess");

#ifndef NO_OPENCV
  ive::IVEImage sub_image;
  ive_instance->roi(&md_output, &sub_image, m_padding.left, m_padding.left + im_width,
                    m_padding.top, m_padding.top + im_height);
  ret = ive_instance->frame_diff(&tmp_src_img_, &background_img, &sub_image, threshold);
#else
  ret = ive_instance->frame_diff(&tmp_src_img_, &background_img, &md_output, threshold);
#endif
  md_timer_.TicToc(2, "tpu_ive");
  if (do_unmap_src) {
    CVI_SYS_Munmap((void *)processed_frame->stVFrame.pu8VirAddr[0], image_size);
  }

  if (ret != CVIAI_SUCCESS) {
    LOGE("failed to do frame difference ret=%d\n", ret);
    return CVIAI_ERR_MD_OPERATION_FAILED;
  }

  md_output.bufRequest(ive_instance);

#ifndef NO_OPENCV
  cv::Mat image(im_height + 2, im_width + 2, CV_8UC1, md_output.getVAddr()[0] + m_padding.left - 1,
                md_output.getStride()[0]);
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
  construct_bbox(bboxes, im_width, im_height, obj_meta);
#else

  memset(obj_meta, 0, sizeof(cvai_object_t));

  int num_boxes = 0;
  int wstride = md_output.getStride()[0];

  int *p_boxes = extract_connected_component(md_output.getVAddr()[0], im_width, im_height, wstride,
                                             min_area, p_ccl_instance, &num_boxes);
  CVI_AI_MemAllocInit(num_boxes, obj_meta);
  obj_meta->height = im_height;
  obj_meta->width = im_width;
  obj_meta->rescale_type = RESCALE_RB;
  memset(obj_meta->info, 0, sizeof(cvai_object_info_t) * num_boxes);
  for (uint32_t i = 0; i < (uint32_t)num_boxes; ++i) {
    obj_meta->info[i].bbox.x1 = p_boxes[i * 5 + 2];
    obj_meta->info[i].bbox.y1 = p_boxes[i * 5 + 1];
    obj_meta->info[i].bbox.x2 = p_boxes[i * 5 + 4];
    obj_meta->info[i].bbox.y2 = p_boxes[i * 5 + 3];
    obj_meta->info[i].bbox.score = 0;
    obj_meta->info[i].classes = -1;
    memset(obj_meta->info[i].name, 0, sizeof(obj_meta->info[i].name));
  }
#endif
  md_timer_.TicToc(3, "post");
  return CVIAI_SUCCESS;
}
