#include "md.hpp"
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "ccl.hpp"

#include <cvi_sys.h>
#include <inttypes.h>

#include "cviai_log.hpp"

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
  ret = tmp_ive_image->fromFrame(src);  // would not allocat image buffer,only warp address
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_VideoFrameInfo2Image fail %x\n", ret);
    return CVI_FAILURE;
  }

  ret = ive_instance->dma(tmp_ive_image, dst);  // dma copy

  if (do_unmap) {
    CVI_SYS_Munmap((void *)src->stVFrame.pu8VirAddr[0], image_size);
  }

  if (ret != CVI_SUCCESS) {
    LOGE("CVI_IVE_DMA fail %x\n", ret);
    return CVI_FAILURE;
  }

  return CVI_SUCCESS;
}

MotionDetection::MotionDetection(IVE *_ive_instance) : ive_instance(_ive_instance) {}

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

CVI_S32 MotionDetection::set_roi(int x1, int y1, int x2, int y2) {
  if (x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0) {
    use_roi_ = false;
    memset(m_roi_, 0, sizeof(md_output));
    return CVI_SUCCESS;
  }
  if (x1 == m_roi_[0] && y1 == m_roi_[1] && x2 == m_roi_[0] && y2 == m_roi_[1]) {
    use_roi_ = true;
    return CVI_SUCCESS;
  }
  int imw = md_output.getWidth();
  int imh = md_output.getHeight();
  if (x2 < x1 || x1 < 0 || x2 >= imw) {
    LOGE("roi x overflow,x1:%d,x2:%d,imgw:%d\n", x1, x2, md_output.getWidth());
    return CVI_FAILURE;
  }
  if (y2 < y1 || y1 < 0 || y2 >= imh) {
    LOGE("roi y overflow,y1:%d,y2:%d,imgw:%d\n", y1, y2, md_output.getHeight());
    return CVI_FAILURE;
  }
  m_roi_[0] = x1;
  m_roi_[1] = y1;
  m_roi_[2] = x2;
  m_roi_[3] = y2;
  use_roi_ = true;
  return md_output.setZero(ive_instance);
}

CVI_S32 MotionDetection::construct_images(VIDEO_FRAME_INFO_S *init_frame) {
  im_width = init_frame->stVFrame.u32Width;
  im_height = init_frame->stVFrame.u32Height;
  uint32_t voWidth = init_frame->stVFrame.u32Width;
  uint32_t voHeight = init_frame->stVFrame.u32Height;
  CVI_S32 ret = CVI_SUCCESS;

  m_padding.left = ive_instance->getAlignedWidth(1);  // 16
  m_padding.right = 1;
  m_padding.top = 1;
  m_padding.bottom = 1;
  memset((void *)&m_padding, 0, sizeof(m_padding));

  // create image with padding (1, 1, 1, 1).
  uint32_t extend_aligned_width = voWidth + m_padding.left + m_padding.right;
  uint32_t extend_aligned_height = voHeight + m_padding.top + m_padding.bottom;
  ret = md_output.create(ive_instance, ImageType::U8C1, extend_aligned_width, extend_aligned_height,
                         true);
  if (ret != CVI_SUCCESS) {
    LOGE("Cannot create buffer image in MotionDetection, ret=0x%x\n", ret);
    return CVI_FAILURE;
  }
  ret = background_img.create(ive_instance, ImageType::U8C1, voWidth, voHeight);
  if (ret != CVI_SUCCESS) {
    LOGE("Cannot create buffer image in MotionDetection, ret=0x%x\n", ret);
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

CVI_S32 MotionDetection::update_background(VIDEO_FRAME_INFO_S *frame) {
  if (p_ccl_instance == nullptr) {
    init(frame);
  }

  if (frame->stVFrame.u32Width != background_img.getWidth() ||
      frame->stVFrame.u32Height != background_img.getHeight()) {
    free_all();
    if (construct_images(frame) != CVI_SUCCESS) {
      return CVI_FAILURE;
    }
  }

  CVI_S32 ret = copy_image(frame, &background_img);

#ifdef DEBUG_MD
  LOGI("MD DEBUG: write: background.yuv\n");
  background_img.write("background.yuv");
#endif

  return ret;
}

CVI_S32 MotionDetection::copy_image(VIDEO_FRAME_INFO_S *srcframe, ive::IVEImage *dst) {
  CVI_S32 ret = VideoFrameCopy2Image(ive_instance, srcframe, &tmp_cpy_img_, dst);
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

CVI_S32 MotionDetection::detect(VIDEO_FRAME_INFO_S *srcframe, std::vector<std::vector<float>> &objs,
                                uint8_t threshold, double min_area) {
  if (srcframe->stVFrame.u32Height != im_height || srcframe->stVFrame.u32Width != im_width) {
    LOGE("Height and width of frame isn't equal to background image in MotionDetection\n");
    return CVI_FAILURE;
  }
  if (srcframe->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_400) {
    LOGE("processed image format should be PIXEL_FORMAT_YUV_400,got %d\n",
         int(srcframe->stVFrame.enPixelFormat));
    return CVI_FAILURE;
    ;
  }

  md_timer_.TicToc("start");
  CVI_S32 ret = CVI_SUCCESS;

  bool do_unmap_src = false;
  size_t image_size = srcframe->stVFrame.u32Length[0] + srcframe->stVFrame.u32Length[1] +
                      srcframe->stVFrame.u32Length[2];
  if (srcframe->stVFrame.pu8VirAddr[0] == NULL) {
    srcframe->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(srcframe->stVFrame.u64PhyAddr[0], image_size);
    do_unmap_src = true;
  }

  ret = tmp_src_img_.fromFrame(srcframe);
  if (ret != CVI_SUCCESS) {
    LOGE("Convert frame to IVE_IMAGE_S fail %x\n", ret);
    return CVI_FAILURE;
  }
  md_timer_.TicToc("preprocess");
  ret = ive_instance->frame_diff(&tmp_src_img_, &background_img, &md_output, threshold);

  md_timer_.TicToc("tpu_ive");
  if (do_unmap_src) {
    CVI_SYS_Munmap((void *)srcframe->stVFrame.pu8VirAddr[0], image_size);
  }

  if (ret != CVI_SUCCESS) {
    LOGE("failed to do frame difference ret=%d\n", ret);
    return CVI_FAILURE;
  }

  md_output.bufRequest(ive_instance);
  int wstride = md_output.getStride()[0];
  int offsetx = 0, offsety = 0, offset = 0;
  int imw = im_width;
  int imh = im_height;
  if (use_roi_) {
    offsetx = m_roi_[0];
    offsety = m_roi_[1];
    offset = m_roi_[1] * wstride + m_roi_[0];
    imw = m_roi_[2] - m_roi_[0];
    imh = m_roi_[3] - m_roi_[1];
  }
  int num_boxes = 0;

  int *p_boxes = extract_connected_component(md_output.getVAddr()[0] + offset, imw, imh, wstride,
                                             min_area, p_ccl_instance, &num_boxes);

  objs.clear();
  for (uint32_t i = 0; i < (uint32_t)num_boxes; ++i) {
    std::vector<float> box;
    box.push_back(p_boxes[i * 5 + 2] + offsetx);
    box.push_back(p_boxes[i * 5 + 1] + offsety);
    box.push_back(p_boxes[i * 5 + 4] + offsetx);
    box.push_back(p_boxes[i * 5 + 3] + offsety);
    objs.push_back(box);
  }
  md_timer_.TicToc("post");
  return CVI_SUCCESS;
}
CVI_S32 MotionDetection::get_motion_map(VIDEO_FRAME_INFO_S *frame) {
  // CVI_S32 ret = CVI_IVE_Image2VideoFrameInfo(&bk_dst, frame, false);
  // return ret;
  return CVI_FAILURE;
}
