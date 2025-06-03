#include <inttypes.h>
#include <sys/time.h>
#include <iostream>
#include <memory>

#include "common/ccl.hpp"
#include "cvi_motion_detect.hpp"
#include "utils/tdl_log.hpp"

using namespace ive;

static int32_t VideoFrameCopy2Image(IVE *ive_instance, VIDEO_FRAME_INFO_S *src,
                                    IVEImage *tmp_ive_image, IVEImage *dst) {
  bool do_unmap = false;
  size_t image_size = src->stVFrame.u32Length[0] + src->stVFrame.u32Length[1] +
                      src->stVFrame.u32Length[2];
  if (src->stVFrame.pu8VirAddr[0] == NULL) {
    src->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_Mmap(src->stVFrame.u64PhyAddr[0], image_size);
    do_unmap = true;
  }
  int32_t ret = CVI_SUCCESS;
  ret = tmp_ive_image->fromFrame(
      src);  // would not allocat image buffer,only warp address
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

int32_t ImageFormatToPixelFormat(ImageFormat &image_format,
                                 PIXEL_FORMAT_E &format) {
  switch (image_format) {
    case ImageFormat::GRAY:
      format = PIXEL_FORMAT_YUV_400;
      break;
    case ImageFormat::RGB_PLANAR:
      format = PIXEL_FORMAT_RGB_888_PLANAR;
      break;
    case ImageFormat::BGR_PLANAR:
      format = PIXEL_FORMAT_BGR_888_PLANAR;
      break;
    case ImageFormat::YUV420SP_UV:
      format = PIXEL_FORMAT_NV12;
      break;
    case ImageFormat::YUV420SP_VU:
      format = PIXEL_FORMAT_NV21;
      break;
    default:
      printf("ImageInfo format not supported: %d\n",
             static_cast<int>(image_format));
      return -1;
  }
  return 0;
}

int32_t BaseImage2VideoFrame(const std::shared_ptr<BaseImage> &image,
                             VIDEO_FRAME_INFO_S &video_frame) {
  if (!image) {
    printf("image is nullptr.\n");
    return CVI_FAILURE;
  }

  // 基本图像信息
  video_frame.stVFrame.u32Width = image->getWidth();
  video_frame.stVFrame.u32Height = image->getHeight();

  ImageFormat base_fmt = image->getImageFormat();
  PIXEL_FORMAT_E format;
  ImageFormatToPixelFormat(base_fmt, format);
  video_frame.stVFrame.enPixelFormat = format;

  uint32_t plane_num = image->getPlaneNum();

  std::vector<uint32_t> strides = image->getStrides();
  std::vector<uint64_t> phy_addrs = image->getPhysicalAddress();
  std::vector<uint8_t *> vir_addrs = image->getVirtualAddress();

  for (uint32_t i = 0; i < plane_num; ++i) {
    video_frame.stVFrame.u32Stride[i] = strides[i];
    video_frame.stVFrame.u64PhyAddr[i] = phy_addrs[i];
    video_frame.stVFrame.pu8VirAddr[i] = vir_addrs[i];
    video_frame.stVFrame.u32Length[i] =
        video_frame.stVFrame.u32Height * strides[i];
  }

  return CVI_SUCCESS;
}

CviMotionDetection::CviMotionDetection()
    : ive_instance_(nullptr),
      ccl_instance_(nullptr),
      im_width_(0),
      im_height_(0),
      use_roi_(false) {
  ive::IVE *ive_handle = new ive::IVE;
  if (ive_handle->init() != CVI_SUCCESS) {
    printf("IVE handle init failed.\n");
  }
  ive_instance_ = ive_handle;
  ccl_instance_ = createConnectInstance();
}

CviMotionDetection::~CviMotionDetection() {
  background_img_.free();
  md_output_.free();
  delete ive_instance_;
  destroyConnectedComponent(ccl_instance_);
}

int32_t CviMotionDetection::setBackground(
    const std::shared_ptr<BaseImage> &background_image) {
  VIDEO_FRAME_INFO_S video_frame;
  BaseImage2VideoFrame(background_image, video_frame);
  bool needs_reconstruction = false;
  if (background_img_.getVAddr()[0] == NULL ||
      md_output_.getVAddr()[0] == NULL ||
      video_frame.stVFrame.u32Width != im_width_ ||
      video_frame.stVFrame.u32Height != im_height_) {
    needs_reconstruction = true;
  }

  if (needs_reconstruction) {
    if (background_img_.getVAddr()[0] != NULL) {
      background_img_.free();
    }
    if (md_output_.getVAddr()[0] != NULL) {
      md_output_.free();
    }
    constructImages(&video_frame);
  }
  int32_t ret = VideoFrameCopy2Image(ive_instance_, &video_frame, &tmp_cpy_img_,
                                     &background_img_);
  return ret;
}

int32_t CviMotionDetection::detect(const std::shared_ptr<BaseImage> &image,
                                   uint8_t threshold, double min_area,
                                   std::vector<ObjectBoxInfo> &objs) {
  VIDEO_FRAME_INFO_S video_frame;
  BaseImage2VideoFrame(image, video_frame);

  if (video_frame.stVFrame.u32Height != im_height_ ||
      video_frame.stVFrame.u32Width != im_width_) {
    LOGE(
        "Height and width of frame isn't equal to background image in "
        "CviMotionDetection\n");
    return CVI_FAILURE;
  }
  if (video_frame.stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_400) {
    LOGE("processed image format should be PIXEL_FORMAT_YUV_400,got %d\n",
         int(video_frame.stVFrame.enPixelFormat));
    return CVI_FAILURE;
  }
  md_timer_.TicToc("start");
  int32_t ret = CVI_SUCCESS;

  bool do_unmap_src = false;
  size_t image_size = video_frame.stVFrame.u32Length[0] +
                      video_frame.stVFrame.u32Length[1] +
                      video_frame.stVFrame.u32Length[2];
  if (video_frame.stVFrame.pu8VirAddr[0] == NULL) {
    video_frame.stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_Mmap(video_frame.stVFrame.u64PhyAddr[0], image_size);
    do_unmap_src = true;
  }

  ret = tmp_src_img_.fromFrame(&video_frame);
  if (ret != CVI_SUCCESS) {
    LOGE("Convert frame to IVE_IMAGE_S fail %x\n", ret);
    return CVI_FAILURE;
  }
  md_timer_.TicToc("preprocess");
  ret = ive_instance_->frameDiff(&tmp_src_img_, &background_img_, &md_output_,
                                 threshold);

  md_timer_.TicToc("tpu_ive");
  if (do_unmap_src) {
    CVI_SYS_Munmap((void *)video_frame.stVFrame.pu8VirAddr[0], image_size);
  }

  if (ret != CVI_SUCCESS) {
    LOGE("failed to do frame difference ret=%d\n", ret);
    return CVI_FAILURE;
  }

  md_output_.bufRequest(ive_instance_);
  int wstride = md_output_.getStride()[0];
  int num_boxes = 0;
  int *p_boxes = nullptr;

  if (use_roi_) {
    int offsetx = 0, offsety = 0, offset = 0;
    int imw = im_width_;
    int imh = im_height_;
    objs.clear();
    for (uint8_t i = 0; i < roi_s_.size(); i++) {
      auto pnt = roi_s_[i];
      offsetx = pnt.x1;
      offsety = pnt.y1;
      offset = pnt.y1 * wstride + pnt.x1;
      imw = pnt.x2 - pnt.x1;
      imh = pnt.y2 - pnt.y1;
      p_boxes = extractConnectedComponent(md_output_.getVAddr()[0] + offset,
                                          imw, imh, wstride, min_area,
                                          ccl_instance_, &num_boxes);
      for (uint32_t i = 0; i < (uint32_t)num_boxes; ++i) {
        ObjectBoxInfo box;
        box.x1 = p_boxes[i * 5 + 2] + offsetx;
        box.y1 = p_boxes[i * 5 + 1] + offsety;
        box.x2 = p_boxes[i * 5 + 4] + offsetx;
        box.y2 = p_boxes[i * 5 + 3] + offsety;
        objs.push_back(box);
      }
    }
  } else {
    p_boxes = extractConnectedComponent(md_output_.getVAddr()[0], im_width_,
                                        im_height_, wstride, min_area,
                                        ccl_instance_, &num_boxes);
    objs.clear();
    for (uint32_t i = 0; i < (uint32_t)num_boxes; ++i) {
      ObjectBoxInfo box;
      box.x1 = p_boxes[i * 5 + 2];
      box.y1 = p_boxes[i * 5 + 1];
      box.x2 = p_boxes[i * 5 + 4];
      box.y2 = p_boxes[i * 5 + 3];
      objs.push_back(box);
    }
  }
  md_timer_.TicToc("post");
  return CVI_SUCCESS;
}

int32_t CviMotionDetection::setROI(const std::vector<ObjectBoxInfo> &_roi_s) {
  if (_roi_s.size() == 0) {
    use_roi_ = false;
    return CVI_FAILURE;
  }
  int imw = md_output_.getWidth();
  int imh = md_output_.getHeight();

  for (auto i = 0; i < _roi_s.size(); i++) {
    auto p = _roi_s[i];
    if (p.x2 < p.x1 || p.x1 < 0 || p.x2 >= imw) {
      use_roi_ = false;
      LOGE("roi[%d] x overflow,x1:%d,x2:%d,imgw:%d\n", i, p.x1, p.x2,
           md_output_.getWidth());
      return CVI_FAILURE;
    }
    if (p.y2 < p.y1 || p.y1 < 0 || p.y2 >= imh) {
      use_roi_ = false;
      LOGE("roi[%d] y overflow,y1:%d,y2:%d,imgw:%d\n", i, p.y1, p.y2,
           md_output_.getHeight());
      return CVI_FAILURE;
    }
  }

  roi_s_ = _roi_s;
  use_roi_ = true;
  return md_output_.setZero(ive_instance_);
}

int32_t CviMotionDetection::constructImages(VIDEO_FRAME_INFO_S *init_frame) {
  im_width_ = init_frame->stVFrame.u32Width;
  im_height_ = init_frame->stVFrame.u32Height;
  int32_t ret = CVI_SUCCESS;

  m_padding_.left = ive_instance_->getAlignedWidth(1);  // 16
  m_padding_.right = 1;
  m_padding_.top = 1;
  m_padding_.bottom = 1;
  memset((void *)&m_padding_, 0, sizeof(m_padding_));

  // create image with padding (1, 1, 1, 1).
  uint32_t extend_aligned_width =
      im_width_ + m_padding_.left + m_padding_.right;
  uint32_t extend_aligned_height =
      im_height_ + m_padding_.top + m_padding_.bottom;
  ret = md_output_.create(ive_instance_, ive::ImageType::U8C1,
                          extend_aligned_width, extend_aligned_height, true);
  if (ret != CVI_SUCCESS) {
    LOGE("Cannot create buffer image in CviMotionDetection, ret=0x%x\n", ret);
    return CVI_FAILURE;
  }
  ret = background_img_.create(ive_instance_, ive::ImageType::U8C1, im_width_,
                               im_height_);
  if (ret != CVI_SUCCESS) {
    LOGE("Cannot create buffer image in CviMotionDetection, ret=0x%x\n", ret);
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

bool CviMotionDetection::isROIEmpty() { return roi_s_.empty(); }
