#include <iostream>
#include <memory>
#include <vector>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl_log.hpp"
#include "cvi_tdl_media.h"
#ifndef USE_TPU_IVE
#include <cvi_ive.h>
#else
#include "ive/ive.h"
#endif
#ifndef NO_OPENCV
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// TODO: use memcpy
inline void BufferRGBPackedCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  if (invert) {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][offset] = ptr_pxl[2];
        vFrame->pu8VirAddr[0][offset + 1] = ptr_pxl[1];
        vFrame->pu8VirAddr[0][offset + 2] = ptr_pxl[0];
      }
    }
  } else {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][offset] = ptr_pxl[0];
        vFrame->pu8VirAddr[0][offset + 1] = ptr_pxl[1];
        vFrame->pu8VirAddr[0][offset + 2] = ptr_pxl[2];
      }
    }
  }
}

inline void BufferRGBPacked2PlanarCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                       uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  if (invert) {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[2];
        vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr_pxl[1];
        vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr_pxl[0];
      }
    }
  } else {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[0];
        vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr_pxl[1];
        vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr_pxl[2];
      }
    }
  }
}
// input is bgr format
inline void BufferRGBPacked2YUVPlanarCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                          uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  CVI_U8 *pY = vFrame->pu8VirAddr[0];
  CVI_U8 *pU = vFrame->pu8VirAddr[1];
  CVI_U8 *pV = vFrame->pu8VirAddr[2];

  for (uint32_t j = 0; j < height; j++) {
    const uint8_t *ptr = buffer + j * stride;
    for (uint32_t i = 0; i < width; i++) {
      const uint8_t *ptr_pxl = i * 3 + ptr;
      int b = ptr_pxl[0];
      int g = ptr_pxl[1];
      int r = ptr_pxl[2];
      if (invert) {
        std::swap(b, r);
      }
      pY[i + j * vFrame->u32Stride[0]] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

      if (j % 2 == 0 && i % 2 == 0) {
        pU[width / 2 + height / 2 * vFrame->u32Stride[1]] =
            ((-38 * r - 74 * g + 112 * b) >> 8) + 128;
        pV[width / 2 + height / 2 * vFrame->u32Stride[2]] =
            ((112 * r - 94 * g - 18 * b) >> 8) + 128;
      }
    }
  }
}
inline void BufferGreyCopy(const uint8_t *buffer, uint32_t width, uint32_t height, uint32_t stride,
                           VIDEO_FRAME_INFO_S *frame) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  for (uint32_t j = 0; j < height; j++) {
    const uint8_t *ptr = buffer + j * stride;
    for (uint32_t i = 0; i < width; i++) {
      const uint8_t *ptr_pxl = ptr + i;
      vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[0];
    }
  }
}

template <typename T>
inline void BufferC12C1Copy(const uint8_t *buffer, uint32_t width, uint32_t height, uint32_t stride,
                            VIDEO_FRAME_INFO_S *frame) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  for (uint32_t j = 0; j < height; j++) {
    const uint8_t *ptr = buffer + j * stride;
    uint8_t *vframec0ptr = vFrame->pu8VirAddr[0] + j * vFrame->u32Stride[0];
    memcpy(vframec0ptr, ptr, width * sizeof(T));
  }
}

void image_buffer_copy(cv::Mat &img, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format) {
  if (CREATE_ION_HELPER(frame, img.cols, img.rows, format, "cvitdl/image") != CVI_SUCCESS) {
    LOGE("alloc ion failed, imgwidth:%d,imgheight:%d\n", img.cols, img.rows);
  }
  switch (format) {
    case PIXEL_FORMAT_RGB_888: {
      BufferRGBPackedCopy(img.data, img.cols, img.rows, img.step, frame, true);
    } break;
    case PIXEL_FORMAT_BGR_888: {
      BufferRGBPackedCopy(img.data, img.cols, img.rows, img.step, frame, false);
    } break;
    case PIXEL_FORMAT_RGB_888_PLANAR: {
      BufferRGBPacked2PlanarCopy(img.data, img.cols, img.rows, img.step, frame, true);
    } break;
    case PIXEL_FORMAT_YUV_400: {
      cv::Mat img2;
      cv::cvtColor(img, img2, cv::COLOR_BGR2GRAY);
      BufferGreyCopy(img2.data, img2.cols, img2.rows, img2.step, frame);
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_420:
      BufferRGBPacked2YUVPlanarCopy(img.data, img.cols, img.rows, img.step, frame, false);
      break;
    default:
      LOGE("Unsupported format: %u.\n", format);
      break;
  }
}

CVI_S32 read_resize_image(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format,
                          uint32_t width, uint32_t height) {
  cv::Mat src_img = cv::imread(filepath);
  if (src_img.empty()) {
    LOGE("Cannot read image %s.\n", filepath);
    return CVI_FAILURE;
  }

  cv::Mat img;
  cv::resize(src_img, img, cv::Size(width, height));
  image_buffer_copy(img, frame, format);
  return CVI_SUCCESS;
}

CVI_S32 read_image(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format) {
  cv::Mat img = cv::imread(filepath);
  if (img.empty()) {
    LOGE("Cannot read image %s.\n", filepath);
    return CVI_FAILURE;
  }
  image_buffer_copy(img, frame, format);
  return CVI_SUCCESS;
}

CVI_S32 release_image(VIDEO_FRAME_INFO_S *frame) {
  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    CVI_SYS_IonFree(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0]);
    frame->stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return CVI_SUCCESS;
}
#endif

class ImageProcessor {
 public:
  virtual ~ImageProcessor() {}
  virtual int read(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format) = 0;
  virtual int read_resize(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format,
                          uint32_t width, uint32_t height) = 0;
  virtual int release(VIDEO_FRAME_INFO_S *frame) = 0;
};

class ImageProcessorNoOpenCV : public ImageProcessor {
 public:
  ImageProcessorNoOpenCV() {
    ive_handle = CVI_IVE_CreateHandle();
    LOGE("ImageProcessorNoOpenCV created.\n");
  }

  ~ImageProcessorNoOpenCV() override {
    if (ive_handle != nullptr) {
      CVI_IVE_DestroyHandle(ive_handle);
      LOGE("ImageProcessorNoOpenCV destroyed..\n");
    }
  }

  int read(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format) override {
    IVE_IMAGE_TYPE_E ive_format;
    int ret = CVI_SUCCESS;
    switch (format) {
      case PIXEL_FORMAT_E::PIXEL_FORMAT_RGB_888:
        ive_format = IVE_IMAGE_TYPE_E::IVE_IMAGE_TYPE_U8C3_PACKAGE;
        break;
      case PIXEL_FORMAT_E::PIXEL_FORMAT_RGB_888_PLANAR:
        ive_format = IVE_IMAGE_TYPE_E::IVE_IMAGE_TYPE_U8C3_PLANAR;
        break;
      default:
        LOGE("format should be PIXEL_FORMAT_RGB_888 or PIXEL_FORMAT_RGB_888_PLANAR\n");
        ive_format = IVE_IMAGE_TYPE_E::IVE_IMAGE_TYPE_U8C1;
        break;
    }

    if (ive_handle != nullptr) {
      image = CVI_IVE_ReadImage(ive_handle, filepath, ive_format);
#ifndef USE_TPU_IVE
      int imgw = image.u32Width;
#else
      int imgw = image.u16Width;
#endif
      if (imgw == 0) {
        printf("Read image failed with %x!\n", ret);
        return CVI_FAILURE;
      }
#ifndef USE_TPU_IVE
      ret = CVI_IVE_Image2VideoFrameInfo(&image, frame);
#else
      ret = CVI_IVE_Image2VideoFrameInfo(&image, frame, false);
#endif
      if (ret != CVI_SUCCESS) {
        LOGE("open img failed with %#x!\n", ret);
        return ret;
      } else {
        printf("image read,width:%d\n", frame->stVFrame.u32Width);
      }
    } else {
      LOGE("ive_handle should valuable.\n");
      return CVI_FAILURE;  // Handle error appropriately
    }
    return CVI_SUCCESS;
  }

  int read_resize(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format,
                  uint32_t width, uint32_t height) {
    return 0;
  }

  int release(VIDEO_FRAME_INFO_S *frame) { CVI_SYS_FreeI(ive_handle, &image); }

 private:
  IVE_HANDLE ive_handle;
  IVE_IMAGE_S image;
};

#ifndef NO_OPENCV
class ImageProcessorWithOpenCV : public ImageProcessor {
 public:
  int read(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format) override {
    return read_image(filepath, frame, format);
  }

  int read_resize(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format,
                  uint32_t width, uint32_t height) override {
    return read_resize_image(filepath, frame, format, width, height);
  }

  int release(VIDEO_FRAME_INFO_S *frame) override { return release_image(frame); }
};
#endif

CVI_S32 CVI_TDL_Create_ImageProcessor(imgprocess_t *hanlde) {
#ifdef NO_OPENCV
  auto imageProcessor = std::make_unique<ImageProcessorNoOpenCV>();
#else
  auto imageProcessor = std::make_unique<ImageProcessorWithOpenCV>();
#endif
  *hanlde = imageProcessor.get();
  return 0;
}

CVI_S32 CVI_TDL_ReadImage(imgprocess_t handle, const char *filepath, VIDEO_FRAME_INFO_S *frame,
                          PIXEL_FORMAT_E format) {
  ImageProcessor *ctx = static_cast<ImageProcessor *>(handle);
  return ctx->read(filepath, frame, format);
}

CVI_S32 CVI_TDL_ReadImage_Resize(imgprocess_t handle, const char *filepath,
                                 VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format, uint32_t width,
                                 uint32_t height) {
  ImageProcessor *ctx = static_cast<ImageProcessor *>(handle);
  return ctx->read_resize(filepath, frame, format, width, height);
}

CVI_S32 CVI_TDL_ReleaseImage(imgprocess_t handle, VIDEO_FRAME_INFO_S *frame) {
  ImageProcessor *ctx = static_cast<ImageProcessor *>(handle);
  return ctx->release(frame);
}
