#include "preprocess/vpss_preprocessor.hpp"

#include "core/utils/vpss_helper.h"
#include "cvi_comm_vb.h"
#include "cvi_tdl_log.hpp"
#include "image/vpss_image.hpp"
#include "memory/cvi_memory_pool.hpp"
VpssPreprocessor::VpssPreprocessor() {
  group_id_ = -1;
  if (!init()) {
    throw std::runtime_error("VpssPreprocessor init failed!\n");
    LOGE("VpssPreprocessor init failed!\n");
  }
}

VpssPreprocessor::~VpssPreprocessor() {
  stop();
  group_id_ = -1;
}

bool VpssPreprocessor::init() {
  int s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_SYS_Init failed!\n");
    return false;
  }

  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_CHN_ATTR_S vpss_chn_attr;
  // Not magic number, only for init.
  uint32_t width = 100;
  uint32_t height = 100;
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, width, height, VI_PIXEL_FORMAT,
                           device_);
  VPSS_CHN_DEFAULT_HELPER(&vpss_chn_attr, width, height,
                          PIXEL_FORMAT_RGB_888_PLANAR, true);

  int id = CVI_VPSS_GetAvailableGrp();
  LOGI("got available groupid:%d", id);
  if (CVI_VPSS_CreateGrp(id, &vpss_grp_attr) != CVI_SUCCESS) {
    LOGE("User assign group id %u failed to create vpss instance.\n", id);
    return false;
  }
  group_id_ = id;

  if (group_id_ == (VPSS_GRP)-1) {
    LOGE("All vpss grp init failed!\n");
    return false;
  }

  LOGI("Create Vpss Group(%d) Dev(%d)\n", group_id_, device_);

  s32Ret = CVI_VPSS_ResetGrp(group_id_);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_ResetGrp(grp:%d) failed with %#x!\n", group_id_, s32Ret);
    return false;
  }
  s32Ret = CVI_VPSS_SetChnAttr(group_id_, 0, &vpss_chn_attr);

  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    return false;
  }

  s32Ret = CVI_VPSS_EnableChn(group_id_, 0);

  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    return false;
  }
  s32Ret = CVI_VPSS_StartGrp(group_id_);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return false;
  }
  memset(&crop_reset_attr_, 0, sizeof(VPSS_CROP_INFO_S));
  return true;
}

bool VpssPreprocessor::stop() {
  int s32Ret = CVI_VPSS_DisableChn(group_id_, 0);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_DisableChn failed with %#x!\n", s32Ret);
    return false;
  }
  s32Ret = CVI_VPSS_StopGrp(group_id_);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_StopGrp failed with %#x!\n", s32Ret);
    return false;
  }

  s32Ret = CVI_VPSS_DestroyGrp(group_id_);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_DestroyGrp failed with %#x!\n", s32Ret);
    return false;
  }
  return true;
}

std::shared_ptr<BaseImage> VpssPreprocessor::preprocess(
    const std::shared_ptr<BaseImage>& image, const PreprocessParams& params,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  std::shared_ptr<VPSSImage> vpss_image = std::make_shared<VPSSImage>();
  std::unique_ptr<MemoryBlock> memory_block;
  int32_t ret = vpss_image->prepareImageInfo(params.dstWidth, params.dstHeight,
                                             params.dstImageFormat,
                                             params.dstPixDataType);
  if (ret != 0) {
    LOGE("VPSSImage prepareImageInfo failed!\n");
    return nullptr;
  }
  if (memory_pool == nullptr) {
    LOGE("memory_pool is nullptr!\n");
    return nullptr;
  }
  memory_block = memory_pool->allocate(vpss_image->getImageByteSize());
  if (memory_block == nullptr) {
    LOGE("VPSSImage allocate memory failed!\n");
    return nullptr;
  }
  ret = vpss_image->setupMemoryBlock(memory_block);
  if (!ret) {
    LOGE("VPSSImage setupMemoryBlock failed!\n");
    return nullptr;
  }
  LOGI("setup output image done");

  preprocessToImage(image, params, vpss_image);
  // CVI_VPSS_GetChnFrame will reset virtual address of output_frame,restore
  // it
  // for (int i = 0; i < vpss_image->getPlaneNum(); i++) {
  //   LOGI("vir_addrs[%d]: %p, output_frame->stVFrame.pu8VirAddr[%d]: %p", i,
  //        vpss_image->getVirtualAddress()[i], i,
  //        output_frame->stVFrame.pu8VirAddr[i]);
  //   output_frame->stVFrame.pu8VirAddr[i] =
  //   vpss_image->getVirtualAddress()[i];
  // }
  LOGI("CVI_VPSS_GetChnFrame done");
  return vpss_image;
}

int32_t VpssPreprocessor::prepareVPSSParams(
    const std::shared_ptr<BaseImage>& src_image,
    const PreprocessParams& params) {
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_CROP_INFO_S vpss_chn_crop_attr;
  VPSS_CHN_ATTR_S vpss_chn_attr;
  bool is_ok = generateVPSSParams(src_image, params, vpss_grp_attr,
                                  vpss_chn_crop_attr, vpss_chn_attr);
  if (!is_ok) {
    LOGE("generateVPSSParams failed!\n");
    return -1;
  }

  int ret = CVI_VPSS_SetGrpAttr(group_id_, &vpss_grp_attr);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetGrpAttr failed with %#x\n", ret);
    return -1;
  }
  ret = CVI_VPSS_SetGrpCrop(group_id_, &crop_reset_attr_);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetGrpCrop failed with %#x\n", ret);
    return -1;
  }
  ret = CVI_VPSS_SetChnAttr(group_id_, 0, &vpss_chn_attr);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetChnAttr failed with %#x\n", ret);
    return -1;
  }
  ret = CVI_VPSS_SetChnCrop(group_id_, 0, &vpss_chn_crop_attr);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetChnCrop failed with %#x\n", ret);
    return -1;
  }
  ret = CVI_VPSS_SetChnScaleCoefLevel(group_id_, 0, VPSS_SCALE_COEF_BILINEAR);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetChnScaleCoefLevel failed with %#x\n", ret);
    return -1;
  }
  return 0;
}

bool VpssPreprocessor::generateVPSSParams(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    VPSS_GRP_ATTR_S& vpss_grp_attr, VPSS_CROP_INFO_S& vpss_chn_crop_attr,
    VPSS_CHN_ATTR_S& vpss_chn_attr) const {
  if (src_image == nullptr) {
    LOGE("src_image is nullptr!\n");
    return false;
  }
  memset(&vpss_grp_attr, 0, sizeof(VPSS_GRP_ATTR_S));
  memset(&vpss_chn_crop_attr, 0, sizeof(VPSS_CROP_INFO_S));
  memset(&vpss_chn_attr, 0, sizeof(VPSS_CHN_ATTR_S));
  const VPSSImage* vpss_image_src =
      dynamic_cast<const VPSSImage*>(src_image.get());
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, vpss_image_src->getWidth(),
                           vpss_image_src->getHeight(),
                           vpss_image_src->getPixelFormat(), device_);

  PIXEL_FORMAT_E dst_format = vpss_image_src->convertPixelFormat(
      params.dstImageFormat, params.dstPixDataType);
  VPSS_CHN_SQ_HELPER(&vpss_chn_attr, params.dstWidth, params.dstHeight,
                     dst_format, params.mean, params.scale, false);
  if (!params.keepAspectRatio) {
    vpss_chn_attr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  }

  if (params.cropWidth > 0 && params.cropHeight > 0) {
    vpss_chn_crop_attr.bEnable = true;
    vpss_chn_crop_attr.stCropRect = {params.cropX, params.cropY,
                                     params.cropWidth, params.cropHeight};
  }
  return true;
}

std::shared_ptr<BaseImage> VpssPreprocessor::resize(
    const std::shared_ptr<BaseImage>& image, int newWidth, int newHeight) {
  PreprocessParams params;
  memset(&params, 0, sizeof(PreprocessParams));
  params.scale[0] = 1;
  params.scale[1] = 1;
  params.scale[2] = 1;
  params.dstWidth = newWidth;
  params.dstHeight = newHeight;
  params.dstImageFormat = image->getImageFormat();
  params.dstPixDataType = image->getPixDataType();
  return preprocess(image, params, nullptr);
}

std::shared_ptr<BaseImage> VpssPreprocessor::crop(
    const std::shared_ptr<BaseImage>& image, int x, int y, int width,
    int height) {
  PreprocessParams params;
  memset(&params, 0, sizeof(PreprocessParams));
  params.cropX = x;
  params.cropY = y;
  params.cropWidth = width;
  params.cropHeight = height;
  params.dstWidth = width;
  params.dstHeight = height;
  params.keepAspectRatio = false;
  params.mean[0] = 0;
  params.mean[1] = 0;
  params.mean[2] = 0;
  params.scale[0] = 1;
  params.scale[1] = 1;
  params.scale[2] = 1;
  params.dstImageFormat = image->getImageFormat();
  params.dstPixDataType = image->getPixDataType();
  return preprocess(image, params, nullptr);
}

int32_t VpssPreprocessor::preprocessToImage(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseImage> image) {
  if (!image->isInitialized()) {
    LOGE("image is not initialized!\n");
    return -1;
  }
  int32_t ret = prepareVPSSParams(image, params);
  if (ret != CVI_SUCCESS) {
    LOGE("prepareVPSSParams failed with %#x\n", ret);
    return ret;
  }
  VIDEO_FRAME_INFO_S* output_frame =
      static_cast<VIDEO_FRAME_INFO_S*>(image->getInternalData());
  const VIDEO_FRAME_INFO_S* input_frame =
      static_cast<const VIDEO_FRAME_INFO_S*>(src_image->getInternalData());

  if (use_vb_pool_) {
    VPSSImage* vpss_image = static_cast<VPSSImage*>(image.get());
    uint32_t vb_pool_id = vpss_image->getVbPoolId();
    if (vb_pool_id == UINT32_MAX) {
      LOGE("VPSSImage getVbPoolId failed!\n");
      return ret;
    }
    LOGI("vb_pool_id: %d", vb_pool_id);
    ret = CVI_VPSS_AttachVbPool(group_id_, 0, vb_pool_id);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_AttachVbPool failed with %#x\n", ret);
      return ret;
    }
    ret = CVI_VPSS_SendFrame(group_id_, input_frame, -1);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_SendFrame failed with %#x\n", ret);
      return ret;
    }
    LOGI("to CVI_VPSS_GetChnFrame");
    ret = CVI_VPSS_DetachVbPool(group_id_, 0);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_DetachVbPool failed with %#x\n", ret);
      return ret;
    }
    ret = CVI_VPSS_GetChnFrame(group_id_, 0, output_frame, -1);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_GetChnFrame failed with %#x\n", ret);
      return ret;
    }

  } else {
    // prepare output frame
    LOGI("to CVI_VPSS_SendChnFrame");
    ret = CVI_VPSS_SendChnFrame(group_id_, 0, output_frame, -1);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_SendChnFrame failed with %#x\n", ret);
      return ret;
    }
    LOGI("to CVI_VPSS_SendFrame");
    ret = CVI_VPSS_SendFrame(group_id_, input_frame, -1);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_SendFrame failed with %#x\n", ret);
      return ret;
    }
    LOGI("to CVI_VPSS_GetChnFrame");

    ret = CVI_VPSS_GetChnFrame(group_id_, 0, output_frame, -1);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_GetChnFrame failed with %#x\n", ret);
      return ret;
    }
  }
  return ret;
}

int32_t VpssPreprocessor::preprocessToTensor(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    const int batch_idx, std::shared_ptr<BaseTensor> tensor) {
  std::shared_ptr<VPSSImage> vpss_image = std::make_shared<VPSSImage>();

  int32_t ret = vpss_image->prepareImageInfo(params.dstWidth, params.dstHeight,
                                             params.dstImageFormat,
                                             params.dstPixDataType);
  if (ret != 0) {
    LOGE("VPSSImage prepareImageInfo failed!\n");
    return -1;
  }
  std::vector<uint32_t> strides = vpss_image->getStrides();
  if (strides[0] == tensor->getWidth()) {
    ret = tensor->constructImage(vpss_image, batch_idx);
    if (ret != 0) {
      LOGE("tensor constructImage failed, ret: %d\n", ret);
      return -1;
    }
  } else {
    ret = vpss_image->allocateMemory();
    if (ret != 0) {
      LOGE("vpss_image allocateMemory failed, ret: %d\n", ret);
      return -1;
    }
  }
  ret = preprocessToImage(src_image, params, vpss_image);
  if (ret != 0) {
    LOGE("preprocessToImage failed, ret: %d\n", ret);
    return -1;
  }
  return ret;
}

std::vector<float> VpssPreprocessor::getRescaleConfig(
    const PreprocessParams& params, const int image_width,
    const int image_height) {
  return {};
}