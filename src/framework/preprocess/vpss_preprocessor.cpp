#include "preprocess/vpss_preprocessor.hpp"

#include <cvi_vb.h>
#include <cvi_vpss.h>
#include <cassert>
// #include "core/utils/vpss_helper.h"
#include "cvi_comm_vb.h"
#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"

void init_vpss_grp_attr(VPSS_GRP_ATTR_S* pstVpssGrpAttr, CVI_U32 srcWidth,
                        CVI_U32 srcHeight, PIXEL_FORMAT_E enSrcFormat,
                        CVI_U8 dev);

void init_vpss_chn_attr(VPSS_CHN_ATTR_S* pastVpssChnAttr, CVI_U32 dst_width,
                        CVI_U32 dst_height, PIXEL_FORMAT_E enDstFormat,
                        CVI_BOOL keep_aspect_ratio);
VpssContext::VpssContext() {
  CVI_S32 s32Ret = CVI_SUCCESS;

#ifdef __CV181X__
  if (!CVI_VB_IsInited()) {
    CVI_VB_Exit();
    VB_CONFIG_S stVbConf;
    memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
    stVbConf.u32MaxPoolCnt = 0;

    s32Ret = CVI_VB_SetConfig(&stVbConf);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("CVI_VB_SetConf failed!\n");
      assert(false);
    }

    s32Ret = CVI_VB_Init();
    if (s32Ret != CVI_SUCCESS) {
      LOGE("CVI_VB_Init failed!\n");
      assert(false);
    }
    LOGI("CVI_VB_Init success");
  }
#endif

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_SYS_Init failed!\n");
    assert(false);
  }
  LOGI("VpssContext init done,ret: %d", s32Ret);
}

VpssContext::~VpssContext() {
  int s32Ret = CVI_SYS_Exit();
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_SYS_Exit failed!\n");
    assert(false);
  }
#ifdef __CV181X__
  CVI_VB_Exit();
#endif
}

VpssContext* VpssContext::GetInstance() {
  static VpssContext instance;
  return &instance;
}

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
  VpssContext::GetInstance();
  CVI_S32 s32Ret = CVI_SUCCESS;

  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_CHN_ATTR_S vpss_chn_attr;
  // Not magic number, only for init.
  uint32_t width = 100;
  uint32_t height = 100;
  init_vpss_grp_attr(&vpss_grp_attr, width, height, VI_PIXEL_FORMAT, device_);
  init_vpss_chn_attr(&vpss_chn_attr, width, height, PIXEL_FORMAT_RGB_888_PLANAR,
                     true);

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
  LOGI("CVI_VPSS_ResetGrp success");
  s32Ret = CVI_VPSS_SetChnAttr(group_id_, 0, &vpss_chn_attr);

  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    return false;
  }
  LOGI("CVI_VPSS_SetChnAttr success");
  s32Ret = CVI_VPSS_EnableChn(group_id_, 0);

  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    return false;
  }
  LOGI("CVI_VPSS_EnableChn success");
  s32Ret = CVI_VPSS_StartGrp(group_id_);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return false;
  }
  LOGI("CVI_VPSS_StartGrp success");
  memset(&crop_reset_attr_, 0, sizeof(VPSS_CROP_INFO_S));
  LOGI("VpssPreprocessor init done");

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
  LOGI("VpssPreprocessor stop done,destroy grp:%d", group_id_);
  return true;
}

std::shared_ptr<BaseImage> VpssPreprocessor::preprocess(
    const std::shared_ptr<BaseImage>& image, const PreprocessParams& params,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  std::shared_ptr<VPSSImage> vpss_image = std::make_shared<VPSSImage>(
      params.dst_width, params.dst_height, params.dst_image_format,
      params.dst_pixdata_type, false, memory_pool);
  std::unique_ptr<MemoryBlock> memory_block;
  if (memory_pool == nullptr) {
    LOGW("input memory_pool is nullptr,use src image memory pool\n");
    memory_pool = image->getMemoryPool();
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
  int32_t ret = vpss_image->setupMemoryBlock(memory_block);
  if (ret != 0) {
    LOGE("VPSSImage setupMemoryBlock failed!\n");
    return nullptr;
  }
  LOGI("setup output image done");

  preprocessToImage(image, params, vpss_image);
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
  LOGI("vpss chn attr ,width:%d,height:%d,format:%d", vpss_chn_attr.u32Width,
       vpss_chn_attr.u32Height, vpss_chn_attr.enPixelFormat);
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

  LOGI(
      "preprocess para "
      "info,dst_width:%d,dst_height:%d,dst_image_format:%d,dst_pixdata_type:%d,"
      "crop_x:"
      "%d,crop_y:%d,crop_width:%d,crop_height:%d,mean[0]:%.2f,mean[1]:%.2f,"
      "mean[2]"
      ":%.2f,"
      "scale[0]:%.2f,scale[1]:%.2f,scale[2]:%.2f,aspectRatio:%d",
      params.dst_width, params.dst_height, (int)params.dst_image_format,
      (int)params.dst_pixdata_type, params.crop_x, params.crop_y,
      params.crop_width, params.crop_height, params.mean[0], params.mean[1],
      params.mean[2], params.scale[0], params.scale[1], params.scale[2],
      params.keep_aspect_ratio);
  memset(&vpss_grp_attr, 0, sizeof(VPSS_GRP_ATTR_S));
  memset(&vpss_chn_crop_attr, 0, sizeof(VPSS_CROP_INFO_S));
  memset(&vpss_chn_attr, 0, sizeof(VPSS_CHN_ATTR_S));

  generateVPSSGrpAttr(src_image, params, vpss_grp_attr);

  generateVPSSChnAttr(src_image, params, vpss_chn_attr);

  if (params.crop_width > 0 && params.crop_height > 0) {
    vpss_chn_crop_attr.bEnable = true;
    vpss_chn_crop_attr.stCropRect = {params.crop_x, params.crop_y,
                                     params.crop_width, params.crop_height};
  }
  return true;
}
int32_t VpssPreprocessor::generateVPSSGrpAttr(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    VPSS_GRP_ATTR_S& vpss_grp_attr) const {
  VPSS_GRP_ATTR_S* pstVpssGrpAttr = &vpss_grp_attr;
  const VPSSImage* vpss_image_src =
      dynamic_cast<const VPSSImage*>(src_image.get());
  memset(pstVpssGrpAttr, 0, sizeof(VPSS_GRP_ATTR_S));
  pstVpssGrpAttr->stFrameRate.s32SrcFrameRate = -1;
  pstVpssGrpAttr->stFrameRate.s32DstFrameRate = -1;
  pstVpssGrpAttr->enPixelFormat = vpss_image_src->getPixelFormat();
  pstVpssGrpAttr->u32MaxW = vpss_image_src->getWidth();
  pstVpssGrpAttr->u32MaxH = vpss_image_src->getHeight();
#if !defined(__CV186X__)
  pstVpssGrpAttr->u8VpssDev = device_;
#endif
  LOGI("vpss grp attr ,width:%d,height:%d,format:%d", vpss_grp_attr.u32MaxW,
       vpss_grp_attr.u32MaxH, vpss_grp_attr.enPixelFormat);
  return 0;
}
int32_t VpssPreprocessor::generateVPSSChnAttr(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    VPSS_CHN_ATTR_S& vpss_chn_attr) const {
  VPSS_CHN_ATTR_S* pastVpssChnAttr = &vpss_chn_attr;

  PIXEL_FORMAT_E dst_format = VPSSImage::convertPixelFormat(
      params.dst_image_format, params.dst_pixdata_type);
  pastVpssChnAttr->u32Width = params.dst_width;
  pastVpssChnAttr->u32Height = params.dst_height;
  pastVpssChnAttr->enVideoFormat = VIDEO_FORMAT_LINEAR;
  pastVpssChnAttr->enPixelFormat = dst_format;
  pastVpssChnAttr->stFrameRate.s32SrcFrameRate = -1;
  pastVpssChnAttr->stFrameRate.s32DstFrameRate = -1;
  pastVpssChnAttr->u32Depth = 1;
  pastVpssChnAttr->bMirror = CVI_FALSE;
  pastVpssChnAttr->bFlip = CVI_FALSE;

  pastVpssChnAttr->stAspectRatio.bEnableBgColor = CVI_TRUE;

  if (!params.keep_aspect_ratio) {
    pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_NONE;
  } else {
    std::vector<float> rescale_params =
        getRescaleConfig(params, src_image->getWidth(), src_image->getHeight());
    int pad_x = (params.crop_x - rescale_params[2]) / rescale_params[0];
    int pad_y = (params.crop_y - rescale_params[3]) / rescale_params[1];

    int resized_w = params.dst_width - pad_x * 2;
    int resized_h = params.dst_height - pad_y * 2;

    pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
    pastVpssChnAttr->stAspectRatio.stVideoRect.s32X = pad_x;
    pastVpssChnAttr->stAspectRatio.stVideoRect.s32Y = pad_y;
    pastVpssChnAttr->stAspectRatio.stVideoRect.u32Width = resized_w;
    pastVpssChnAttr->stAspectRatio.stVideoRect.u32Height = resized_h;
  }

  pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  // pastVpssChnAttr->stAspectRatio.u32BgColor =
  //     RGB_8BIT((int)(params.mean[0] / params.scale[0]),
  //              (int)(params.mean[1] / params.scale[1]),
  //              (int)(params.mean[2] / params.scale[2]));

  bool enable_normalize = params.scale[0] != 1 || params.scale[1] != 1 ||
                          params.scale[2] != 1 || params.mean[0] != 0 ||
                          params.mean[1] != 0 || params.mean[2] != 0;
  pastVpssChnAttr->stNormalize.bEnable = enable_normalize;
  if (enable_normalize) {
    for (uint32_t i = 0; i < 3; i++) {
      pastVpssChnAttr->stNormalize.factor[i] = params.scale[i];
    }
    for (uint32_t i = 0; i < 3; i++) {
      pastVpssChnAttr->stNormalize.mean[i] = params.mean[i];
    }
    pastVpssChnAttr->stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
  }
  LOGI(
      "vpss chn attr "
      ",width:%d,height:%d,format:%d,normalize:%d,factor[0]:%.2f,factor[1]:%."
      "2f,factor[2]:%.2f,mean[0]:%.2f,mean[1]:%.2f,mean[2]:%.2f",
      vpss_chn_attr.u32Width, vpss_chn_attr.u32Height,
      vpss_chn_attr.enPixelFormat, vpss_chn_attr.stNormalize.bEnable,
      vpss_chn_attr.stNormalize.factor[0], vpss_chn_attr.stNormalize.factor[1],
      vpss_chn_attr.stNormalize.factor[2], vpss_chn_attr.stNormalize.mean[0],
      vpss_chn_attr.stNormalize.mean[1], vpss_chn_attr.stNormalize.mean[2]);
  return 0;
}

int32_t VpssPreprocessor::preprocessToImage(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseImage> image) {
  if (!image->isInitialized()) {
    LOGE("image is not initialized!\n");
    return -1;
  }
  if (src_image->getImageType() != ImageType::VPSS_FRAME) {
    LOGE("src_image is not VPSSImage! image type: %d\n",
         src_image->getImageType());
    return -1;
  }
  int32_t ret = prepareVPSSParams(src_image, params);
  if (ret != CVI_SUCCESS) {
    LOGE("prepareVPSSParams failed with %#x\n", ret);
    return ret;
  }

  VIDEO_FRAME_INFO_S* output_frame =
      static_cast<VIDEO_FRAME_INFO_S*>(image->getInternalData());
  const VIDEO_FRAME_INFO_S* input_frame =
      static_cast<const VIDEO_FRAME_INFO_S*>(src_image->getInternalData());
  VPSSImage* vpss_image = static_cast<VPSSImage*>(image.get());
  vpss_image->checkToSwapRGB();

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

  vpss_image->restoreVirtualAddress(true);

  return ret;
}

int32_t VpssPreprocessor::preprocessToTensor(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    const int batch_idx, std::shared_ptr<BaseTensor> tensor) {
  std::shared_ptr<VPSSImage> vpss_image = std::make_shared<VPSSImage>(
      params.dst_width, params.dst_height, params.dst_image_format,
      params.dst_pixdata_type, false);

  std::vector<uint32_t> strides = vpss_image->getStrides();
  int32_t ret = 0;
  int tensor_stride = tensor->getWidth() * tensor->getElementSize();
  if (strides[0] == tensor_stride) {
    LOGI("vpss preprocessor, construct image from input tensor");
    ret = tensor->constructImage(vpss_image, batch_idx);
    if (ret != 0) {
      LOGE("tensor constructImage failed, ret: %d\n", ret);
      return -1;
    }
  } else {
    LOGI("vpss preprocessor, image stride:%d, tensor stride:%d", strides[0],
         tensor_stride);
    ret = vpss_image->allocateMemory();
    if (ret != 0) {
      LOGE("vpss_image allocateMemory failed, ret: %d\n", ret);
      return -1;
    }
  }
  LOGI(
      "to "
      "preprocessToImage,scale:%f,%f,%f,mean:%f,%f,%f,dst_height:%d,dst_width:%"
      "d,"
      "dst_pixdata_type:%d,dstStride:%d",
      params.scale[0], params.scale[1], params.scale[2], params.mean[0],
      params.mean[1], params.mean[2], params.dst_height, params.dst_width,
      (int)params.dst_pixdata_type, strides[0]);

  ret = preprocessToImage(src_image, params, vpss_image);
  if (ret != 0) {
    LOGE("preprocessToImage failed, ret: %d\n", ret);
    return -1;
  }
  if (strides[0] != tensor->getWidth()) {
    // copy vpss image to tensor
    LOGI("copy vpss image to tensor");
    vpss_image->invalidateCache();
    ret = tensor->copyFromImage(vpss_image, batch_idx);
    if (ret != 0) {
      LOGE("tensor copyFromImage failed, ret: %d\n", ret);
      return -1;
    }
  }
  return ret;
}

void init_vpss_grp_attr(VPSS_GRP_ATTR_S* pstVpssGrpAttr, CVI_U32 srcWidth,
                        CVI_U32 srcHeight, PIXEL_FORMAT_E enSrcFormat,
                        CVI_U8 dev) {
  memset(pstVpssGrpAttr, 0, sizeof(VPSS_GRP_ATTR_S));
  pstVpssGrpAttr->stFrameRate.s32SrcFrameRate = -1;
  pstVpssGrpAttr->stFrameRate.s32DstFrameRate = -1;
  pstVpssGrpAttr->enPixelFormat = enSrcFormat;
  pstVpssGrpAttr->u32MaxW = srcWidth;
  pstVpssGrpAttr->u32MaxH = srcHeight;
#if !defined(__CV186X__)
  pstVpssGrpAttr->u8VpssDev = dev;
#endif
}
void init_vpss_chn_attr(VPSS_CHN_ATTR_S* pastVpssChnAttr, CVI_U32 dst_width,
                        CVI_U32 dst_height, PIXEL_FORMAT_E enDstFormat,
                        CVI_BOOL keep_aspect_ratio) {
  pastVpssChnAttr->u32Width = dst_width;
  pastVpssChnAttr->u32Height = dst_height;
  pastVpssChnAttr->enVideoFormat = VIDEO_FORMAT_LINEAR;
  pastVpssChnAttr->enPixelFormat = enDstFormat;

  pastVpssChnAttr->stFrameRate.s32SrcFrameRate = -1;
  pastVpssChnAttr->stFrameRate.s32DstFrameRate = -1;
  pastVpssChnAttr->u32Depth = 1;
  pastVpssChnAttr->bMirror = CVI_FALSE;
  pastVpssChnAttr->bFlip = CVI_FALSE;
  if (keep_aspect_ratio) {
    pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_AUTO;
    pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  } else {
    pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_NONE;
  }
  pastVpssChnAttr->stNormalize.bEnable = CVI_FALSE;
  pastVpssChnAttr->stNormalize.factor[0] = 0;
  pastVpssChnAttr->stNormalize.factor[1] = 0;
  pastVpssChnAttr->stNormalize.factor[2] = 0;
  pastVpssChnAttr->stNormalize.mean[0] = 0;
  pastVpssChnAttr->stNormalize.mean[1] = 0;
  pastVpssChnAttr->stNormalize.mean[2] = 0;
  pastVpssChnAttr->stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
}