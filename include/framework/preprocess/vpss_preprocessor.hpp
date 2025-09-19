#ifndef VPSS_PREPROCESSOR_H
#define VPSS_PREPROCESSOR_H

#include <cvi_comm_vpss.h>

#include "preprocess/base_preprocessor.hpp"

class VpssContext {
 public:
  VpssContext();
  ~VpssContext();

  static VpssContext* GetInstance();

 private:
  static VpssContext instance_;
};
class VpssPreprocessor : public BasePreprocessor {
 public:
  VpssPreprocessor(int device = 0);
  ~VpssPreprocessor();

  std::shared_ptr<BaseImage> preprocess(
      const std::shared_ptr<BaseImage>& image, const PreprocessParams& params,
      std::shared_ptr<BaseMemoryPool> memory_pool = nullptr) override;
  int32_t preprocessToImage(const std::shared_ptr<BaseImage>& src_image,
                            const PreprocessParams& params,
                            std::shared_ptr<BaseImage> dst_image) override;
  int32_t preprocessToTensor(const std::shared_ptr<BaseImage>& src_image,
                             const PreprocessParams& params,
                             const int batch_idx,
                             std::shared_ptr<BaseTensor> tensor) override;

  void setUseVbPool(bool use_vb_pool) { use_vb_pool_ = use_vb_pool; }

 private:
  bool init();
  bool stop();
  int32_t prepareVPSSParams(const std::shared_ptr<BaseImage>& src_image,
                            const PreprocessParams& params);
  int32_t generateVPSSGrpAttr(const std::shared_ptr<BaseImage>& src_image,
                              const PreprocessParams& params,
                              VPSS_GRP_ATTR_S& vpss_grp_attr) const;
  int32_t generateVPSSChnAttr(const std::shared_ptr<BaseImage>& src_image,
                              const PreprocessParams& params,
                              VPSS_CHN_ATTR_S& vpss_chn_attr) const;
  bool generateVPSSParams(const std::shared_ptr<BaseImage>& src_image,
                          const PreprocessParams& params,
                          VPSS_GRP_ATTR_S& vpss_grp_attr,
                          VPSS_CROP_INFO_S& vpss_chn_crop_attr,
                          VPSS_CHN_ATTR_S& vpss_chn_attr) const;
  int group_id_;
  int device_;
  VPSS_CROP_INFO_S crop_reset_attr_;
  bool use_vb_pool_ = false;
};

#endif  // VPSS_PREPROCESSOR_H