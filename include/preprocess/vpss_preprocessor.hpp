#ifndef VPSS_PREPROCESSOR_H
#define VPSS_PREPROCESSOR_H

#include "cvi_comm_vpss.h"
#include "preprocess/base_preprocessor.hpp"
class VpssPreprocessor : public BasePreprocessor {
 public:
  VpssPreprocessor();
  ~VpssPreprocessor();
  std::shared_ptr<BaseImage> resize(const std::shared_ptr<BaseImage>& image, int newWidth,
                                    int newHeight) override;
  std::shared_ptr<BaseImage> crop(const std::shared_ptr<BaseImage>& image, int x, int y, int width,
                                  int height) override;

  std::shared_ptr<BaseImage> preprocess(const std::shared_ptr<BaseImage>& image,
                                        const PreprocessParams& params,
                                        std::shared_ptr<BaseMemoryPool> memory_pool) override;
  bool stop();

  void setUseVbPool(bool use_vb_pool) { use_vb_pool_ = use_vb_pool; }

 private:
  bool init();
  int32_t prepareVPSSParams(const std::shared_ptr<BaseImage>& src_image,
                            const PreprocessParams& params);
  bool generateVPSSParams(const std::shared_ptr<BaseImage>& src_image,
                          const PreprocessParams& params, VPSS_GRP_ATTR_S& vpss_grp_attr,
                          VPSS_CROP_INFO_S& vpss_chn_crop_attr,
                          VPSS_CHN_ATTR_S& vpss_chn_attr) const;
  int group_id_;
  int device_;
  VPSS_CROP_INFO_S crop_reset_attr_;
  bool use_vb_pool_ = false;
};

#endif  // VPSS_PREPROCESSOR_H