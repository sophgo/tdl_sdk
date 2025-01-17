#ifndef FACE_SPOOF_HPP_
#define FACE_SPOOF_HPP_
#include "framework/base_model.hpp"

typedef std::vector<cv::Mat> BGR_IR_MAT;

class FaceSpoof : public BaseModel {
public:
  FaceSpoof(const stNetParam &param) {net_param_ = param;}

  ~FaceSpoof() {}

  bmStatus_t setup();

  bmStatus_t classify(const std::vector<BGR_IR_MAT> &bgr_ir_images,
                      std::vector<bool> &is_real_faces);

private:
  bmStatus_t preprocess(std::vector<BGR_IR_MAT>::const_iterator &img_iter,
                        int batch_size);

  bmStatus_t postprocess(std::vector<bool> &results);

  cv::Mat tmp_resized_img_;
  const float thresh_ = 0.55;
};

#endif
