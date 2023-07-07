#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iterator>

#include <core/core/cvai_errno.h>
#include <error_msg.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include "coco_utils.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_sys.h"
#include "object_utils.hpp"
#include "yolov5.hpp"

int yolov5_argmax(float *ptr, int start_idx, int arr_len) {
  int max_idx = start_idx;
  for (int i = start_idx + 1; i < start_idx + arr_len; i++) {
    if (ptr[i] > ptr[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx - start_idx;
}

int max_val(int x, int y) {
  if (x > y) return x;
  return y;
}

namespace cviai {

static void convert_det_struct(const Detections &dets, cvai_object_t *obj, int im_height,
                               int im_width) {
  CVI_AI_MemAllocInit(dets.size(), obj);
  obj->height = im_height;
  obj->width = im_width;
  memset(obj->info, 0, sizeof(cvai_object_info_t) * obj->size);

  for (uint32_t i = 0; i < obj->size; ++i) {
    obj->info[i].bbox.x1 = dets[i]->x1;
    obj->info[i].bbox.y1 = dets[i]->y1;
    obj->info[i].bbox.x2 = dets[i]->x2;
    obj->info[i].bbox.y2 = dets[i]->y2;
    obj->info[i].bbox.score = dets[i]->score;
    obj->info[i].classes = dets[i]->label;
  }
}

Yolov5::Yolov5() : Core(CVI_MEM_DEVICE) {}

int Yolov5::onModelOpened() {
  for (size_t j = 0; j < getNumOutputTensor(); j++) {
    TensorInfo oinfo = getOutputTensorInfo(j);
    std::string channel_name = oinfo.tensor_name.c_str();
    CVI_SHAPE output_shape = oinfo.shape;
    if (output_shape.dim[1] == 1200) {
      out_names_["output_1200"] = oinfo.tensor_name;
      out_len_ = output_shape.dim[2];
    } else if (output_shape.dim[1] == 4800) {
      out_names_["output_4800"] = oinfo.tensor_name;
    } else {
      out_names_["output_19200"] = oinfo.tensor_name;
    }
  }

  if (out_names_.count("output_1200") == 0 || out_names_.count("output_4800") == 0 ||
      out_names_.count("output_19200") == 0) {
    return CVIAI_FAILURE;
  }

  return CVIAI_SUCCESS;
}

Yolov5::~Yolov5() {}

int Yolov5::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Yolov5 only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  (*data)[0].factor[0] = p_preprocess_cfg_->factor[0];
  (*data)[0].factor[1] = p_preprocess_cfg_->factor[1];
  (*data)[0].factor[2] = p_preprocess_cfg_->factor[2];
  (*data)[0].mean[0] = p_preprocess_cfg_->mean[0];
  (*data)[0].mean[1] = p_preprocess_cfg_->mean[1];
  (*data)[0].mean[2] = p_preprocess_cfg_->mean[2];
  (*data)[0].format = p_preprocess_cfg_->format;
  (*data)[0].use_quantize_scale = p_preprocess_cfg_->use_quantize_scale;
  return CVIAI_SUCCESS;
}

int dump_frame_result_yolov5(const std::string &filepath, VIDEO_FRAME_INFO_S *frame) {
  FILE *fp = fopen(filepath.c_str(), "wb");
  if (fp == nullptr) {
    LOGE("failed to open: %s.\n", filepath.c_str());
    return CVI_FAILURE;
  }

  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    size_t image_size =
        frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
  }
  for (int c = 0; c < 3; c++) {
    uint8_t *paddr = (uint8_t *)frame->stVFrame.pu8VirAddr[c];
    // std::cout << "towrite channel:" << c << ",towritelen:" << frame->stVFrame.u32Length[c]
    //           << ",addr:" << (void *)paddr << std::endl;
    fwrite(paddr, frame->stVFrame.u32Length[c], 1, fp);
  }
  fclose(fp);
  return CVI_SUCCESS;
}

int Yolov5::vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                           VPSSConfig &vpss_config) {
  auto &vpssChnAttr = vpss_config.chn_attr;
  auto &factor = vpssChnAttr.stNormalize.factor;
  auto &mean = vpssChnAttr.stNormalize.mean;

  // set dump config
  vpssChnAttr.stNormalize.bEnable = false;
  vpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;

  VPSS_CHN_SQ_RB_HELPER(&vpssChnAttr, srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                        vpssChnAttr.u32Width, vpssChnAttr.u32Height, PIXEL_FORMAT_RGB_888_PLANAR,
                        factor, mean, false);
  int ret = mp_vpss_inst->sendFrame(srcFrame, &vpssChnAttr, &vpss_config.chn_coeff, 1);
  if (ret != CVI_SUCCESS) {
    LOGE("vpssPreprocess Send frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_SEND_FRAME;
  }

  ret = mp_vpss_inst->getFrame(dstFrame, 0, m_vpss_timeout);
  if (ret != CVI_SUCCESS) {
    LOGE("get frame failed: %s!\n", get_vpss_error_msg(ret));
    return CVIAI_ERR_VPSS_GET_FRAME;
  }
  // dump_frame_result_yolov5("vpss_processed", dstFrame);
  return CVIAI_SUCCESS;
}

void Yolov5::set_param(Yolov5PreParam *p_preprocess_cfg, YOLOV5AlgParam *p_yolov5_param) {
  p_preprocess_cfg_ = p_preprocess_cfg;
  p_yolov5_param_ = p_yolov5_param;
}

int Yolov5::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  if (ret != CVIAI_SUCCESS) {
    LOGE("Yolov5 run inference failed\n");
    return ret;
  }

  CVI_SHAPE shape = getInputShape(0);

  outputParser(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
               srcFrame->stVFrame.u32Height, obj_meta);
  model_timer_.TicToc("post");
  return CVIAI_SUCCESS;
}

void xywh2xxyy(float x, float y, float w, float h, PtrDectRect &det) {
  det->x1 = x - w / 2;
  det->y1 = y - h / 2;
  det->x2 = x + w / 2;
  det->y2 = y + h / 2;
}

void Yolov5::getYolov5Detections(float *ptr, int stride, int grid_len, uint32_t *anchor,
                                 Detections &vec_obj) {
  for (int anchor_idx = 0; anchor_idx < 3; anchor_idx++) {
    float pw = anchor[anchor_idx * 2];
    float ph = anchor[anchor_idx * 2 + 1];
    for (int grid_y = 0; grid_y < grid_len; grid_y++) {
      for (int grid_x = 0; grid_x < grid_len; grid_x++) {
        int start_idx = (anchor_idx * grid_len * grid_len + grid_y * grid_len + grid_x) * out_len_;
        float sigmoid_x = ptr[start_idx];
        float sigmoid_y = ptr[start_idx + 1];
        float sigmoid_w = ptr[start_idx + 2];
        float sigmoid_h = ptr[start_idx + 3];
        float obj_conf = ptr[start_idx + 4];
        int cls = yolov5_argmax(ptr, start_idx + 5, out_len_ - 5);
        float cls_conf = ptr[start_idx + 5 + cls];
        float object_score = cls_conf * obj_conf;

        // filter detections lowwer than conf_thresh
        if (obj_conf < p_yolov5_param_->conf_thresh) {
          continue;
        }
        PtrDectRect det = std::make_shared<object_detect_rect_t>();
        det->score = object_score;
        det->label = cls;
        // decode predicted bounding box of each grid to whole image
        float x = (2 * sigmoid_x - 0.5 + (float)grid_x) * (float)stride;
        float y = (2 * sigmoid_y - 0.5 + (float)grid_y) * (float)stride;
        float w = pow((sigmoid_w * 2), 2) * pw;
        float h = pow((sigmoid_h * 2), 2) * ph;
        xywh2xxyy(x, y, w, h, det);
        vec_obj.push_back(det);
      }
    }
  }
}

void Yolov5::clip_bbox(int frame_width, int frame_height, cvai_bbox_t *bbox) {
  if (bbox->x1 < 0) {
    bbox->x1 = 0;
  } else if (bbox->x1 > frame_width) {
    bbox->x1 = frame_width;
  }

  if (bbox->x2 < 0) {
    bbox->x2 = 0;
  } else if (bbox->x2 > frame_width) {
    bbox->x2 = frame_width;
  }

  if (bbox->y1 < 0) {
    bbox->y1 = 0;
  } else if (bbox->y1 > frame_height) {
    bbox->y1 = frame_height;
  }

  if (bbox->y2 < 0) {
    bbox->y2 = 0;
  } else if (bbox->y2 > frame_height) {
    bbox->y2 = frame_height;
  }
}

cvai_bbox_t Yolov5::yolov5_box_rescale(int frame_width, int frame_height, int width, int height,
                                       cvai_bbox_t bbox) {
  cvai_bbox_t rescale_bbox;
  int max_board = max_val(frame_width, frame_height);
  float ratio = float(max_board) / float(width);
  rescale_bbox.x1 = int(bbox.x1 * ratio);
  rescale_bbox.x2 = int(bbox.x2 * ratio);
  rescale_bbox.y1 = int(bbox.y1 * ratio);
  rescale_bbox.y2 = int(bbox.y2 * ratio);
  rescale_bbox.score = bbox.score;
  clip_bbox(frame_width, frame_height, &rescale_bbox);
  return rescale_bbox;
}

void Yolov5::Yolov5PostProcess(Detections &dets, int frame_width, int frame_height,
                               cvai_object_t *obj_meta) {
  Detections final_dets = nms_multi_class(dets, p_yolov5_param_->nms_thresh);
  CVI_SHAPE shape = getInputShape(0);
  convert_det_struct(final_dets, obj_meta, shape.dim[2], shape.dim[3]);
  // rescale bounding box to original image
  if (!hasSkippedVpssPreprocess()) {
    for (uint32_t i = 0; i < obj_meta->size; ++i) {
      obj_meta->info[i].bbox = yolov5_box_rescale(frame_width, frame_height, obj_meta->width,
                                                  obj_meta->height, obj_meta->info[i].bbox);
    }
    obj_meta->width = frame_width;
    obj_meta->height = frame_height;
  }
}

void Yolov5::outputParser(const int image_width, const int image_height, const int frame_width,
                          const int frame_height, cvai_object_t *obj_meta) {
  Detections vec_obj;
  float *ptr0 = getOutputRawPtr<float>(out_names_["output_19200"]);
  getYolov5Detections(ptr0, 8, 80, *p_yolov5_param_->anchors[0], vec_obj);

  float *ptr1 = getOutputRawPtr<float>(out_names_["output_4800"]);
  getYolov5Detections(ptr1, 16, 40, *p_yolov5_param_->anchors[1], vec_obj);

  float *ptr2 = getOutputRawPtr<float>(out_names_["output_1200"]);
  getYolov5Detections(ptr2, 32, 20, *p_yolov5_param_->anchors[2], vec_obj);

  Yolov5PostProcess(vec_obj, frame_width, frame_height, obj_meta);
}

// namespace cviai
}  // namespace cviai
