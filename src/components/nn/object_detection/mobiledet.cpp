#include "object_detection/mobiledet.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

static const float STD_R = (255.0 * 0.229);
static const float STD_G = (255.0 * 0.224);
static const float STD_B = (255.0 * 0.225);
static const float MODEL_MEAN_R = 0.485 * 255.0;
static const float MODEL_MEAN_G = 0.456 * 255.0;
static const float MODEL_MEAN_B = 0.406 * 255.0;

#define FACTOR_R (1.0 / STD_R)
#define FACTOR_G (1.0 / STD_G)
#define FACTOR_B (1.0 / STD_B)
#define MEAN_R (MODEL_MEAN_R / STD_R)
#define MEAN_G (MODEL_MEAN_G / STD_G)
#define MEAN_B (MODEL_MEAN_B / STD_B)

#define unlikely(x) __builtin_expect(!!(x), 0)

using namespace std;

using MDetV2Config = MobileDetV2Detection::CvimodelInfo;
int get_level(int val) {
  int num_level = 0;
  while (val != 1) {
    val = val / 2;
    num_level += 1;
  }
  return num_level;
}

static void decode_box(const float *const box, const AnchorBox &anchor,
                       ObjectBoxInfo &det) {
  float ycenter_a = anchor.y + anchor.h / 2;
  float xcenter_a = anchor.x + anchor.w / 2;

  float ty = box[0];
  float tx = box[1];
  float th = box[2];
  float tw = box[3];

  float w = std::exp(tw) * anchor.w;
  float h = std::exp(th) * anchor.h;
  float ycenter = ty * anchor.h + ycenter_a;
  float xcenter = tx * anchor.w + xcenter_a;
  det.x1 = xcenter - w / 2;
  det.y1 = ycenter - h / 2;
  det.x2 = xcenter + w / 2;
  det.y2 = ycenter + h / 2;
}

void DequantizeScale(const int8_t *q_data, float *data, float dequant_scale,
                     size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = float(q_data[i]) * dequant_scale;
  }
}

void clip_bbox(const size_t image_width, const size_t image_height,
               ObjectBoxInfo &box) {
  if (box.x1 < 0) box.x1 = 0;
  if (box.y1 < 0) box.y1 = 0;
  if (box.x2 < 0) box.x2 = 0;
  if (box.y2 < 0) box.y2 = 0;

  if (box.x1 >= image_width) box.x1 = image_width - 1;
  if (box.y1 >= image_height) box.y1 = image_height - 1;
  if (box.x2 >= image_width) box.x2 = image_width - 1;
  if (box.y2 >= image_height) box.y2 = image_height - 1;
}

static std::vector<int8_t> constructInverseThresh(
    float threshld, std::vector<int> strides,
    std::map<int, float> dequant_thresh) {
  std::vector<int8_t> inverse_threshold;
  float inverse_th = std::log(threshld / (1 - threshld));
  for (int stride : strides) {
    int8_t quant_score_thresh =
        static_cast<int8_t>(round(inverse_th / dequant_thresh[stride]));

    inverse_threshold.push_back(quant_score_thresh);
  }

  return inverse_threshold;
}

MobileDetV2Detection::MobileDetV2Detection(
    MobileDetV2Detection::Category category, float iou_thresh)
    : m_model_config(CvimodelInfo::create_config(category)),
      m_iou_threshold(iou_thresh) {
  m_model_threshold = m_model_config.default_score_threshold;
  m_quant_inverse_score_threshold =
      constructInverseThresh(m_model_threshold, m_model_config.strides,
                             m_model_config.class_dequant_thresh);

  // m_filter.set();

  net_param_.pre_params.scale[0] = static_cast<float>(FACTOR_R);
  net_param_.pre_params.scale[1] = static_cast<float>(FACTOR_G);
  net_param_.pre_params.scale[2] = static_cast<float>(FACTOR_B);
  net_param_.pre_params.mean[0] = static_cast<float>(MEAN_R);
  net_param_.pre_params.mean[1] = static_cast<float>(MEAN_G);
  net_param_.pre_params.mean[2] = static_cast<float>(MEAN_B);

  net_param_.pre_params.dst_image_format = ImageFormat::RGB_PLANAR;
  //   preprocess_params_[0].rescale_type = RESCALE_RB;
  // #ifndef __CV186X__
  //   preprocess_params_[0].resize_method = VPSS_SCALE_COEF_OPENCV_BILINEAR;
  // #endif
}

MobileDetV2Detection::~MobileDetV2Detection() {}

int32_t MobileDetV2Detection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  auto input_shape = net_->getTensorInfo(input_layer).shape;

  m_model_config.image_height = input_shape[2];
  m_model_config.image_width = input_shape[3];

  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();

  int not_named_tensor = 0;
  for (size_t j = 0; j < num_output; j++) {
    if (output_layers[j].find("class_stride") == std::string::npos &&
        output_layers[j].find("box_stride") == std::string::npos) {
      printf("found not named tensor:%s\n", output_layers[j].c_str());
      not_named_tensor += 1;
    }
    printf("named tensor:%s\n", output_layers[j].c_str());
  }
  printf("not_named_tensor:%d\n", not_named_tensor);
  if (not_named_tensor > 0) {
    int input_w = input_shape[3];
    int input_h = input_shape[2];
    int num_per_grid =
        m_model_config.num_scales * m_model_config.aspect_ratios.size();
    int num_cls = m_model_config.num_classes;
    m_model_config.bbox_out_names.clear();
    m_model_config.class_out_names.clear();
    m_model_config.strides.clear();
    for (size_t j = 0; j < num_output; j++) {
      auto oinfo = net_->getTensorInfo(output_layers[j]);
      int feat_w = oinfo.shape[2];
      int feat_h = oinfo.shape[1];
      int channel = oinfo.shape[3];
      int stridew = input_w / feat_w;
      int strideh = input_h / feat_h;
      if (stridew != strideh) {
        LOGE("stride not equal,stridew:%d,strideh:%d,featw:%d,feath:%d\n",
             stridew, strideh, feat_w, feat_h);
      }
      if (channel == num_cls * num_per_grid) {
        m_model_config.class_out_names[stridew] = output_layers[j];
        printf("parse class tensor name:%s,stride:%d\n",
               output_layers[j].c_str(), stridew);
        m_model_config.strides.push_back(stridew);
      } else if (channel == 4 * num_per_grid) {
        m_model_config.bbox_out_names[stridew] = output_layers[j];
        printf("parse bbox tensor name:%s,stride:%d\n",
               output_layers[j].c_str(), stridew);
      } else {
        printf("error parse bbox tensor name:%s,stride:%d\n",
               output_layers[j].c_str(), stridew);
        LOGE("unexpected branch,channel:%d,name:%s", channel,
             output_layers[j].c_str());
      }
    }

    std::sort(m_model_config.strides.begin(), m_model_config.strides.end(),
              [](const int a, const int b) { return a < b; });
    m_model_config.min_level = get_level(m_model_config.strides[0]);
    m_model_config.max_level =
        get_level(m_model_config.strides[m_model_config.strides.size() - 1]);
    printf("minlevel:%d,maxlevel:%d\n", int(m_model_config.min_level),
           int(m_model_config.max_level));
  }

  RetinaNetAnchorGenerator generator = RetinaNetAnchorGenerator(
      m_model_config.min_level, m_model_config.max_level,
      m_model_config.num_scales, m_model_config.aspect_ratios,
      m_model_config.anchor_scale, m_model_config.image_width,
      m_model_config.image_height);
  m_anchors = generator.get_anchor_boxes();

  for (auto pair : m_model_config.class_out_names) {
    int stride = pair.first;
    string name = pair.second;
    m_model_config.class_dequant_thresh[stride] =
        net_->getTensorInfo(name).qscale;
  }

  for (auto pair : m_model_config.bbox_out_names) {
    int stride = pair.first;
    string name = pair.second;
    m_model_config.bbox_dequant_thresh[stride] =
        net_->getTensorInfo(name).qscale;
  }

  m_quant_inverse_score_threshold =
      constructInverseThresh(m_model_threshold, m_model_config.strides,
                             m_model_config.class_dequant_thresh);
  return 0;
}

int32_t MobileDetV2Detection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);

  printf("outputParse,batch size:%d,input shape:%d,%d,%d,%d\n", images.size(),
         input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
         input_tensor.shape[3]);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;

    generate_dets_for_each_stride(lb_boxes);

    DetectionHelper::nmsObjects(lb_boxes, m_iou_threshold);

    std::vector<float> scale_params = batch_rescale_params_[b];
    int num_obj = 0;
    std::shared_ptr<ModelBoxInfo> obj = std::make_shared<ModelBoxInfo>();
    obj->image_width = image_width;
    obj->image_height = image_height;
    for (auto &bbox : lb_boxes) {
      num_obj += bbox.second.size();
      for (auto &b : bbox.second) {
        DetectionHelper::rescaleBbox(b, scale_params,
                                     net_param_.pre_params.crop_x,
                                     net_param_.pre_params.crop_y);
        if (type_mapping_.count(b.class_id)) {
          LOGI("class_id: %d, object_type: %d\n", b.class_id,
               type_mapping_[b.class_id]);
          b.object_type = type_mapping_[b.class_id];
        }
        obj->bboxes.push_back(b);
      }
    }
    LOGI("batch:%d,num_obj:%d", b, num_obj);
    out_datas.push_back(obj);
  }
  return 0;
}

void MobileDetV2Detection::generate_dets_for_tensor(
    std::map<int, std::vector<ObjectBoxInfo>> &det_vec,
    float class_dequant_thresh, float bbox_dequant_thresh, int8_t quant_thresh,
    const int8_t *logits, const int8_t *objectness, int8_t *bboxes,
    size_t class_tensor_size, const vector<AnchorBox> &anchors) {
  for (size_t obj_index = 0; obj_index < class_tensor_size; obj_index++) {
    if (unlikely(*(objectness + obj_index) >= quant_thresh)) {
      // create detection if any object exists in this grid
      size_t score_index = obj_index * m_model_config.num_classes;
      size_t end = score_index + m_model_config.num_classes;

      // find objects in this grid
      for (size_t class_idx = score_index; class_idx < end; class_idx++) {
        if (logits[class_idx] >= quant_thresh) {
          size_t anchor_index = class_idx / m_model_config.num_classes;
          size_t box_index = anchor_index * 4;
          // PtrDectRect det = make_shared<object_detect_rect_t>();
          ObjectBoxInfo det;
          int det_label = class_idx - score_index;
          // det_label =
          // m_model_config.class_id_map(det_label);//TODO(fuquan.ke):fix with
          // typeMapping
          float dequant_logits = logits[class_idx] * class_dequant_thresh;
          det.score = 1.0 / (1.0 + std::exp(-dequant_logits));

          float dequant_box[4];
          DequantizeScale(bboxes + box_index, dequant_box, bbox_dequant_thresh,
                          4);
          decode_box(dequant_box, anchors[box_index / 4], det);
          clip_bbox(m_model_config.image_width, m_model_config.image_height,
                    det);
          float width = det.x2 - det.x1;
          float height = det.y2 - det.y1;
          det.class_id = det_label;
          if (width > 1 && height > 1) {
            det_vec[det_label].push_back(det);
          }
        }
      }
    }
  }
}

void MobileDetV2Detection::generate_dets_for_each_stride(
    std::map<int, std::vector<ObjectBoxInfo>> &det_vec) {
  vector<pair<int8_t *, size_t>> cls_raw_out;
  vector<pair<int8_t *, size_t>> objectness_raw_out;
  vector<pair<int8_t *, size_t>> bbox_raw_out;
  get_raw_outputs(&cls_raw_out, &objectness_raw_out, &bbox_raw_out);

  auto class_thresh_iter = m_model_config.class_dequant_thresh.begin();
  auto bbox_thresh_iter = m_model_config.bbox_dequant_thresh.begin();

  for (size_t stride_index = 0; stride_index < cls_raw_out.size();
       stride_index++) {
    generate_dets_for_tensor(
        det_vec, class_thresh_iter->second, bbox_thresh_iter->second,
        m_quant_inverse_score_threshold[stride_index],
        cls_raw_out[stride_index].first, objectness_raw_out[stride_index].first,
        bbox_raw_out[stride_index].first,
        static_cast<int>(objectness_raw_out[stride_index].second),
        m_anchors[stride_index]);

    class_thresh_iter++;
    bbox_thresh_iter++;
  }
}

void MobileDetV2Detection::get_raw_outputs(
    std::vector<pair<int8_t *, size_t>> *cls_tensor_ptr,
    std::vector<pair<int8_t *, size_t>> *objectness_tensor_ptr,
    std::vector<pair<int8_t *, size_t>> *bbox_tensor_ptr) {
  for (auto stride : m_model_config.strides) {
    {
      const TensorInfo &info =
          net_->getTensorInfo(m_model_config.class_out_names[stride]);
      cls_tensor_ptr->push_back(
          {reinterpret_cast<int8_t *>(info.sys_mem), info.tensor_elem});
    }

    {
      const TensorInfo &info =
          net_->getTensorInfo(m_model_config.bbox_out_names[stride]);
      bbox_tensor_ptr->push_back(
          {reinterpret_cast<int8_t *>(info.sys_mem), info.tensor_elem});
    }

    {
      const string tname = m_model_config.num_classes > 1
                               ? m_model_config.obj_max_names[stride]
                               : m_model_config.class_out_names[stride];
      const TensorInfo &info = net_->getTensorInfo(tname);
      objectness_tensor_ptr->push_back(
          {reinterpret_cast<int8_t *>(info.sys_mem), info.tensor_elem});
    }
  }
}

void MobileDetV2Detection::setModelThreshold(const float &threshold) {
  if (m_model_threshold != threshold) {
    m_model_threshold = threshold;
    m_quant_inverse_score_threshold =
        constructInverseThresh(m_model_threshold, m_model_config.strides,
                               m_model_config.class_dequant_thresh);
  }
}

MDetV2Config MDetV2Config::create_config(MobileDetV2Detection::Category model) {
  MDetV2Config config;
  config.min_level = 3;
  config.max_level = 7;
  config.num_scales = 3;
  config.aspect_ratios = {{1.0, 1.0}, {1.4, 0.7}, {0.7, 1.4}};
  config.anchor_scale = 4.0;
  config.strides = {8, 16, 32, 64, 128};
  config.class_out_names = {{8, "class_stride_8"},
                            {16, "class_stride_16"},
                            {32, "class_stride_32"},
                            {64, "class_stride_64"},
                            {128, "class_stride_128"}};

  config.obj_max_names = {{8, "class_stride_8_obj_max"},
                          {16, "class_stride_16_obj_max"},
                          {32, "class_stride_32_obj_max"},
                          {64, "class_stride_64_obj_max"},
                          {128, "class_stride_128_obj_max"}};

  config.bbox_out_names = {{8, "box_stride_8"},
                           {16, "box_stride_16"},
                           {32, "box_stride_32"},
                           {64, "box_stride_64"},
                           {128, "box_stride_128"}};
  config.default_score_threshold = 0.4;
  switch (model) {
    case Category::coco80:
      config.num_classes = 90;
      // config.class_id_map = [](int orig_id) { return orig_id; };
      break;
    case Category::person_vehicle:
      config.num_classes = 6;
      // config.class_id_map = [](int orig_id) {
      //   if (orig_id == 0) return static_cast<int>(TDL_DET_TYPE_PERSON);
      //   if (orig_id == 1) return static_cast<int>(TDL_DET_TYPE_BICYCLE);
      //   if (orig_id == 2) return static_cast<int>(TDL_DET_TYPE_CAR);
      //   if (orig_id == 3) return
      //   static_cast<int>(TDL_DET_TYPE_MOTORBIKE); if (orig_id == 4)
      //   return static_cast<int>(TDL_DET_TYPE_BUS); if (orig_id == 5)
      //   return static_cast<int>(TDL_DET_TYPE_TRUCK); return
      //   static_cast<int>(TDL_DET_TYPE_END);
      // };
      break;
    case Category::person_pets:
      config.num_classes = 3;
      // config.class_id_map = [](int orig_id) {
      //   if (orig_id == 0) return static_cast<int>(TDL_DET_TYPE_PERSON);
      //   if (orig_id == 1) return static_cast<int>(TDL_DET_TYPE_CAT);
      //   if (orig_id == 2) return static_cast<int>(TDL_DET_TYPE_DOG);
      //   return static_cast<int>(TDL_DET_TYPE_END);
      // };
      break;
    case Category::vehicle:
      config.num_classes = 3;
      // config.class_id_map = [](int orig_id) {
      //   if (orig_id == 0) return static_cast<int>(TDL_DET_TYPE_CAR);
      //   if (orig_id == 1) return static_cast<int>(TDL_DET_TYPE_TRUCK);
      //   if (orig_id == 2) return
      //   static_cast<int>(TDL_DET_TYPE_MOTORBIKE); return
      //   static_cast<int>(TDL_DET_TYPE_END);
      // };
      break;
    case Category::pedestrian:
      config.num_classes = 1;
      // config.class_id_map = [](int orig_id) {
      //   if (orig_id == 0) return static_cast<int>(TDL_DET_TYPE_PERSON);
      //   return static_cast<int>(TDL_DET_TYPE_END);
      // };
      break;
  }

  return config;
}
