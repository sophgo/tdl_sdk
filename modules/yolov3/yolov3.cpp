#include "yolov3.hpp"
#include "core_utils.hpp"

#define YOLOV3_CLASSES 80
#define YOLOV3_CONF_THRESHOLD 0.5
#define YOLOV3_NMS_THRESHOLD 0.45
#define YOLOV3_ANCHOR_NUM 3
#define YOLOV3_COORDS 4
#define YOLOV3_DEFAULT_DET_BUFFER 100
#define YOLOV3_SCALE (float)((1 / 255.0) * (128.0 / 1.00000488758))
#define YOLOV3_OUTPUT1 "layer82-conv_dequant"
#define YOLOV3_OUTPUT2 "layer94-conv_dequant"
#define YOLOV3_OUTPUT3 "layer106-conv_dequant"

static std::vector<std::string> names = {
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

using namespace std;

namespace cviai {

Yolov3::Yolov3() {
  mp_config = new ModelConfig;
  mp_config->skip_preprocess = true;
  mp_config->input_mem_type = 2;

  m_input_scale = YOLOV3_SCALE;

  m_yolov3_param = {
      YOLOV3_CLASSES,                                                                  // m_classes
      {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326},  // m_biases
      YOLOV3_CONF_THRESHOLD,             // m_threshold
      YOLOV3_NMS_THRESHOLD,              // m_nms_threshold
      YOLOV3_ANCHOR_NUM,                 // m_anchor_nums
      YOLOV3_COORDS,                     // m_coords
      1,                                 // m_batch
      v3,                                // type
      {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}}  // m_mask
  };
}

Yolov3::~Yolov3() {}

int Yolov3::inference(VIDEO_FRAME_INFO_S *srcFrame, cvi_object_t *obj,
                      cvi_obj_det_type_t det_type) {
  int ret = run(srcFrame);

  outputParser(obj, det_type);

  return ret;
}

void Yolov3::outputParser(cvi_object_t *obj, cvi_obj_det_type_t det_type) {
  vector<float *> features;
  vector<string> output_name = {YOLOV3_OUTPUT1, YOLOV3_OUTPUT2, YOLOV3_OUTPUT3};
  vector<CVI_SHAPE> output_shape;
  for (int i = 0; i < m_output_num; i++) {
    CVI_TENSOR *out =
        CVI_NN_GetTensorByName(output_name[i].c_str(), mp_output_tensors, m_output_num);
    output_shape.push_back(CVI_NN_TensorShape(out));
    features.push_back((float *)CVI_NN_TensorPtr(out));
  }

  CVI_TENSOR *input = getInputTensor(0);
  int yolov3_w = input->shape.dim[2];
  int yolov3_h = input->shape.dim[3];

  // Yolov3 has 3 different size outputs
  vector<YOLOLayer> net_outputs;
  for (unsigned int i = 0; i < features.size(); i++) {
    YOLOLayer l = {features[i], int(output_shape[i].dim[0]), int(output_shape[i].dim[1]),
                   int(output_shape[i].dim[2]), int(output_shape[i].dim[3])};
    net_outputs.push_back(l);
  }

  vector<object_detect_rect_t> results;
  static int det_buf_size = YOLOV3_DEFAULT_DET_BUFFER;
  static detection *total_dets = new detection[det_buf_size];
  int total_boxes = 0;

  for (size_t i = 0; i < net_outputs.size(); i++) {
    int nboxes = 0;

    doYolo(net_outputs.at(i));
    detection *dets =
        GetNetworkBoxes(net_outputs.at(i), m_yolov3_param.m_classes, yolov3_w, yolov3_h,
                        m_yolov3_param.m_threshold, 1, &nboxes, m_yolov3_param, i);

    int next_size = total_boxes + nboxes;
    if (next_size > det_buf_size) {
      total_dets = (detection *)realloc(total_dets, next_size * sizeof(detection));
      det_buf_size = next_size;
    }

    memcpy(total_dets + total_boxes, dets, sizeof(detection) * nboxes);
    total_boxes += nboxes;

    // we do not use FreeDetections because we use just use memcpy,
    // FreeDetections will free det.prob
    delete[] dets;
  }

  DoNmsSort(total_dets, total_boxes, m_yolov3_param.m_classes, m_yolov3_param.m_nms_threshold);
  getYOLOResults(total_dets, total_boxes, m_yolov3_param.m_threshold, yolov3_h, yolov3_w, results,
                 det_type);
  for (int i = 0; i < total_boxes; ++i) {
    delete[] total_dets[i].prob;
  }

  // fill obj
  obj->size = results.size();
  obj->objects = (cvi_object_info_t *)malloc(sizeof(cvi_object_info_t) * obj->size);
  obj->width = yolov3_w;
  obj->height = yolov3_h;

  memset(obj->objects, 0, sizeof(cvi_object_info_t) * obj->size);
  for (int i = 0; i < obj->size; ++i) {
    obj->objects[i].bbox.x1 = results[i].x1;
    obj->objects[i].bbox.y1 = results[i].y1;
    obj->objects[i].bbox.x2 = results[i].x2;
    obj->objects[i].bbox.y2 = results[i].y2;
    obj->objects[i].bbox.score = results[i].score;
    obj->objects[i].classes = results[i].label;
    strncpy(obj->objects[i].name, names[results[i].label].c_str(), sizeof(obj->objects[i].name));
    printf("YOLO3: %s (%d): %lf %lf %lf %lf, score=%.2f\n", obj->objects[i].name,
           obj->objects[i].classes, obj->objects[i].bbox.x1, obj->objects[i].bbox.x2,
           obj->objects[i].bbox.y1, obj->objects[i].bbox.y2, results[i].score);
  }
}

void Yolov3::doYolo(YOLOLayer &l) {
  float *data = l.data;
  int w = l.width;
  int h = l.height;
  int output_size = l.norm * l.channels * w * h;

  for (int b = 0; b < m_yolov3_param.m_batch; ++b) {
    for (int p = 0; p < w * h; ++p) {
      for (int n = 0; n < m_yolov3_param.m_anchor_nums; ++n) {
        int obj_index = EntryIndex(w, h, m_yolov3_param.m_classes, b, n * w * h + p,
                                   m_yolov3_param.m_coords, output_size);
        ActivateArray(data + obj_index, 1, true);
        float objectness = data[obj_index];

        if (objectness >= m_yolov3_param.m_threshold) {
          int box_index =
              EntryIndex(w, h, m_yolov3_param.m_classes, b, n * w * h + p, 0, output_size);
          ActivateArray(data + box_index, 1, true);
          ActivateArray(data + box_index + (w * h), 1, true);

          for (int j = 0; j < m_yolov3_param.m_classes; ++j) {
            int class_index = EntryIndex(w, h, m_yolov3_param.m_classes, b, n * w * h + p,
                                         4 + 1 + j, output_size);
            ActivateArray(data + class_index, 1, true);
          }
        }
      }
    }
  }
}

void Yolov3::getYOLOResults(detection *dets, int num, float threshold, int ori_w, int ori_h,
                            vector<object_detect_rect_t> &results, cvi_obj_det_type_t det_type) {
  for (int i = 0; i < num; ++i) {
    std::string labelstr = "";
    int obj_class = -1;
    object_detect_rect_t obj_result;
    obj_result.score = 0;
    obj_result.label = obj_class;
    for (int j = 0; j < m_yolov3_param.m_classes; ++j) {
      if (dets[i].prob[j] > threshold) {
        if (obj_class < 0) {
          labelstr = names[j];
          obj_class = j;
          obj_result.label = obj_class;
          obj_result.score = dets[i].prob[j];
        } else {
          labelstr += ", " + names[j];
          if (dets[i].prob[j] > obj_result.score) {
            obj_result.score = dets[i].prob[j];
            obj_result.label = obj_class;
          }
        }
      }
    }

    if (obj_class < 0) {
      continue;
    }

    bool skip_class = (det_type & CVI_DET_TYPE_ALL);
    if ((det_type & CVI_DET_TYPE_VEHICLE)) {
      if ((obj_result.label >= 1) && obj_result.label <= 7) skip_class = false;
    }
    if ((det_type & CVI_DET_TYPE_PEOPLE)) {
      if (obj_result.label == 0) skip_class = false;
    }
    if ((det_type & CVI_DET_TYPE_PET)) {
      if ((obj_result.label == 16) || (obj_result.label == 17)) skip_class = false;
    }
    if (skip_class) continue;

    box b = dets[i].bbox;
    int left = (b.x - b.w / 2.) * ori_w;
    int right = (b.x + b.w / 2.) * ori_w;
    int top = (b.y - b.h / 2.) * ori_h;
    int bot = (b.y + b.h / 2.) * ori_h;
    if (left < 0) left = 0;
    if (right > ori_w - 1) right = ori_w - 1;
    if (top < 0) top = 0;
    if (bot > ori_h - 1) bot = ori_h - 1;

    object_detect_rect_t rect;

    rect.x1 = left;
    rect.y1 = top;
    rect.x2 = right;
    rect.y2 = bot;
    rect.label = obj_result.label;
    rect.score = obj_result.score;

    results.emplace_back(move(rect));
  }
}

}  // namespace cviai
