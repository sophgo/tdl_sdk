#include "retina_face.hpp"

#include "cviai_types_free.h"
#include "retina_face_utils.hpp"

#define FACE_THRESHOLD 0.5
#define NAME_BBOX "face_rpn_bbox_pred_"
#define NAME_SCORE "face_rpn_cls_score_reshape_"
#define NAME_LANDMARK "face_rpn_landmark_pred_"
#define FACE_POINTS_SIZE 5

namespace cviai {

RetinaFace::RetinaFace() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_preprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
}

RetinaFace::~RetinaFace() {}

int RetinaFace::initAfterModelOpened() {
  std::vector<anchor_cfg> cfg;
  anchor_cfg tmp;
  tmp.SCALES = {32, 16};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 32;
  cfg.push_back(tmp);

  tmp.SCALES = {8, 4};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 16;
  cfg.push_back(tmp);

  tmp.SCALES = {2, 1};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 8;
  cfg.push_back(tmp);

  std::vector<std::vector<anchor_box>> anchors_fpn = generate_anchors_fpn(false, cfg);
  std::map<std::string, std::vector<anchor_box>> anchors_fpn_map;
  for (size_t i = 0; i < anchors_fpn.size(); i++) {
    std::string key = "stride" + std::to_string(m_feat_stride_fpn[i]) + "_dequant";
    anchors_fpn_map[key] = anchors_fpn[i];
    m_num_anchors[key] = anchors_fpn[i].size();
  }

  for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
    std::string key = "stride" + std::to_string(m_feat_stride_fpn[i]) + "_dequant";
    std::string landmark_str = NAME_LANDMARK + key;
    CVI_TENSOR *out = CVI_NN_GetTensorByName(landmark_str.c_str(), mp_output_tensors, m_output_num);
    CVI_SHAPE landmark_shape = CVI_NN_TensorShape(out);
    int stride = m_feat_stride_fpn[i];

    m_anchors[landmark_str] =
        anchors_plane(landmark_shape.dim[2], landmark_shape.dim[3], stride, anchors_fpn_map[key]);
  }

  return CVI_RC_SUCCESS;
}

int RetinaFace::inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_face_t *meta, int *face_count) {
  int ret = run(srcFrame);

  CVI_TENSOR *input = getInputTensor(0);
  float ratio = 1.0;
  int image_width = input->shape.dim[3];
  int image_height = input->shape.dim[2];
  std::vector<cvai_face_info_t> faceList;
  std::vector<cvai_face_info_t> BBoxes;
  outputParser(ratio, image_width, image_height, &BBoxes, &faceList);

  initFaceMeta(meta, faceList.size());
  meta->width = image_width;
  meta->height = image_height;

  *face_count = meta->size;
  for (int i = 0; i < meta->size; ++i) {
    meta->face_info[i].bbox.x1 = faceList[i].bbox.x1;
    meta->face_info[i].bbox.x2 = faceList[i].bbox.x2;
    meta->face_info[i].bbox.y1 = faceList[i].bbox.y1;
    meta->face_info[i].bbox.y2 = faceList[i].bbox.y2;

    for (int j = 0; j < 5; ++j) {
      meta->face_info[i].face_pts.x[j] = faceList[i].face_pts.x[j];
      meta->face_info[i].face_pts.y[j] = faceList[i].face_pts.y[j];
    }
  }

  for (size_t i = 0; i < BBoxes.size(); ++i) {
    CVI_AI_FreeCpp(&BBoxes[i].face_pts);
  }

  return ret;
}

void RetinaFace::outputParser(float ratio, int image_width, int image_height,
                              std::vector<cvai_face_info_t> *BBoxes,
                              std::vector<cvai_face_info_t> *bboxes_nms) {
  for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
    std::string key = "stride" + std::to_string(m_feat_stride_fpn[i]) + "_dequant";

    std::string score_str = NAME_SCORE + key;
    CVI_TENSOR *out = CVI_NN_GetTensorByName(score_str.c_str(), mp_output_tensors, m_output_num);
    float *score_blob = (float *)CVI_NN_TensorPtr(out);
    CVI_SHAPE score_shape = CVI_NN_TensorShape(out);
    size_t score_size =
        score_shape.dim[0] * score_shape.dim[1] * score_shape.dim[2] * score_shape.dim[3];
    softmax_by_channel(
        score_blob, score_blob,
        {score_shape.dim[0], score_shape.dim[1], score_shape.dim[2], score_shape.dim[3]});
    score_blob += score_size / 2;

    std::string bbox_str = NAME_BBOX + key;
    out = CVI_NN_GetTensorByName(bbox_str.c_str(), mp_output_tensors, m_output_num);
    float *bbox_blob = (float *)CVI_NN_TensorPtr(out);

    std::string landmark_str = NAME_LANDMARK + key;
    out = CVI_NN_GetTensorByName(landmark_str.c_str(), mp_output_tensors, m_output_num);
    float *landmark_blob = (float *)CVI_NN_TensorPtr(out);
    CVI_SHAPE landmark_shape = CVI_NN_TensorShape(out);

    int width = landmark_shape.dim[3];
    int height = landmark_shape.dim[2];
    size_t count = width * height;
    size_t num_anchor = m_num_anchors[key];

    std::vector<anchor_box> anchors = m_anchors[landmark_str];
    for (size_t num = 0; num < num_anchor; num++) {
      for (size_t j = 0; j < count; j++) {
        float conf = score_blob[j + count * num];
        if (conf <= FACE_THRESHOLD) {
          continue;
        }
        cvai_face_info_t box;
        box.face_pts.size = 5;
        box.face_pts.x = (float *)malloc(sizeof(float) * box.face_pts.size);
        box.face_pts.y = (float *)malloc(sizeof(float) * box.face_pts.size);
        box.bbox.score = conf;

        cv::Vec4f regress;
        float dx = bbox_blob[j + count * (0 + num * 4)];
        float dy = bbox_blob[j + count * (1 + num * 4)];
        float dw = bbox_blob[j + count * (2 + num * 4)];
        float dh = bbox_blob[j + count * (3 + num * 4)];
        regress = cv::Vec4f(dx, dy, dw, dh);
        bbox_pred(anchors[j + count * num], regress, ratio, box.bbox);

        for (size_t k = 0; k < box.face_pts.size; k++) {
          box.face_pts.x[k] = landmark_blob[j + count * (num * 10 + k * 2)];
          box.face_pts.y[k] = landmark_blob[j + count * (num * 10 + k * 2 + 1)];
        }
        landmark_pred(anchors[j + count * num], ratio, box.face_pts);

        BBoxes->push_back(box);
      }
    }
  }

  bboxes_nms->clear();
  NonMaximumSuppression(*BBoxes, *bboxes_nms, 0.4, 'u');

  for (auto &box : *bboxes_nms) {
    clip_boxes(image_width, image_height, box.bbox);
  }
}

void RetinaFace::initFaceMeta(cvai_face_t *meta, int size) {
  meta->size = size;
  if (meta->size == 0) return;

  meta->face_info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * meta->size);

  memset(meta->face_info, 0, sizeof(cvai_face_info_t) * meta->size);

  for (int i = 0; i < meta->size; ++i) {
    meta->face_info[i].bbox.x1 = -1;
    meta->face_info[i].bbox.x2 = -1;
    meta->face_info[i].bbox.y1 = -1;
    meta->face_info[i].bbox.y2 = -1;

    meta->face_info[i].name[0] = '\0';
    meta->face_info[i].emotion = EMOTION_UNKNOWN;
    meta->face_info[i].gender = GENDER_UNKNOWN;
    meta->face_info[i].race = RACE_UNKNOWN;
    meta->face_info[i].age = -1;
    meta->face_info[i].liveness_score = -1;
    meta->face_info[i].mask_score = -1;

    CVI_AI_FreeCpp(&meta->face_info[i].face_pts);
    meta->face_info[i].face_pts.size = FACE_POINTS_SIZE;
    meta->face_info[i].face_pts.x =
        (float *)malloc(sizeof(float) * meta->face_info[i].face_pts.size);
    meta->face_info[i].face_pts.y =
        (float *)malloc(sizeof(float) * meta->face_info[i].face_pts.size);
    for (uint32_t j = 0; j < meta->face_info[i].face_pts.size; ++j) {
      meta->face_info[i].face_pts.x[j] = -1;
      meta->face_info[i].face_pts.y[j] = -1;
    }
  }
}

}  // namespace cviai
