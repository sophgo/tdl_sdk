#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "retina_face.h"
#include "anchor_generator.h"
#include "cvi_face_utils.hpp"
#include "cv183x_facelib_v0.0.1.h"

#define RETINA_FACE_N               1
#define RETINA_FACE_C               3
#define RETINA_FACE_WIDTH           608
#define RETINA_FACE_HEIGHT          608
#define RETINA_FACE_SCALE_FACTOR    (float(128) / 255.001236)
#define FACE_THRESHOLD              0.5
#define NAME_BBOX                   "face_rpn_bbox_pred_"
#define NAME_SCORE                  "face_rpn_cls_score_reshape_"
#define NAME_LANDMARK               "face_rpn_landmark_pred_"

static CVI_MODEL_HANDLE model_handle;
static CVI_TENSOR *input_tensors;
static CVI_TENSOR *output_tensors;
static int32_t input_num;
static int32_t output_num;

using namespace std;

static vector<int> _feat_stride_fpn = {32, 16, 8};
static map<string, vector<anchor_box>> _anchors;
static map<string, int> _num_anchors;
static bool bNetworkInited = false;

static void bbox_pred(const anchor_box &anchor, cv::Vec4f regress, float ratio,
                      cvi_face_detect_rect_t &bbox) {
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = FastExp(regress[2]) * width;
    float pred_h = FastExp(regress[3]) * height;

    bbox.x1 = (pred_ctr_x - 0.5 * (pred_w - 1.0)) * ratio;
    bbox.y1 = (pred_ctr_y - 0.5 * (pred_h - 1.0)) * ratio;
    bbox.x2 = (pred_ctr_x + 0.5 * (pred_w - 1.0)) * ratio;
    bbox.y2 = (pred_ctr_y + 0.5 * (pred_h - 1.0)) * ratio;
}

static void landmark_pred(const anchor_box &anchor, float ratio, cvi_face_pts_t &facePt) {
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for(size_t j = 0; j < 5; j ++) {
        facePt.x[j] = (facePt.x[j] * width + ctr_x) * ratio;
        facePt.y[j] = (facePt.y[j] * height + ctr_y) * ratio;
    }
}

static int softmax_by_channel(float *input, float *output, const std::vector<int64_t>& shape) {
    int *iter = new int[shape[1]];
    float *ex = new float[shape[1]];

    for (int N = 0; N < shape[0]; ++N) {
        for (int H = 0; H < shape[2]; ++H) {
            for (int W = 0; W < shape[3]; ++W) {
                float max_val = std::numeric_limits<float>::lowest();
                for (int C = 0; C < shape[1]; ++C) {
                    iter[C] = (N * shape[1] * shape[2] * shape[3]) 
                           + (C * shape[2] * shape[3]) + (H * shape[3]) + W;
                }

                for (int C = 0; C < shape[1]; ++C) {
                    max_val = std::max(input[iter[C]], max_val);
                }

                // find softmax divisor
                float sum_of_ex = 0.0f;
                for (int C = 0; C < shape[1]; ++C) {
                    float x = input[iter[C]] - max_val;
                    ex[C] = FastExp(x);
                    sum_of_ex += ex[C];
                }

                // calculate softmax
                for (int C = 0; C < shape[1]; ++C) {
                    output[iter[C]] = ex[C] / sum_of_ex;
                }
            }
        }
    }

    delete[] iter;
    delete[] ex;

    return 0;
}

static vector<cvi_face_info_t> retina_face_parse(float ratio, int image_width, int image_height) {
    std::vector<cvi_face_info_t> BBoxes;

    for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]) + "_dequant";

        string score_str = NAME_SCORE + key;
        CVI_TENSOR *out = CVI_NN_GetTensorByName(score_str.c_str(), output_tensors, output_num);
        float *score_blob = (float *)CVI_NN_TensorPtr(out);
        CVI_SHAPE score_shape = CVI_NN_TensorShape(out);
        size_t score_size = score_shape.dim[0] * score_shape.dim[1] * score_shape.dim[2] * score_shape.dim[3];
        softmax_by_channel(score_blob, score_blob,
                           {score_shape.dim[0], score_shape.dim[1], score_shape.dim[2], score_shape.dim[3]});
        score_blob += score_size / 2;

        string bbox_str = NAME_BBOX + key;
        out = CVI_NN_GetTensorByName(bbox_str.c_str(), output_tensors, output_num);
        float *bbox_blob = (float *)CVI_NN_TensorPtr(out);

        string landmark_str = NAME_LANDMARK + key;
        out = CVI_NN_GetTensorByName(landmark_str.c_str(), output_tensors, output_num);
        float *landmark_blob = (float *)CVI_NN_TensorPtr(out);
        CVI_SHAPE landmark_shape = CVI_NN_TensorShape(out);

        int width = landmark_shape.dim[3];
        int height = landmark_shape.dim[2];
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

        vector<anchor_box> anchors = _anchors[landmark_str];
        for (size_t num = 0; num < num_anchor; num++) {
            for (size_t j = 0; j < count; j++) {
                cvi_face_info_t box;

                float conf = score_blob[j + count * num];
                if (conf <= FACE_THRESHOLD) {
                    continue;
                }
                box.bbox.score = conf;

                cv::Vec4f regress;
                float dx = bbox_blob[j + count * (0 + num * 4)];
                float dy = bbox_blob[j + count * (1 + num * 4)];
                float dw = bbox_blob[j + count * (2 + num * 4)];
                float dh = bbox_blob[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);
                bbox_pred(anchors[j + count * num], regress, ratio, box.bbox);

                for (size_t k = 0; k < 5; k++) {
                    box.face_pts.x[k] = landmark_blob[j + count * (num * 10 + k * 2)];
                    box.face_pts.y[k] = landmark_blob[j + count * (num * 10 + k * 2 + 1)];
                }
                landmark_pred(anchors[j + count * num], ratio, box.face_pts);

                BBoxes.push_back(box);
            }
        }
    }

    std::vector<cvi_face_info_t> bboxes_nms;
    NonMaximumSuppression(BBoxes, bboxes_nms, 0.4, 'u');

    for (auto &box : bboxes_nms) {
        clip_boxes(image_width, image_height, box.bbox);
    }

    return bboxes_nms;
}

void init_network_retina(char *model_path)
{
    int ret = CVI_NN_RegisterModel(model_path, &model_handle);
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        return;
    }

    CVI_NN_SetConfig(model_handle, OPTION_SKIP_PREPROCESS, true);
    CVI_NN_SetConfig(model_handle, OPTION_INPUT_MEM_TYPE, 2);

    if (CVI_RC_SUCCESS != CVI_NN_GetInputOutputTensors(model_handle, &input_tensors, &input_num, &output_tensors, &output_num)) {
        printf("CVI_NN_GetINputsOutputs failed\n");
    }

    vector<anchor_cfg> cfg;
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

    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(false, cfg);
    map<string, vector<anchor_box>> anchors_fpn_map;
    for(size_t i = 0; i < anchors_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]) + "_dequant";
        anchors_fpn_map[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
    }

    for(size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]) + "_dequant";
        string landmark_str = NAME_LANDMARK + key;
        CVI_TENSOR *out = CVI_NN_GetTensorByName(landmark_str.c_str(), output_tensors, output_num);
        CVI_SHAPE landmark_shape = CVI_NN_TensorShape(out);
        int stride = _feat_stride_fpn[i];

        _anchors[landmark_str] = anchors_plane(landmark_shape.dim[2], landmark_shape.dim[3], stride, anchors_fpn_map[key]);
    }
    bNetworkInited = true;
}

void clean_network() {
    if (bNetworkInited) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }
    }
}

void init_face_meta(cvi_face_t *meta, int size)
{
    meta->size = size;
    meta->face_info = (cvi_face_info_t *)malloc(sizeof(cvi_face_info_t) * meta->size);

    memset(meta->face_info, 0, sizeof(cvi_face_info_t) * meta->size);

    for (int i = 0; i < meta->size; ++i) {
        meta->face_info[i].bbox.x1 = -1;
        meta->face_info[i].bbox.x2 = -1;
        meta->face_info[i].bbox.y1 = -1;
        meta->face_info[i].bbox.y2 = -1;

        meta->face_info[i].name[0] = '\0';
        meta->face_info[i].emotion[0] = '\0';
        meta->face_info[i].gender[0] = '\0';
        meta->face_info[i].race[0] = '\0';
        meta->face_info[i].age = -1;
        meta->face_info[i].face_liveness = -1;
        meta->face_info[i].mask_score = -1;

        for (int j = 0; j < 5; ++j) {
            meta->face_info[i].face_pts.x[j] = -1;
            meta->face_info[i].face_pts.y[j] = -1;
        }
    }
}

void retina_face_inference(VIDEO_FRAME_INFO_S *stDstFrame, cvi_face_t *meta, int *face_count) {

    //printf("zhxjun width:%d stride:%d height:%d u64PhyAddr:%lx u32Length:%d\n",stDstFrame->stVFrame.u32Width,stDstFrame->stVFrame.u32Stride[0],stDstFrame->stVFrame.u32Height,stDstFrame->stVFrame.u64PhyAddr[0],stDstFrame->stVFrame.u32Length[0]);
    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);

    CVI_VIDEO_FRAME_INFO info;
    info.type = CVI_FRAME_PLANAR;
    info.shape.dim_size = 4;
    info.shape.dim[0] = RETINA_FACE_N;
    info.shape.dim[1] = RETINA_FACE_C;
    info.shape.dim[2] = RETINA_FACE_HEIGHT;
    info.shape.dim[3] = RETINA_FACE_WIDTH;
    info.fmt = CVI_FMT_INT8;
    for (size_t i = 0; i < 3; ++i) {
        info.stride[i] = stDstFrame->stVFrame.u32Stride[i];
        info.pyaddr[i] = stDstFrame->stVFrame.u64PhyAddr[i];
    }
    CVI_NN_SetTensorWithVideoFrame(input_tensors, &info);

    CVI_NN_Forward(model_handle, input_tensors, input_num, output_tensors, output_num);

    float ratio = 1.0;
    vector<cvi_face_info_t> faceList = retina_face_parse(ratio, RETINA_FACE_WIDTH, RETINA_FACE_HEIGHT);

    init_face_meta(meta, faceList.size());
    meta->width = RETINA_FACE_WIDTH;
    meta->height = RETINA_FACE_HEIGHT;

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
}

void clean_network_retina() {
    if (bNetworkInited) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }
    }
}
