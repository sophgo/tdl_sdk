#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "thermal_face.h"
#include "cvi_face_utils.hpp"
#include "cv183x_facelib_v0.0.1.h"
//#include <cvi_tracer.h>

#define THERMAL_FACE_N                  1
#define THERMAL_FACE_C                  3
#define THERMAL_FACE_WIDTH              512
#define THERMAL_FACE_HEIGHT             640
#define THERMAL_FACE_SCALE_FACTOR       (float(128) / 255.001236)
#define FACE_THRESHOLD                  0.05
#define NAME_BBOX                       "regression_Concat_dequant"
#define NAME_SCORE                      "classification_Sigmoid_dequant"

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

static CVI_MODEL_HANDLE model_handle;
static CVI_TENSOR *input_tensors;
static CVI_TENSOR *output_tensors;
static int32_t input_num;
static int32_t output_num;
static bool bNetworkInited;

using namespace std;

static vector<anchor_box> all_anchors;

static vector<anchor_box> generate_anchors(int base_size, const vector<float> &ratios, const vector<float> &scales) {
    int num_anchors = ratios.size() * scales.size();
    vector<anchor_box> anchors(num_anchors, anchor_box());
    vector<float> areas(num_anchors, 0);

    for (int i = 0; i < anchors.size(); i++) {
        anchors[i].x2 = base_size * scales[i%scales.size()];
        anchors[i].y2 = base_size * scales[i%scales.size()];
        areas[i] = anchors[i].x2 * anchors[i].y2;

        anchors[i].x2 = sqrt(areas[i] / ratios[i/scales.size()]);
        anchors[i].y2 = anchors[i].x2 * ratios[i/scales.size()];

        anchors[i].x1 -= anchors[i].x2 * 0.5;
        anchors[i].x2 -= anchors[i].x2 * 0.5;
        anchors[i].y1 -= anchors[i].y2 * 0.5;
        anchors[i].y2 -= anchors[i].y2 * 0.5;
    }

    return anchors;
}

static vector<anchor_box> shift(const vector<int> &shape, int stride, const vector<anchor_box> &anchors) {
    vector<int> shift_x(shape[0]*shape[1], 0);
    vector<int> shift_y(shape[0]*shape[1], 0);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            shift_x[i*shape[1] + j] = (j + 0.5) * stride;
        }
    }
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            shift_y[i*shape[1] + j] = (i + 0.5) * stride;
        }
    }

    vector<anchor_box> shifts(shape[0]*shape[1], anchor_box());
    for (int i = 0; i < shifts.size(); i++) {
        shifts[i].x1 = shift_x[i];
        shifts[i].y1 = shift_y[i];
        shifts[i].x2 = shift_x[i];
        shifts[i].y2 = shift_y[i];
    }

    vector<anchor_box> all_anchors(anchors.size() * shifts.size(), anchor_box());
    for (int i = 0; i < shifts.size(); i++) {
        for (int j = 0; j < anchors.size(); j++) {
            all_anchors[i*anchors.size() + j].x1 = anchors[j].x1 + shifts[i].x1;
            all_anchors[i*anchors.size() + j].y1 = anchors[j].y1 + shifts[i].y1;
            all_anchors[i*anchors.size() + j].x2 = anchors[j].x2 + shifts[i].x2;
            all_anchors[i*anchors.size() + j].y2 = anchors[j].y2 + shifts[i].y2;
        }
    }

    return all_anchors;
}

static void bbox_pred(const anchor_box &anchor, cv::Vec4f regress, vector<float> std,
                      cvi_face_detect_rect_t &bbox) {
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    regress[0] *= std[0];
    regress[1] *= std[1];
    regress[2] *= std[2];
    regress[3] *= std[3];

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = FastExp(regress[2]) * width;
    float pred_h = FastExp(regress[3]) * height;

    bbox.x1 = (pred_ctr_x - 0.5 * (pred_w - 1.0));
    bbox.y1 = (pred_ctr_y - 0.5 * (pred_h - 1.0));
    bbox.x2 = (pred_ctr_x + 0.5 * (pred_w - 1.0));
    bbox.y2 = (pred_ctr_y + 0.5 * (pred_h - 1.0));
}

static vector<cvi_face_info_t> thermal_face_parse(int image_width, int image_height) {
    std::vector<cvi_face_info_t> BBoxes;

    string score_str = NAME_SCORE;
    CVI_TENSOR *out = CVI_NN_GetTensorByName(score_str.c_str(), output_tensors, output_num);
    float *score_blob = (float *)CVI_NN_TensorPtr(out);
    CVI_SHAPE score_shape = CVI_NN_TensorShape(out);
 
    string bbox_str = NAME_BBOX;
    out = CVI_NN_GetTensorByName(bbox_str.c_str(), output_tensors, output_num);
    float *bbox_blob = (float *)CVI_NN_TensorPtr(out);
    size_t bbox_size = CVI_NN_TensorCount(out);

    for (size_t i = 0; i < all_anchors.size(); i++) {
        cvi_face_info_t box;

        float conf = score_blob[i];
        if (conf <= FACE_THRESHOLD) {
            continue;
        }
        box.bbox.score = conf;

        cv::Vec4f regress;
        float dx = bbox_blob[0 + i * 4];
        float dy = bbox_blob[1 + i * 4];
        float dw = bbox_blob[2 + i * 4];
        float dh = bbox_blob[3 + i * 4];
        regress = cv::Vec4f(dx, dy, dw, dh);
        bbox_pred(all_anchors[i], regress, {0.1, 0.1, 0.2, 0.2}, box.bbox);

        BBoxes.push_back(box);
    }

    std::vector<cvi_face_info_t> bboxes_nms;
    NonMaximumSuppression(BBoxes, bboxes_nms, 0.5, 'u');

    for (auto &box : bboxes_nms) {
        clip_boxes(image_width, image_height, box.bbox);
    }

    return bboxes_nms;
}

static void init_face_meta(cvi_face_t *meta, int size)
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

void init_network_thermal(char *model_path) {

    int ret = CVI_NN_RegisterModel(model_path, &model_handle);
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        return;
    }

    // CVI_NN_SetConfig(model_handle, OPTION_SKIP_PREPROCESS, true);
    // CVI_NN_SetConfig(model_handle, OPTION_INPUT_MEM_TYPE, 2);

    if (CVI_RC_SUCCESS != CVI_NN_GetInputOutputTensors(model_handle, &input_tensors, &input_num, &output_tensors, &output_num)) {
        printf("CVI_NN_GetINputsOutputs failed\n");
    }

    vector<int> pyramid_levels = {3, 4, 5, 6, 7};
    vector<int> strides = {8, 16, 32, 64, 128};
    vector<int> sizes = {24, 48, 96, 192, 384};
    vector<float> ratios = {1, 2};
    vector<float> scales = {1, 1.25992105, 1.58740105};

    vector<vector<int>> image_shapes;
    for (int s : strides) {
        image_shapes.push_back({(THERMAL_FACE_WIDTH + s - 1) / s, (THERMAL_FACE_HEIGHT + s - 1) / s});
    }

    for (int i = 0; i < sizes.size(); i++) {
        vector<anchor_box> anchors = generate_anchors(sizes[i], ratios, scales);
        vector<anchor_box> shifted_anchors = shift(image_shapes[i], strides[i], anchors);
        all_anchors.insert(all_anchors.end(), shifted_anchors.begin(), shifted_anchors.end());
    }

    bNetworkInited = true;
}

void thermal_face_inference(VIDEO_FRAME_INFO_S *stDstFrame, cvi_face_t *meta, int *face_count)
{
    int img_width = stDstFrame->stVFrame.u32Width;
    int img_height = stDstFrame->stVFrame.u32Height;
    cv::Mat image(img_height, img_width, CV_8UC3);
    stDstFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(stDstFrame->stVFrame.u64PhyAddr[0], stDstFrame->stVFrame.u32Length[0]);
    char *va_rgb = (char *)stDstFrame->stVFrame.pu8VirAddr[0];
    int dst_width = image.cols;
    int dst_height = image.rows;

    for (size_t i = 0; i < dst_height; i++) {
        memcpy(image.ptr(i, 0), va_rgb + stDstFrame->stVFrame.u32Stride[0] * i, dst_width * 3);
    }
    CVI_SYS_Munmap((void *)stDstFrame->stVFrame.pu8VirAddr[0], stDstFrame->stVFrame.u32Length[0]);

    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);

    // image = cv::imread("/mnt/data/thermal.jpg");
    cv::Mat tmpchannels[3];
    cv::split(image, tmpchannels);

    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std = {0.229, 0.224, 0.225};
    for (int i = 0; i < 3; i++) {
        tmpchannels[i].convertTo(tmpchannels[i], CV_32F, 1 / (255.0 * std[i]), -(mean[i] / std[i]));
        copyMakeBorder(tmpchannels[i], tmpchannels[i], 0, 32, 0, 0, cv::BORDER_CONSTANT, 0.0);

        int size = tmpchannels[i].rows * tmpchannels[i].cols;
        for (int r = 0; r < tmpchannels[i].rows; ++r) {
            memcpy((float *)CVI_NN_TensorPtr(input) + size*i + tmpchannels[i].cols*r,
                                    tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
        }
    }

    // CVI_VIDEO_FRAME_INFO info;
    // info.type = CVI_FRAME_PLANAR;
    // info.shape.dim_size = 4;
    // info.shape.dim[0] = THERMAL_FACE_N;
    // info.shape.dim[1] = THERMAL_FACE_C;
    // info.shape.dim[2] = THERMAL_FACE_HEIGHT;
    // info.shape.dim[3] = THERMAL_FACE_WIDTH;
    // info.fmt = CVI_FMT_INT8;
    // for (size_t i = 0; i < 3; ++i) {
    //     info.stride[i] = stDstFrame->stVFrame.u32Stride[i];
    //     info.pyaddr[i] = stDstFrame->stVFrame.u64PhyAddr[i];
    // }
    // CVI_NN_SetTensorWithVideoFrame(input_tensors, &info);

    CVI_NN_Forward(model_handle, input_tensors, input_num, output_tensors, output_num);

    vector<cvi_face_info_t> faceList = thermal_face_parse(THERMAL_FACE_WIDTH, THERMAL_FACE_HEIGHT);

    init_face_meta(meta, faceList.size());
    meta->width = THERMAL_FACE_WIDTH;
    meta->height = THERMAL_FACE_HEIGHT;
    for (int i = 0; i < meta->size; ++i) {
        meta->face_info[i].bbox.x1 = faceList[i].bbox.x1;
        meta->face_info[i].bbox.x2 = faceList[i].bbox.x2;
        meta->face_info[i].bbox.y1 = faceList[i].bbox.y1;
        meta->face_info[i].bbox.y2 = faceList[i].bbox.y2;
    }
}

void clean_network_thermal() {
    if (bNetworkInited) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }
    }
}
