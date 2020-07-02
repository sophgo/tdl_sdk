
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "template_matching.h"
#include "liveness.h"
#include "cv183x_facelib_v0.0.1.h"
#include "cviruntime.h"
#include "cvi_face_utils.hpp"

using namespace std;

#define RESIZE_SIZE             112
#define LIVENESS_SCALE          (1 / 255.0)
#define LIVENESS_N              1
#define LIVENESS_C              6
#define LIVENESS_WIDTH          32
#define LIVENESS_HEIGHT         32
#define CROP_NUM                9
#define MIN_FACE_WIDTH          25
#define MIN_FACE_HEIGHT         25
#define OUTPUT_NAME             "fc2_dequant"

static CVI_MODEL_HANDLE model_handle;
static CVI_TENSOR *input_tensors;
static CVI_TENSOR *output_tensors;
static int32_t input_num;
static int32_t output_num;
static bool bInit = false;

static void prepare_input_tensor(vector<cv::Mat> &input_mat) {
    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    float *input_ptr = (float *)CVI_NN_TensorPtr(input);

    for (int j = 0; j < CROP_NUM; j++) {
        cv::Mat tmpchannels[LIVENESS_C];
        cv::split(input_mat[j], tmpchannels);

        for (int c = 0; c < LIVENESS_C; ++c) {
            tmpchannels[c].convertTo(tmpchannels[c], CV_32F, LIVENESS_SCALE, 0);

            int size = tmpchannels[c].rows * tmpchannels[c].cols;
            for (int r = 0; r < tmpchannels[c].rows; ++r) {
                memcpy(input_ptr + size * c + tmpchannels[c].cols * r, tmpchannels[c].ptr(r, 0),
                       tmpchannels[c].cols * sizeof(float));
            }
        }
        input_ptr += CVI_NN_TensorCount(input) / CROP_NUM;
    }
}

static vector<vector<cv::Mat>> image_preprocess(VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *sink_buffer, cvi_face_t *meta) {
    // printf("image_preprocess 1 u32Height:%d,u32Width:%d u32Stride:%d\n",frame->stVFrame.u32Height,frame->stVFrame.u32Width,frame->stVFrame.u32Stride[0]);
    cv::Mat rgb_frame(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3);
    frame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(frame->stVFrame.u64PhyAddr[0],
                                                           frame->stVFrame.u32Length[0]);
    char *va_rgb = (char *)frame->stVFrame.pu8VirAddr[0];
    for (int i = 0; i < rgb_frame.rows; i++) {
        memcpy(rgb_frame.ptr(i, 0), va_rgb + frame->stVFrame.u32Stride[0] * i, rgb_frame.cols * 3);
    }
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);

    if(!rgb_frame.data) {
        printf("src Image is empty!\n");
        return vector<vector<cv::Mat>> {};
    }
    //printf("image_preprocess 2 sink_buffer u32Height:%d u32Width:%d u32Stride:%d\n",sink_buffer->stVFrame.u32Height,sink_buffer->stVFrame.u32Width,sink_buffer->stVFrame.u32Stride[0]);

    cv::Mat ir_frame(sink_buffer->stVFrame.u32Height, sink_buffer->stVFrame.u32Width, CV_8UC3);
    sink_buffer->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(sink_buffer->stVFrame.u64PhyAddr[0],
                                                                 sink_buffer->stVFrame.u32Length[0]);
    va_rgb = (char *)sink_buffer->stVFrame.pu8VirAddr[0];
    for (int i = 0; i < ir_frame.rows; i++) {
	    memcpy(ir_frame.ptr(i, 0), va_rgb + sink_buffer->stVFrame.u32Stride[0] * i, ir_frame.cols * 3);
    }
    CVI_SYS_Munmap((void *)sink_buffer->stVFrame.pu8VirAddr[0], sink_buffer->stVFrame.u32Length[0]);

    if(!ir_frame.data) {
        printf("sink Image is empty!\n");
        return vector<vector<cv::Mat>> {};
    }

    vector<vector<cv::Mat>> input_mat(meta->size, vector<cv::Mat> ());
    for (int i = 0; i < meta->size; i++) {
        cvi_face_info_t face_info = bbox_rescale(frame, meta, i);
        cv::Rect box;
        box.x = face_info.bbox.x1;
        box.y = face_info.bbox.y1;
        box.width = face_info.bbox.x2 - box.x;
        box.height = face_info.bbox.y2 - box.y;

        if (box.width <= MIN_FACE_WIDTH || box.height <= MIN_FACE_HEIGHT) continue;
        //printf("image_preprocess 7 box x:%d y:%d width:%d height:%d\n",box.x,box.y,box.width,box.height);
        cv::Mat crop_rgb_frame = rgb_frame(box);
        cv::Mat crop_ir_frame = template_matching(crop_rgb_frame, ir_frame, box);

        // cv::imwrite("/mnt/data/out.jpg", crop_rgb_frame);
        // cv::imwrite("/mnt/data/out2.jpg", crop_ir_frame);
        // cv::Mat crop_rgb_frame = cv::imread("/home/terry/calibration/RGBIRLiveness/rgb.jpg");
        // cv::Mat crop_ir_frame = cv::imread("/home/terry/calibration/RGBIRLiveness/ir.jpg");

        cv::Mat color, ir;
        cv::resize(crop_rgb_frame, color, cv::Size(RESIZE_SIZE, RESIZE_SIZE));
        cv::resize(crop_ir_frame, ir, cv::Size(RESIZE_SIZE, RESIZE_SIZE));

        vector<cv::Mat> colors = TTA_9_cropps(color);
        vector<cv::Mat> irs = TTA_9_cropps(ir);

        vector<cv::Mat> input_v;
        for (int i = 0; i < colors.size(); i++) {
            cv::Mat temp;
            cv::merge(vector<cv::Mat> {colors[i], irs[i]}, temp);
            input_v.push_back(temp);
        }
        input_mat[i] = input_v;
    }

    return input_mat;
}

int init_network_liveness(char *model_path) {

    int ret = CVI_NN_RegisterModel(model_path, &model_handle);
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        return -1;
    }

    CVI_NN_SetConfig(model_handle, OPTION_BATCH_SIZE, 9);
    if (CVI_RC_SUCCESS != CVI_NN_GetInputOutputTensors(model_handle, &input_tensors, &input_num,
                                                       &output_tensors, &output_num)) {
        printf("CVI_NN_GetINputsOutputs failed\n");
    }

    printf("init_network_liveness successed\n");
    bInit = true;
    
    return 0;
}

void liveness_inference(VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *sink_buffer, cvi_face_t *meta) {
    if (meta->size <= 0)
    {
        printf("meta->size <= 0\n");
        return;
    }

    vector<vector<cv::Mat>> input_mats = image_preprocess(frame, sink_buffer, meta);
    if (input_mats.empty())
    {
        printf("input_mat.empty \n");
         return;
    }

    for (int i = 0; i < meta->size; i++) {
        float conf0 = 0.0;
        float conf1 = 0.0;

        vector<cv::Mat> input = input_mats[i];
        if (input.empty()) continue;

        prepare_input_tensor(input);

        CVI_NN_Forward(model_handle, input_tensors, input_num, output_tensors, output_num);

        CVI_TENSOR *out = CVI_NN_GetTensorByName(OUTPUT_NAME, output_tensors, output_num);
        float *out_data = (float *)CVI_NN_TensorPtr(out);
        for (int j = 0; j < CROP_NUM; j++) {
            conf0 += out_data[j * 2];
            conf1 += out_data[(j * 2) + 1];
        }

        conf0 /= input.size();
        conf1 /= input.size();

        float max = std::max(conf0, conf1);
        float f0 = std::exp(conf0 - max);
        float f1 = std::exp(conf1 - max);
        float score = f1 / (f0 + f1);

        meta->face_info[i].face_liveness = score;
    }
}


void clean_network_liveness() {
    if (bInit) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }
        bInit = false;
    }
}