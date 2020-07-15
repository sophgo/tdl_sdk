#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "face_quality.h"
#include "cvi_face_utils.hpp"
#include "cv183x_facelib_v0.0.1.h"
#include <unistd.h>

#define FACE_QUALITY_WIDTH              112
#define FACE_QUALITY_HEIGHT             112
#define SCALE_B                         (1.0 / (255.0 * 0.229))
#define SCALE_G                         (1.0 / (255.0 * 0.224))
#define SCALE_R                         (1.0 / (255.0 * 0.225))
#define MEAN_B                          -(0.485 / 0.229)
#define MEAN_G                          -(0.456 / 0.224)
#define MEAN_R                          -(0.406 / 0.225)
#define NAME_SCORE                      "score_Softmax_dequant"

static CVI_MODEL_HANDLE model_handle;
static CVI_TENSOR *input_tensors;
static CVI_TENSOR *output_tensors;
static int32_t input_num;
static int32_t output_num;
static bool bNetworkInited;

using namespace std;

void init_network_face_quality(char *model_path) {

    int ret = CVI_NN_RegisterModel(model_path, &model_handle);
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        return;
    }

    if (CVI_RC_SUCCESS != CVI_NN_GetInputOutputTensors(model_handle, &input_tensors, &input_num, &output_tensors, &output_num)) {
        printf("CVI_NN_GetINputsOutputs failed\n");
    }

    bNetworkInited = true;
}

void face_quality_inference(VIDEO_FRAME_INFO_S *stDstFrame, cvi_face_t *meta)
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

    for (int i = 0; i < meta->size; i++) {
        cvi_face_info_t face_info = bbox_rescale(stDstFrame, meta, i);
        cv::Mat crop_frame(FACE_QUALITY_HEIGHT, FACE_QUALITY_WIDTH, image.type());
        face_align(image, crop_frame, face_info, FACE_QUALITY_WIDTH, FACE_QUALITY_HEIGHT);

        cv::Mat tmpchannels[3];
        cv::split(crop_frame, tmpchannels);

        vector<float> mean = {MEAN_B, MEAN_G, MEAN_R};
        vector<float> scale = {SCALE_B, SCALE_G, SCALE_R};
        for (int i = 0; i < 3; i++) {
            tmpchannels[i].convertTo(tmpchannels[i], CV_32F, scale[i], mean[i]);
            int size = tmpchannels[i].rows * tmpchannels[i].cols;
            for (int r = 0; r < tmpchannels[i].rows; ++r) {
                memcpy((float *)CVI_NN_TensorPtr(input) + size*i + tmpchannels[i].cols*r,
                                        tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
            }
        }

        CVI_NN_Forward(model_handle, input_tensors, input_num, output_tensors, output_num);

        CVI_TENSOR *out = CVI_NN_GetTensorByName(NAME_SCORE, output_tensors, output_num);
        float* score = (float *)CVI_NN_TensorPtr(out);
        meta->face_info[i].face_quality = score[1];
        cout << score[0] << "," << score[1] << endl;
    }
    sleep(3);
}

void clean_network_face_quality() {
    if (bNetworkInited) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }
    }
}
