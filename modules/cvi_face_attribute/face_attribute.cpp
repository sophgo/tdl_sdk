#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "cv183x_facelib_v0.0.1.h"
#include "cvi_face_utils.hpp"
#include "face_attribute.h"

using namespace std;

//#define ENABLE_FACE_ATTRIBUTE_DEBUG

#define FACE_OUT_NAME           "BMFace_dense_MatMul_folded"
#define AGE_OUT_NAME            "Age_Conv_Conv2D"
#define EMOTION_OUT_NAME        "Emotion_Conv_Conv2D"
#define GENDER_OUT_NAME         "Gender_Conv_Conv2D"
#define RACE_OUT_NAME           "Race_Conv_Conv2D"

#define RACE_OUT_THRESH         (16.9593105)
#define GENDER_OUT_THRESH       (6.53386879)
#define AGE_OUT_THRESH          (35.0413513)
#define EMOTION_OUT_THRESH      (9.25036811)

static bool attribute_init = false;
static CVI_MODEL_HANDLE model_handle;
static CVI_TENSOR *input_tensors;
static CVI_TENSOR *output_tensors;
static int32_t input_num;
static int32_t output_num;
static float *attribute_buffer = NULL;
static bool bInit = false;


int init_network_face_attribute(char *model_path) {

    int ret = CVI_NN_RegisterModel(model_path, &model_handle);
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        return -1;
    }
    printf("CVI_NN_RegisterModel successed\n");

    CVI_NN_SetConfig(model_handle, OPTION_SKIP_POSTPROCESS, true);
    if (CVI_RC_SUCCESS != CVI_NN_GetInputOutputTensors(model_handle, &input_tensors, &input_num, &output_tensors, &output_num)) {
        printf("CVI_NN_GetINputsOutputs failed\n");
    }

    attribute_buffer = new float[NUM_AGE_FEATURE_DIM];

    bInit = true;
    return 0;
}

template <typename U, typename V>
std::pair<U, V> ExtractFeatures(float *out, int size) {
    SoftMaxForBuffer(out, out, size);
    size_t max_idx = 0;
    float max_val = -1e3;

    for (int i = 0; i < size; i++) {
        if (out[i] > max_val) {
            max_val = out[i];
            max_idx = i;
        }
    }

    V features;
    memcpy(features.features.data(), out, sizeof(float) * size);
    return std::make_pair(U(max_idx + 1), features);
}

std::pair<float, AgeFeature> ExtractAge(float *out, const int age_prob_size) {
    float expect_age = 0;
    if (age_prob_size == 1) {
        expect_age = out[0];
    } else if (age_prob_size == 101) {
        SoftMaxForBuffer(out, out, age_prob_size);
        for (size_t i = 0; i < 101; i++) {
            expect_age += (i * out[i]);
        }
    } else {
        expect_age = -1.0;
    }

    AgeFeature features;
    memcpy(features.features.data(), out, sizeof(float) * age_prob_size);
    return std::make_pair(expect_age, features);
}

FaceFeature ExtractFaceFeature(int8_t *out, const int face_feature_size) {
    FaceFeature features;
    memcpy(features.features.data(), out, sizeof(float) * face_feature_size);
    return features;
}

void face_attribute_parse(cvi_face_t *meta, int meta_i)
{
    FaceAttributeInfo result;

    // feature
    CVI_TENSOR *out = CVI_NN_GetTensorByName(FACE_OUT_NAME, output_tensors, output_num);
    int8_t* face_blob = (int8_t *)CVI_NN_TensorPtr(out);
    size_t face_feature_size = CVI_NN_TensorCount(out);
    memcpy(meta->face_info[meta_i].face_feature, face_blob, face_feature_size);

    // race
    out = CVI_NN_GetTensorByName(RACE_OUT_NAME, output_tensors, output_num);
    int8_t* race_blob = (int8_t *)CVI_NN_TensorPtr(out);
    size_t race_prob_size = CVI_NN_TensorCount(out);
    Dequantize(race_blob, attribute_buffer, RACE_OUT_THRESH, race_prob_size);
    auto race = ExtractFeatures<FaceRace, RaceFeature>(attribute_buffer, race_prob_size);
    result.race = race.first;
    result.race_prob = std::move(race.second);

    // gender
    out = CVI_NN_GetTensorByName(GENDER_OUT_NAME, output_tensors, output_num);
    int8_t* gender_blob = (int8_t *)CVI_NN_TensorPtr(out);
    size_t gender_prob_size = CVI_NN_TensorCount(out);
    Dequantize(gender_blob, attribute_buffer, GENDER_OUT_THRESH, gender_prob_size);
    auto gender = ExtractFeatures<FaceGender, GenderFeature>(attribute_buffer, gender_prob_size);
    result.gender = gender.first;
    result.gender_prob = std::move(gender.second);

    // age
    out = CVI_NN_GetTensorByName(AGE_OUT_NAME, output_tensors, output_num);
    int8_t* age_blob = (int8_t *)CVI_NN_TensorPtr(out);
    size_t age_prob_size = CVI_NN_TensorCount(out);
    Dequantize(age_blob, attribute_buffer, AGE_OUT_THRESH, age_prob_size);
    auto age = ExtractAge(attribute_buffer, age_prob_size);
    result.age = age.first;
    result.age_prob = std::move(age.second);

    // emotion
    out = CVI_NN_GetTensorByName(EMOTION_OUT_NAME, output_tensors, output_num);
    int8_t* emotion_blob = (int8_t *)CVI_NN_TensorPtr(out);
    size_t emotion_prob_size = CVI_NN_TensorCount(out);
    Dequantize(emotion_blob, attribute_buffer, EMOTION_OUT_THRESH, emotion_prob_size);
    auto emotion = ExtractFeatures<FaceEmotion, EmotionFeature>(attribute_buffer, emotion_prob_size);
    result.emotion = emotion.first;
    result.emotion_prob = std::move(emotion.second);

    strncpy(meta->face_info[meta_i].race, GetRaceString(result.race).c_str(), sizeof(meta->face_info[meta_i].race));
    strncpy(meta->face_info[meta_i].gender, GetGenderString(result.gender).c_str(), sizeof(meta->face_info[meta_i].gender));
    strncpy(meta->face_info[meta_i].emotion, GetEmotionString(result.emotion).c_str(), sizeof(meta->face_info[meta_i].emotion));
    meta->face_info[meta_i].age = result.age;

#ifdef ENABLE_FACE_ATTRIBUTE_DEBUG
    std::cout << "Emotion   |" << GetEmotionString(result.emotion) << std::endl;
    std::cout << "Age       |" << result.age << std::endl;
    std::cout << "Gender    |" << GetGenderString(result.gender) << std::endl;
    std::cout << "Race      |" << GetRaceString(result.race) << std::endl;
#endif
}

static uint8_t *prepare_face_attribute_input_tensor(cv::Mat src_image, cvi_face_info_t &face_info)
{
    cv::Mat image(FACE_ATTRIBUTE_HEIGHT, FACE_ATTRIBUTE_WIDTH, image.type());

    face_align(src_image, image, face_info, FACE_ATTRIBUTE_WIDTH, FACE_ATTRIBUTE_HEIGHT);


    // cv::imwrite("/mnt/data/src_image.jpg", src_image);
    // cv::imwrite("/mnt/data/align_image.jpg", image);



    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    std::vector<float> mean = {FACE_ATTRIBUTE_MEAN, FACE_ATTRIBUTE_MEAN, FACE_ATTRIBUTE_MEAN};

    cv::Mat tmpchannels[3];
    cv::split(image, tmpchannels);

    for (int i = 0; i < 3; ++i) {
       tmpchannels[i].convertTo(tmpchannels[i], CV_32F, FACE_ATTRIBUTE_INPUT_THRESHOLD, mean[i]);

        int size = tmpchannels[i].rows * tmpchannels[i].cols;
        for (int r = 0; r < tmpchannels[i].rows; ++r) {
            memcpy((float *)CVI_NN_TensorPtr(input) + size*i + tmpchannels[i].cols*r,
                                    tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
        }
    }
}

void face_attribute_inference(VIDEO_FRAME_INFO_S *stOutFrame, cvi_face_t *meta)
{
    int img_width = stOutFrame->stVFrame.u32Width;
    int img_height = stOutFrame->stVFrame.u32Height;
    cv::Mat image(img_height, img_width, CV_8UC3);
    stOutFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(stOutFrame->stVFrame.u64PhyAddr[0], stOutFrame->stVFrame.u32Length[0]);
    char *va_rgb = (char *)stOutFrame->stVFrame.pu8VirAddr[0];
    int dst_width = image.cols;
    int dst_height = image.rows;

    for (size_t i = 0; i < dst_height; i++) {
        memcpy(image.ptr(i, 0), va_rgb + stOutFrame->stVFrame.u32Stride[0] * i, dst_width * 3);
    }

    CVI_SYS_Munmap((void *)stOutFrame->stVFrame.pu8VirAddr[0], stOutFrame->stVFrame.u32Length[0]);

    for (int i = 0; i < meta->size; ++i) {
        cvi_face_info_t face_info = bbox_rescale(stOutFrame, meta, i);

        prepare_face_attribute_input_tensor(image, face_info);

        CVI_NN_Forward(model_handle, input_tensors, input_num, output_tensors, output_num);

        face_attribute_parse(meta, i);
    }
}

void clean_network_face_attribute() {
    if (bInit) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }

        delete [] attribute_buffer;
        bInit = false;
    }
}
