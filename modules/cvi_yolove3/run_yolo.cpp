#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cv183x_facelib_v0.0.1.h"
//#include "cvi_utils/utils.hpp"
#include "cviruntime.h"
#include <opencv2/opencv.hpp>
#include "run_yolo.h"
#include "yolo.h"

using namespace std;

#define YOLOV3_N 1
#define YOLOV3_C 3
#define YOLOV3_SCALE                (float)((1 / 255.0) * (128.0 / 1.00000488758))
#define YOLOV3_MEAN                 0
#define YOLOV3_CLASSES              80
#define YOLOV3_CONF_THRESHOLD       0.5
#define YOLOV3_NMS_THRESHOLD        0.45
#define YOLOV3_ANCHOR_NUM           3
#define YOLOV3_COORDS               4
#define YOLOV3_DEFAULT_DET_BUFFER   100
#define YOLOV3_OUTPUT1              "layer82-conv_dequant"
#define YOLOV3_OUTPUT2              "layer94-conv_dequant"
#define YOLOV3_OUTPUT3              "layer106-conv_dequant"

static int yolov3_h = 320;
static int yolov3_w = 320;
//static int yolov3_h = 416;
//static int yolov3_w = 416;
static bool bInit = false;
static CVI_MODEL_HANDLE model_handle;
static CVI_TENSOR *input_tensors;
static CVI_TENSOR *output_tensors;
static int32_t input_num;
static int32_t output_num;

typedef enum {
    CVI_YOLOV3_DET_TYPE_ALL,
    CVI_YOLOV3_DET_TYPE_VEHICLE,
    CVI_YOLOV3_DET_TYPE_PEOPLE,
    CVI_YOLOV3_DET_TYPE_PET
} CviYolov3DetType;

static std::vector<std::string> names = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

void set_model_size(int model_size) {
    yolov3_w = model_size;
    yolov3_h = model_size;
}

int init_network_yolov3(char *model_path) {

    int ret = CVI_NN_RegisterModel(model_path, &model_handle);
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        return -1;
    }
    printf("init_network_yolov3 successed\n");

    CVI_NN_SetConfig(model_handle, OPTION_SKIP_PREPROCESS, true);

    if (CVI_RC_SUCCESS != CVI_NN_GetInputOutputTensors(model_handle, &input_tensors, &input_num, &output_tensors, &output_num)) {
        printf("CVI_NN_GetINputsOutputs failed\n");
    }

    bInit = true;

    return 0;
}

void free_cnn_env()
{
    if (bInit) {
        int ret = CVI_NN_CleanupModel(model_handle);
        if (ret != CVI_RC_SUCCESS) {
            printf("CVI_NN_CleanupModel failed, err %d\n", ret);
        }
    }
}

static void prepare_yolo_input_tensor(VIDEO_FRAME_INFO_S *frame, CVI_TENSOR *input) {

    //printf("zhxjun prepare_yolo_input_tensor width:%d height:%d u64PhyAddr:%lx length:%d\n",
    //frame->stVFrame.u32Width,frame->stVFrame.u32Height,frame->stVFrame.u64PhyAddr[0],frame->stVFrame.u32Length[0]);

    void *dst = CVI_NN_TensorPtr(input);

    int cp_size = frame->stVFrame.u32Width * frame->stVFrame.u32Height;
    for (size_t i = 0; i < 3; i++) {
        frame->stVFrame.pu8VirAddr[i] = (CVI_U8 *)CVI_SYS_Mmap(frame->stVFrame.u64PhyAddr[i], frame->stVFrame.u32Length[i]);
        const void* src = (const void*)frame->stVFrame.pu8VirAddr[i];

        memcpy(dst, src, cp_size);
        dst += cp_size;

        CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[i], frame->stVFrame.u32Length[i]);
    }

}

static void DoYolo(YOLOLayer &l, YOLOParamter yolov3_param) {
    float *data = l.data;
    int w = l.width;
    int h = l.height;
    int output_size = l.norm * l.channels * w * h;

    for (int b = 0; b < yolov3_param.m_batch; ++b) {
        for (int p = 0; p < w * h; ++p) {
            for (int n = 0; n < yolov3_param.m_anchor_nums; ++n) {
                // int index = EntryIndex(w, h, yolov3_param.m_classes, b, n * w * h, 0, output_size);
                // ActivateArray(data + index, 2 * w * h, true);

                // index = EntryIndex(w, h, yolov3_param.m_classes, b, n * w * h, yolov3_param.m_coords, output_size);
                // ActivateArray(data + index, (1 + yolov3_param.m_classes) * w * h, true);

                int obj_index  = EntryIndex(w, h, yolov3_param.m_classes, b, n * w * h + p, yolov3_param.m_coords, output_size);
                ActivateArray(data + obj_index, 1, true);
                float objectness = data[obj_index];

                if (objectness >= yolov3_param.m_threshold) {
                    int box_index = EntryIndex(w, h, yolov3_param.m_classes, b, n * w * h + p, 0, output_size);
                    ActivateArray(data + box_index, 1, true);
                    ActivateArray(data + box_index + (w * h), 1, true);

                    for (int j = 0; j < yolov3_param.m_classes; ++j) {
                        int class_index = EntryIndex(w, h, yolov3_param.m_classes, b, n* w * h + p, 4 + 1 + j, output_size);
                        ActivateArray(data + class_index, 1, true);
                    }
                }
            }
        }
    }
}

void GetYOLOResults(detection *dets, int num, float threshold, YOLOParamter yolo_param, int ori_w, int ori_h,
        vector<object_detect_rect_t> &results, int det_type) {
    for (int i = 0; i < num; ++i) {
        std::string labelstr = "";
        int obj_class = -1;
        object_detect_rect_t obj_result;
        obj_result.score = 0;
        obj_result.label = obj_class;
        for (int j = 0; j < yolo_param.m_classes; ++j) {
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

        if (det_type == CVI_YOLOV3_DET_TYPE_VEHICLE) {
            if ((obj_result.label < 1) || obj_result.label > 7) continue;
        } else if (det_type == CVI_YOLOV3_DET_TYPE_PEOPLE) {
            if (obj_result.label != 0) continue;
        } else if (det_type == CVI_YOLOV3_DET_TYPE_PET) {
            if ((obj_result.label != 16) && (obj_result.label != 17)) continue;
        }

        box b = dets[i].bbox;
        int left  = (b.x-b.w/2.)*ori_w;
        int right = (b.x+b.w/2.)*ori_w;
        int top   = (b.y-b.h/2.)*ori_h;
        int bot   = (b.y+b.h/2.)*ori_h;
        if (left < 0) left = 0;
        if (right > ori_w-1) right = ori_w-1;
        if (top < 0) top = 0;
        if (bot > ori_h-1) bot = ori_h-1;
        int cls = obj_result.label;
         //std::cout << std::setprecision(2) << names[obj_result.label].c_str() << ": " << obj_result.score*100 << "%" << std::endl;
         //std::cout << "left: " << left
         //          << ", right: " << right
         //          << ", top: " << top
         //          << ", bottom" << bot << std::endl;

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

static void YoloV3Parse(vector<float*> features, vector<CVI_SHAPE> &output_shape, cvi_object_meta_t *meta,
                        int det_type, VIDEO_FRAME_INFO_S *frame) {
    static YOLOParamter yolov3_param = {
        YOLOV3_CLASSES, //m_classes
        {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326}, // m_biases
        YOLOV3_CONF_THRESHOLD, // m_threshold
        YOLOV3_NMS_THRESHOLD, //m_nms_threshold
        YOLOV3_ANCHOR_NUM, // m_anchor_nums
        YOLOV3_COORDS, // m_coords
        1, // m_batch
        yolov3_w, //yolo_size
        v3, // type
        {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}} // m_mask
    };

    // Yolov3 has 3 different size outputs
    vector<YOLOLayer> net_outputs;
    for (int i = 0; i < features.size(); i++){
        YOLOLayer l = {
            features[i],
            int(output_shape[i].dim[0]),
            int(output_shape[i].dim[1]),
            int(output_shape[i].dim[2]),
            int(output_shape[i].dim[3])
        };
        net_outputs.push_back(l);
    }

    vector<object_detect_rect_t> results;
    static int det_buf_size = YOLOV3_DEFAULT_DET_BUFFER;
    static detection *total_dets = new detection[det_buf_size];
    int total_boxes = 0;

    for (size_t i = 0; i < net_outputs.size(); i++) {
        int nboxes = 0;
        // Do yolo layer
        DoYolo(net_outputs.at(i), yolov3_param);
        detection *dets = GetNetworkBoxes(net_outputs.at(i), yolov3_param.m_classes, yolov3_h, yolov3_w, yolov3_param.m_threshold, 1, &nboxes, yolov3_param, i);

        int next_size = total_boxes + nboxes;
        if (next_size > det_buf_size) {
            total_dets = (detection *)realloc(total_dets, next_size * sizeof(detection));
            det_buf_size = next_size;
        }

        memcpy(total_dets + total_boxes, dets, sizeof(detection) * nboxes);
        total_boxes += nboxes;

        // we do not use FreeDetections because we use just use memcpy,
        // FreeDetections will free det.prob
        delete []dets;
    }

    DoNmsSort(total_dets, total_boxes, yolov3_param.m_classes, yolov3_param.m_nms_threshold);
    GetYOLOResults(total_dets, total_boxes,  yolov3_param.m_threshold, yolov3_param, yolov3_h, yolov3_w, results, det_type);
    for (int i = 0; i < total_boxes; ++i) {
       delete []total_dets[i].prob;
    }

    float frame_width = frame->stVFrame.u32Width;
    float frame_height = frame->stVFrame.u32Height;
    float ratio_w = frame_width / yolov3_w;
    float ratio_h = frame_height / yolov3_h;

    // fill meta
    meta->size = results.size();
    meta->objects = (cvi_object_info_t*)malloc(sizeof(cvi_object_info_t) * meta->size);
    meta->width = frame->stVFrame.u32Width;
    meta->height = frame->stVFrame.u32Height;

    memset(meta->objects, 0, sizeof(cvi_object_info_t) * meta->size);

    for (int i = 0; i < meta->size; ++i) {
        meta->objects[i].bbox.x1 = results[i].x1 * ratio_w;
        meta->objects[i].bbox.y1 = results[i].y1 * ratio_h;
        meta->objects[i].bbox.x2 = results[i].x2 * ratio_w;
        meta->objects[i].bbox.y2 = results[i].y2 * ratio_h;
        meta->objects[i].bbox.score = results[i].score;
        strncpy(meta->objects[i].name, names[results[i].label].c_str(), sizeof(meta->objects[i].name));
        meta->objects[i].classes = results[i].label;

         printf("YOLO3: %s (%d): %lf %lf %lf %lf, score=%.2f\n", meta->objects[i].name, meta->objects[i].classes,
                 meta->objects[i].bbox.x1, meta->objects[i].bbox.x2, meta->objects[i].bbox.y1, meta->objects[i].bbox.y2, results[i].score);
    }
}

void yolov3_inference(VIDEO_FRAME_INFO_S *frame, cvi_object_meta_t *meta, int det_type) {
    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    prepare_yolo_input_tensor(frame, input);

    CVI_NN_Forward(model_handle, input_tensors, input_num, output_tensors, output_num);

    vector<float*> features;
    vector<string> output_name = {YOLOV3_OUTPUT1, YOLOV3_OUTPUT2, YOLOV3_OUTPUT3};
    vector<CVI_SHAPE> output_shape;
    for (int i = 0; i < output_num; i++) {
        CVI_TENSOR *out = CVI_NN_GetTensorByName(output_name[i].c_str(), output_tensors, output_num);
        output_shape.push_back(CVI_NN_TensorShape(out));
        features.push_back((float *)CVI_NN_TensorPtr(out));
    }

    YoloV3Parse(features, output_shape, meta, det_type, frame);
}
