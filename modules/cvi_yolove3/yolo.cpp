#include "yolo.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include "cvi_face_utils.hpp"

using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::vector;
using std::make_pair;



int EntryIndex(int w, int h, int classes, int batch, int location, int entry, int outputs) {
  //w, h is the feature map width and height
  int n =   location / (w * h); // The number of anchor, n: [0, max_anchors_num), default max is 5;
  int loc = location % (w * h); // The loc bounding box of the anchor loc: [0, w * h), which grid in w * h
  // The nth of anchors, n * w * h * (l.coords+l.classes+1), the 1 is conf.
  return batch * outputs + n * w * h * (4 + classes + 1) + entry * w * h + loc;
};

void ActivateArray(float *x, const int n, bool fast_exp = true) {
  int i;
  for (i = 0; i < n; ++i) {
    if (fast_exp) {
        x[i] = 1./(1. + FastExp(-x[i]));
    } else {
        x[i] = 1./(1. + exp(-x[i]));
    }
  }
};

float Overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
};

float BoxIntersection(box a, box b) {
    float w = Overlap(a.x, a.w, b.x, b.w);
    float h = Overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
};

float BoxUnion(box a, box b) {
    float i = BoxIntersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
};

float BoxIou(box a, box b) {
    return BoxIntersection(a, b)/BoxUnion(a, b);
};

int NmsComparator(const void *pa, const void *pb) {
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
};

box GetRegionBox(float *data, float *biases, int n, int index, int i, int j, int w, int h, int stride, bool fast_exp) {
    box b;
    b.x = (i + data[index + 0 * stride]) / w;
    b.y = (j + data[index + 1 * stride]) / h;
    if (fast_exp) {
        b.w = FastExp(data[index + 2 * stride]) * biases[2 * n] / w;
        b.h = FastExp(data[index + 3 * stride]) * biases[2 * n + 1] / h;
    } else {
        b.w = exp(data[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(data[index + 3 * stride]) * biases[2 * n + 1] / h;
    }
    return b;
};

box GetYoloBox(float *data, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, bool fast_exp = true) {
    box b;
    b.x = (i + data[index + 0 * stride]) / lw;
    b.y = (j + data[index + 1 * stride]) / lh;
    if (fast_exp) {
        b.w = FastExp(data[index + 2 * stride]) * biases[2 * n] / w;
        b.h = FastExp(data[index + 3 * stride]) * biases[2 * n + 1] / h;
    } else {
        b.w = exp(data[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(data[index + 3 * stride]) * biases[2 * n + 1] / h;
    }

    return b;
}

void Softmax(float *input, int n, float temp, int stride, float *output, bool fast_exp) {
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i*stride] > largest) largest = input[i*stride];
    }
    for (i = 0; i < n; ++i) {
        float e;
        if (fast_exp) {
            e = FastExp(input[i*stride]/temp - largest/temp);
        } else {
            e = exp(input[i*stride]/temp - largest/temp);
        }
        sum += e;
        output[i*stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i*stride] /= sum;
    }
};

void SoftmaxCpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output) {
    int g, b;
    for (b = 0; b < batch; ++b) {
        for (g = 0; g < groups; ++g) {
            Softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset, true);
        }
    }
};

void DoNmsSort(detection *dets, int total, int classes, float threshold) {
    int i, j, k;
    k = total-1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), NmsComparator);
        for (i = 0; i < total; ++i) {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i+1; j < total; ++j) {
                box b = dets[j].bbox;
                if (BoxIou(a, b) > threshold) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
};


int GetNumOfDetections(YOLOLayer &net_output, YOLOParamter yolo_param) {
    if (yolo_param.type == v2){
        return net_output.width * net_output.height * yolo_param.m_anchor_nums;
    }else if(yolo_param.type == v3){
        int i, n;
        int count = 0;
        float *output = net_output.data;
        int w = net_output.width;
        int h = net_output.height;
        int output_size = net_output.norm * net_output.channels * w * h;
        for (i = 0; i < w*h; ++i) {
            for (n = 0; n < yolo_param.m_anchor_nums; ++n) {
                int obj_index  = EntryIndex(w, h, yolo_param.m_classes, 0, n*w*h + i, 4, output_size);
                if (output[obj_index] > yolo_param.m_threshold) {
                    ++count;
                }
            }
        }
        return count;
    }else{
        cout << "Type is not v2 or v3" << endl;
        return 0;
    }

}

detection *MakeNetworkBoxes(YOLOLayer &net_output, float threshold, int *num, YOLOParamter yolo_param) {
    int i;
    int nboxes = GetNumOfDetections(net_output, yolo_param);
    if (num) *num = nboxes;
    detection *dets = new detection[nboxes];
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = new float[yolo_param.m_classes];
    }
    return dets;
}

void CorrectBoundingBox(detection *dets, int n, int w, int h, int net_width, int net_height, int relative) {
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)net_width/w) < ((float)net_height/h)) {
        new_w = net_width;
        new_h = (h * net_width)/w;
    } else {
        new_h = net_height;
        new_w = (w * net_height)/h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x =  (b.x - (net_width - new_w)/2./net_width) / ((float)new_w/net_width); 
        b.y =  (b.y - (net_height - new_h)/2./net_height) / ((float)new_h/net_height); 
        b.w *= (float)net_width/new_w;
        b.h *= (float)net_height/new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void GetRegionDetections(YOLOLayer &net_out, int w, int h, int net_width, int net_height, float threshold, int relative, detection *dets, YOLOParamter yolov2_param) {
    int i,j,n;
    float *predictions = net_out.data;
    int layer_w = net_out.width;
    int layer_h = net_out.height;
    int output_size = net_out.norm * net_out.channels * layer_w * layer_h;
    for (i = 0; i < layer_w * layer_h; ++i) {
        int row = i / layer_w;
        int col = i % layer_w;
        for (n = 0; n < yolov2_param.m_anchor_nums; ++n) {
            int index = n * layer_w * layer_h + i;
            for (j = 0; j < yolov2_param.m_classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index  = EntryIndex(layer_w, layer_h, yolov2_param.m_classes, 0, n*layer_w*layer_h + i, yolov2_param.m_coords, output_size);
            int box_index  = EntryIndex(layer_w, layer_h, yolov2_param.m_classes, 0, n*layer_w*layer_h + i, 0, output_size);
            float scale = predictions[obj_index];
            dets[index].bbox = GetRegionBox(predictions, yolov2_param.m_biases, n, box_index, col, row, layer_w, layer_h, layer_w * layer_h, true);
            dets[index].objectness = scale > threshold ? scale : 0;

            if (dets[index].objectness) {
                for (j = 0; j < yolov2_param.m_classes; ++j) {
                    int class_index = EntryIndex(layer_w, layer_h, yolov2_param.m_classes, 0, n*layer_w*layer_h + i, yolov2_param.m_coords + 1 + j, output_size);
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > threshold) ? prob : 0;
                }
            }
        }
    }
    CorrectBoundingBox(dets, layer_w * layer_h * yolov2_param.m_anchor_nums, w, h, net_width, net_height, relative);
}




void GetYoloDetections(YOLOLayer l, int w, int h, int net_width, int net_height, float threshold, int relative, detection *dets, YOLOParamter yolov3_param, vector<int> m_mask) {
    int i,j,n;
    float *predictions = l.data;
    int layer_w = l.width;
    int layer_h = l.height;
    int count = 0;

    int output_size = l.norm * l.channels * layer_w * layer_h;
    for (i = 0; i < layer_w * layer_h; ++i) {
        int row = i / layer_w;
        int col = i % layer_w;
        for (n = 0; n < yolov3_param.m_anchor_nums; ++n) {
            int obj_index  = EntryIndex(layer_w, layer_h, yolov3_param.m_classes, 0, n*layer_w*layer_h + i, yolov3_param.m_coords, output_size);
            float objectness = predictions[obj_index];
            if (objectness <= yolov3_param.m_threshold) continue;
            int box_index  = EntryIndex(layer_w, layer_h, yolov3_param.m_classes, 0, n*layer_w*layer_h + i, 0, output_size);
            // Need use right biases(anchors)
            int biases_idx = m_mask.empty()? n : m_mask[n];
            dets[count].bbox = GetYoloBox(predictions, yolov3_param.m_biases, biases_idx, box_index, col, row, layer_w, layer_h, yolov3_param.yolo_size, yolov3_param.yolo_size, layer_w * layer_h);
            dets[count].objectness = objectness;
            dets[count].classes = yolov3_param.m_classes;
            for (j = 0; j < yolov3_param.m_classes; ++j) {
                int class_index = EntryIndex(layer_w, layer_h, yolov3_param.m_classes, 0, n*layer_w*layer_h + i, 4 + 1 + j, output_size);
                float prob = objectness * predictions[class_index];
                dets[count].prob[j] = (prob > yolov3_param.m_threshold) ? prob : 0;
            }
            ++count;
        }
    }
    CorrectBoundingBox(dets, count, w, h, net_width, net_height, relative);
}

detection *GetNetworkBoxes(YOLOLayer net_output, int classes, int w, int h, float threshold, int relative, int *num, YOLOParamter yolo_param, int index) {
    detection *dets = MakeNetworkBoxes(net_output, threshold, num, yolo_param);
    if (yolo_param.type == v2){
        GetRegionDetections(net_output, w, h, yolo_param.yolo_size, yolo_param.yolo_size, threshold, relative, dets, yolo_param);
    }else if (yolo_param.type == v3){
        vector<int> mask;
         for (int i = 0; i < 3; i++){
                mask.push_back(yolo_param.m_mask[index][i]);
            }
        GetYoloDetections(net_output, w, h, yolo_param.yolo_size, yolo_param.yolo_size, threshold, relative, dets, yolo_param, mask);
    }
    return dets;
}

void GetYOLOResults(detection *dets, int num, float threshold, YOLOParamter yolo_param, int ori_w, int ori_h, void (*call_back)(int, int, int, int, int, float)) {
    std::vector<std::string> names = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
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
        if (obj_class >= 0) {
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
            std::cout << std::setprecision(2) << names[obj_result.label].c_str() << ": " << obj_result.score*100 << "%" << std::endl;
            std::cout << "left: " << left
                      << ", right: " << right
                      << ", top: " << top
                      << ", bottom" << bot << std::endl;
            if(call_back != nullptr){
                call_back(cls, left, top, right-left, bot-top, obj_result.score);
            }
        }
    }
}



void FreeDetections(detection *dets, int n) {
    int i;
    for (i = 0; i < n; ++i) {
       delete []dets[i].prob;
       dets[i].prob = nullptr;
    }
    delete [] dets;
    dets = nullptr;
     
};
