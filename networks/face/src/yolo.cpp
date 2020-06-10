// Copyright 2019 Bitmain Inc.
// License
// Author Rain Wang <rain.wang@bitmain.com>

#include "yolo.hpp"
#include "utils/face_debug.hpp"
#include "utils/math_utils.hpp"

namespace qnn {
namespace vision {

#define YOLO_SIZE 608
#define YOLO_CHANNEL_NUM 3

static const std::vector<NetShape> kPossibleInputShapes = {
    NetShape(1, YOLO_CHANNEL_NUM, YOLO_SIZE, YOLO_SIZE)
};

static float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

static inline int EntryIndex(int w, int h, int classes, int batch, int location, int entry, int outputs) {
  int n =   location / (w * h);
  int loc = location % (w * h);
  return batch * outputs + n * w * h * (4 + classes + 1) + entry * w * h + loc;
}

static inline box GetRegionBox(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride, bool fast_exp = true) {
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    if (fast_exp) {
        b.w = qnn::math::FastExp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = qnn::math::FastExp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    } else {
        b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    }

    return b;
}

static inline box GetYoloBox(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, bool fast_exp = true) {
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    if (fast_exp) {
        b.w = qnn::math::FastExp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = qnn::math::FastExp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    } else {
        b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    }

    return b;
}

static inline float GetColor(int c, int x, int max) {
    float ratio = ((float)x/max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio)*colors[i][c] + ratio*colors[j][c];
    return r;
}

static inline float Overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static inline float BoxIntersection(box a, box b) {
    float w = Overlap(a.x, a.w, b.x, b.w);
    float h = Overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

static inline float BoxUnion(box a, box b) {
    float i = BoxIntersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

static inline float BoxIou(box a, box b) {
    return BoxIntersection(a, b)/BoxUnion(a, b);
}

static inline int NmsComparator(const void *pa, const void *pb) {
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
}

static void ActivateArray(float *x, const int n, bool fast_exp = true) {
  int i;
  for (i = 0; i < n; ++i) {
    if (fast_exp) {
        x[i] = 1./(1. + qnn::math::FastExp(-x[i]));
    } else {
        x[i] = 1./(1. + exp(-x[i]));
    }
  }
}

static void CorrectBoundingBox(detection *dets, int n, int w, int h, int net_width, int net_height, int relative) {
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

static void Softmax(float *input, int n, float temp, int stride, float *output, bool fast_exp = true) {
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i*stride] > largest) largest = input[i*stride];
    }
    for (i = 0; i < n; ++i) {
        float e;
        if (fast_exp) {
            e = qnn::math::FastExp(input[i*stride]/temp - largest/temp);
        } else {
            e = exp(input[i*stride]/temp - largest/temp);
        }
        sum += e;
        output[i*stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i*stride] /= sum;
    }
}

static void SoftmaxCpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output) {
    int g, b;
    for (b = 0; b < batch; ++b) {
        for (g = 0; g < groups; ++g) {
            Softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

static void DoNmsSort(detection *dets, int total, int classes, float threshold) {
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
}

static void FreeDetections(detection *dets, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
    }
    free(dets);
}

void Yolo2::FillNetworkBoxes(OutputTensor &net_output, int w, int h, float threshold, int relative, detection *dets) {
    GetRegionDetections(net_output, w, h, YOLO_SIZE, YOLO_SIZE, threshold, relative, dets);
}

detection *Yolo2::GetNetworkBoxes(OutputTensor &net_output, int w, int h, float threshold, int relative, int *num) {
    detection *dets = MakeNetworkBoxes(net_output, threshold, num);
    FillNetworkBoxes(net_output, w, h, threshold, relative, dets);
    return dets;
}

int Yolo2::GetNumOfDetections(OutputTensor &net_output) {
    return net_output.shape.w * net_output.shape.h * m_num;
}

void Yolo2::DoRegion(OutputTensor &net_output, int batch) {
    float *data = net_output.data;
    int w = net_output.shape.w;
    int h = net_output.shape.h;
    int output_size = net_output.shape.n * net_output.shape.c * w * h;
    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < m_num; ++n) {
            int index = EntryIndex(w, h, m_classes, b, n * w * h, 0, output_size);
            ActivateArray(data + index, 2 * w * h);
            index = EntryIndex(w, h, m_classes, b, n * w * h, m_coords, output_size);
            ActivateArray(data + index, w * h);
        }
    }

    // Softmax
    int index = EntryIndex(w, h, m_classes, 0, 0, m_coords + 1, output_size);
    SoftmaxCpu(data + index, m_classes, batch * m_num, output_size / m_num, w * h, 1, w * h, 1, data + index);
}

detection *Yolo2::MakeNetworkBoxes(OutputTensor &net_output, float threshold, int *num) {
    int i;
    int nboxes = GetNumOfDetections(net_output);
    if (num) *num = nboxes;
    detection *dets = (detection *)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float *)calloc(m_classes, sizeof(float));
    }
    return dets;
}

void Yolo2::GetRegionDetections(OutputTensor &net_out, int w, int h, int net_width, int net_height, float threshold, int relative, detection *dets) {
    int i,j,n;
    float *predictions = net_out.data;
    int layer_w = net_out.shape.w;
    int layer_h = net_out.shape.h;

    int output_size = net_out.shape.n * net_out.shape.c * layer_w * layer_h;
    for (i = 0; i < layer_w * layer_h; ++i) {
        int row = i / layer_w;
        int col = i % layer_w;
        for (n = 0; n < m_num; ++n) {
            int index = n * layer_w * layer_h + i;
            for (j = 0; j < m_classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index  = EntryIndex(layer_w, layer_h, m_classes, 0, n*layer_w*layer_h + i, m_coords, output_size);
            int box_index  = EntryIndex(layer_w, layer_h, m_classes, 0, n*layer_w*layer_h + i, 0, output_size);
            float scale = predictions[obj_index];
            dets[index].bbox = GetRegionBox(predictions, m_biases, n, box_index, col, row, layer_w, layer_h, layer_w * layer_h);
            dets[index].objectness = scale > threshold ? scale : 0;

            if (dets[index].objectness) {
                for (j = 0; j < m_classes; ++j) {
                    int class_index = EntryIndex(layer_w, net_out.shape.h, m_classes, 0, n*layer_w*layer_h + i, m_coords + 1 + j, output_size);
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > threshold) ? prob : 0;
                }
            }
        }
    }
    CorrectBoundingBox(dets, layer_w * layer_h * m_num, w, h, net_width, net_height, relative);
}

std::vector<object_detect_rect_t> Yolo2::GetResults(cv::Mat &im, detection *dets, int num, float threshold, std::vector<std::string> &names) {
    std::vector<object_detect_rect_t> results;

    for (int i = 0; i < num; ++i) {
        std::string labelstr = "";
        int obj_class = -1;
        object_detect_rect_t obj_result;

        for (int j = 0; j < m_classes; ++j) {
            if (dets[i].prob[j] > threshold) {
                if (obj_class < 0) {
                    labelstr = names[j];
                    obj_class = j;
                    obj_result.classes = obj_class;
                    obj_result.prob = dets[i].prob[j];
                } else {
                    labelstr += ", " + names[j];
                    if (dets[i].prob[j] > obj_result.prob) {
                        obj_result.prob = dets[i].prob[j];
                        obj_result.classes = obj_class;
                    }
                }
                std::cout << std::setprecision(2) << names[j].c_str() << ": " << dets[i].prob[j]*100 << "%" << std::endl;
            }
        }

        if (obj_class >= 0) {
            box b = dets[i].bbox;

            int left  = (b.x-b.w/2.)*im.cols;
            int right = (b.x+b.w/2.)*im.cols;
            int top   = (b.y-b.h/2.)*im.rows;
            int bot   = (b.y+b.h/2.)*im.rows;

            if (left < 0) left = 0;
            if (right > im.cols-1) right = im.cols-1;
            if (top < 0) top = 0;
            if (bot > im.rows-1) bot = im.rows-1;

            obj_result.x1 = left;
            obj_result.x2 = right;
            obj_result.y1 = top;
            obj_result.y2 = bot;
            obj_result.label = labelstr;
            results.push_back(obj_result);
        }
    }

    return results;
}

Yolo2::Yolo2(const std::string &model_path, const std::string dataset_type, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, 3, YOLO_SIZE, YOLO_SIZE, true,
               cv::INTER_LINEAR, qnn_ctx) {
    // Check dataset type
    if (!dataset_type.compare("COCO")) {
        // Dataset anchors
        float biases[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
        m_biases = (float*)calloc(sizeof(biases), sizeof(float));
        int idx = 0;
        for (auto biase : biases) {
            m_biases[idx++] = biase;
        }

        // Dataset class names
        m_names = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
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
    } else {
        std::cerr << "Can't interpret dataset parameter: " << dataset_type << std::endl;
        exit(-1);
    }

    // Get threshold and set quantize value
    float threshold = GetInPutThreshold();

    SetQuanParams({std::vector<float>{0, 0, 0},
                   std::vector<float>{(float)((1 / 255.0) * (128.0 / threshold))}});
    EnableDequantize(true);
}

void Yolo2::SetThreshold(const float threshold) {
    m_threshold = threshold;
}

void Yolo2::Detect(const cv::Mat &image, std::vector<object_detect_rect_t> &object_dets) {
    if (image.empty()) {
        return;
    }

    object_dets.clear();
    std::vector<cv::Mat> images;
    images.emplace_back(image);

    ImageNet::Detect(images, [&](OutTensors &out, std::vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= (int)images.size());

        OutputTensor &net_output = out["conv22"];
        for (auto img : images) {
            int nboxes = 0;
            // Do region
            DoRegion(net_output, m_batch);
            // Get object detection results
            detection *dets = GetNetworkBoxes(net_output, img.cols, img.rows, m_threshold, 1, &nboxes);
            DoNmsSort(dets, nboxes, m_classes, m_nms_threshold);
            object_dets = GetResults(img, dets, nboxes, m_threshold, m_names);
            FreeDetections(dets, nboxes);
        }
    });
}

void Yolo2::Draw(const std::vector<object_detect_rect_t> &object_dets, cv::Mat &img) {
    for (auto det : object_dets) {
        int offset = det.classes * 123457 % m_classes;
        float red = GetColor(2, offset, m_classes) * 255;
        float green = GetColor(1, offset, m_classes) * 255;
        float blue = GetColor(0, offset, m_classes) * 255;
        float text_y = det.y1 > 10 ?  det.y1 - 10 : det.y1 + 20;

        cv::rectangle(img, cv::Point(det.x1, det.y1),cv::Point(det.x2, det.y2), cv::Scalar(blue, green, red), 3, 8, 0);
        cv::putText(img, det.label, cv::Point(det.x1, text_y), 0, 1, cv::Scalar(blue, green, red), 3);
    }
}

void Yolo2::PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) {
    cv::Scalar pad_scalar = cv::Scalar(127, 127, 127);
    ratio = ResizeImageCenter(image, prepared, net_width, net_height, m_ibuf, m_resize_policy,
                                preserve_ratio, pad_scalar);
}

Yolo3::Yolo3(const std::string &model_path, const std::string dataset_type, QNNCtx *qnn_ctx)
    : Yolo2(model_path, dataset_type, qnn_ctx) {
    // Check dataset type
    if (!dataset_type.compare("COCO")) {
        // Dataset anchors
        float biases[18] = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
        m_biases = (float*)calloc(sizeof(biases), sizeof(float));
        int idx = 0;
        for (auto biase : biases) {
            m_biases[idx++] = biase;
        }
        // Predict 3 box for every size
        m_num = 3;
    } else {
        std::cerr << "Can't interpret dataset parameter: " << dataset_type << std::endl;
        exit(-1);
    }
}

void Yolo3::DoYolo(OutputTensor &net_output, int batch) {
    float *data = net_output.data;
    int w = net_output.shape.w;
    int h = net_output.shape.h;
    int output_size = net_output.shape.n * net_output.shape.c * w * h;

    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < m_num; ++n) {
            int index = EntryIndex(w, h, m_classes, b, n * w * h, 0, output_size);
            ActivateArray(data + index, 2 * w * h);
            index = EntryIndex(w, h, m_classes, b, n * w * h, m_coords, output_size);
            ActivateArray(data + index, (1 + m_classes) * w * h);
        }
    }
}

int Yolo3::GetNumOfDetections(OutputTensor &net_output) {
    int i, n;
    int count = 0;
    float *output = net_output.data;
    int w = net_output.shape.w;
    int h = net_output.shape.h;
    int output_size = net_output.shape.n * net_output.shape.c * w * h;
    for (i = 0; i < w*h; ++i) {
        for (n = 0; n < m_num; ++n) {
            int obj_index  = EntryIndex(w, h, m_classes, 0, n*w*h + i, 4, output_size);
            if (output[obj_index] > m_threshold) {
                ++count;
            }
        }
    }

    return count;
}

void Yolo3::GetYoloDetections(OutputTensor &net_out, int w, int h, int net_width, int net_height, float threshold, int relative, detection *dets) {
    int i,j,n;
    float *predictions = net_out.data;
    int layer_w = net_out.shape.w;
    int layer_h = net_out.shape.h;
    int count = 0;

    int output_size = net_out.shape.n * net_out.shape.c * layer_w * layer_h;
    for (i = 0; i < layer_w * layer_h; ++i) {
        int row = i / layer_w;
        int col = i % layer_w;
        for (n = 0; n < m_num; ++n) {
            int obj_index  = EntryIndex(layer_w, layer_h, m_classes, 0, n*layer_w*layer_h + i, m_coords, output_size);
            float objectness = predictions[obj_index];
            if (objectness <= m_threshold) continue;
            int box_index  = EntryIndex(layer_w, layer_h, m_classes, 0, n*layer_w*layer_h + i, 0, output_size);
            // Need use right biases(anchors)
            int biases_idx = m_mask.empty()? n : m_mask[n];
            dets[count].bbox = GetYoloBox(predictions, m_biases, biases_idx, box_index, col, row, layer_w, layer_h, YOLO_SIZE, YOLO_SIZE, layer_w * layer_h);
            dets[count].objectness = objectness;
            dets[count].classes = m_classes;
            for (j = 0; j < m_classes; ++j) {
                int class_index = EntryIndex(layer_w, layer_h, m_classes, 0, n*layer_w*layer_h + i, 4 + 1 + j, output_size);
                float prob = objectness * predictions[class_index];
                dets[count].prob[j] = (prob > m_threshold) ? prob : 0;
            }
            ++count;
        }
    }
    CorrectBoundingBox(dets, count, w, h, net_width, net_height, relative);
}

void Yolo3::FillNetworkBoxes(OutputTensor &net_output, int w, int h, float threshold, int relative, detection *dets) {
    GetYoloDetections(net_output, w, h, YOLO_SIZE, YOLO_SIZE, threshold, relative, dets);
}

void Yolo3::Detect(const cv::Mat &image, std::vector<object_detect_rect_t> &object_dets) {
    if (image.empty()) {
        return;
    }

    object_dets.clear();
    std::vector<cv::Mat> images;
    images.emplace_back(image);

    ImageNet::Detect(images, [&](OutTensors &out, std::vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= (int)images.size());

        // Yolov3 has 3 different size outputs
        std::vector<OutputTensor> net_outputs;
        std::queue<std::vector<int>> layer_mask_q;
        net_outputs.push_back(out["layer82-conv"]);
        layer_mask_q.emplace(std::vector<int>{6, 7, 8});
        net_outputs.push_back(out["layer94-conv"]);
        layer_mask_q.emplace(std::vector<int>{3, 4, 5});
        net_outputs.push_back(out["layer106-conv"]);
        layer_mask_q.emplace(std::vector<int>{0, 1, 2});

        for (auto img : images) {
            for (auto &net_output : net_outputs) {
                int nboxes = 0;
                // Get right mask(biases)
                if (!layer_mask_q.empty()) {
                     m_mask = layer_mask_q.front();
                     layer_mask_q.pop();
                }
                // Do yolo layer
                DoYolo(net_output, m_batch);
                detection *dets = GetNetworkBoxes(net_output, img.cols, img.rows, m_threshold, 1, &nboxes);
                DoNmsSort(dets, nboxes, m_classes, m_nms_threshold);
                // Get object detection results
                std::vector<object_detect_rect_t> results = GetResults(img, dets, nboxes, m_threshold, m_names);
                object_dets.insert(object_dets.end(), results.begin(), results.end());
            }
        }
    });
}

}  // namespace vision
}  // namespace qnn
