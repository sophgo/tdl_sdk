#ifndef _YOLO_H_
#define _YOLO_H_

typedef enum { v2, v3 } YoloType;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
} object_detect_rect_t;

typedef struct {
  float x, y, w, h;
} box;

typedef struct {
  box bbox;
  int classes;
  float *prob;
  float objectness;
  int sort_class;
} detection;

typedef struct {
  float *data;
  int norm;
  int channels;
  int width;
  int height;
} YOLOLayer;

typedef struct {
  int m_classes;
  float m_biases[20];
  float m_threshold;
  float m_nms_threshold;
  int m_anchor_nums;
  int m_coords;
  int m_batch;
  YoloType type;
  int m_mask[3][3];
} YOLOParamter;

/**
 * @breif  calculate index in outputs with some parameter with anchor,
 *         one anchor include [x, y, w, h, c, C1, C2....,Cn] information,
 *         x, y ,w, h using with location, c is mean confidence, and C1 to Cn is classes
 *         which index of anchor we need get, is use the parameter of entry.
 *         when entry = 0,  geting the anchor in outputs index, when entry = 4, geting the
 *confidence of this anchor entry = 5, get the index of C1,
 * @param w feature map width
 * @param h feature map height
 * @param batch batch value
 * @param location use to get Nth anchor and loc(class + coords + 1)
 * @param entry entry of offset]
 **/
int EntryIndex(int w, int h, int classes, int batch, int location, int entry, int outputs);

void ActivateArray(float *x, const int n, bool fast_exp);

float Overlap(float x1, float w1, float x2, float w2);
float BoxIntersection(box a, box b);
float BoxUnion(box a, box b);
float BoxIou(box a, box b);
int NmsComparator(const void *pa, const void *pb);

box GetRegionBox(float *data, float *biases, int n, int index, int i, int j, int w, int h,
                 int stride, bool fast_exp);
box GetYoloBox(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w,
               int h, int stride, bool fast_exp);
void Softmax(float *input, int n, float temp, int stride, float *output, bool fast_exp);
void SoftmaxCpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset,
                int stride, float temp, float *output);
void DoNmsSort(detection *dets, int total, int classes, float threshold);
void FillNetworkBoxes(YOLOLayer net_output, int w, int h, float threshold, int relative,
                      detection *dets, YOLOParamter yolo_param);
detection *GetNetworkBoxes(YOLOLayer net_output, int classes, int w, int h, float threshold,
                           int relative, int *num, YOLOParamter yolo_param, int index);
void GetYOLOResults(detection *dets, int num, float threshold, YOLOParamter yolo_param, int ori_w,
                    int ori_h, void (*call_back)(int, int, int, int, int, float));
void FreeDetections(detection *dets, int n);

#endif