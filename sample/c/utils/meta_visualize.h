#ifndef META_VISUALIZE_H
#define META_VISUALIZE_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
} box_t;

typedef struct {
  float x;
  float y;
} point_t;

typedef int int32_t;

int32_t TDL_VisualizeRectangle(box_t *box,
                               int32_t num,
                               char *input_path,
                               char *output_path);

int32_t TDL_VisualizePoint(point_t *point,
                           int32_t num,
                           char *input_path,
                           char *output_path);

int32_t TDL_VisualizeLine(box_t *box,
                          int32_t num,
                          char *input_path,
                          char *output_path);

int32_t TDL_VisualizePolylines(point_t *point,
                               int32_t num,
                               char *input_path,
                               char *output_path);

int32_t TDL_CropImage(
    int x, int y, int weight, int height, char *input_path, char *output_path);

int32_t TDL_MatToImage(
    int **mat, int weight, int height, char *output_path, int scale);

int32_t TDL_VisualizText(
    int32_t x, int32_t y, char *text, char *input_path, char *output_path);

#ifdef __cplusplus
}
#endif

#endif