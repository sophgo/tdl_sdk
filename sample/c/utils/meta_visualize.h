#ifndef META_VISUALIZE_H
#define META_VISUALIZE_H

#include <stdio.h>
#include "tdl_types.h"
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
#include "cvi_comm_vpss.h"
#include "cvi_sys.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_RECT_COLOR_R (53. / 255.)
#define DEFAULT_RECT_COLOR_G (208. / 255.)
#define DEFAULT_RECT_COLOR_B (217. / 255.)
#define DEFAULT_TEXT_THICKNESS 1

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

int32_t TDL_VisualizeRectangle(box_t *box, int32_t num, char *input_path,
                               char *output_path);

int32_t TDL_VisualizePoint(point_t *point, int32_t num, char *input_path,
                           char *output_path);

int32_t TDL_VisualizeLine(box_t *box, int32_t num, char *input_path,
                          char *output_path);

int32_t TDL_VisualizePolylines(point_t *point, int32_t num, char *input_path,
                               char *output_path);

int32_t TDL_CropImage(int x, int y, int weight, int height, char *input_path,
                      char *output_path);

int32_t TDL_MatToImage(int **mat, int weight, int height, char *output_path,
                       int scale);

int32_t TDL_VisualizText(int32_t x, int32_t y, char *text, char *input_path,
                         char *output_path);

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
int32_t TDL_DrawObjRect(const TDLObject *meta, void *frame, const bool drawText,
                        TDLBrush brush);

int32_t TDL_DrawFaceRect(const TDLFace *meta, void *frame, const bool drawText,
                         TDLBrush brush);
int32_t TDL_ObjectWriteText(char *name, int x, int y, void *frame, float r,
                            float g, float b);
#endif

#ifdef __cplusplus
}
#endif

#endif