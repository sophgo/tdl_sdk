#ifndef _TEXT_RENDER_H_
#define _TEXT_RENDER_H_

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct {
    unsigned char *y_addr;
    int y_width;
    int y_stride;
    int y_height;

    unsigned char *u_addr;
    int u_width;
    int u_stride;
    int u_height;

    unsigned char *v_addr;
    int v_width;
    int v_stride;
    int v_height;
} i420_image_info;

typedef struct {
    int x1;
    int x2;
    int y1;
    int y2;
} image_rect;

typedef struct {
    float r;
    float g;
    float b;
} color_rgb;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum {
    DRAW_POLICY_BOTTOM_LEFT_OUTSIDE,
    DRAW_POLICY_BOTTOM_RIGHT_INSIDE
} DRAW_POLICY;

#ifdef __cplusplus
}
#endif

#endif
