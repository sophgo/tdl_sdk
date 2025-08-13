#include "meta_visualize.h"
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "utils/tdl_log.hpp"

#define min(x, y) (((x) <= (y)) ? (x) : (y))
#define max(x, y) (((x) >= (y)) ? (x) : (y))

int32_t TDL_VisualizeRectangle(box_t *box, int32_t num, char *input_path,
                               char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    LOGE("input path is empty\n");
    return -1;
  }

  for (int32_t i = 0; i < num; i++) {
    cv::rectangle(image,
                  cv::Rect(int32_t(box[i].x1), int32_t(box[i].y1),
                           int32_t(box[i].x2 - box[i].x1),
                           int32_t(box[i].y2 - box[i].y1)),
                  cv::Scalar(0, 0, 255), 2);
  }

  cv::imwrite(output_path, image);

  return 0;
}

int32_t TDL_VisualizePoint(point_t *point, int32_t num, char *input_path,
                           char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    LOGE("input path is empty\n");
    return -1;
  }

  for (int32_t i = 0; i < num; i++) {
    cv::circle(image, cv::Point(int32_t(point[i].x), int32_t(point[i].y)), 7,
               cv::Scalar(0, 0, 255), -1);
  }

  cv::imwrite(output_path, image);

  return 0;
}

int32_t TDL_VisualizeLine(box_t *box, int32_t num, char *input_path,
                          char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    LOGE("input path is empty\n");
    return -1;
  }

  for (int32_t i = 0; i < num; i++) {
    cv::line(image, cv::Point(int32_t(box[i].x1), int32_t(box[i].y1)),
             cv::Point(int32_t(box[i].x2), int32_t(box[i].y2)),
             cv::Scalar(0, 0, 255), 2);
  }

  cv::imwrite(output_path, image);

  return 0;
}

int32_t TDL_VisualizePolylines(point_t *point, int32_t num, char *input_path,
                               char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    LOGE("input path is empty\n");
    return -1;
  }

  std::vector<cv::Point> points;
  for (int32_t i = 0; i < num; i++) {
    points.push_back(cv::Point(static_cast<int32_t>(point[i].x),
                               static_cast<int32_t>(point[i].y)));
  }

  cv::polylines(image, points, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

  cv::imwrite(output_path, image);

  return 0;
}

int32_t TDL_CropImage(int x, int y, int weight, int height, char *input_path,
                      char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    LOGE("input path is empty\n");
    return -1;
  }

  cv::Rect roi(x, y, weight, height);
  if (roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
    LOGE("The cropping area exceeds the original image range");
    return -1;
  }

  cv::Mat cropped = image(roi);

  cv::imwrite(output_path, cropped);

  return 0;
}

cv::Vec3b getColor(int value) {
  switch (value) {
    case 1:  // blue
      return cv::Vec3b(255, 0, 0);
    case 2:  // red
      return cv::Vec3b(0, 0, 255);
    case 3:  // green
      return cv::Vec3b(0, 255, 0);
    case 4:  // yellow
      return cv::Vec3b(255, 255, 0);
    default:  // black
      return cv::Vec3b(0, 0, 0);
  }
}

int32_t TDL_MatToImage(int **mat, int weight, int height, char *output_path,
                       int scale) {
  cv::Mat image(weight * scale, height * scale, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int i = 0; i < weight; i++) {
    for (int j = 0; j < height; j++) {
      cv::Vec3b color = getColor(mat[i][j]);
      cv::Rect roi(j * scale, i * scale, scale, scale);
      cv::rectangle(image, roi, cv::Scalar(color[0], color[1], color[2]),
                    cv::FILLED);
    }
  }
  imwrite(output_path, image);
  return 0;
}

int32_t TDL_VisualizText(int32_t x, int32_t y, char *text, char *input_path,
                         char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    LOGE("input path is empty\n");
    return -1;
  }

  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 1.0;
  int thickness = 2;
  cv::Scalar color(0, 255, 0);  // 绿色
  cv::Point textOrg(x, y);

  cv::putText(image, text, textOrg, fontFace, fontScale, color, thickness);

  cv::imwrite(output_path, image);

  return 0;
}

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
typedef enum {
  FORMAT_YUV_420P,
  FORMAT_NV21,
} PixelFormat;

enum { PLANE_Y = 0, PLANE_U, PLANE_V, PLANE_NUM };

static float GetYuvColor(int chanel, color_rgb *color) {
  if (color == NULL) {
    return 0;
  }

  float yuv_color = 0;
  if (chanel == PLANE_Y) {
    yuv_color =
        (0.257 * color->r) + (0.504 * color->g) + (0.098 * color->b) + 16;
  } else if (chanel == PLANE_U) {
    yuv_color =
        -(.148 * color->r) - (.291 * color->g) + (.439 * color->b) + 128;
  } else if (chanel == PLANE_V) {
    yuv_color =
        (0.439 * color->r) - (0.368 * color->g) - (0.071 * color->b) + 128;
  }

  return (yuv_color < 0) ? 0 : ((yuv_color > 255.) ? 255 : yuv_color);
}

// TODO: Need refactor
int _WriteText(VIDEO_FRAME_INFO_S *frame, int x, int y, const char *name,
               color_rgb color, int thickness) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV12 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
    LOGE(
        "Only PIXEL_FORMAT_NV21 and PIXEL_FORMAT_YUV_PLANAR_420 are supported "
        "in DrawPolygon\n");
    return -1;
  }
  std::string name_str = name;
  int width = frame->stVFrame.u32Width;
  int height = frame->stVFrame.u32Height;
  x = max(min(x, width - 1), 0);
  y = max(min(y, height - 1), 0);

  char color_y = GetYuvColor(PLANE_Y, &color);
  char color_u = GetYuvColor(PLANE_U, &color);
  char color_v = GetYuvColor(PLANE_V, &color);

  size_t image_size = frame->stVFrame.u32Length[0] +
                      frame->stVFrame.u32Length[1] +
                      frame->stVFrame.u32Length[2];
  bool do_unmap = false;
  for (int i = 0; i < 3; ++i) {
    CVI_U32 u32DataLen =
        frame->stVFrame.u32Stride[i] * frame->stVFrame.u32Height;
    if (u32DataLen == 0) {
      continue;
    }

    frame->stVFrame.pu8VirAddr[i] = (uint8_t *)CVI_SYS_Mmap(
        frame->stVFrame.u64PhyAddr[i], frame->stVFrame.u32Length[i]);

    CVI_SYS_IonFlushCache(frame->stVFrame.u64PhyAddr[i],
                          frame->stVFrame.pu8VirAddr[i],
                          frame->stVFrame.u32Length[i]);
    do_unmap = true;
  }

  if (frame->stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_420) {
    // 0: Y-plane, 1: U-plane, 2: V-plane
    for (int i = PLANE_Y; i < PLANE_NUM; i++) {
      char draw_color;
      if (i == PLANE_Y) {
        draw_color = color_y;
      } else if (i == PLANE_U) {
        draw_color = color_u;
      } else {
        draw_color = color_v;
      }

      cv::Size cv_size =
          cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
      cv::Point cv_point = cv::Point(x, y - 2);
      double font_scale = 1;
      if (i != 0) {
        cv_size = cv::Size(frame->stVFrame.u32Width / 2,
                           frame->stVFrame.u32Height / 2);
        cv_point = cv::Point(x / 2, (y - 2) / 2);
        font_scale /= 2;
        // FIXME: Should div but don't know why it's not correct.
        // thickness /= 2;
      }
      // FIXME: Color incorrect.
      cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i],
                    frame->stVFrame.u32Stride[i]);
      cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_COMPLEX_SMALL,
                  font_scale, cv::Scalar(draw_color), thickness, cv::LINE_AA);
    }
  } else { /* PIXEL_FORMAT_NV21 */
    // 0: Y-plane, 1: VU-plane
    for (int i = 0; i < 2; i++) {
      cv::Size cv_size =
          cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
      cv::Point cv_point = cv::Point(x, y - 2);
      double font_scale = 1;
      int text_thickness = max(thickness, 2);
      if (i != 0) {
        cv_size = cv::Size(frame->stVFrame.u32Width / 2,
                           frame->stVFrame.u32Height / 2);
        cv_point = cv::Point(x / 2, (y - 2) / 2);
        font_scale /= 2;
        text_thickness /= 2;
      }

      if (i == 0) {
        cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i],
                      frame->stVFrame.u32Stride[i]);
        cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(static_cast<uint8_t>(color_y)),
                    text_thickness, 8);
      } else {
        cv::Mat image(cv_size, CV_8UC2, frame->stVFrame.pu8VirAddr[i],
                      frame->stVFrame.u32Stride[i]);
        cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    cv::Scalar(static_cast<uint8_t>(color_v),
                               static_cast<uint8_t>(color_u)),
                    text_thickness, 8);
      }
    }
  }
  if (do_unmap) {
    for (int i = 0; i < 3; i++) {
      CVI_U32 u32DataLen =
          frame->stVFrame.u32Stride[i] * frame->stVFrame.u32Height;
      if (u32DataLen == 0) {
        continue;
      }
      CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[i],
                     frame->stVFrame.u32Length[i]);
    }
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }

  return 0;
}

template <PixelFormat format>
void DrawRect(VIDEO_FRAME_INFO_S *frame, float x1, float x2, float y1, float y2,
              const char *name, color_rgb color, int rect_thickness,
              const bool draw_text);

template <>
void DrawRect<FORMAT_YUV_420P>(VIDEO_FRAME_INFO_S *frame, float x1, float x2,
                               float y1, float y2, const char *name,
                               color_rgb color, int rect_thickness,
                               const bool draw_text) {
  std::string name_str = name;
  int width = frame->stVFrame.u32Width;
  int height = frame->stVFrame.u32Height;
  x1 = max(min(x1, width - 1), 0);
  x2 = max(min(x2, width - 1), 0);
  y1 = max(min(y1, height - 1), 0);
  y2 = max(min(y2, height - 1), 0);

  uint8_t color_y = GetYuvColor(PLANE_Y, &color);
  uint8_t color_u = GetYuvColor(PLANE_U, &color);
  uint8_t color_v = GetYuvColor(PLANE_V, &color);

  for (int i = PLANE_Y; i < PLANE_NUM; i++) {
    int stride = frame->stVFrame.u32Stride[i];

    int draw_x1 = ((int)x1 >> 2) << 2;
    int draw_x2 = ((int)x2 >> 2) << 2;
    int draw_y1 = ((int)y1 >> 2) << 2;
    int draw_y2 = ((int)y2 >> 2) << 2;
    int draw_rect_thickness = rect_thickness;
    uint8_t draw_color;
    if (i == PLANE_Y) {
      draw_color = color_y;
    } else if (i == PLANE_U) {
      draw_color = color_u;
    } else {
      draw_color = color_v;
    }

    if (i > PLANE_Y) {
      // uv plane has half size
      draw_x1 /= 2;
      draw_x2 /= 2;
      draw_y1 /= 2;
      draw_y2 /= 2;
      draw_rect_thickness /= 2;
    }

    // draw rect vertical line
    for (int h = draw_y1; h < draw_y2; ++h) {
      for (int w = draw_x1; w < draw_x1 + draw_rect_thickness; ++w) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w),
               draw_color, sizeof(draw_color));
      }
      for (int w = draw_x2 - draw_rect_thickness; (w < draw_x2) && (w >= 0);
           ++w) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w),
               draw_color, sizeof(draw_color));
      }
    }

    // draw rect horizontal line
    for (int w = draw_x1; w < draw_x2; ++w) {
      for (int h = draw_y1; h < draw_y1 + draw_rect_thickness; ++h) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w),
               draw_color, sizeof(draw_color));
      }
      for (int h = draw_y2 - draw_rect_thickness; (h < draw_y2) && (h >= 0);
           ++h) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w),
               draw_color, sizeof(draw_color));
      }
    }

    if (!draw_text) {
      continue;
    }

    cv::Size cv_size =
        cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
    cv::Point cv_point = cv::Point(x1, y1 - 2);
    double font_scale = 2;
    int thickness = rect_thickness;
    if (i != 0) {
      cv_size =
          cv::Size(frame->stVFrame.u32Width / 2, frame->stVFrame.u32Height / 2);
      cv_point = cv::Point(x1 / 2, (y1 - 2) / 2);
      font_scale /= 2;
      thickness /= 2;
    }

    cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i],
                  frame->stVFrame.u32Stride[i]);
    cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(draw_color), thickness, 8);
  }
}

template <>
void DrawRect<FORMAT_NV21>(VIDEO_FRAME_INFO_S *frame, float x1, float x2,
                           float y1, float y2, const char *name,
                           color_rgb color, int rect_thickness,
                           const bool draw_text) {
  std::string name_str = name;
  int width = frame->stVFrame.u32Width;
  int height = frame->stVFrame.u32Height;
  x1 = max(min(x1, width - 1), 0);
  x2 = max(min(x2, width - 1), 0);
  y1 = max(min(y1, height - 1), 0);
  y2 = max(min(y2, height - 1), 0);
  uint8_t color_y = GetYuvColor(PLANE_Y, &color);
  uint8_t color_u = GetYuvColor(PLANE_U, &color);
  uint8_t color_v = GetYuvColor(PLANE_V, &color);
  // 0: Y-plane, 1: VU-plane
  for (int i = 0; i < 2; i++) {
    int stride = frame->stVFrame.u32Stride[i];
    int draw_x1 = ((int)x1 >> 2) << 2;
    int draw_x2 = ((int)x2 >> 2) << 2;
    int draw_y1 = ((int)y1 >> 2) << 2;
    int draw_y2 = ((int)y2 >> 2) << 2;
    int draw_thickness_width = rect_thickness;
    int color = 0;
    if (i == 0) {
      color = color_y;
    } else {
      color = ((uint16_t)color_u << 8) | color_v;
    }

    if (i > 0) {
      // vu plane has half size
      draw_x1 /= 2;
      draw_x2 /= 2;
      draw_y1 /= 2;
      draw_y2 /= 2;
      draw_thickness_width /= 2;
    }
    // draw rect vertical line
    for (int h = draw_y1; h < draw_y2; ++h) {
      if (i > 0) {
        int offset = h * stride + draw_x1 * 2;
        if (uint32_t(offset + draw_thickness_width * 2) >=
            frame->stVFrame.u32Length[i]) {
          LOGE("draw_rect overflow\n");
          break;
        }
        if (draw_x1 * 2 + draw_thickness_width * 2 >= stride) {
          int overflow = (draw_x1 * 2 + draw_thickness_width * 2) - stride + 2;
          offset = max(offset - overflow, 0);
        }
        std::fill_n((uint16_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    draw_thickness_width, (uint16_t)color);
        // this would not be overflowed
        offset = h * stride + (draw_x2 - draw_thickness_width) * 2;
        std::fill_n((uint16_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    draw_thickness_width, (uint16_t)color);
      } else {
        int offset = h * stride + draw_x1;
        std::fill_n((uint8_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    draw_thickness_width, (uint8_t)color);
        offset = h * stride + (draw_x2 - draw_thickness_width);
        std::fill_n((uint8_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    draw_thickness_width, (uint8_t)color);
      }
    }
    // draw rect horizontal line
    int hstart = draw_y1;
    if (hstart + draw_thickness_width >= height) {
      hstart = height - draw_thickness_width;
    }
    for (int h = hstart; h < hstart + draw_thickness_width; ++h) {
      if (i > 0) {
        int offset = h * stride + draw_x1 * 2;
        int box_width = ((draw_x2 - draw_thickness_width) - draw_x1);
        if (box_width < 0) {
          box_width = 0;
        }
        std::fill_n((uint16_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    box_width, (uint16_t)color);
      } else {
        int offset = h * stride + draw_x1;
        int box_width = ((draw_x2 - draw_thickness_width) - draw_x1);
        if (box_width < 0) {
          box_width = 0;
        }
        std::fill_n((uint8_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    box_width, (uint8_t)color);
      }
    }

    for (int h = draw_y2 - draw_thickness_width; (h < draw_y2) && (h >= 0);
         ++h) {
      if (i > 0) {
        int offset = h * stride + draw_x1 * 2;
        int box_width = ((draw_x2 - draw_thickness_width) - draw_x1);
        if (box_width < 0) {
          box_width = 0;
        }
        std::fill_n((uint16_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    box_width, (uint16_t)color);
      } else {
        int offset = h * stride + draw_x1;
        int box_width = ((draw_x2 - draw_thickness_width) - draw_x1);
        if (box_width < 0) {
          box_width = 0;
        }
        std::fill_n((uint8_t *)(frame->stVFrame.pu8VirAddr[i] + offset),
                    box_width, (uint8_t)color);
      }
    }

    if (!draw_text) {
      continue;
    }

    cv::Size cv_size =
        cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
    cv::Point cv_point = cv::Point(x1, y1 - 2);
    double font_scale = 2;
    int thickness = rect_thickness;
    if (i != 0) {
      cv_size =
          cv::Size(frame->stVFrame.u32Width / 2, frame->stVFrame.u32Height / 2);
      cv_point = cv::Point(x1 / 2, (y1 - 2) / 2);
      font_scale /= 2;
      thickness /= 2;
    }

    if (i == 0) {
      cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i],
                    frame->stVFrame.u32Stride[i]);
      cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX,
                  font_scale, cv::Scalar(static_cast<uint8_t>(color)),
                  thickness, 8);
    } else {
      cv::Mat image(cv_size, CV_8UC2, frame->stVFrame.pu8VirAddr[i],
                    frame->stVFrame.u32Stride[i]);
      cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX,
                  font_scale,
                  cv::Scalar(static_cast<uint8_t>(color_v),
                             static_cast<uint8_t>(color_u)),
                  thickness, 8);
    }
  }
}

int WriteText(char *name, int x, int y, VIDEO_FRAME_INFO_S *drawFrame,
              TDLBrush brush) {
  color_rgb rgb_color = brush.color;
  if (rgb_color.r == -1) rgb_color.r = DEFAULT_RECT_COLOR_R;
  if (rgb_color.g == -1) rgb_color.g = DEFAULT_RECT_COLOR_G;
  if (rgb_color.b == -1) rgb_color.b = DEFAULT_RECT_COLOR_B;

  return _WriteText(drawFrame, x, y, name, rgb_color, brush.size);
}

template <typename MetaType>
int DrawMeta(const MetaType *meta, VIDEO_FRAME_INFO_S *drawFrame,
             const bool drawText, const std::vector<TDLBrush> &brushes) {
  if (drawFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21 &&
      drawFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV12 &&
      drawFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
    LOGE(
        "Only PIXEL_FORMAT_NV21 and PIXEL_FORMAT_YUV_PLANAR_420 are supported "
        "in DrawMeta\n");
    return -1;
  }

  if (meta->size == 0) {
    return 0;
  }

  size_t image_size = drawFrame->stVFrame.u32Length[0] +
                      drawFrame->stVFrame.u32Length[1] +
                      drawFrame->stVFrame.u32Length[2];

  bool do_unmap = false;
  for (int i = 0; i < 3; ++i) {
    CVI_U32 u32DataLen =
        drawFrame->stVFrame.u32Stride[i] * drawFrame->stVFrame.u32Height;
    if (u32DataLen == 0) {
      continue;
    }

    drawFrame->stVFrame.pu8VirAddr[i] = (uint8_t *)CVI_SYS_Mmap(
        drawFrame->stVFrame.u64PhyAddr[i], drawFrame->stVFrame.u32Length[i]);

    CVI_SYS_IonFlushCache(drawFrame->stVFrame.u64PhyAddr[i],
                          drawFrame->stVFrame.pu8VirAddr[i],
                          drawFrame->stVFrame.u32Length[i]);
    do_unmap = true;
  }

  for (size_t i = 0; i < meta->size; i++) {
    TDLBrush brush = brushes[i];
    color_rgb rgb_color;
    rgb_color.r = brush.color.r;
    rgb_color.g = brush.color.g;
    rgb_color.b = brush.color.b;

    int thickness = max(brush.size, 2);
    if ((brush.size % 2) != 0) {
      brush.size += 1;
    }

    TDLBox bbox = meta->info[i].box;

    if (drawFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_NV21 ||
        drawFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_NV12) {
      DrawRect<FORMAT_NV21>(drawFrame, bbox.x1, bbox.x2, bbox.y1, bbox.y2,
                            meta->info[i].name, rgb_color, thickness, drawText);
    } else {
      DrawRect<FORMAT_YUV_420P>(drawFrame, bbox.x1, bbox.x2, bbox.y1, bbox.y2,
                                meta->info[i].name, rgb_color, thickness,
                                drawText);
    }
  }

  if (do_unmap) {
    for (int i = 0; i < 3; i++) {
      CVI_U32 u32DataLen =
          drawFrame->stVFrame.u32Stride[i] * drawFrame->stVFrame.u32Height;
      if (u32DataLen == 0) {
        continue;
      }
      CVI_SYS_Munmap((void *)drawFrame->stVFrame.pu8VirAddr[i],
                     drawFrame->stVFrame.u32Length[i]);
    }

    drawFrame->stVFrame.pu8VirAddr[0] = NULL;
    drawFrame->stVFrame.pu8VirAddr[1] = NULL;
    drawFrame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return 0;
}

int32_t TDL_DrawObjRect(const TDLObject *meta, void *frame, const bool drawText,
                        TDLBrush brush) {
  if (meta->size <= 0) return 0;

  std::vector<TDLBrush> brushes(meta->size, brush);
  return DrawMeta(meta, (VIDEO_FRAME_INFO_S *)frame, drawText, brushes);
}

int32_t TDL_DrawFaceRect(const TDLFace *meta, void *frame, const bool drawText,
                         TDLBrush brush) {
  if (meta->size <= 0) return 0;

  std::vector<TDLBrush> brushes(meta->size, brush);
  return DrawMeta(meta, (VIDEO_FRAME_INFO_S *)frame, drawText, brushes);
}

int32_t TDL_ObjectWriteText(char *name, int x, int y, void *frame,
                            TDLBrush brush) {
  return WriteText(name, x, y, (VIDEO_FRAME_INFO_S *)frame, brush);
}
#endif