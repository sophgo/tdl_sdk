#include "meta_visualize.h"
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "tdl_type_internal.hpp"
#include "utils/tdl_log.hpp"

#define min(x, y) (((x) <= (y)) ? (x) : (y))
#define max(x, y) (((x) >= (y)) ? (x) : (y))

int32_t VisualizeRectangleFromFile(box_t *box, int32_t num, char *input_path,
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

int32_t VisualizeRectangle(box_t *box, int32_t num, TDLImage image_handle,
                           char *output_path, int *colors) {
  int32_t ret = DrawRectangle(box, num, image_handle, colors);
  if (ret != 0) {
    return ret;
  }

  cv::Mat mat;
  bool is_rgb;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  ImageFactory::convertToMat(image_context->image, mat, is_rgb);

  // If it's RGB, we need to convert it to BGR for imwrite to save correctly
  // imwrite expects BGR by default for .jpg
  if (is_rgb) {
    cv::Mat bgr_mat;
    cv::cvtColor(mat, bgr_mat, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, bgr_mat);
  } else {
    cv::imwrite(output_path, mat);
  }

  return 0;
}

int32_t DrawRectangle(box_t *box, int32_t num, TDLImage image_handle,
                      int *colors) {
  cv::Mat mat;
  bool is_rgb;

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  int32_t ret = ImageFactory::convertToMat(image_context->image, mat, is_rgb);
  if (ret != 0) {
    std::cout << "Failed to convert to mat" << std::endl;
    return -1;
  }

  for (int32_t i = 0; i < num; i++) {
    int c1 = colors[i * 3];
    int c2 = colors[i * 3 + 1];
    int c3 = colors[i * 3 + 2];
    // OpenCV scalar order depends on Mat format.
    // If mat is RGB, Scalar(R, G, B) gives RGB.
    // If mat is BGR, Scalar(B, G, R) gives BGR.
    // The user provided colors are assumed to match the Mat format or be BGR.
    // To be safe and consistent with previous behavior, we assume user provides
    // BGR and if mat is RGB, we swap them.
    if (is_rgb) {
      cv::rectangle(mat,
                    cv::Rect(int32_t(box[i].x1), int32_t(box[i].y1),
                             int32_t(box[i].x2 - box[i].x1),
                             int32_t(box[i].y2 - box[i].y1)),
                    cv::Scalar(c3, c2, c1), 2);
    } else {
      cv::rectangle(mat,
                    cv::Rect(int32_t(box[i].x1), int32_t(box[i].y1),
                             int32_t(box[i].x2 - box[i].x1),
                             int32_t(box[i].y2 - box[i].y1)),
                    cv::Scalar(c1, c2, c3), 2);
    }
  }

  return 0;
}

int32_t DrawText(TDLImage image_handle, int32_t x, int32_t y, const char *text,
                 int *colors) {
  cv::Mat mat;
  bool is_rgb;

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  int32_t ret = ImageFactory::convertToMat(image_context->image, mat, is_rgb);
  if (ret != 0) {
    std::cout << "Failed to convert to mat" << std::endl;
    return -1;
  }

  cv::Scalar color(colors[2], colors[1], colors[0]);
  cv::putText(mat, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, color,
              2);

  return 0;
}

int32_t VisualizePointFromFile(point_t *point, int32_t num, char *input_path,
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

int32_t VisualizePoint(point_t *point, int32_t num, TDLImage image_handle,
                       char *output_path) {
  cv::Mat mat;
  bool is_rgb;

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  int32_t ret = ImageFactory::convertToMat(image_context->image, mat, is_rgb);
  if (ret != 0) {
    std::cout << "Failed to convert to mat" << std::endl;
    return -1;
  }
  if (is_rgb) {
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  }

  for (int32_t i = 0; i < num; i++) {
    cv::circle(mat, cv::Point(int32_t(point[i].x), int32_t(point[i].y)), 7,
               cv::Scalar(0, 0, 255), -1);
  }

  cv::imwrite(output_path, mat);

  return 0;
}

int32_t VisualizeLine(box_t *box, int32_t num, char *input_path,
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

int32_t VisualizePolylines(point_t *point, int32_t num, char *input_path,
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

int32_t CropImage(int x, int y, int weight, int height, char *input_path,
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

int32_t MatToImage(int **mat, int weight, int height, char *output_path,
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

int32_t VisualizTextFromFile(int32_t x, int32_t y, char *text, char *input_path,
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

int32_t VisualizText(int32_t x, int32_t y, TDLImage image_handle,
                     const char *text, char *output_path, int *colors) {
  int32_t ret = DrawText(image_handle, x, y, text, colors);
  if (ret != 0) {
    return ret;
  }

  cv::Mat mat;
  bool is_rgb;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  ImageFactory::convertToMat(image_context->image, mat, is_rgb);

  if (is_rgb) {
    cv::Mat bgr_mat;
    cv::cvtColor(mat, bgr_mat, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, bgr_mat);
  } else {
    cv::imwrite(output_path, mat);
  }

  return 0;
}

int32_t VisualizeDepthMap(TDLDepthLogits *depth_logits,
                          const char *output_path) {
  int width = depth_logits->w;
  int height = depth_logits->h;
  float *logits = depth_logits->logits;

  if (!logits) {
    LOGE("Depth logits is NULL\n");
    return -1;
  }

  // 找到深度值的最小值和最大值用于归一化
  float min_val = logits[0];
  float max_val = logits[0];
  for (int i = 1; i < width * height; i++) {
    if (logits[i] < min_val) min_val = logits[i];
    if (logits[i] > max_val) max_val = logits[i];
  }

  // 创建 float 类型的 Mat，包装原始数据
  cv::Mat depth_map(height, width, CV_32FC1, logits);

  // 归一化到 0-255
  cv::Mat depth_map_vis;
  cv::normalize(depth_map, depth_map_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  // 保存图像
  cv::imwrite(output_path, depth_map_vis);

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
      double font_scale = thickness;
      if (i != 0) {
        cv_size = cv::Size(frame->stVFrame.u32Width / 2,
                           frame->stVFrame.u32Height / 2);
        cv_point = cv::Point(x / 2, (y - 2) / 2);
        font_scale /= 2;
      }
      // FIXME: Color incorrect.
      cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i],
                    frame->stVFrame.u32Stride[i]);
      cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_COMPLEX_SMALL,
                  font_scale, cv::Scalar(draw_color), 1, cv::LINE_AA);
    }
  } else { /* PIXEL_FORMAT_NV21 */
    // 0: Y-plane, 1: VU-plane
    for (int i = 0; i < 2; i++) {
      cv::Size cv_size =
          cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
      cv::Point cv_point = cv::Point(x, y - 2);
      double font_scale = thickness;
      if (i != 0) {
        cv_size = cv::Size(frame->stVFrame.u32Width / 2,
                           frame->stVFrame.u32Height / 2);
        cv_point = cv::Point(x / 2, (y - 2) / 2);
        font_scale /= 2;
      }

      if (i == 0) {
        cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i],
                      frame->stVFrame.u32Stride[i]);
        cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(static_cast<uint8_t>(color_y)), 1,
                    8);
      } else {
        cv::Mat image(cv_size, CV_8UC2, frame->stVFrame.pu8VirAddr[i],
                      frame->stVFrame.u32Stride[i]);
        cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    cv::Scalar(static_cast<uint8_t>(color_v),
                               static_cast<uint8_t>(color_u)),
                    1, 8);
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
    double font_scale = 0.5;
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
    double font_scale = 0.8;
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

    int thickness = 1;
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

int32_t DrawObjRect(const TDLObject *meta, void *frame, const bool drawText,
                    TDLBrush brush) {
  if (meta->size <= 0) return 0;

  std::vector<TDLBrush> brushes(meta->size, brush);
  return DrawMeta(meta, (VIDEO_FRAME_INFO_S *)frame, drawText, brushes);
}

int32_t DrawFaceRect(const TDLFace *meta, void *frame, const bool drawText,
                     TDLBrush brush) {
  if (meta->size <= 0) return 0;

  std::vector<TDLBrush> brushes(meta->size, brush);
  return DrawMeta(meta, (VIDEO_FRAME_INFO_S *)frame, drawText, brushes);
}

int32_t ObjectWriteText(char *name, int x, int y, void *frame, TDLBrush brush) {
  return WriteText(name, x, y, (VIDEO_FRAME_INFO_S *)frame, brush);
}

int32_t DrawLine(box_t *lines, int32_t num, void *frame, TDLBrush brush) {
  if (num <= 0 || lines == NULL || frame == NULL) {
    return 0;
  }

  VIDEO_FRAME_INFO_S *drawFrame = (VIDEO_FRAME_INFO_S *)frame;
  if (drawFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21 &&
      drawFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV12 &&
      drawFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
    LOGE("Only NV21, NV12 and YUV_PLANAR_420 are supported in DrawLine\n");
    return -1;
  }

  color_rgb rgb_color = brush.color;
  if (rgb_color.r == -1) rgb_color.r = DEFAULT_RECT_COLOR_R;
  if (rgb_color.g == -1) rgb_color.g = DEFAULT_RECT_COLOR_G;
  if (rgb_color.b == -1) rgb_color.b = DEFAULT_RECT_COLOR_B;

  uint8_t color_y = (uint8_t)GetYuvColor(PLANE_Y, &rgb_color);
  uint8_t color_u = (uint8_t)GetYuvColor(PLANE_U, &rgb_color);
  uint8_t color_v = (uint8_t)GetYuvColor(PLANE_V, &rgb_color);
  int thickness = max(brush.size, 2);

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

  bool is_planar =
      (drawFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_420);

  for (int k = 0; k < num; k++) {
    int x1 = (int)lines[k].x1, y1 = (int)lines[k].y1;
    int x2 = (int)lines[k].x2, y2 = (int)lines[k].y2;

    // Y plane - full resolution
    cv::Mat y_mat(drawFrame->stVFrame.u32Height, drawFrame->stVFrame.u32Width,
                  CV_8UC1, drawFrame->stVFrame.pu8VirAddr[0],
                  drawFrame->stVFrame.u32Stride[0]);
    cv::line(y_mat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(color_y),
             thickness, cv::LINE_AA);

    if (is_planar) {
      int thickness_half = max(thickness / 2, 1);
      // U plane - half resolution
      cv::Mat u_mat(drawFrame->stVFrame.u32Height / 2,
                    drawFrame->stVFrame.u32Width / 2, CV_8UC1,
                    drawFrame->stVFrame.pu8VirAddr[1],
                    drawFrame->stVFrame.u32Stride[1]);
      cv::line(u_mat, cv::Point(x1 / 2, y1 / 2), cv::Point(x2 / 2, y2 / 2),
               cv::Scalar(color_u), thickness_half, cv::LINE_AA);

      // V plane - half resolution
      cv::Mat v_mat(drawFrame->stVFrame.u32Height / 2,
                    drawFrame->stVFrame.u32Width / 2, CV_8UC1,
                    drawFrame->stVFrame.pu8VirAddr[2],
                    drawFrame->stVFrame.u32Stride[2]);
      cv::line(v_mat, cv::Point(x1 / 2, y1 / 2), cv::Point(x2 / 2, y2 / 2),
               cv::Scalar(color_v), thickness_half, cv::LINE_AA);
    } else {
      int thickness_half = max(thickness / 2, 1);
      // UV plane (interleaved) - half resolution
      cv::Mat uv_mat(drawFrame->stVFrame.u32Height / 2,
                     drawFrame->stVFrame.u32Width / 2, CV_8UC2,
                     drawFrame->stVFrame.pu8VirAddr[1],
                     drawFrame->stVFrame.u32Stride[1]);
      cv::line(uv_mat, cv::Point(x1 / 2, y1 / 2), cv::Point(x2 / 2, y2 / 2),
               cv::Scalar(color_v, color_u), thickness_half, cv::LINE_AA);
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
#endif
