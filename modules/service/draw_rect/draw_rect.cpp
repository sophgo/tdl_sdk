#include "draw_rect.hpp"

#include "core_utils.hpp"
#include "opencv2/opencv.hpp"

#include <cvi_sys.h>
#include <string.h>

#define min(x, y) (((x) <= (y)) ? (x) : (y))
#define max(x, y) (((x) >= (y)) ? (x) : (y))

static std::vector<std::pair<int, int>> l_pair = {{0, 1},   {0, 2},   {1, 3},   {2, 4},   {5, 6},
                                                  {5, 7},   {7, 9},   {6, 8},   {8, 10},  {17, 11},
                                                  {17, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};

static std::vector<cv::Scalar> p_color = {
    {0, 255, 255},  {0, 191, 255},  {0, 255, 102},  {0, 77, 255},   {0, 255, 0},    {77, 255, 255},
    {77, 255, 204}, {77, 204, 255}, {191, 255, 77}, {77, 191, 255}, {191, 255, 77}, {204, 77, 255},
    {77, 255, 204}, {191, 77, 255}, {77, 255, 191}, {127, 77, 255}, {77, 255, 127}, {0, 255, 255}};

static std::vector<cv::Scalar> line_color = {
    {0, 215, 255},   {0, 255, 204},  {0, 134, 255},  {0, 255, 50},  {77, 255, 222},
    {77, 196, 255},  {77, 135, 255}, {191, 255, 77}, {77, 255, 77}, {77, 222, 255},
    {255, 156, 127}, {0, 127, 255},  {255, 127, 77}, {0, 77, 255},  {255, 77, 36}};

namespace cviai {
namespace service {

enum { PLANE_Y = 0, PLANE_U, PLANE_V, PLANE_NUM };

static float GetYuvColor(int chanel, color_rgb *color) {
  if (color == NULL) {
    return 0;
  }

  float yuv_color = 0;
  if (chanel == PLANE_Y) {
    yuv_color = (0.257 * color->r) + (0.504 * color->g) + (0.098 * color->b) + 16;
  } else if (chanel == PLANE_U) {
    yuv_color = -(.148 * color->r) - (.291 * color->g) + (.439 * color->b) + 128;
  } else if (chanel == PLANE_V) {
    yuv_color = (0.439 * color->r) - (0.368 * color->g) - (0.071 * color->b) + 128;
  }

  return (yuv_color < 0) ? 0 : ((yuv_color > 255.) ? 255 : yuv_color);
}
// TODO: Need refactor
void WriteText(VIDEO_FRAME_INFO_S *frame, int x, int y, const char *name, color_rgb color,
               int rect_thinkness, const bool draw_text) {
  std::string name_str = name;
  int width = frame->stVFrame.u32Width;
  int height = frame->stVFrame.u32Height;
  x = max(min(x, width - 1), 0);
  y = max(min(y, height - 1), 0);

  color.r *= 255;
  color.g *= 255;
  color.b *= 255;
  char color_y = GetYuvColor(PLANE_Y, &color);
  char color_u = GetYuvColor(PLANE_U, &color);
  char color_v = GetYuvColor(PLANE_V, &color);

  CVI_VOID *vir_addr = CVI_NULL;
  size_t image_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  vir_addr = CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
  CVI_U32 plane_offset = 0;

  for (int i = PLANE_Y; i < PLANE_NUM; i++) {
    frame->stVFrame.pu8VirAddr[i] = ((CVI_U8 *)vir_addr) + plane_offset;
    plane_offset += frame->stVFrame.u32Length[i];

    char draw_color;
    if (i == PLANE_Y) {
      draw_color = color_y;
    } else if (i == PLANE_U) {
      draw_color = color_u;
    } else {
      draw_color = color_v;
    }

    cv::Size cv_size = cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
    cv::Point cv_point = cv::Point(x, y - 2);
    double font_scale = 1;
    int thickness = 1;
    if (i != 0) {
      cv_size = cv::Size(frame->stVFrame.u32Width / 2, frame->stVFrame.u32Height / 2);
      cv_point = cv::Point(x / 2, (y - 2) / 2);
      font_scale /= 2;
      // FIXME: Should div but don't know why it's not correct.
      // thickness /= 2;
    }
    // FIXME: Color incorrect.
    cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i], frame->stVFrame.u32Stride[i]);
    cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale,
                cv::Scalar(draw_color), thickness, cv::LINE_AA);
    frame->stVFrame.pu8VirAddr[i] = NULL;
  }
  CVI_SYS_IonFlushCache(frame->stVFrame.u64PhyAddr[0], vir_addr, image_size);
  CVI_SYS_Munmap(vir_addr, image_size);
}

// TODO: Need refactor
void DrawRect(VIDEO_FRAME_INFO_S *frame, float x1, float x2, float y1, float y2, const char *name,
              color_rgb color, int rect_thinkness, const bool draw_text) {
  std::string name_str = name;
  int width = frame->stVFrame.u32Width;
  int height = frame->stVFrame.u32Height;
  x1 = max(min(x1, width - 1), 0);
  x2 = max(min(x2, width - 1), 0);
  y1 = max(min(y1, height - 1), 0);
  y2 = max(min(y2, height - 1), 0);

  color.r *= 255;
  color.g *= 255;
  color.b *= 255;
  char color_y = GetYuvColor(PLANE_Y, &color);
  char color_u = GetYuvColor(PLANE_U, &color);
  char color_v = GetYuvColor(PLANE_V, &color);

  CVI_VOID *vir_addr = CVI_NULL;
  size_t image_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  vir_addr = CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
  CVI_U32 plane_offset = 0;

  for (int i = PLANE_Y; i < PLANE_NUM; i++) {
    frame->stVFrame.pu8VirAddr[i] = ((CVI_U8 *)vir_addr) + plane_offset;
    plane_offset += frame->stVFrame.u32Length[i];
    int stride = frame->stVFrame.u32Stride[i];

    int draw_x1 = x1;
    int draw_x2 = x2;
    int draw_y1 = y1;
    int draw_y2 = y2;
    int draw_rect_thinkness = rect_thinkness;
    char draw_color;
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
      draw_rect_thinkness /= 2;
    }

    // draw rect vertical line
    for (int h = draw_y1; h <= draw_y2; ++h) {
      for (int w = draw_x1; w < draw_x1 + draw_rect_thinkness; ++w) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w), draw_color,
               sizeof(draw_color));
      }
      for (int w = draw_x2 - draw_rect_thinkness; (w < draw_x2) && (w >= 0); ++w) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w), draw_color,
               sizeof(draw_color));
      }
    }

    // draw rect horizontal line
    for (int w = draw_x1; w <= draw_x2; ++w) {
      for (int h = draw_y1; h < draw_y1 + draw_rect_thinkness; ++h) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w), draw_color,
               sizeof(draw_color));
      }
      for (int h = draw_y2 - draw_rect_thinkness; (h < draw_y2) && (h >= 0); ++h) {
        memset((void *)(frame->stVFrame.pu8VirAddr[i] + h * stride + w), draw_color,
               sizeof(draw_color));
      }
    }

    if (!draw_text) {
      continue;
    }
    cv::Size cv_size = cv::Size(frame->stVFrame.u32Width, frame->stVFrame.u32Height);
    cv::Point cv_point = cv::Point(x1, y1 - 2);
    double font_scale = 2;
    int thickness = 8;
    if (i != 0) {
      cv_size = cv::Size(frame->stVFrame.u32Width / 2, frame->stVFrame.u32Height / 2);
      cv_point = cv::Point(x1 / 2, (y1 - 2) / 2);
      font_scale /= 2;
      // FIXME: Should div but don't know why it's not correct.
      // thickness /= 2;
    }
    // FIXME: Color incorrect.
    cv::Mat image(cv_size, CV_8UC1, frame->stVFrame.pu8VirAddr[i], frame->stVFrame.u32Stride[i]);
    cv::putText(image, name_str, cv_point, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(draw_color), thickness, 8);
    frame->stVFrame.pu8VirAddr[i] = NULL;
  }
  CVI_SYS_IonFlushCache(frame->stVFrame.u64PhyAddr[0], vir_addr, image_size);
  CVI_SYS_Munmap(vir_addr, image_size);
}

int WriteMeta(char *name, int x, int y, VIDEO_FRAME_INFO_S *drawFrame, const bool drawText) {
  color_rgb rgb_color;
  rgb_color.r = DEFAULT_RECT_COLOR_R;
  rgb_color.g = DEFAULT_RECT_COLOR_G;
  rgb_color.b = DEFAULT_RECT_COLOR_B;
  for (size_t i = 0; i < 1; i++) {
    WriteText(drawFrame, x, y, name, rgb_color, DEFAULT_RECT_THINKNESS, drawText);
  }
  return CVI_SUCCESS;
}

template <typename T>
int DrawMeta(const T *meta, VIDEO_FRAME_INFO_S *drawFrame, const bool drawText) {
  if (meta->size == 0) {
    return CVI_SUCCESS;
  }
  color_rgb rgb_color;
  rgb_color.r = DEFAULT_RECT_COLOR_R;
  rgb_color.g = DEFAULT_RECT_COLOR_G;
  rgb_color.b = DEFAULT_RECT_COLOR_B;
  for (size_t i = 0; i < meta->size; i++) {
    cvai_bbox_t bbox =
        box_rescale(drawFrame->stVFrame.u32Width, drawFrame->stVFrame.u32Height, meta->width,
                    meta->height, meta->info[i].bbox, meta->rescale_type);
    DrawRect(drawFrame, bbox.x1, bbox.x2, bbox.y1, bbox.y2, meta->info[i].name, rgb_color,
             DEFAULT_RECT_THINKNESS, drawText);
  }
  return CVI_SUCCESS;
}

template int DrawMeta<cvai_face_t>(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *drawFrame,
                                   const bool drawText);
template int DrawMeta<cvai_object_t>(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *drawFrame,
                                     const bool drawText);

int DrawPose17(const cvai_object_t *obj, VIDEO_FRAME_INFO_S *frame) {
  frame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.u32Length[0]);
  cv::Mat img(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
              frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  if (img.data == nullptr) {
    return CVI_FAILURE;
  }

  for (uint32_t i = 0; i < obj->size; ++i) {
    std::vector<cv::Point2f> kp_preds(17);
    std::vector<float> kp_scores(17);

    cvai_pose17_meta_t pose = obj->info[i].pose_17;
    for (int i = 0; i < 17; ++i) {
      kp_preds[i].x = pose.x[i];
      kp_preds[i].y = pose.y[i];
      kp_scores[i] = pose.score[i];
    }

    cv::Point2f extra_pred;
    extra_pred.x = (kp_preds[5].x + kp_preds[6].x) / 2;
    extra_pred.y = (kp_preds[5].y + kp_preds[6].y) / 2;
    kp_preds.push_back(extra_pred);

    float extra_score = (kp_scores[5] + kp_scores[6]) / 2;
    kp_scores.push_back(extra_score);

    // Draw keypoints
    std::unordered_map<int, std::pair<int, int>> part_line;
    for (uint32_t n = 0; n < kp_scores.size(); n++) {
      if (kp_scores[n] <= 0.35) continue;

      int cor_x = kp_preds[n].x;
      int cor_y = kp_preds[n].y;
      part_line[n] = std::make_pair(cor_x, cor_y);

      cv::Mat bg;
      img.copyTo(bg);
      cv::circle(bg, cv::Size(cor_x, cor_y), 2, p_color[n], -1);
      float transparency = max(float(0.0), min(float(1.0), kp_scores[n]));
      cv::addWeighted(bg, transparency, img, 1 - transparency, 0, img);
    }

    // Draw limbs
    for (uint32_t i = 0; i < l_pair.size(); i++) {
      int start_p = l_pair[i].first;
      int end_p = l_pair[i].second;
      if (part_line.count(start_p) > 0 && part_line.count(end_p) > 0) {
        std::pair<int, int> start_xy = part_line[start_p];
        std::pair<int, int> end_xy = part_line[end_p];

        float mX = (start_xy.first + end_xy.first) / 2;
        float mY = (start_xy.second + end_xy.second) / 2;
        float length = sqrt(pow((start_xy.second - end_xy.second), 2) +
                            pow((start_xy.first - end_xy.first), 2));
        float angle =
            atan2(start_xy.second - end_xy.second, start_xy.first - end_xy.first) * 180.0 / M_PI;
        float stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1;
        std::vector<cv::Point> polygon;
        cv::ellipse2Poly(cv::Point(int(mX), int(mY)), cv::Size(int(length / 2), stickwidth),
                         int(angle), 0, 360, 1, polygon);

        cv::Mat bg;
        img.copyTo(bg);
        cv::fillConvexPoly(bg, polygon, line_color[i]);
        float transparency =
            max(float(0.0), min(float(1.0), float(0.5) * (kp_scores[start_p] + kp_scores[end_p])));
        cv::addWeighted(bg, transparency, img, 1 - transparency, 0, img);
      }
    }
  }

  CVI_SYS_IonFlushCache(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0],
                        frame->stVFrame.u32Length[0]);
  CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Length[0]);
  frame->stVFrame.pu8VirAddr[0] = NULL;

  // frame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0],
  //                                                             frame->stVFrame.u32Length[0]);
  // cv::Mat draw_img(frame->stVFrame.u32Height, frame->stVFrame.u32Width, CV_8UC3,
  //             frame->stVFrame.pu8VirAddr[0], frame->stVFrame.u32Stride[0]);
  // cv::cvtColor(draw_img, draw_img, CV_RGB2BGR);
  // cv::imwrite("/mnt/data/out2.jpg", draw_img);

  return CVI_SUCCESS;
}

}  // namespace service
}  // namespace cviai
