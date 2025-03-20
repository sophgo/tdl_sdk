#include <cstdlib>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "meta_visualize.h"

int32_t TDL_VisualizeRectangle(box_t *box,
                                   int32_t num,
                                   char *input_path,
                                   char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    printf("input path is empty\n");
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

int32_t TDL_VisualizePoint(point_t *point,
                               int32_t num,
                               char *input_path,
                               char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    printf("input path is empty\n");
    return -1;
  }

  for (int32_t i = 0; i < num; i++) {
      cv::circle(image,
                 cv::Point(int32_t(point[i].x), int32_t(point[i].y)),
                 7,
                 cv::Scalar(0, 0, 255),
                 -1);
  }

  cv::imwrite(output_path, image);

  return 0;
}

int32_t TDL_VisualizeLine(box_t *box,
                              int32_t num,
                              char *input_path,
                              char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    printf("input path is empty\n");
    return -1;
  }

  for (int32_t i = 0; i < num; i++) {
      cv::line(image,
                 cv::Point(int32_t(box[i].x1), int32_t(box[i].y1)),
                 cv::Point(int32_t(box[i].x2), int32_t(box[i].y2)),
                 cv::Scalar(0, 0, 255), 2);
  }

  cv::imwrite(output_path, image);

  return 0;
}

int32_t TDL_VisualizePolylines(point_t *point,
                                   int32_t num,
                                   char *input_path,
                                   char *output_path) {
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    printf("input path is empty\n");
    return -1;
  }

  std::vector<cv::Point> points;
  for (int32_t i = 0; i < num; i++) {
    points.push_back(cv::Point(
      static_cast<int32_t>(point[i].x),
      static_cast<int32_t>(point[i].y)));
  }

  cv::polylines(image, points, true,
                cv::Scalar(0, 255, 0),
                2, cv::LINE_AA);

  cv::imwrite(output_path, image);

  return 0;
}