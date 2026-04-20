#include <image/base_image.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "cv/target_search/grabcut_segment.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cerr << "用法: " << argv[0] << " <输入目录> <输出目录>" << endl;
    cerr << "示例: " << argv[0] << " /path/to/input /path/to/output" << endl;
    return -1;
  }

  string input_dir = argv[1];
  string save_dir = argv[2];

  if (!input_dir.empty() && input_dir.back() != '/') {
    input_dir += '/';
  }
  if (!save_dir.empty() && save_dir.back() != '/') {
    save_dir += '/';
  }

  vector<pair<string, vector<Point>>> test_cases = {
      {"1280x720.png", {{567, 395}, {622, 394}, {758, 274}, {1189, 385}}}};

  cvtdl_grabcut_params_t params;
  params.iter_count = 5;

  GrabCutSegmentor segmentor(params);

  int count = 0;
  for (const auto& tc : test_cases) {
    string img_path = input_dir + tc.first;

    std::shared_ptr<BaseImage> image = ImageFactory::readImage(img_path);
    if (!image) {
      cerr << "Failed to load image: " << img_path << endl;
      continue;
    }

    Mat img = imread(img_path);
    if (img.empty()) {
      cerr << "Failed to load image for visualization: " << img_path << endl;
      continue;
    }

    for (const Point& seed : tc.second) {
      cvtdl_grabcut_result_t result;

      int ret = segmentor.segment(image, seed, &result);

      if (ret != 0) {
        cerr << "Failed to segment image: " << img_path << endl;
        continue;
      }
      if (ret != 0 || !result.success) {
        cout << "Segmentation failed for seed: (" << seed.x << "," << seed.y
             << ") with error code: " << ret << endl;
      } else {
        Mat vis = img.clone();
        rectangle(vis, result.bbox, Scalar(0, 255, 0), 2);
        circle(vis, Point(seed.x, seed.y), 3, Scalar(0, 0, 255), -1);

        string base_name = save_dir + tc.first + "_" + to_string(count);
        imwrite(base_name + "_bbox.jpg", vis);
        imwrite(base_name + "_mask.jpg", result.fg_mask * 255);

        cout << "Saved result " << count << " at: " << base_name << endl;
      }
      count++;
    }
  }

  return 0;
}
