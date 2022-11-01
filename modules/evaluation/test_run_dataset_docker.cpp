
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include "core.hpp"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "sys_utils.hpp"

std::string g_model_root;
cvai_bbox_t box;

std::string run_image_person_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  CVI_S32 ret;
  if (model_init == 0) {
    std::cout << "to init Person model" << std::endl;
    std::string str_person_model = g_model_root;

    ret = CVI_AI_OpenModel_InDocker(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN,
                                    str_person_model.c_str());
    if (ret != CVI_SUCCESS) {
      std::cout << "open model failed:" << str_person_model << std::endl;
      return "";
    }
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, 0.01);
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, true);
    CVI_AI_UseInpuSysMem(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN);
    model_init = 1;
  }
  cvai_object_t person_obj;
  memset(&person_obj, 0, sizeof(cvai_object_t));
  ret = CVI_AI_MobileDetV2_Pedestrian(ai_handle, p_frame, &person_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect face failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < person_obj.size; i++) {
    box = person_obj.info[i].bbox;
    ss << (person_obj.info[i].classes + 1) << " " << box.score << " " << box.x1 << " " << box.y1
       << " " << box.x2 << " " << box.y2 << "\n";
  }

  CVI_AI_Free(&person_obj);
  return ss.str();
}

int main(int argc, char *argv[]) {
  g_model_root = std::string(argv[1]);
  std::string image_root(argv[2]);
  std::string image_list(argv[3]);
  std::string dst_root(argv[4]);
  std::string process_flag(argv[5]);

  if (image_root.at(image_root.size() - 1) != '/') {
    image_root = image_root + std::string("/");
  }
  if (dst_root.at(dst_root.size() - 1) != '/') {
    dst_root = dst_root + std::string("/");
  }
  create_directory(dst_root);
  int starti = 0;
  CVI_S32 ret = 0;
  if (argc > 7) starti = atoi(argv[6]);

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::cout << "to read imagelist:" << image_list << std::endl;
  std::vector<std::string> image_files = read_file_lines(image_list);
  if (image_root.size() == 0) {
    std::cout << ",imageroot empty\n";
    return -1;
  }

  std::map<std::string, std::function<std::string(VIDEO_FRAME_INFO_S *, cviai_handle_t)>>
      process_funcs = {{"person", run_image_person_detection}};
  if (process_funcs.count(process_flag) == 0) {
    std::cout << "error flag:" << process_flag << std::endl;
    return -1;
  }

  for (uint32_t i = starti; i < image_files.size(); i++) {
    std::cout << "processing :" << i << "/" << image_files.size() << std::endl;
    std::string strf = image_root + image_files[i];
    std::string dstf = dst_root + replace_file_ext(image_files[i], "txt");

    int32_t height = 256;
    int32_t width = 384;

    // imread
    cv::Mat image;
    image = cv::imread(strf.c_str());
    if (!image.data) {
      printf("Could not open or find the image\n");
      return -1;
    }

    // resize
    cv::resize(image, image, cv::Size(height, width));  // linear is default
    // crop
    cv::Size size = cv::Size(height, width);
    cv::Rect crop(cv::Point(0.5 * (image.cols - size.width), 0.5 * (image.rows - size.height)),
                  size);
    image = image(crop);
    cv::imwrite(strf.c_str(), image);

    VIDEO_FRAME_INFO_S fdFrame;
    ret = CVI_AI_ReadImage_Docker(strf.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
    if (ret != CVI_SUCCESS) {
      std::cout << "Convert to video frame failed with:" << ret << ",file:" << strf << std::endl;
      continue;
    } else {
      std::cout << "load image, Width:" << fdFrame.stVFrame.u32Width
                << "  Height:" << fdFrame.stVFrame.u32Height << std::endl;
    }

    std::string str_res = process_funcs[process_flag](&fdFrame, ai_handle);
    if (str_res.size() > 0) {
      FILE *fp = fopen(dstf.c_str(), "w");
      fwrite(str_res.c_str(), str_res.size(), 1, fp);
      fclose(fp);
    }
    CVI_AI_ReleaseImage_Docker(&fdFrame);
  }
  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
