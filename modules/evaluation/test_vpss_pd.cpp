
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
#include "mapi.hpp"
#include "opencv2/opencv.hpp"
#include "sys_utils.hpp"

std::string g_model_root;
cvai_bbox_t box;

std::string run_image_person_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  CVI_S32 ret;
  if (model_init == 0) {
    std::cout << "to init Person model" << std::endl;
    std::string str_person_model = g_model_root;

    ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN,
                           str_person_model.c_str());
    if (ret != CVI_SUCCESS) {
      std::cout << "open model failed:" << str_person_model << std::endl;
      return "";
    }
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, 0.01);
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, true);
    CVI_AI_SetTpuFusePreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, true);
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

static void get_frame_from_mat(VIDEO_FRAME_INFO_S &in_frame, const cv::Mat &mat) {
  CVI_MAPI_AllocateFrame(&in_frame, mat.cols, mat.rows, PIXEL_FORMAT_BGR_888);
  CVI_MAPI_FrameMmap(&in_frame, true);
  uint8_t *src_ptr = mat.data;
  uint8_t *dst_ptr = in_frame.stVFrame.pu8VirAddr[0];
  for (int h = 0; h < mat.rows; ++h) {
    memcpy(dst_ptr, src_ptr, mat.cols * mat.elemSize());
    src_ptr += mat.step[0];
    dst_ptr += in_frame.stVFrame.u32Stride[0];
  }
  CVI_MAPI_FrameFlushCache(&in_frame);
  CVI_MAPI_FrameMunmap(&in_frame);
}

int main(int argc, char *argv[]) {
  CVI_S32 ret = 0;
  g_model_root = std::string(argv[1]);
  std::string image_root(argv[2]);
  std::string process_flag(argv[3]);

  if (argc != 4) {
    printf("need 3 arg, eg ./test_run_dataset_docker xxxx.cvimodel xxx.jpg person\n");
    return CVIAI_FAILURE;
  }

  // imread
  cv::Mat image;
  image = cv::imread(image_root);
  if (!image.data) {
    printf("Could not open or find the image\n");
    return -1;
  }

  // init vb
  CVI_MAPI_Media_Init(image.cols, image.rows, 2);
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::map<std::string, std::function<std::string(VIDEO_FRAME_INFO_S *, cviai_handle_t)>>
      process_funcs = {{"person", run_image_person_detection}};
  if (process_funcs.count(process_flag) == 0) {
    std::cout << "error flag:" << process_flag << std::endl;
    return -1;
  }

  int32_t height = 256;
  int32_t width = 384;

  // init vpss
  PreprocessArg arg;
  arg.width = width;
  arg.height = height;
  // attention:transform model need set pixel format YUV420_PLANAR
  arg.yuv_type = YUV420_PLANAR;
  init_vpss(image.cols, image.rows, &arg);
  VIDEO_FRAME_INFO_S frame_in;
  VIDEO_FRAME_INFO_S frame_preprocessed;
  memset(&frame_in, 0x00, sizeof(frame_in));

  get_frame_from_mat(frame_in, image);

  if (CVI_SUCCESS != CVI_VPSS_SendFrame(0, &frame_in, -1)) {
    printf("send frame failed\n");
    return -1;
  }
  if (CVI_SUCCESS != CVI_VPSS_GetChnFrame(0, 0, &frame_preprocessed, 1000)) {
    printf("get frame failed\n");
    return -1;
  }
  CVI_MAPI_ReleaseFrame(&frame_in);

  std::string str_res = process_funcs[process_flag](&frame_preprocessed, ai_handle);
  if (str_res.size() > 0) {
    FILE *fp = fopen("test.txt", "w");
    fwrite(str_res.c_str(), str_res.size(), 1, fp);
    fclose(fp);
  }
  if (CVI_SUCCESS != CVI_VPSS_ReleaseChnFrame(0, 0, &frame_preprocessed)) {
    printf("release frame failed!\n");
    return -1;
  }

  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
