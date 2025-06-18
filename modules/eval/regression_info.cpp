#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "sys_utils.hpp"
#include <dirent.h>
#include <iomanip>
#include <type_traits>

struct ModelProcess {
    CVI_TDL_SUPPORTED_MODEL_E model_index;
    std::function<std::string(VIDEO_FRAME_INFO_S*, cvitdl_handle_t, CVI_TDL_SUPPORTED_MODEL_E)> process_func;
};

template <typename ObjectType>
void output_info(std::stringstream &ss, ObjectType *obj, uint32_t i) {

}

template <>
void output_info<cvtdl_object_t>(std::stringstream &ss, cvtdl_object_t *obj, uint32_t i) {
    ss << " " << obj->info[i].classes;  
}

template <typename ObjectType>
void object_info(ObjectType *obj, std::stringstream &ss) {

    ss << std::fixed << std::setprecision(2);
    
    for (uint32_t i = 0; i < obj->size; i++) {
        cvtdl_bbox_t box = obj->info[i].bbox;
        ss << " " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2;
        output_info(ss, obj, i);
        ss << " " << box.score << "\n";
    }
}


std::string common_object_detection(VIDEO_FRAME_INFO_S *p_frame, cvitdl_handle_t tdl_handle,
                                       CVI_TDL_SUPPORTED_MODEL_E model_index) {

  cvtdl_object_t obj = {0};
  CVI_S32 ret = CVI_TDL_Detection(tdl_handle, p_frame, model_index, &obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect object failed:" << ret << std::endl;
  }

  std::stringstream ss;
  object_info(&obj, ss);
  CVI_TDL_Free(&obj);
  return ss.str();
}

std::string common_face_detection(VIDEO_FRAME_INFO_S *p_frame, cvitdl_handle_t tdl_handle,
                                       CVI_TDL_SUPPORTED_MODEL_E model_index) {

  cvtdl_face_t obj = {0};
  CVI_S32 ret = CVI_TDL_FaceDetection(tdl_handle, p_frame, model_index, &obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect face failed:" << ret << std::endl;
  }

  std::stringstream ss;
  object_info(&obj, ss);
  CVI_TDL_Free(&obj);
  return ss.str();
}



std::map<std::string, ModelProcess> process_info = {

  // object detection
  {"yolov5", {CVI_TDL_SUPPORTED_MODEL_YOLOV5, common_object_detection}},
  {"yolov8", {CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, common_object_detection}},
  {"yolov10", {CVI_TDL_SUPPORTED_MODEL_YOLOV10_DETECTION, common_object_detection}},
  {"fire_smoke", {CVI_TDL_SUPPORTED_MODEL_YOLOV8_FIRE_SMOKE, common_object_detection}},
  {"hand", {CVI_TDL_SUPPORTED_MODEL_HAND_DETECTION, common_object_detection}},
  {"person_pet", {CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION, common_object_detection}},
  {"person_vehicle", {CVI_TDL_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION, common_object_detection}},
  {"hand_face_person", {CVI_TDL_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION, common_object_detection}},
  {"head_person", {CVI_TDL_SUPPORTED_MODEL_HEAD_PERSON_DETECTION, common_object_detection}},
  {"mobiledetv2_pedestrian", {CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, common_object_detection}},

  // face detection
  {"scrfdface", {CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, common_face_detection}},
  {"retinaface", {CVI_TDL_SUPPORTED_MODEL_RETINAFACE, common_face_detection}},

};


int main(int argc, char *argv[]) {

  std::string process_flag(argv[1]);
  std::string image_dir(argv[2]);
  std::string dst_root(argv[3]);
  std::string model_path(argv[4]);

  if (image_dir.at(image_dir.size() - 1) != '/') {
    image_dir = image_dir + std::string("/");
  }
  if (dst_root.at(dst_root.size() - 1) != '/') {
    dst_root = dst_root + std::string("/");
  }

  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }
  std::cout << "to read image dir:" << image_dir << std::endl;

  std::vector<std::string> image_files;
  ret = gen_file_names(image_dir, image_files);
  if (ret != CVI_SUCCESS) {
    printf("gen file names failed with %#x!\n", ret);
    return ret;
  }

  if (image_files.size() == 0) {
    std::cout << ", imageroot empty\n";
    return -1;
  }

  if (process_info.count(process_flag) == 0) {
    std::cout << "error flag:" << process_flag << std::endl;
    return -1;
  }

  CVI_TDL_SUPPORTED_MODEL_E model_index = process_info[process_flag].model_index;

  ret = CVI_TDL_OpenModel(tdl_handle, model_index, argv[4]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  for (size_t i = 0; i < image_files.size(); i++) {
    std::cout << "processing :" << i << "/" << image_files.size() << "\t" << image_files[i]
              << std::endl;
    std::string strf = image_dir + image_files[i];
    std::string dstf = dst_root + replace_file_ext(image_files[i], "txt");
    VIDEO_FRAME_INFO_S fdFrame;

    ret = CVI_TDL_ReadImage(img_handle, strf.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);

    std::cout << "CVI_TDL_ReadImage done\t";

    if (ret != CVI_SUCCESS) {
      std::cout << "Convert to video frame failed with:" << ret << ",file:" << strf << std::endl;
      continue;
    }
    std::string str_res;

    str_res = process_info[process_flag].process_func(&fdFrame, tdl_handle, model_index);
    std::cout << "process_info done\t";
    // std::cout << "str_res.size():" << str_res.size() << std::endl;

    std::cout << "writing file:" << dstf << std::endl;
    FILE *fp = fopen(dstf.c_str(), "w");
    fwrite(str_res.c_str(), str_res.size(), 1, fp);
    fclose(fp);

    CVI_TDL_ReleaseImage(img_handle, &fdFrame);
  }
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
