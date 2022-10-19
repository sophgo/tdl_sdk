
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
// #include "ive/ive.h"
#include "sys_utils.hpp"
std::string g_model_root;
/*
std::string run_image_license_plate_detection(VIDEO_FRAME_INFO_S *p_frame,
                                              cviai_handle_t ai_handle) {
  static int lpd_model_init = 0;
  if (lpd_model_init == 0) {
    std::cout << "to init lpd model" << std::endl;
    std::string str_vehicle_model =
        g_model_root + std::string("/mobiledetv2-vehicle-d0-ls.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0,
                        str_vehicle_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, 0.1);
    std::string str_lpd_model = g_model_root + std::string("/wpodnet_v0_bf16.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, str_lpd_model.c_str());
    // CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, 0.01);
    lpd_model_init = 1;
  }
  cvai_object_t vehicle_obj;
  memset(&vehicle_obj, 0, sizeof(cvai_object_t));
  CVI_AI_MobileDetV2_Vehicle_D0(ai_handle, p_frame, &vehicle_obj);
  CVI_AI_LicensePlateDetection(ai_handle, p_frame, &vehicle_obj);

  // generate detection result
  std::stringstream ss;
  std::cout << "vehicle_obj.size:" << vehicle_obj.size << std::endl;
  for (uint32_t i = 0; i < vehicle_obj.size; i++) {
    if (vehicle_obj.info[i].vehicle_properity) {
      cvai_bbox_t box = vehicle_obj.info[i].vehicle_properity->license_bbox;
      std::cout << "license_plate"
                << " " << box.score << " " << box.x1 << " " << box.y1 << " " << box.x2 << " "
                << box.y2 << std::endl;
      ss << "license_plate"
         << " " << box.score << " " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2
         << "\n";
    }
  }
  CVI_AI_Free(&vehicle_obj);
  return ss.str();
}

std::string run_image_vehicle_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init vehicle model" << std::endl;
    std::string str_vehicle_model =
        g_model_root + std::string("/mobiledetv2-vehicle-d0-ls.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0,
                        str_vehicle_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, 0.01);
    model_init = 1;
  }
  cvai_object_t vehicle_obj;
  memset(&vehicle_obj, 0, sizeof(cvai_object_t));
  CVI_S32 ret = CVI_AI_MobileDetV2_Vehicle_D0(ai_handle, p_frame, &vehicle_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect vehicle failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < vehicle_obj.size; i++) {
    cvai_bbox_t box = vehicle_obj.info[i].bbox;
    ss << (vehicle_obj.info[i].classes + 1) << " " << box.score << " " << box.x1 << " " << box.y1
       << " " << box.x2 << " " << box.y2 << "\n";
  }
  CVI_AI_Free(&vehicle_obj);
  return ss.str();
}*/

std::string run_image_face_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  CVI_S32 ret;
  if (model_init == 0) {
    std::string str_face_model =
        g_model_root + std::string("/retinaface_mnet0.25_342_608.cvimodel");
    // std::string("scrfd_DW_conv_432_768_int8.cvimodel");
    std::cout << "to init face detection model:" << str_face_model << std::endl;
    ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, str_face_model.c_str());
    if (ret != CVI_SUCCESS) {
      std::cout << "open model failed:" << str_face_model << std::endl;
      return "";
    }
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, 0.01);
    model_init = 1;
  }
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  std::cout << "to do process\n";
  ret = CVI_AI_RetinaFace(ai_handle, p_frame, &face);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect face failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < face.size; i++) {
    cvai_bbox_t box = face.info[i].bbox;
    cvai_pts_t pts = face.info[i].pts;
    ss << "face"
       << " " << box.score << " " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2
       << ";";

    for (uint32_t k = 0; k < pts.size; k++) {
      ss << pts.x[k] << " " << pts.y[k] << " ";
    }
    ss << "\n";
  }
  CVI_AI_Free(&face);
  return ss.str();
}
/*
std::string run_image_facemask_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init face mask detection model" << std::endl;
    std::string str_fdmask_model =
        g_model_root + std::string("/yolox_RetinafaceMask_lm_432_768_int8_0705.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                        str_fdmask_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION, 0.01);
    model_init = 1;
  }
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  CVI_S32 ret = CVI_AI_FaceMaskDetection(ai_handle, p_frame, &face);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect face mask failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < face.size; i++) {
    std::string label = face.info[i].mask_score > 0.5 ? "mask" : "no_mask";
    cvai_bbox_t box = face.info[i].bbox;
    ss << label << " " << box.score << " " << box.x1 << " " << box.y1 << " " << box.x2 << " "
       << box.y2 << "\n";
  }
  CVI_AI_Free(&face);
  return ss.str();
}

std::string run_image_person_vehicle_detection(VIDEO_FRAME_INFO_S *p_frame,
                                               cviai_handle_t ai_handle) {
  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init Person vehicle model" << std::endl;
    std::string str_person_vehicle_model =
        g_model_root + std::string("/mobiledetv2-person-vehicle-ls-768.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0,
                        str_person_vehicle_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, 0.01);
    model_init = 1;
  }
  cvai_object_t person_vehicle_obj;
  memset(&person_vehicle_obj, 0, sizeof(cvai_object_t));
  CVI_S32 ret = CVI_AI_MobileDetV2_Person_Vehicle(ai_handle, p_frame, &person_vehicle_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect person vehicle failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < person_vehicle_obj.size; i++) {
    cvai_bbox_t box = person_vehicle_obj.info[i].bbox;
    ss << (person_vehicle_obj.info[i].classes + 1) << " " << box.score << " " << box.x1 << " "
       << box.y1 << " " << box.x2 << " " << box.y2 << "\n";
  }
  CVI_AI_Free(&person_vehicle_obj);
  return ss.str();
}

std::string run_image_person_pets_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init Person cat dog model" << std::endl;
    std::string str_person_pets_model =
        g_model_root + std::string("/mobiledetv2-lite-person-pets-ls.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS,
                        str_person_pets_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS, 0.01);
    model_init = 1;
  }
  cvai_object_t person_pets_obj;
  memset(&person_pets_obj, 0, sizeof(cvai_object_t));
  CVI_S32 ret = CVI_AI_MobileDetV2_Lite_Person_Pets(ai_handle, p_frame, &person_pets_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect person pets failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < person_pets_obj.size; i++) {
    cvai_bbox_t box = person_pets_obj.info[i].bbox;
    ss << (person_pets_obj.info[i].classes + 1) << " " << box.score << " " << box.x1 << " "
       << box.y1 << " " << box.x2 << " " << box.y2 << "\n";
  }
  CVI_AI_Free(&person_pets_obj);
  return ss.str();
}

std::string run_image_person_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init Person model" << std::endl;
    std::string str_person_model =
        g_model_root + std::string("/mobiledetv2-pedestrian-d0-ls-640.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0,
                        str_person_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, 0.01);
    model_init = 1;
  }
  cvai_object_t person_obj;
  memset(&person_obj, 0, sizeof(cvai_object_t));
  CVI_S32 ret = CVI_AI_MobileDetV2_Pedestrian_D0(ai_handle, p_frame, &person_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect person failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;
  for (uint32_t i = 0; i < person_obj.size; i++) {
    cvai_bbox_t box = person_obj.info[i].bbox;
    ss << (person_obj.info[i].classes + 1) << " " << box.score << " " << box.x1 << " " << box.y1
       << " " << box.x2 << " " << box.y2 << "\n";
  }
  CVI_AI_Free(&person_obj);
  return ss.str();
}

std::string run_image_face_recognition_directly(VIDEO_FRAME_INFO_S *p_frame,
                                                cviai_handle_t ai_handle) {
  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init face recognition model" << std::endl;
    std::string str_fa_model = g_model_root + std::string("/cviface-v5-s-0827.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, str_fa_model.c_str());
    model_init = 1;
  }
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  // face.size = 1;
  CVI_AI_MemAllocInit(1, 5, &face);

  CVI_S32 ret = CVI_AI_FaceRecognition(ai_handle, p_frame, &face);

  if (ret != CVI_SUCCESS) {
    std::cout << "face recognition failed:" << ret << std::endl;
  }
  // generate detection result
  std::stringstream ss;
  if (face.size != 1) {
    std::cout << "facesize,error:" << face.size << std::endl;
  }

  cvai_feature_t feature = face.info[0].feature;
  std::cout << "featsize:" << feature.size << std::endl;
  for (uint32_t i = 0; i < feature.size - 1; i++) {
    ss << (int32_t)feature.ptr[i] << " ";
  }
  ss << (int32_t)feature.ptr[feature.size - 1] << "\n";

  CVI_AI_Free(&face);
  std::cout << "done\n";
  return ss.str();
}

std::string run_image_face_recognition(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle) {
  return run_image_face_recognition_directly(p_frame, ai_handle);

  static int model_init = 0;
  if (model_init == 0) {
    std::cout << "to init face recognition model" << std::endl;
    std::string str_face_model =
        g_model_root + std::string("/retinaface_mnet0.25_342_608.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, str_face_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, 0.01);
    std::string str_fa_model = g_model_root + std::string("/cviface-v5-s.cvimodel");
    CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, str_fa_model.c_str());
    model_init = 1;
  }
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  CVI_S32 ret = CVI_AI_RetinaFace(ai_handle, p_frame, &face);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect face failed:" << ret << std::endl;
  }
  ret = CVI_AI_FaceRecognition(ai_handle, p_frame, &face);
  face.size = 1;
  if (ret != CVI_SUCCESS) {
    std::cout << "face recognition failed:" << ret << std::endl;
  }
  // generate detection result
  std::stringstream ss;
  cvai_feature_t feature = face.info[0].feature;
  for (uint32_t i = 0; i < feature.size - 1; i++) {
    ss << (int32_t)feature.ptr[i] << " ";
  }
  ss << (int32_t)feature.ptr[feature.size - 1] << "\n";
  CVI_AI_Free(&face);
  return ss.str();
}*/

int main(int argc, char *argv[]) {
  // if (argc != 6) {
  //   printf("Usage: %s <model_root>  <image_root> <image_list> <dst_root> <process_flag>\n",
  //          argv[0]);
  //   // return CVI_FAILURE;
  // }
  g_model_root =
      "/mnt/data/admin1_data/AI_CV/cv182x/ai_models/output/cv182x/";  // std::string(argv[1]);
  std::string image_root("/mnt/data/admin1_data/alios_test");         // argv[2]);
  std::string image_list("/mnt/data/admin1_data/alios_test/image_list.txt");  // argv[3]);
  std::string dst_root("/mnt/data/admin1_data/alios_test_predict");           // argv[4]);
  std::string process_flag("fd");                                             // argv[5]);

  if (image_root.at(image_root.size() - 1) != '/') {
    image_root = image_root + std::string("/");
  }
  if (dst_root.at(dst_root.size() - 1) != '/') {
    dst_root = dst_root + std::string("/");
  }
  create_directory(dst_root);
  int starti = 0;
  if (argc > 6) starti = atoi(argv[6]);
  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1960;
  const CVI_S32 vpssgrp_height = 1080;

  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

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
  // std::map<std::string, std::function<std::string(VIDEO_FRAME_INFO_S *, cviai_handle_t)>>
  //     process_funcs = {{"lpd", run_image_license_plate_detection},
  //                      {"vehicle", run_image_vehicle_detection},
  //                      {"fd", run_image_face_detection},
  //                      {"fdmask", run_image_facemask_detection},
  //                      {"person_vehicle", run_image_person_vehicle_detection},
  //                      {"person_pets", run_image_person_pets_detection},
  //                      {"person", run_image_person_detection},
  //                      {"face_recognition", run_image_face_recognition}};
  std::map<std::string, std::function<std::string(VIDEO_FRAME_INFO_S *, cviai_handle_t)>>
      process_funcs = {
          {"fd", run_image_face_detection},
      };
  if (process_funcs.count(process_flag) == 0) {
    std::cout << "error flag:" << process_flag << std::endl;
    return -1;
  }

  for (uint32_t i = starti; i < image_files.size(); i++) {
    std::cout << "processing :" << i << "/" << image_files.size() << std::endl;
    std::string strf = image_root + image_files[i];
    std::string dstf = dst_root + replace_file_ext(image_files[i], "txt");
    std::cout << "to read:" << strf << std::endl;
    VIDEO_FRAME_INFO_S fdFrame;
    ret = CVI_AI_ReadImage(strf.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
    if (ret != CVI_SUCCESS) {
      std::cout << "Convert to video frame failed with:" << ret << ",file:" << strf << std::endl;
      continue;
    } else {
      std::cout << "load image,width:" << fdFrame.stVFrame.u32Width << std::endl;
    }

    std::string str_res = process_funcs[process_flag](&fdFrame, ai_handle);
    if (str_res.size() > 0) {
      FILE *fp = fopen(dstf.c_str(), "w");
      fwrite(str_res.c_str(), str_res.size(), 1, fp);
      fclose(fp);
    }

    CVI_AI_ReleaseImage(&fdFrame);
  }

  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
