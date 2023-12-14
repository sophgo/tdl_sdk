#include <signal.h>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <vector>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
static bool stopped = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    stopped = true;
  }
}

struct vpssPair {
  CVI_U32 groupId;
  VIDEO_FRAME_INFO_S frame;
};

void timer();
void SWBinding(std::vector<vpssPair> vpss_vec, cvtdl_vpssconfig_t *vpssConfig);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <retina_model_path> <image> <lanes>.\n", argv[0]);
    return CVI_TDL_FAILURE;
  }
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);
  CVI_S32 ret = CVI_TDL_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 3840;
  const CVI_S32 vpssgrp_height = 2160;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 8, 608, 608,
                         PIXEL_FORMAT_RGB_888_PLANAR, 10);
#ifndef CV186X
  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);
#endif
  std::string arg_val = argv[3];
  uint32_t total_lanes = std::stoi(arg_val);
  if (!(total_lanes >= 1 && total_lanes < 10)) {
    total_lanes = 1;
  }
  printf("Lanes: %u\n", total_lanes);
  std::vector<vpssPair> vpss_vec;
  for (uint32_t i = 0; i < total_lanes; i++) {
    ret = VPSS_INIT_HELPER2(i, 100, 100, PIXEL_FORMAT_RGB_888, 100, 100, PIXEL_FORMAT_RGB_888, 2,
                            true);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Init sys failed with %#x!\n", ret);
      return ret;
    }
    vpssPair vp;
    vp.groupId = i;
    vpss_vec.push_back(vp);
  }

  // Init cvitdl handle.
  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  // Setup model path and model config.
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_TDL_SetSkipVpssPreprocess(tdl_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, true);

  for (uint32_t i = 0; i < vpss_vec.size(); i++) {
    CVI_TDL_ReadImage(argv[2], &vpss_vec[i].frame, PIXEL_FORMAT_RGB_888);
  }

  cvtdl_vpssconfig_t vpssConfig;
  CVI_TDL_GetVpssChnConfig(tdl_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE,
                           vpss_vec[0].frame.stVFrame.u32Width,
                           vpss_vec[0].frame.stVFrame.u32Height, 0, &vpssConfig);
  std::thread t1(SWBinding, vpss_vec, &vpssConfig);
  std::thread t2(timer);
  // Run inference and print result.
  while (!stopped) {
    for (uint32_t i = 0; i < vpss_vec.size(); i++) {
      VIDEO_FRAME_INFO_S frame, frFrame;
      int ret = CVI_VPSS_GetChnFrame(vpss_vec[i].groupId, 0, &frame, 1000);
      ret |= CVI_VPSS_GetChnFrame(vpss_vec[i].groupId, 1, &frFrame, 1000);
      if (ret != CVI_TDL_SUCCESS) {
        if (stopped) {
          break;
        }
        continue;
      }
      cvtdl_face_t face;
      memset(&face, 0, sizeof(cvtdl_face_t));
      CVI_TDL_RetinaFace(tdl_handle, &frame, &face);
      printf("Face found %x.\n", face.size);
      CVI_TDL_Free(&face);
      CVI_VPSS_ReleaseChnFrame(0, 0, &frame);
      CVI_VPSS_ReleaseChnFrame(0, 0, &frFrame);
    }
  }

  t1.join();
  t2.join();

  // Free image and handles.
  for (uint32_t i = 0; i < vpss_vec.size(); i++) {
    CVI_TDL_ReleaseImage(&vpss_vec[i].frame);
  }
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}

int i = 0;
std::condition_variable cv;
std::mutex cv_m;

void timer() {
  while (!stopped) {
    {
      std::lock_guard<std::mutex> lk(cv_m);
      i = 1;
    }
    cv.notify_all();
    std::this_thread::sleep_for(std::chrono::microseconds(40000));
  }
  std::lock_guard<std::mutex> lk(cv_m);
  i = 1;
  cv.notify_all();
}

void SWBinding(std::vector<vpssPair> vpss_vec, cvtdl_vpssconfig_t *vpssConfig) {
  while (!stopped) {
    for (uint32_t i = 0; i < vpss_vec.size(); i++) {
      VIDEO_FRAME_INFO_S *fdFrame = &vpss_vec[i].frame;
      VPSS_GRP_ATTR_S vpss_grp_attr;
      VPSS_CHN_ATTR_S vpss_chn_attr;
      VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, fdFrame->stVFrame.u32Width,
                              fdFrame->stVFrame.u32Height, fdFrame->stVFrame.enPixelFormat);
      VPSS_CHN_DEFAULT_HELPER(&vpss_chn_attr, fdFrame->stVFrame.u32Width,
                              fdFrame->stVFrame.u32Height, fdFrame->stVFrame.enPixelFormat, true);
      // FIXME: Setting to 1 cause timeout
      // vpss_grp_attr.u8VpssDev = 1;
      CVI_VPSS_SetGrpAttr(vpss_vec[i].groupId, &vpss_grp_attr);
      CVI_VPSS_SetChnAttr(vpss_vec[i].groupId, 0, &vpssConfig->chn_attr);
      CVI_VPSS_SetChnAttr(vpss_vec[i].groupId, 1, &vpss_chn_attr);
      CVI_VPSS_SetChnScaleCoefLevel(vpss_vec[i].groupId, 0, vpssConfig->chn_coeff);
      CVI_VPSS_SendFrame(vpss_vec[i].groupId, &vpss_vec[i].frame, -1);
    }
    std::unique_lock<std::mutex> lk(cv_m);
    cv.wait(lk, [] { return i == 1; });
    i = 0;
  }
}
