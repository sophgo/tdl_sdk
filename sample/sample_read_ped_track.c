#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"

int ReleaseImage(VIDEO_FRAME_INFO_S *frame) {
  CVI_S32 ret = CVI_SUCCESS;
  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    ret = CVI_SYS_IonFree(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0]);
    frame->stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}
void export_img_result(const char *sz_dstf, cvai_object_t *p_objinfo, int imgw, int imgh) {
  FILE *fp = fopen(sz_dstf, "w");

  for (uint32_t i = 0; i < p_objinfo->size; i++) {
    // if(p_objinfo->info[i].unique_id != 0){
    // sprintf( buf, "\nOD DB File Size = %" PRIu64 " bytes \t"
    char szinfo[128];
    float ctx = (p_objinfo->info[i].bbox.x1 + p_objinfo->info[i].bbox.x2) / 2 / imgw;
    float cty = (p_objinfo->info[i].bbox.y1 + p_objinfo->info[i].bbox.y2) / 2 / imgh;
    float ww = (p_objinfo->info[i].bbox.x2 - p_objinfo->info[i].bbox.x1) / imgw;
    float hh = (p_objinfo->info[i].bbox.y2 - p_objinfo->info[i].bbox.y1) / imgh;
    float score = p_objinfo->info[i].bbox.score;
    sprintf(szinfo, "%d %f %f %f %f %" PRIu64 " %f\n", 4, ctx, cty, ww, hh,
            p_objinfo->info[i].unique_id, score);
    int tid = p_objinfo->info[i].unique_id;
    printf("trackid:%d\n", tid);

    fwrite(szinfo, 1, strlen(szinfo), fp);
  }
  fclose(fp);
}
int main(int argc, char *argv[]) {
  CVI_S32 ret = CVI_SUCCESS;

  cviai_handle_t ai_handle = NULL;

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);

  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 1);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, argv[1]);
  CVI_AI_DeepSORT_Init(ai_handle, true);

  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  ds_conf.ktracker_conf.accreditation_threshold = 4;
  ds_conf.ktracker_conf.P_beta[2] = 0.01;
  ds_conf.ktracker_conf.P_beta[6] = 1e-5;
  ds_conf.kfilter_conf.Q_beta[2] = 0.01;
  ds_conf.kfilter_conf.Q_beta[6] = 1e-5;
  ds_conf.kfilter_conf.R_beta[2] = 0.1;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, true);

  for (int img_idx = 0; img_idx < 10; img_idx++) {
    char szimg[256];
    sprintf(szimg, "%s/%08d.bin", argv[2], img_idx);
    VIDEO_FRAME_INFO_S fdFrame;
    ret = CVI_AI_LoadBinImage(szimg, &fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
    int imgw = fdFrame.stVFrame.u32Width;
    int imgh = fdFrame.stVFrame.u32Width;
    printf("start to process:%s,width:%d,height:%d\n", szimg, imgw, imgh);
    cvai_object_t obj_meta;
    memset(&obj_meta, 0, sizeof(cvai_object_t));
    cvai_tracker_t tracker_meta;
    memset(&tracker_meta, 0, sizeof(cvai_tracker_t));
    CVI_AI_MobileDetV2_Pedestrian(ai_handle, &fdFrame, &obj_meta);
    int objnum = obj_meta.size;
    printf("objnum:%d\n", objnum);
    CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, false);
    char dstf[256];
    sprintf(dstf, "%s/%08d.txt", argv[3], img_idx);

    export_img_result(dstf, &obj_meta, imgw, imgh);
    ReleaseImage(&fdFrame);
  }

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
  CVI_VB_Exit();
  return 0;
}
