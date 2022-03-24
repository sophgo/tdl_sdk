#include <signal.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_mot_utils.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <inttypes.h>

// #define OUTPUT_MOT_DATA
// #define VISUAL_UNSTABLE_TRACKER

const char DUMP_DATA_DIR[] = "cviai_MOT_data";
const char DUMP_DATA_INFO_NAME[] = "MOT_data_info.txt";

typedef struct {
  int R;
  int G;
  int B;
} COLOR_RGB_t;

COLOR_RGB_t GET_RANDOM_COLOR(uint64_t seed, int min);
void GENERATE_VISUAL_COLOR_OBJ(cvai_object_t *faces, cvai_tracker_t *trackers, COLOR_RGB_t *colors);
void GENERATE_VISUAL_COLOR_FACE(cvai_face_t *faces, cvai_tracker_t *trackers, COLOR_RGB_t *colors);

/* global variables */
static volatile bool bExit = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 11) {
    printf(
        "Usage: %s <face/ person/ vehicle/ pet>\n"
        "          <obj detection model name>\n"
        "          <obj detection model path>\n"
        "          <face detection model path>\n"
        "          <reid model path>\n"
        "          <face recognition model path>\n"
        "          <det threshold>\n"
        "          <enable DeepSORT>\n"
        "          <DeepSORT config path>\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  TARGET_TYPE_e target_type;
  char *target_type_name = argv[1];
  if (CVI_SUCCESS != GET_TARGET_TYPE(&target_type, target_type_name)) {
    printf("GET_TARGET_TYPE error!\n");
    return CVI_FAILURE;
  }
  char *od_model_name = argv[2];
  char *od_model_path = argv[3];
  char *fd_model_path = argv[4];
  char *reid_model_path = argv[5];
  char *fr_model_path = argv[6];
  float det_threshold = atof(argv[7]);
  bool enable_DeepSORT = atoi(argv[8]) == 1;
  char *deepsort_config_path = argv[9];
  CVI_S32 voType = atoi(argv[10]);

  CVI_S32 s32Ret = CVI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  PIXEL_FORMAT_E aiInputFormat = PIXEL_FORMAT_RGB_888;
  // PIXEL_FORMAT_E aiInputFormat = PIXEL_FORMAT_NV21;
  if (InitVideoSystem(&vs_ctx, &aiInputSize, aiInputFormat, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;

  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(od_model_name, &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);
  ret |= CVI_AI_OpenModel(ai_handle, od_model_id, od_model_path);
  ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, fd_model_path);
  ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, reid_model_path);
  ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, fr_model_path);
  if (ret != CVI_SUCCESS) {
    printf("model open failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, od_model_id, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, false);

  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  switch (target_type) {
    case FACE:
      break;
    case PERSON:
      CVI_AI_SelectDetectClass(ai_handle, od_model_id, 1, CVI_AI_DET_TYPE_PERSON);
      break;
    case VEHICLE:
    case PET:
    default:
      printf("not support target type[%d] now\n", target_type);
      return CVI_FAILURE;
  }
  CVI_AI_SetModelThreshold(ai_handle, od_model_id, det_threshold);
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, det_threshold);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle, false);
  cvai_deepsort_config_t ds_conf;
  FILE *inFile_mot_config;
  inFile_mot_config = fopen(deepsort_config_path, "r");
  if (inFile_mot_config == NULL) {
    printf("failed to read DeepSORT config file: %s\n", deepsort_config_path);
    printf("use predefined config...\n");
    if (CVI_SUCCESS != GET_PREDEFINED_CONFIG(&ds_conf, target_type)) {
      printf("GET_PREDEFINED_CONFIG error!\n");
      return CVI_FAILURE;
    }
  } else {
    fread(&ds_conf, sizeof(cvai_deepsort_config_t), 1, inFile_mot_config);
    fclose(inFile_mot_config);
  }
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, true);

#ifdef OUTPUT_MOT_DATA
  char outFile_data_path[256];
  sprintf(outFile_data_path, "%s/%s", DUMP_DATA_DIR, DUMP_DATA_INFO_NAME);
  FILE *outFile_data;
  outFile_data = fopen(outFile_data_path, "w");
  if (outFile_data == NULL) {
    printf("There is a problem opening the output file: %s.\n", outFile_data_path);
    return CVI_FAILURE;
  }
  fprintf(outFile_data, "%d %d\n", -1, (int)enable_DeepSORT);
#endif

  cvai_object_t obj_meta;
  cvai_face_t face_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&face_meta, 0, sizeof(cvai_face_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

  VIDEO_FRAME_INFO_S stVIFrame;
  VIDEO_FRAME_INFO_S stVOFrame;

  uint32_t frame_counter;
  frame_counter = 0;
  while (bExit == false) {
    frame_counter += 1;
    printf("\nFrame counter[%u]\n", frame_counter);

    if (CVI_SUCCESS != CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                            vs_ctx.vpssConfigs.vpssChnAI, &stVIFrame, 2000)) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    switch (target_type) {
      case PERSON: {
        inference(ai_handle, &stVIFrame, &obj_meta);
        if (enable_DeepSORT) {
          CVI_AI_OSNet(ai_handle, &stVIFrame, &obj_meta);
        }
        CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, enable_DeepSORT);
      } break;
      case FACE: {
        CVI_AI_RetinaFace(ai_handle, &stVIFrame, &face_meta);
        if (enable_DeepSORT) {
          CVI_AI_FaceRecognition(ai_handle, &stVIFrame, &face_meta);
        }
        CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, enable_DeepSORT);
      } break;
      default:
        break;
    }
    if (CVI_SUCCESS != CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                                vs_ctx.vpssConfigs.vpssChnAI, &stVIFrame)) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

#ifdef OUTPUT_MOT_DATA
    fprintf(outFile_data, "%u\n", tracker_meta.size);
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      cvai_bbox_t *target_bbox =
          (target_type == FACE) ? &face_meta.info[i].bbox : &obj_meta.info[i].bbox;
      fprintf(outFile_data, "%f,%f,%f,%f\n", target_bbox->x1, target_bbox->y1, target_bbox->x2,
              target_bbox->y2);
    }
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      cvai_feature_t *target_feature =
          (target_type == FACE) ? &face_meta.info[i].feature : &obj_meta.info[i].feature;
      fprintf(outFile_data, "%u %d %s\n", target_feature->size, target_feature->type, "NULL");
    }
#endif

    if (voType) {
      if (CVI_SUCCESS != CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                              vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame,
                                              1000)) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }

      uint32_t image_size = stVOFrame.stVFrame.u32Length[0] + stVOFrame.stVFrame.u32Length[1] +
                            stVOFrame.stVFrame.u32Length[2];
      stVOFrame.stVFrame.pu8VirAddr[0] =
          (uint8_t *)CVI_SYS_MmapCache(stVOFrame.stVFrame.u64PhyAddr[0], image_size);
      stVOFrame.stVFrame.pu8VirAddr[1] =
          stVOFrame.stVFrame.pu8VirAddr[0] + stVOFrame.stVFrame.u32Length[0];
      stVOFrame.stVFrame.pu8VirAddr[2] =
          stVOFrame.stVFrame.pu8VirAddr[1] + stVOFrame.stVFrame.u32Length[1];

      cvai_face_t face_p;
      cvai_object_t obj_p;
      COLOR_RGB_t *colors = (COLOR_RGB_t *)malloc(tracker_meta.size * sizeof(COLOR_RGB_t));
      switch (target_type) {
        case PERSON: {
          obj_p.size = 1;
          obj_p.height = obj_meta.height;
          obj_p.width = obj_meta.width;
          obj_p.rescale_type = obj_meta.rescale_type;
          GENERATE_VISUAL_COLOR_OBJ(&obj_meta, &tracker_meta, colors);
        } break;
        case FACE: {
          face_p.size = 1;
          face_p.height = face_meta.height;
          face_p.width = face_meta.width;
          face_p.rescale_type = face_meta.rescale_type;
          GENERATE_VISUAL_COLOR_FACE(&face_meta, &tracker_meta, colors);
        } break;
        default:
          break;
      }

      for (uint32_t i = 0; i < tracker_meta.size; i++) {
#ifndef VISUAL_UNSTABLE_TRACKER
        if (tracker_meta.info[i].state != CVI_TRACKER_STABLE) continue;
#endif
        cvai_service_brush_t brush;
        brush.color.r = (float)colors[i].R;
        brush.color.g = (float)colors[i].G;
        brush.color.b = (float)colors[i].B;
        brush.size = 2;
        cvai_bbox_t *bbox_p = NULL;
        char id_num[64];
        uint64_t u_id =
            (target_type == FACE) ? face_meta.info[i].unique_id : obj_meta.info[i].unique_id;
        sprintf(id_num, "%" PRIu64 "", u_id);
        switch (target_type) {
          case PERSON: {
            obj_p.info = &obj_meta.info[i];
            CVI_AI_Service_ObjectDrawRect(service_handle, &obj_p, &stVOFrame, false, brush);
            bbox_p = &obj_meta.info[i].bbox;
          } break;
          case FACE: {
            face_p.info = &face_meta.info[i];
            CVI_AI_Service_FaceDrawRect(service_handle, &face_p, &stVOFrame, false, brush);
            bbox_p = &face_meta.info[i].bbox;
          } break;
          default:
            break;
        }
        CVI_AI_Service_ObjectWriteText(id_num, bbox_p->x1, bbox_p->y1, &stVOFrame,
                                       (float)colors[i].R / 255., (float)colors[i].G / 255.,
                                       (float)colors[i].B / 255.);
      }

      CVI_SYS_Munmap((void *)stVOFrame.stVFrame.pu8VirAddr[0], image_size);
      stVOFrame.stVFrame.pu8VirAddr[0] = NULL;
      stVOFrame.stVFrame.pu8VirAddr[1] = NULL;
      stVOFrame.stVFrame.pu8VirAddr[2] = NULL;

      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    switch (target_type) {
      case FACE:
        CVI_AI_Free(&face_meta);
        break;
      case PERSON:
      case VEHICLE:
      case PET:
        CVI_AI_Free(&obj_meta);
        break;
      default:
        break;
    }
    CVI_AI_Free(&tracker_meta);
  }

#ifdef OUTPUT_MOT_DATA
  fclose(outFile_data);
#endif

  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}

COLOR_RGB_t GET_RANDOM_COLOR(uint64_t seed, int min) {
  float scale = (256. - (float)min) / 256.;
  srand((uint32_t)seed);
  COLOR_RGB_t color;
  color.R = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  color.G = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  color.B = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  return color;
}

void GENERATE_VISUAL_COLOR_OBJ(cvai_object_t *objects, cvai_tracker_t *trackers,
                               COLOR_RGB_t *colors) {
  for (uint32_t i = 0; i < objects->size; i++) {
    if (trackers->info[i].state == CVI_TRACKER_NEW) {
      colors[i].R = 255;
      colors[i].G = 255;
      colors[i].B = 255;
      continue;
    }
    COLOR_RGB_t tmp_color = GET_RANDOM_COLOR(objects->info[i].unique_id, 64);
    if (trackers->info[i].state == CVI_TRACKER_UNSTABLE) {
      tmp_color.R = tmp_color.R / 2;
      tmp_color.G = tmp_color.G / 2;
      tmp_color.B = tmp_color.B / 2;
    }
    colors[i] = tmp_color;
  }
}

void GENERATE_VISUAL_COLOR_FACE(cvai_face_t *faces, cvai_tracker_t *trackers, COLOR_RGB_t *colors) {
  for (uint32_t i = 0; i < faces->size; i++) {
    if (trackers->info[i].state == CVI_TRACKER_NEW) {
      colors[i].R = 255;
      colors[i].G = 255;
      colors[i].B = 255;
      continue;
    }
    COLOR_RGB_t tmp_color = GET_RANDOM_COLOR(faces->info[i].unique_id, 64);
    if (trackers->info[i].state == CVI_TRACKER_UNSTABLE) {
      tmp_color.R = tmp_color.R / 2;
      tmp_color.G = tmp_color.G / 2;
      tmp_color.B = tmp_color.B / 2;
    }
    colors[i] = tmp_color;
  }
}