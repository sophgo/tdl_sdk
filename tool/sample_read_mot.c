#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "sample_mot_utils.h"
#include "utils/od.h"

#include <inttypes.h>

#define OUTPUT_MOT_RESULT
#define OUTPUT_MOT_DATA

const char RESULT_FILE_PATH[] = "sample_MOT_result.txt";
const char DATA_FILE_PATH[] = "sample_MOT_data.txt";
const char DATA_FEATURE_DIR[] = "MOT_data_feature";

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
        "          <sample imagelist path>\n"
        "          <inference count>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

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
  char *imagelist_file_path = argv[9];
  int inference_count = atoi(argv[10]);

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;

  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(od_model_name, &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);

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
  if (CVI_SUCCESS != GET_PREDEFINED_CONFIG(&ds_conf, target_type)) {
    printf("GET_PREDEFINED_CONFIG error!\n");
    return CVI_FAILURE;
  }
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, true);

#ifdef OUTPUT_MOT_RESULT
  FILE *outFile_result;
  outFile_result = fopen(RESULT_FILE_PATH, "w");
  if (outFile_result == NULL) {
    printf("There is a problem opening the output file: %s.\n", RESULT_FILE_PATH);
    return CVI_FAILURE;
  }
#endif

#ifdef OUTPUT_MOT_DATA
  FILE *outFile_data;
  outFile_data = fopen(DATA_FILE_PATH, "w");
  if (outFile_data == NULL) {
    printf("There is a problem opening the output file: %s.\n", DATA_FILE_PATH);
    return CVI_FAILURE;
  }
#endif

  char text_buf[256];
  FILE *inFile = fopen(imagelist_file_path, "r");
  fscanf(inFile, "%s", text_buf);
  int img_num = atoi(text_buf);
  printf("Images Num: %d\n", img_num);

#ifdef OUTPUT_MOT_RESULT
  fprintf(outFile_result, "%u\n", img_num);
#endif

#ifdef OUTPUT_MOT_DATA
  fprintf(outFile_data, "%u %d\n", img_num, (int)enable_DeepSORT);
#endif

  cvai_object_t obj_meta;
  cvai_face_t face_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&face_meta, 0, sizeof(cvai_face_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

  for (int counter = 0; counter < img_num; counter++) {
    if (counter == inference_count) {
      break;
    }
    fscanf(inFile, "%s", text_buf);
    printf("[%i] image path = %s\n", counter, text_buf);

    VIDEO_FRAME_INFO_S frame;
    CVI_AI_ReadImage(text_buf, &frame, PIXEL_FORMAT_RGB_888);

    switch (target_type) {
      case PERSON: {
        inference(ai_handle, &frame, &obj_meta);
        if (enable_DeepSORT) {
          CVI_AI_OSNet(ai_handle, &frame, &obj_meta);
        }
        CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, enable_DeepSORT);
      } break;
      case FACE: {
        CVI_AI_RetinaFace(ai_handle, &frame, &face_meta);
        if (enable_DeepSORT) {
          CVI_AI_FaceRecognition(ai_handle, &frame, &face_meta);
        }
        CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, enable_DeepSORT);
      } break;
      default:
        break;
    }

#ifdef OUTPUT_MOT_RESULT
    fprintf(outFile_result, "%u\n", tracker_meta.size);
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      cvai_bbox_t *target_bbox =
          (target_type == FACE) ? &face_meta.info[i].bbox : &obj_meta.info[i].bbox;
      uint64_t u_id =
          (target_type == FACE) ? face_meta.info[i].unique_id : obj_meta.info[i].unique_id;
      fprintf(outFile_result, "%" PRIu64 ",%d,%d,%d,%d,%d,%d,%d,%d,%d\n", u_id,
              (int)target_bbox->x1, (int)target_bbox->y1, (int)target_bbox->x2,
              (int)target_bbox->y2, tracker_meta.info[i].state, (int)tracker_meta.info[i].bbox.x1,
              (int)tracker_meta.info[i].bbox.y1, (int)tracker_meta.info[i].bbox.x2,
              (int)tracker_meta.info[i].bbox.y2);
    }
    cvai_tracker_t inact_trackers;
    memset(&inact_trackers, 0, sizeof(cvai_tracker_t));
    CVI_AI_DeepSORT_GetTracker_Inactive(ai_handle, &inact_trackers);
    fprintf(outFile_result, "%u\n", inact_trackers.size);
    for (uint32_t i = 0; i < inact_trackers.size; i++) {
      fprintf(outFile_result, "%" PRIu64 ",-1,-1,-1,-1,%d,%d,%d,%d,%d\n", inact_trackers.info[i].id,
              inact_trackers.info[i].state, (int)inact_trackers.info[i].bbox.x1,
              (int)inact_trackers.info[i].bbox.y1, (int)inact_trackers.info[i].bbox.x2,
              (int)inact_trackers.info[i].bbox.y2);
    }
    CVI_AI_Free(&inact_trackers);
#endif

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
      fprintf(outFile_data, "%u %d %s/feature_%d_%u.bin\n", target_feature->size,
              target_feature->type, DATA_FEATURE_DIR, counter, i);
    }
#endif

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
    CVI_AI_ReleaseImage(&frame);
  }

#ifdef OUTPUT_MOT_RESULT
  fclose(outFile_result);
#endif
#ifdef OUTPUT_MOT_DATA
  fclose(outFile_data);
#endif
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}
