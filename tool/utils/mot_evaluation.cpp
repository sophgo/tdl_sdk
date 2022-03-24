#include "mot_evaluation.hpp"
#include <cmath>

#define DEFAULT_DATA_INFO_NAME "MOT_data_info.txt"

MOT_Evaluation::MOT_Evaluation() {}

MOT_Evaluation::~MOT_Evaluation() {}

CVI_S32 MOT_Evaluation::update(cvai_tracker_t &trackers, cvai_tracker_t &inact_trackers) {
  time_counter += 1;
  bbox_counter += trackers.size;
  std::set<uint64_t> check_list = alive_stable_id;
  for (uint32_t i = 0; i < trackers.size; i++) {
    if (trackers.info[i].state == CVI_TRACKER_NEW) {
      new_id_counter += 1;
      continue;
    }
    if (trackers.info[i].state == CVI_TRACKER_UNSTABLE) {
      continue;
    }
    stable_tracking_counter += 1;
    uint64_t t_id = trackers.info[i].id;
    if (alive_stable_id.find(t_id) == alive_stable_id.end()) {
      alive_stable_id.insert(t_id);
      stable_id.insert(t_id);
      tracking_performance[t_id] = std::pair<uint32_t, uint32_t>(1, time_counter);
    } else {
      tracking_performance[t_id].first += 1;
      check_list.erase(t_id);
    }
  }
  std::set<uint64_t> inact_id;
  for (uint32_t i = 0; i < inact_trackers.size; i++) {
    inact_id.insert(inact_trackers.info[i].id);
  }
  for (std::set<uint64_t>::iterator it = check_list.begin(); it != check_list.end(); it++) {
    if (inact_id.find(*it) == inact_id.end()) {
      alive_stable_id.erase(*it);
      tracking_performance[*it].second = time_counter - tracking_performance[*it].second;
      entropy += -log2((double)tracking_performance[*it].first / tracking_performance[*it].second);
    }
  }

  return CVI_SUCCESS;
}

void MOT_Evaluation::summary(MOT_Performance_t &performance) {
  double total_entropy = entropy;
  for (const auto &id : alive_stable_id) {
    total_entropy += -log2((double)tracking_performance[id].first /
                           (time_counter - tracking_performance[id].second + 1));
  }
  double coverage_rate = (double)stable_tracking_counter / (double)bbox_counter;

  performance.stable_id_num = (uint32_t)stable_id.size();
  performance.total_entropy = total_entropy;
  performance.coverage_rate = coverage_rate;
  performance.score = (1. - coverage_rate) * total_entropy;

#if 0
  printf("\n@@@ MOT Evaluation Summary @@@\n");
  printf("Stable ID Num : %u\n", (uint32_t)stable_id.size());
  printf("Total Entropy : %.4lf\n", total_entropy);
  printf("Coverage Rate : %.4lf ( %u / %u )\n", coverage_rate, stable_tracking_counter,
         bbox_counter);
  printf("Score : %.4lf\n", performance.score);
#endif
}

GridIndexGenerator::GridIndexGenerator(const std::vector<uint32_t> &r) {
  ranges = r;
  idx.resize(r.size());
  std::fill(idx.begin(), idx.end(), 0);
}

GridIndexGenerator::~GridIndexGenerator() {}

CVI_S32 GridIndexGenerator::next(std::vector<uint32_t> &next_idx) {
  if (counter == 0) {
    next_idx = idx;
    counter += 1;
    return CVI_SUCCESS;
  }
  bool carry = true;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (idx[i] + 1 == ranges[i]) {
      idx[i] = 0;
    } else {
      idx[i] += 1;
      carry = false;
    }
    if (!carry) {
      next_idx = idx;
      counter += 1;
      return CVI_SUCCESS;
    }
  }
  return CVI_FAILURE;
}

CVI_S32 RUN_MOT_EVALUATION(cviai_handle_t ai_handle, const MOT_EVALUATION_ARGS_t &args,
                           MOT_Performance_t &performance) {
  char text_buffer[256];
  char text_buffer_tmp[256];
  char inFile_data_path[256];
  sprintf(inFile_data_path, "%s/%s", args.mot_data_path, DEFAULT_DATA_INFO_NAME);
  FILE *inFile_data = fopen(inFile_data_path, "r");
  if (inFile_data == NULL) {
    printf("fail to open file: %s\n", inFile_data_path);
    return CVI_FAILURE;
  }
  fscanf(inFile_data, "%s %s", text_buffer, text_buffer_tmp);
  int frame_num = atoi(text_buffer);
#if 0
  bool output_features = atoi(text_buffer_tmp) == 1;
  printf("Frame Num: %d ( Output Features: %s )\n", frame_num, output_features ? "true" :
         "false");
#endif

  cvai_object_t obj_meta;
  cvai_face_t face_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&face_meta, 0, sizeof(cvai_face_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

  MOT_Evaluation mot_eval_data;

  for (int counter = 0; counter < frame_num; counter++) {
    fscanf(inFile_data, "%s", text_buffer);
    uint32_t bbox_num = (uint32_t)atoi(text_buffer);

    switch (args.target_type) {
      case PERSON: {
        obj_meta.size = bbox_num;
        obj_meta.info = (cvai_object_info_t *)malloc(obj_meta.size * sizeof(cvai_object_info_t));
        for (uint32_t i = 0; i < bbox_num; i++) {
          char text_buffer_bbox[4][32];
          /* read bbox info */
          fscanf(inFile_data, "%s %s %s %s", text_buffer_bbox[0], text_buffer_bbox[1],
                 text_buffer_bbox[2], text_buffer_bbox[3]);
          obj_meta.info[i].bbox.x1 = atof(text_buffer_bbox[0]);
          obj_meta.info[i].bbox.y1 = atof(text_buffer_bbox[1]);
          obj_meta.info[i].bbox.x2 = atof(text_buffer_bbox[2]);
          obj_meta.info[i].bbox.y2 = atof(text_buffer_bbox[3]);
          /* read feature info (ignore now) */
        }
        CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, args.enable_DeepSORT);
      } break;
      case FACE: {
        face_meta.rescale_type = RESCALE_CENTER;
        face_meta.size = bbox_num;
        face_meta.info = (cvai_face_info_t *)malloc(face_meta.size * sizeof(cvai_face_info_t));
        memset(face_meta.info, 0, face_meta.size * sizeof(cvai_face_info_t));
        for (uint32_t i = 0; i < bbox_num; i++) {
          char text_buffer_bbox[4][32];
          /* read bbox info */
          fscanf(inFile_data, "%s %s %s %s", text_buffer_bbox[0], text_buffer_bbox[1],
                 text_buffer_bbox[2], text_buffer_bbox[3]);
          face_meta.info[i].bbox.x1 = atof(text_buffer_bbox[0]);
          face_meta.info[i].bbox.y1 = atof(text_buffer_bbox[1]);
          face_meta.info[i].bbox.x2 = atof(text_buffer_bbox[2]);
          face_meta.info[i].bbox.y2 = atof(text_buffer_bbox[3]);
#if 0
          printf("face[%u] bbox: x1[%.2f], y1[%.2f], x2[%.2f], y2[%.2f]\n", i,
                 face_meta.info[i].bbox.x1, face_meta.info[i].bbox.y1, face_meta.info[i].bbox.x2,
                 face_meta.info[i].bbox.y2);
#endif
          /* read feature info (ignore now) */
          char text_buffer_feature[3][64];  // size, type, bin data path
          fscanf(inFile_data, "%s %s %s", text_buffer_feature[0], text_buffer_feature[1],
                 text_buffer_feature[2]);
        }
        CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, args.enable_DeepSORT);
      } break;
      default:
        break;
    }

    cvai_tracker_t inact_trackers;
    memset(&inact_trackers, 0, sizeof(cvai_tracker_t));
    CVI_AI_DeepSORT_GetTracker_Inactive(ai_handle, &inact_trackers);
    mot_eval_data.update(tracker_meta, inact_trackers);

    CVI_AI_Free(&inact_trackers);

    switch (args.target_type) {
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

  mot_eval_data.summary(performance);

  fclose(inFile_data);

  return CVI_SUCCESS;
}