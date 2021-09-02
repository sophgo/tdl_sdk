#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <audio_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;
  std::string model_dir = std::string(argv[1]);
  std::string audio_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  std::string model_name = std::string(m_json_read["reg_config"][0]["model_name"]);
  std::string model_path = model_dir + "/" + model_name;
  int audio_num = int(m_json_read["reg_config"][0]["audio_num"]);

  // Init VB pool size.
  cviai_handle_t facelib_handle = NULL;
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle2(&facelib_handle, 1, 0);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION,
                             model_path.c_str());

  if (ret != CVIAI_SUCCESS) {
    printf("Set face quality model failed with %#x!\n", ret);
    return ret;
  }

  bool pass = true;
  for (int audio_idx = 0; audio_idx < audio_num; audio_idx++) {
    std::string audio_path =
        audio_dir + "/" + std::string(m_json_read["reg_config"][0]["test_audios"][audio_idx]);
    int expected_res = int(m_json_read["reg_config"][0]["expected_results"][audio_idx]);

    FILE *fp = fopen(audio_path.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    int size = (int)ftell(fp) * sizeof(char);
    CVI_U8 *temp = (CVI_U8 *)malloc(size);
    fseek(fp, 0, SEEK_SET);
    fread(temp, 1, size, fp);
    fclose(fp);
    VIDEO_FRAME_INFO_S frame;
    frame.stVFrame.pu8VirAddr[0] = temp;
    frame.stVFrame.u32Height = 1;
    frame.stVFrame.u32Width = size;
    int index = -1;
    CVI_AI_SoundClassification(facelib_handle, &frame, &index);

    pass &= (index == expected_res);
    free(temp);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
  return pass ? CVIAI_SUCCESS : CVIAI_FAILURE;
}
