
#include <cvi_comm_vpss.h>
#include <pthread.h>
#include <signal.h>
#include "cvi_audio.h"
#include "speech_recognition/zipformer_encoder.hpp"
#include "tdl_model_factory.hpp"

#define AUDIOFORMATSIZE 2
#define SECOND 1
#define CVI_AUDIO_BLOCK_MODE -1
#define PERIOD_SIZE 640
#define SAMPLE_RATE 16000
#define FRAME_SIZE \
  SAMPLE_RATE *AUDIOFORMATSIZE *SECOND  // PCM_FORMAT_S16_LE (2bytes) 3 seconds

bool gRun = true;  // signal

typedef struct {
  std::shared_ptr<ZipformerEncoder> zipformer;
} RUN_TDL_THREAD_ARG_S;

typedef struct {
  uint8_t buf[2][FRAME_SIZE];
  bool ready[2];  // 标记该槽是否有新数据
  int write_idx;  // 0/1
  uint64_t seq[2];
  pthread_mutex_t mtx;
  pthread_cond_t cv;
} PingPong;

static PingPong pp;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    gRun = false;
  }
}

void *capture(void *_) {
  uint64_t seq = 0;
  CVI_S32 s32Ret;
  AUDIO_FRAME_S stFrame;
  AEC_FRAME_S stAecFrm;
  int loop = SAMPLE_RATE / PERIOD_SIZE * SECOND;  // 3 seconds
  int size = PERIOD_SIZE * AUDIOFORMATSIZE;       // PCM_FORMAT_S16_LE (2bytes)

  // Set video frame interface
  uint8_t buffer[FRAME_SIZE];  // 1 seconds
  memset(buffer, 0, FRAME_SIZE);

  while (gRun) {
    for (int i = 0; i < loop; ++i) {
      s32Ret = CVI_AI_GetFrame(0, 0, &stFrame, &stAecFrm,
                               CVI_AUDIO_BLOCK_MODE);  // Get audio frame
      if (s32Ret != 0) {
        printf("CVI_AI_GetFrame --> none!!\n");
        continue;
      } else {
        memcpy(buffer + i * size, (uint8_t *)stFrame.u64VirAddr[0],
               size);  // Set the period size date to global buffer
      }
    }

    pthread_mutex_lock(&pp.mtx);
    int idx = pp.write_idx;
    memcpy(pp.buf[idx], buffer, FRAME_SIZE);
    pp.seq[idx] = seq++;
    pp.ready[idx] = true;
    pp.write_idx = 1 - idx;  // 翻转
    pthread_cond_signal(&pp.cv);
    pthread_mutex_unlock(&pp.mtx);
  }
  s32Ret = CVI_AI_ReleaseFrame(0, 0, &stFrame, &stAecFrm);
  if (s32Ret != 0) printf("CVI_AI_ReleaseFrame Failed!!\n");
  return NULL;
}

void *infer(void *args) {
  RUN_TDL_THREAD_ARG_S *pstArgs = (RUN_TDL_THREAD_ARG_S *)args;

  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      SAMPLE_RATE * 2, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  uint8_t *data_buffer = bin_data->getVirtualAddress()[0];

  std::shared_ptr<ModelASRInfo> asr_meta = std::make_shared<ModelASRInfo>();
  std::shared_ptr<ModelOutputInfo> output_data =
      std::static_pointer_cast<ModelOutputInfo>(asr_meta);

  while (gRun) {
    pthread_mutex_lock(&pp.mtx);
    // 等待任一槽 ready
    while (!pp.ready[0] && !pp.ready[1]) {
      pthread_cond_wait(&pp.cv, &pp.mtx);
    }
    if (!gRun) {
      pthread_mutex_unlock(&pp.mtx);
      break;
    }
    if (!gRun && !pp.ready[0] && !pp.ready[1]) {
      pthread_mutex_unlock(&pp.mtx);
      break;
    }
    int i = pp.ready[0] ? 0 : 1;

    memcpy(data_buffer, pp.buf[i], FRAME_SIZE);

    pp.ready[i] = false;
    pthread_mutex_unlock(&pp.mtx);

    pstArgs->zipformer->inference(bin_data, output_data);

    if (asr_meta->text_info) {
      printf("%s", asr_meta->text_info);
      fflush(stdout);
    }
  }

  return NULL;
}

CVI_S32 SET_AUDIO_ATTR(CVI_VOID) {
  // STEP 1: cvitek_audin_set
  //_update_audin_config
  AIO_ATTR_S AudinAttr;
  AudinAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)SAMPLE_RATE;
  AudinAttr.u32ChnCnt = 1;
  AudinAttr.enSoundmode = AUDIO_SOUND_MODE_MONO;
  AudinAttr.enBitwidth = AUDIO_BIT_WIDTH_16;
  AudinAttr.enWorkmode = AIO_MODE_I2S_MASTER;
  AudinAttr.u32EXFlag = 0;
  AudinAttr.u32FrmNum = 10;                /* only use in bind mode */
  AudinAttr.u32PtNumPerFrm = PERIOD_SIZE;  // sample rate / fps
  AudinAttr.u32ClkSel = 0;
  AudinAttr.enI2sType = AIO_I2STYPE_INNERCODEC;
  CVI_S32 s32Ret;
  // STEP 2:cvitek_audin_uplink_start
  //_set_audin_config
  s32Ret = CVI_AI_SetPubAttr(0, &AudinAttr);
  if (s32Ret != 0) printf("CVI_AI_SetPubAttr failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_Enable(0);
  if (s32Ret != 0) printf("CVI_AI_Enable failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_EnableChn(0, 0);
  if (s32Ret != 0) printf("CVI_AI_EnableChn failed with %#x!\n", s32Ret);

  s32Ret = CVI_AI_SetVolume(0, 12);
  if (s32Ret != 0) printf("CVI_AI_SetVolume failed with %#x!\n", s32Ret);

  printf("SET_AUDIO_ATTR success!!\n");
  return 0;
}

// 改进点1：增加条件变量初始化（在main函数中）
int main(int argc, char **argv) {
  memset(&pp, 0, sizeof(pp));
  pthread_mutex_init(&pp.mtx, NULL);  // 新增
  pthread_cond_init(&pp.cv, NULL);    // 新增

  if (argc != 3) {
    printf("Usage: %s <model_dir> <tokens_path>\n", argv[0]);
    return -1;
  }

  std::string model_dir = argv[1];
  std::string tokens_path = argv[2];

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_zipformer_encoder =
      model_factory.getModel("RECOGNITION_SPEECH_ZIPFORMER_ENCODER");
  if (!model_zipformer_encoder) {
    printf("Failed to create model_zipformer_encoder\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_zipformer_decoder =
      model_factory.getModel("RECOGNITION_SPEECH_ZIPFORMER_DECODER");
  if (!model_zipformer_decoder) {
    printf("Failed to create model_zipformer_decoder\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_zipformer_joiner =
      model_factory.getModel("RECOGNITION_SPEECH_ZIPFORMER_JOINER");
  if (!model_zipformer_joiner) {
    printf("Failed to create model_zipformer_joiner\n");
    return -1;
  }

  std::shared_ptr<ZipformerEncoder> zipformer =
      std::dynamic_pointer_cast<ZipformerEncoder>(model_zipformer_encoder);

  zipformer->setTokensPath(tokens_path);
  zipformer->setModel(model_zipformer_decoder, model_zipformer_joiner);

  RUN_TDL_THREAD_ARG_S tdl_args = {.zipformer = zipformer};

  if (CVI_AUDIO_INIT() == 0) {
    printf("CVI_AUDIO_INIT success!!\n");
  } else {
    printf("CVI_AUDIO_INIT failure!!\n");
    return 0;
  }
  SET_AUDIO_ATTR();

  printf("Initialization completed, please start speaking:\n");

  pthread_t stFrameThread, stTDLThread;
  pthread_create(&stFrameThread, NULL, capture, NULL);
  pthread_create(&stTDLThread, NULL, infer, &tdl_args);

  pthread_join(stFrameThread, NULL);
  pthread_join(stTDLThread, NULL);

  return 0;
}
