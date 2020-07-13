#include <unistd.h>
#include <memory>
#include <string>
#include <vector>

#include "cviai.h"

#include "yolov3/yolov3.hpp"

using namespace std;
using namespace cviai;

typedef struct {
  Yolov3 *yolov3;
} cviai_context_t;

int CVI_AI_InitHandle(const cviai_config_t *config, cviai_handle_t *handle) {
  cviai_context_t *context = new cviai_context_t;

  if (config->model_yolo3 != NULL) {
    context->yolov3 = new Yolov3();
    context->yolov3->modelOpen(config->model_yolo3);
    printf("init_network_yolov3 done\n");
  }

  *handle = context;
  return 0;
}

int CVI_AI_ObjDetect(cviai_handle_t handle, VIDEO_FRAME_INFO_S *stObjDetFrame, cvi_object_t *obj,
                     cvi_obj_det_type_t det_type) {
  cviai_context_t *fctx = static_cast<cviai_context_t *>(handle);

  return fctx->yolov3->inference(stObjDetFrame, obj, det_type);
}
