
#include "bmodel_runner.hpp"
#include "bmtap2.h"
#include "utils/function_tracer.h"
#include "utils/log_common.h"
#include <cassert>

namespace qnn {

using namespace std;

BModelRunner::BModelRunner(bmctx_t handle, const string &model)
    : ModelRunner(), ctxt_handle(handle), net_handle(nullptr) {
    bmerr_t err = bmnet_register_bmodel(ctxt_handle, model.c_str(), &net_handle);
    LOGD << "bmnet_register_bmodel ret=" << err;
    if (BM_SUCCESS != err) {
        std::exit(EXIT_FAILURE);
    }
}

BModelRunner::~BModelRunner() {
    assert(net_handle);
    bmnet_cleanup(net_handle);
}

net_err_t BModelRunner::SetInputShape(const NetShape &shape) {
    assert(net_handle);
    shape_t input_shape = shape_t4(shape.n, shape.c, shape.h, shape.w);
    bmerr_t ret = bmnet_set_input_shape(net_handle, input_shape);
    if (ret != BM_SUCCESS) return RET_UNSUPPORTED_SHAPE;
    return RET_SUCCESS;
}

net_err_t BModelRunner::Inference(NetShape &shape, char *input, char *output) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    assert(net_handle);
    net_err_t err = SetInputShape(shape);
    if (err != RET_SUCCESS) {
        assert(false);
        return err;
    }

    bmerr_t ret = bmnet_inference(net_handle, (uint8_t *)input, (uint8_t *)output);
    assert(ret == BM_SUCCESS);
    if (ret != BM_SUCCESS) return RET_INFERENCE_ERROR;

    return RET_SUCCESS;
}
// Currently the BSP release has the different name of bmmet_get_input_threshold
// instead of bmnet_get_input_threshold, temperally use compile flag to WA it,
// once the BSP release name change in the future, need to refine it
float BModelRunner::GetInPutThreshold() {
    assert(net_handle);
#if USE_LEGACY_BMTAP2 == 0
    bmnet_input_info_t *input_info = bmnet_get_input_info(net_handle);
    if (input_info == nullptr) {
        std::cerr << "Input info is null. Aborting...";
        std::exit(EXIT_FAILURE);
    }
    return input_info->threshold_array[0];
#else
#    if defined(USE_BSPSDK)
    return bmmet_get_input_threshold(net_handle);
#    else
    return bmnet_get_input_threshold(net_handle);
#    endif
#endif
}

net_err_t BModelRunner::GetOutputInfo(bmnet_output_info_t *outputInfo) {
    assert(net_handle);
    bmerr_t ret = bmnet_get_output_info(net_handle, outputInfo);
    assert(ret == BM_SUCCESS);
    if (ret != BM_SUCCESS) return RET_GET_OUTPUT_INFO_ERROR;
    return RET_SUCCESS;
}

const ModelInfo *BModelRunner::GetModelInfo() {
#if USE_LEGACY_BMTAP2 == 1
    std::cerr << "Legacy bmtap2 does not support getting bmodel information from file.";
    const ModelInfo *model_info = nullptr;
#else
    const ModelInfo *model_info = bmnet_get_model_info(net_handle);
    if (model_info == nullptr) {
        std::cerr << "Model info is null. Aborting...";
        std::exit(EXIT_FAILURE);
    }
#endif
    return model_info;
}

}  // namespace qnn
