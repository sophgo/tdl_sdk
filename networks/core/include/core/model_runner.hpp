
/**
 * @file model_runner.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once

#include "bmtap2.h"
#include "net_types.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <cstring>

struct bmnet_model_info_s;
typedef struct bmnet_model_info_s ModelInfo;

namespace qnn {

typedef NetShape InputShape;
typedef NetShape OutputShape;

/**
 * @brief Model runner base class
 *
 */
class ModelRunner {
    public:
    /**
     * @brief Destroy the ModelRunner object.
     *
     */
    virtual ~ModelRunner() {}

    /**
     * @brief Set the input shape object. (virtual function)
     *
     * @param shape Input shape
     * @return net_err_t Error status
     */
    virtual net_err_t SetInputShape(const NetShape &shape) = 0;

    /**
     * @brief Send the cmdbuf to bmkernel with bmtap2's API. (virtual function)
     *
     * @param shape The input shape
     * @param input The input buffer
     * @param output The output buffer
     * @return net_err_t Error status
     */
    virtual net_err_t Inference(NetShape &shape, char *input, char *output) = 0;

    /**
     * @brief Get the input threshold from selected input shape. (virtual function)
     *
     * @return float The input quantize threshold
     */
    virtual float GetInPutThreshold() = 0;

    /**
     * @brief Get the ciorresponding output info from the chosen input layer. (virtual function)
     *
     * @param outputInfo The output info
     * @return net_err_t Error status
     */
    virtual net_err_t GetOutputInfo(bmnet_output_info_t *outputInfo) = 0;

    /**
     * @brief Get the model info from bmodel file. (virtual function)
     *
     * @return ModelInfo The model info
     */
    virtual const ModelInfo *GetModelInfo() = 0;
};

}  // namespace qnn
