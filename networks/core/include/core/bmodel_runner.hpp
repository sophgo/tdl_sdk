/**
 * @file bmodel_runner.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once

#include "bmtap2.h"
#include "model_runner.hpp"
#include <string>

namespace qnn {

/**
 * @brief The bmodel container
 *
 */
class BModelRunner : public ModelRunner {
    public:
    /**
     * @brief Construct a new BModelRunner object.
     *
     * @param handle The bm context handler
     * @param model The model path
     */
    BModelRunner(bmctx_t handle, const std::string &model);

    /**
     * @brief Destroy the BModelRunner object.
     *
     */
    virtual ~BModelRunner();

    /**
     * @brief Set the input shape to bmodel handler.
     *
     * @param shape The input shape
     * @return net_err_t Error status
     */
    virtual net_err_t SetInputShape(const NetShape &shape) override;

    /**
     * @brief Send the cmdbuf to bmkernel with bmtap2's API.
     *
     * @param shape The input shape
     * @param input The input buffer
     * @param output The output buffer
     * @return net_err_t Error status
     */
    virtual net_err_t Inference(NetShape &shape, char *input, char *output) override;

    /**
     * @brief Get the input threshold from selected input shape.
     *
     * @return float The input quantize threshold
     */
    virtual float GetInPutThreshold() override;

    /**
     * @brief Get the corresponding output info from the chosen input layer.
     *
     * @param outputInfo The output info
     * @return net_err_t Error status
     */
    virtual net_err_t GetOutputInfo(bmnet_output_info_t *outputInfo) override;

    /**
     * @brief Get the model info from the bmodel. Only works when build with bmtap2_v2.
     *
     * @return ModelInfo bmodel info
     */
    const ModelInfo *GetModelInfo() override;

    private:
    bmctx_t ctxt_handle; /**< The bm context handler */
    bmnet_t net_handle;  /**< The bmnet context handler */
};

}  // namespace qnn
