// Copyright 2018 Bitmain Inc.
// License
// Author Lester Chen <lester.chen@bitmain.com>
/**
 * @file base_net.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once

#include "bmodel_runner.hpp"
#include "net_types.hpp"
#include "qnnctx.hpp"
#include "tensor.hpp"
#include "utils/log_common.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qnn {

typedef std::unordered_map<std::string, OutputTensor> OutTensors;
typedef std::vector<std::pair<NetShape, std::vector<NetShape>>> SupportedShapes;

/**
 * @brief The base class of the NN
 *
 */
class BaseNet {
    public:
    /**
     * @brief Construct a new BaseNet object
     *
     * @param model_path The path of the nn model.
     * @param supported_input_shapes Supported input shapes. Deprecated in bmtap2 BM1880_v2.
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager.
     */
    BaseNet(const std::string &model_path, const std::vector<NetShape> &supported_input_shapes,
            QNNCtx *qnn_ctx);

    /**
     * @brief Destroy the BaseNet object
     *
     */
    virtual ~BaseNet() {}

    // TODO: This might change if the interface in bmtap2 changed
    /**
     * @brief Get the input quantize threshold from the chosen input layer. A wrapper of
     * bmodel_runner.
     *
     * @return float Return the chosen input layer's quantize threshold
     */
    float GetInPutThreshold() { return model_runner->GetInPutThreshold(); }

    /**
     * @brief Get the ciorresponding output info from the chosen input layer. A wrapper of
     * bmodel_runner.
     *
     * @param outputInfo The output info
     * @return net_err_t Error status
     */
    net_err_t GetOutputInfo(bmnet_output_info_t *outputInfo) {
        return model_runner->GetOutputInfo(outputInfo);
    }

    void EnableDequantize(bool enabled) { is_dequantize_enabled = enabled; }

    friend class ImageNet;

    protected:
    /**
     * @brief Call bmtap2's inference with preallocated input and output buffers.
     *
     * @param shape The input tensor shape. Note that you'll have to set tensor first.
     * @return net_err_t Error status
     */
    net_err_t Inference(NetShape &shape) {
        return model_runner->Inference(shape, m_in_buffer_ptr, m_out_buffer_ptr);
    }

    /**
     * @brief Map the input buffer to vector of Mats
     *
     * @param channels The vector of Mats
     * @param input The input buffer address pointer
     * @param C Channel of the input shape
     * @param H Height of the input shape
     * @param W Width of the input shape
     * @return char* The input buffer address pointer with offset (C * H * W)
     */
    char *WrapInputLayer(std::vector<cv::Mat> &channels, char *input, int C, int H, int W);

    /**
     * @brief Select tensor with the given shape. Assert if fail.
     *
     * @param shape The input nchw shape
     * @return std::pair<InputTensor *, OutTensors *> The input and output tensors in std::pair
     */
    std::pair<InputTensor *, OutTensors *> SelectTensor(const NetShape &shape);

    // Note: TODO: This might change if the API of bmtap2 changed.
    /**
     * @brief Get all the possible output shapes with the input shapes array from constructor.
     *
     */
    void SolveSupportedShapes();

    /**
     * @brief Setup input output tensors using the supported shapes from SolveSupportedShapes.
     *
     */
    void AllocateSupportedTensors();

    /**
     * @brief Dequantize std::map of output tensors.
     *
     * @param out_tensors The output tensor.
     */
    void DequantizeOutputTensors(OutTensors &out_tensors);

    /**
     * @brief Dequantize output tensors with the threshold from bmodel. Original datum is saved in
     * q_data, while
     * dequantized datum is saved in data.
     *
     * @param out_tensors The output tensor.
     */
    void DequantizeTensor(OutputTensor &tensor);

    /**
     * @brief Get the maximum batch size from the supported shapes.
     *
     * @param total The total images left
     * @return int The batch value
     */
    int GetNumberPerBatch(int total);

    private:
    /**
     * @brief Allocate buffer if not set
     *
     */
    void MaybeAllocBuffer();

    /**
     * @brief Allocate tensor if not set
     *
     */
    void MaybeAllocTensor();

    /**
     * @brief Get the maximum buffer size from the supported shape lists.
     *
     * @param input_size The input tensor buffer size
     * @param output_size The output tnesor buffer size
     */
    void GetBufferSize(int &input_size, int &output_size);

    QNNCtx *m_qnn_ctx = nullptr;     /**< The QNN Context instance */
    CtxBufferInfo m_internal_buffer; /**< Buffer handler instance */

    /**
     * @defgroup basenetptrgroup The BaseNet pointers
     * These ptrs points to QNNCtx if available. By default the internal buffer will be used.
     * @{
     */
    char *m_in_buffer_ptr = nullptr;  /**< The input buffer pointer */
    char *m_out_buffer_ptr = nullptr; /**< This output buffer pointer */
    /** @} */                         // End of basenetptrgroup

    std::shared_ptr<ModelRunner> model_runner; /**< The bmodel container */

    int max_batch_num; /**< The maximum supported batch num from input shape lists. */
    std::vector<NetShape> supported_input_shapes;     /**< The supported input shapes */
    SupportedShapes supported_shapes;                 /**< The supported shapes in std::pair */
    std::vector<InputTensor> supported_input_tensors; /**< The supported input tensors */
    std::vector<OutTensors> supported_output_tensors; /**< The supported output tensors */
    std::vector<int> supported_batches;               /**< The supported batches */
    bool is_dequantize_enabled = true;
};

}  // namespace qnn
