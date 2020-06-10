// Copyright 2018 Bitmain Inc.
// License
// Author Lester Chen <lester.chen@bitmain.com>
#include "base_net.hpp"
#include "net_loader.hpp"
#include "utils/function_tracer.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define MAX_BUF_SIZE 0x1000000

namespace qnn {

using cv::Mat;
using std::max;
using std::min;
using std::pair;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

BaseNet::BaseNet(const string &model_path, const vector<NetShape> &supported_input_shapes,
                 QNNCtx *qnn_ctx)
    : max_batch_num(0) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    model_runner = NetLoader::Get().Load(model_path);
    std::string model_name = "";
#if USE_LEGACY_BMTAP2 == 0
    LOGI << "Get input size info from model...";
    const ModelInfo *model_info = model_runner->GetModelInfo();
    bmnet_input_info_t *input_info = model_info->input_info_array;
    for (size_t i = 0; i < model_info->command_num; i++) {
        for (size_t j = 0; j < input_info[i].input_num; j++) {
            NetShape shape(input_info[i].shape_array[j].n, input_info[i].shape_array[j].c,
                           input_info[i].shape_array[j].h, input_info[i].shape_array[j].w);
            this->supported_input_shapes.push_back(shape);
            LOGI << shape;
        }
    }
    model_name = model_info->net_name;
#else
    LOGW << "This SDK is built with legacy bmtap2.";
    this->supported_input_shapes = supported_input_shapes;
#endif
    SolveSupportedShapes();
    if (qnn_ctx) {
        LOGD << "Ctx found";
        m_qnn_ctx = qnn_ctx;
        int input_size = 0;
        int output_size = 0;
        GetBufferSize(input_size, output_size);
        int idx = m_qnn_ctx->Register(model_name, input_size, output_size);
        LOGI << "Current QNNCtx global idx: " << idx;
    } else {
        MaybeAllocBuffer();
    }
}

char *BaseNet::WrapInputLayer(vector<Mat> &channels, char *input, int C, int H, int W) {
#if defined(NPU_INT8)
    unsigned char *input_data = reinterpret_cast<unsigned char *>(input);
    for (int i = 0; i < C; ++i) {
        cv::Mat channel(H, W, CV_8SC1, input_data);
        channels.emplace_back(channel);
        input_data += W * H;
    }
    return (char *)input_data;
#elif defined(NPU_FLOAT32)
    float *input_data = reinterpret_cast<float *>(input);

    for (int i = 0; i < C; ++i) {
        cv::Mat channel(H, W, CV_32FC1, input_data);
        channels.emplace_back(channel);
        input_data += W * H;
    }
    return (char *)input_data;
#endif
}

void BaseNet::SolveSupportedShapes() {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    supported_shapes.reserve(supported_input_shapes.size());

    bmnet_output_info_t output_info;
    vector<NetShape> output_shapes;
    for (size_t i = 0; i < supported_input_shapes.size(); i++) {
        model_runner->SetInputShape(supported_input_shapes[i]);

        GetOutputInfo(&output_info);

        output_shapes.clear();
        for (size_t j = 0; j < output_info.output_num; j++) {
            NetShape _shape(output_info.shape_array[j].n, output_info.shape_array[j].c,
                            output_info.shape_array[j].h, output_info.shape_array[j].w);
            output_shapes.emplace_back(_shape);
        }
        supported_shapes.emplace_back(std::make_pair(supported_input_shapes[i], output_shapes));
    }
}

void BaseNet::AllocateSupportedTensors() {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    supported_input_tensors.reserve(supported_shapes.size());
    supported_output_tensors.reserve(supported_shapes.size());
    supported_batches.reserve(supported_shapes.size());

    bmnet_output_info_t output_info;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
        model_runner->SetInputShape(supported_shapes[i].first);

        GetOutputInfo(&output_info);

        float *dequantized_buffer_ptr = nullptr;
        if (m_qnn_ctx) {
            auto aptr_tuple = m_qnn_ctx->Request();
            m_in_buffer_ptr = std::get<0>(aptr_tuple);
            m_out_buffer_ptr = std::get<1>(aptr_tuple);
            dequantized_buffer_ptr = std::get<2>(aptr_tuple);
        } else {
            m_in_buffer_ptr = m_internal_buffer.GetInputPtr();
            m_out_buffer_ptr = m_internal_buffer.GetOutputPtr();
            dequantized_buffer_ptr = m_internal_buffer.GetDequantizePtr();
        }
        char *in_buffer_ptr = m_in_buffer_ptr;
        char *out_buffer_ptr = m_out_buffer_ptr;

        // Allocate input tensor
        InputTensor input_tensor(supported_shapes[i].first);
        input_tensor.data = in_buffer_ptr;

        // Allocate output tensors
        OutTensors output_tensors;
        for (size_t j = 0; j < output_info.output_num; j++) {
            OutputTensor tensor(supported_shapes[i].second[j]);
            tensor.name = output_info.name_array[j];
            tensor.quantize_threshold = output_info.threshold_array[j];
            tensor.q_data = out_buffer_ptr;
            tensor.data = dequantized_buffer_ptr;
            out_buffer_ptr += tensor.count;
            dequantized_buffer_ptr += tensor.count;

            output_tensors[tensor.name] = tensor;
        }
        supported_input_tensors.emplace_back(input_tensor);
        supported_output_tensors.emplace_back(output_tensors);
        supported_batches.emplace_back(supported_shapes[i].first.n);
    }
    assert(supported_batches.size() > 0);
    std::sort(supported_batches.begin(), supported_batches.end(), std::greater<int>());
    max_batch_num = supported_batches.front();
}

int BaseNet::GetNumberPerBatch(int total) {
    assert(total >= 0);
    if (total == 0) {
        return 0;
    }
    MaybeAllocTensor();
    assert(supported_batches.size() > 0);
    vector<int>::iterator it = supported_batches.begin();
    while (total < *it) {
        ++it;
    }
    assert(*it > 0 && *it <= max_batch_num);
    return *it;
}

pair<InputTensor *, OutTensors *> BaseNet::SelectTensor(const NetShape &shape) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);

    MaybeAllocTensor();
    InputTensor *input = NULL;
    OutTensors *output = NULL;

    for (size_t i = 0; i < supported_input_tensors.size(); ++i) {
        if (supported_input_tensors[i].shape == shape) {
            input = &supported_input_tensors[i];
            output = &supported_output_tensors[i];
            break;
        }
    }
    // doesn't support the shape
    assert(input != NULL);
    assert(output != NULL);
    return make_pair(input, output);
}

void BaseNet::MaybeAllocBuffer() {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (m_internal_buffer.IsInitialized()) {
        return;
    }

    int input_size = 0;
    int output_size = 0;
    GetBufferSize(input_size, output_size);
    m_internal_buffer.SetInputSize(input_size);
    m_internal_buffer.SetOutputSize(output_size);
    m_internal_buffer.Init();
}

// FIXME: Remove someday?
void BaseNet::MaybeAllocTensor() {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    assert(supported_input_tensors.size() == supported_output_tensors.size());
    if (supported_input_tensors.size() != 0) {
        return;
    }

    // SolveSupportedShapes(); This is moved to the constructor.
    // MaybeAllocBuffer(); This is moved to the constructor
    AllocateSupportedTensors();
}

void BaseNet::DequantizeTensor(OutputTensor &tensor) {
    if (tensor.count == 0) {
        return;
    }

    // Don't put if inside for loop if the condition does not changed. (performance issue.)
    if (is_dequantize_enabled) {
        for (int i = 0; i < tensor.count; i++) {
            tensor.data[i] = float(tensor.q_data[i]) * tensor.quantize_threshold / 128.0;
        }
    } else {
        for (int i = 0; i < tensor.count; i++) {
            tensor.data[i] = tensor.q_data[i];
        }
    }
}

void BaseNet::DequantizeOutputTensors(OutTensors &out_tensors) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (out_tensors.size() == 0) {
        return;
    }

    for (auto it = out_tensors.begin(); it != out_tensors.end(); it++) {
        DequantizeTensor(it->second);
    }
}

void BaseNet::GetBufferSize(int &input_size, int &output_size) {
    input_size = 0;
    output_size = 0;
    for (auto shapes : supported_shapes) {
        // input tensor
        input_size = max(input_size, NetTensor(shapes.first).size);

        // output tensors
        int sum = 0;
        for (NetShape &shape : shapes.second) {
            sum += NetTensor(shape).size;
        }
        output_size = max(output_size, sum);
    }
    assert(input_size > 0 && output_size > 0 && input_size < MAX_BUF_SIZE &&
           output_size < MAX_BUF_SIZE);
}

}  // namespace qnn
