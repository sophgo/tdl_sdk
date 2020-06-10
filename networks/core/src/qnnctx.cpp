// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
#include "qnnctx.hpp"

#include "utils/log_common.h"
#include <iostream>
#define QNN_16_ALIGN 16
#define QNN_ALIGN QNN_16_ALIGN

namespace qnn {

template <class T>
unique_aptr<T> allocate_aligned(int alignment, int length, bool aligned) {
    T *ptr = 0;  // Make sure no one gets garbage if allocation fails.
    // Default uses aligned memory, but has option to fallback to malloc if aligned memory fails.
    if (aligned) {
        int error = posix_memalign((void **)&ptr, alignment, sizeof(T) * length);
        if (error != 0) {
            LOGE << "Allocate buffer size  = " << length << " * (" << typeid(*ptr).name()
                 << ") bytes failed.";
            exit(-1);
        }
    } else {
        ptr = (T *)malloc(sizeof(T) * length);
        if (ptr == NULL) {
            LOGE << "Allocate buffer size  = " << length << " * (" << typeid(*ptr).name()
                 << ") bytes failed.";
            exit(-1);
        }
    }
    return unique_aptr<T>{ptr};
}

template unique_aptr<char> allocate_aligned(int, int, bool);
template unique_aptr<int> allocate_aligned(int, int, bool);
template unique_aptr<float> allocate_aligned(int, int, bool);
template unique_aptr<double> allocate_aligned(int, int, bool);

void CtxBufferInfo::Init() {
    int padded_multiplier = std::ceil((float)m_max_in_size / QNN_ALIGN);
    m_max_in_size = (padded_multiplier * QNN_ALIGN);  // Updated to padded size
    int &&char_buffer_length = m_max_in_size + m_max_out_size;
    // Make sure the allocated memories are aligned.
    m_char_buffer = allocate_aligned<char>(QNN_ALIGN, char_buffer_length);
    m_float_buffer = allocate_aligned<float>(QNN_ALIGN, m_max_out_size);
    m_is_initialized = true;
}

const int CtxBufferInfo::GetIdInc() { return m_idx++; }

int QNNCtx::Register(const std::string model_name, const int max_input_size,
                     const int max_output_size) {
    if (m_info.IsInitialized()) {
        std::cerr << "Error! Please register any model before starts running!";
        exit(-1);
    }

    m_info.SetInputSize(max_input_size);
    m_info.SetOutputSize(max_output_size);

    LOGI << "Registering model: " << model_name << " to context.";
    return m_info.GetIdInc();
}

std::tuple<char *, char *, float *> QNNCtx::Request() {
    if (!m_info.IsInitialized()) {
        m_info.Init();
    }
    return std::tuple<char *, char *, float *>(m_info.GetInputPtr(), m_info.GetOutputPtr(),
                                               m_info.GetDequantizePtr());
}
}  // namespace qnn