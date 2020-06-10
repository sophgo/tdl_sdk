// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file qnnctx.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once
#include <cmath>
#include <functional>
#include <memory>

namespace qnn {

/**
 * @brief Custom destructor for unique pointer for using posix_memalign based allocated buffer.
 *
 * @tparam T Allocated variable type.
 */
template <class T>
struct malloc_delete {
    void operator()(T *data) const { free(data); }
};

/**
 * @brief Alias name for aligned std::unique_ptr
 *
 * @tparam T Allocated variable type.
 */
template <class T>
using unique_aptr = std::unique_ptr<T[], malloc_delete<T>>;

/**
 * @brief Function for allocating aligned memory for unique pointer.
 *
 * @tparam T Allocated variable type.
 */
template <class T>
extern unique_aptr<T> allocate_aligned(int alignment, int length, bool aligned = true);

/**
 * @brief A Buffer Handler
 *
 */
class CtxBufferInfo {
    public:
    /**
     * @brief Initialize and create buffer
     *
     */
    void Init();

    /** @defgroup ctxbufferinfogettergroup Context Getter Functions
     *  @{
     */

    /**
     * @brief Tell if the buffer context is initialized.
     *
     * @return true Is initialized.
     * @return false Not initialized.
     */
    const bool &IsInitialized() { return m_is_initialized; }

    /**
     * @brief Get the current id. (value+1) represents the number of model class registered.
     *
     * @return const int The id value
     */
    const int GetId() { return m_idx; }

    /**
     * @brief Get the input array address.
     *
     * @return char* The head of the input address
     */
    char *GetInputPtr() { return m_char_buffer.get(); }

    /**
     * @brief Get the output array address. The output and the input address uses the same buffer
     * array, so an offset is added.
     *
     * @return char* The head of the output address.
     */
    char *GetOutputPtr() { return m_char_buffer.get() + m_max_in_size; }

    /**
     * @brief Get the dequantize array address.
     *
     * @return float* The head of the dequantize address.
     */
    float *GetDequantizePtr() { return m_float_buffer.get(); }

    /**
     * @brief Get the input size of the input array buffer.
     *
     * @return const int& The input array size
     */
    const int &GetInputSize() { return m_max_in_size; }

    /**
     * @brief Get the output size of the output array buffer.
     *
     * @return const int&  The output array size.
     */
    const int &GetOutputSize() { return m_max_out_size; }
    /** @}*/  // End of ctxbufferinfogettergroup

    /**
     * @brief Set the maximum input size of a network.
     *
     * @param in The maximum input size of a network
     */
    inline void SetInputSize(const int &in) { m_max_in_size = std::max(m_max_in_size, in); }

    /**
     * @brief Set the maximum output size of a network.
     *
     * @param out The maximum output size of a network
     */
    inline void SetOutputSize(const int &out) { m_max_out_size = std::max(m_max_out_size, out); }

    protected:
    /**
     * @brief Get the current id while increament by 1. This is handy if registering multiple
     * models.
     *
     * @return const int The id value
     */
    const int GetIdInc();
    friend class QNNCtx;

    private:
    bool m_is_initialized = false; /**< The initialized flag */
    int m_idx = 0;                 /**< id idx. */
    int m_max_in_size = 0;  /**< Possible maximum input size of all the registered networks */
    int m_max_out_size = 0; /**< Possible maximum output size of all the registered networks */
    unique_aptr<char> m_char_buffer;   /**< The unique ptr of buffer in char */
    unique_aptr<float> m_float_buffer; /**< The unique ptr of buffer in float */
};

/**
 * @brief The base class of QNNCtx
 *
 */
class QNNCtx {
    public:
    /**
     * @brief Register a network to QNN Context. Note that QNNCtx does not supports multithreading.
     * All registered models with same QNNCtx must run in sequence.
     *
     * @param model_name The name of the registered model
     * @param max_input_size The possible maximum input size of the model
     * @param max_output_size The possible maximum output size of the model
     * @return int The registered index of this QNNCtx
     */
    int Register(const std::string model_name, const int max_input_size, const int max_output_size);

    /**
     * @brief Request buffers from QNNCtx.
     *
     * @return std::tuple<char *, char *, float *> The buffers in the sequence of char input, char
     * output, float dequantize array
     */
    std::tuple<char *, char *, float *> Request();

    private:
    CtxBufferInfo m_info; /**< Buffer handler instance */
};
}  // namespace qnn
