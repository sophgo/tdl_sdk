// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>
/**
 * @file tensor.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace qnn {

/**
 * @addtogroup TENSOR_OBJ Tensor Objects
 * \ingroup NETWORKS_CORE
 * @{
 */
// TODO: FIXME: bmkernel is able to transpose when send buffer to npu. But currently bmtap2 did not
// support.
/**
 * @brief Shape sequence.
 *
 */
enum TENSORSEQ {
    NCHW,
    NCWH,
};

struct NetShape {
    /**
     * @brief Construct a new NetShape object. (Default constrcutor)
     *
     */
    NetShape() {}

    /**
     * @brief Construct a new NetShape object.
     *
     * @param N Batch size
     * @param C Channel size
     * @param H Height
     * @param W Width
     */
    NetShape(int N, int C, int H, int W) : n(N), c(C), h(H), w(W) {}

    /**
     * @brief Construct a new NetShape object.
     *
     * @param other Input netshape object
     */
    NetShape(const NetShape &other) : n(other.n), c(other.c), h(other.h), w(other.w) {}

    /**
     * @brief Compare operator.
     *
     * @param rhs Rhs Netshape object
     * @return true memcmp are the same
     * @return false memcmp are not the same
     */
    bool operator==(const NetShape &rhs) { return 0 == memcmp(this, &rhs, sizeof(NetShape)); }

    /**
     * @brief Assign operator. (Copy, not reference.)
     *
     * @param other Netshape object.
     * @return NetShape& New Netshape object.
     */
    NetShape &operator=(const NetShape &other) {
        this->n = other.n;
        this->c = other.c;
        this->h = other.h;
        this->w = other.w;
        return *this;
    }

    /**
     * @brief ostream operator.
     *
     * @param rhs Rhs NetTensor object
     * @return ostream ostream
     */
    friend std::ostream &operator<<(std::ostream &s, const NetShape &rhs) {
        s << "( " << rhs.n << ", " << rhs.c << ", " << rhs.h << ", " << rhs.w << ")";
        return s;
    }

    int n; /**< Batch size */
    int c; /**< Channel size */
    int h; /**< Height */
    int w; /* <Width */
};

/**
 * @brief The NetTensor base class. Saves the tensor information from bmmodel's layer.
 *
 */
struct NetTensor {
    /**
     * @brief Construct a new NetTensor object. (Default constructor)
     *
     */
    NetTensor() {}

    /**
     * @brief Construct a new NetTensor object.
     *
     * @param s Input NetShape
     */
    NetTensor(const NetShape &s) : shape(s) {
        count = shape.n * shape.c * shape.h * shape.w;
#if defined(NPU_INT8)
        size = count * sizeof(int8_t);
#elif (NPU_FLOAT32)
        size = count * sizeof(float);
#endif
    }

    /**
     * @brief Construct a new NetTensor object from existing object.
     *
     * @param other Another NetTensor
     */
    NetTensor(const NetTensor &other)
        : shape(other.shape), size(other.size), count(other.count), name(other.name) {}

    /**
     * @brief Destroy the NetTensor object.
     *
     */
    virtual ~NetTensor() {}

    /**
     * @brief Compare operator.
     *
     * @param rhs Rhs NetTensor object
     * @return true shape same
     * @return false shape not match
     */
    bool operator==(const NetTensor &rhs) { return shape == rhs.shape; }

    NetShape shape;   /**< Shape of the tensor */
    int size;         /**< Size of the tensor inside memory */
    int count;        /**< Size of shape n * c * h * w */
    int dim = 4;      /**< hardcode dimension = 4 at this moment */
    std::string name; /**< Name of the layer */
};

/**
 * @brief The Input Tensor structure.
 *
 */
struct InputTensor : public NetTensor {
    /**
     * @brief Construct a new InputTensor object from given NetShape.
     *
     * @param s Input NetShape
     */
    InputTensor(const NetShape &s) : NetTensor(s) {}

    /**
     * @brief Construct a new InputTensor object from another tensor object.
     *
     * @param other InputTensor
     */
    InputTensor(const InputTensor &other) : NetTensor(other) { data = other.data; }

    virtual ~InputTensor() override {}

    char *data; /**< Data buffer in char* */
};

/**
 * @brief The Output Tensor structure.
 *
 */
struct OutputTensor : public NetTensor {
    /**
     * @brief Construct a new OutputTensor object. (Default constructor)
     *
     */
    OutputTensor() {}

    /**
     * @brief Construct a new OutputTensor object from given NetShape.
     *
     * @param s Input Netshape
     */
    OutputTensor(const NetShape &s) : NetTensor(s) {}

    /**
     * @brief Construct a new OutputTensor object from another tensor object.
     *
     * @param other OutputTensor
     */
    OutputTensor(const OutputTensor &other)
        : NetTensor(other), quantize_threshold(other.quantize_threshold) {
        q_data = other.q_data;
        data = other.data;
    }

    virtual ~OutputTensor() override {}

    float *data;                    /**< Data buffer in float*, dequantized output */
    char *q_data;                   /**< Data buffer in char*, quantized output */
    float quantize_threshold = 0.0; /**< Dequantize threshold */
};
/** @} */
}  // namespace qnn