#ifndef OPENPOSE_NET_RESIZE_AND_MERGE_BASE_HPP
#define OPENPOSE_NET_RESIZE_AND_MERGE_BASE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    void resizeAndMergeCpu(
        T* targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<T>& scaleInputToNetInputs = {1.f});
}

#endif // OPENPOSE_NET_RESIZE_AND_MERGE_BASE_HPP