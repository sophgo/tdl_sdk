#ifndef OPENPOSE_NET_NMS_BASE_HPP
#define OPENPOSE_NET_NMS_BASE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    void nmsCpu(
      T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize,
      const std::array<int, 4>& sourceSize, const Point<T>& offset);
}

#endif // OPENPOSE_NET_NMS_BASE_HPP