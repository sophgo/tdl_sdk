#ifndef PYTHON_IMAGE_HPP_
#define PYTHON_IMAGE_HPP_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image/base_image.hpp"

namespace py = pybind11;
namespace pytdl {

class PyImage {
 public:
  // 构造函数
  PyImage();
  PyImage(std::shared_ptr<BaseImage>& image);
  PyImage(const py::array& numpy_array,
          ImageFormat format = ImageFormat::RGB_PACKED);

  // 静态创建方法
  static PyImage fromNumpy(const py::array& numpy_array,
                           ImageFormat format = ImageFormat::RGB_PACKED);

  std::tuple<int, int> getSize() const;
  ImageFormat getFormat() const;
  std::shared_ptr<BaseImage> getImage() const { return image_; }
  // py::array numpy() const;

 private:
  std::shared_ptr<BaseImage> image_;
};

// 模块级函数接口定义
void write(const PyImage& image, const std::string& path);
PyImage read(const std::string& path);
PyImage resize(const PyImage& src, int width, int height);
PyImage crop(const PyImage& src, const std::tuple<int, int, int, int>& roi);
PyImage cropResize(const PyImage& src,
                   const std::tuple<int, int, int, int>& roi, int width,
                   int height);
// PyImage convert(const PyImage& src, ImageFormat dst_format,
//                 ImageDataType dst_type);
}  // namespace pytdl
#endif
