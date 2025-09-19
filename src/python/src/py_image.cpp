#include "py_image.hpp"
#include <string>
#include "preprocess/base_preprocessor.hpp"
#include "py_utils.hpp"
#include "utils/common_utils.hpp"
namespace pytdl {
std::shared_ptr<BasePreprocessor> gPreprocessor =
    PreprocessorFactory::createPreprocessor(InferencePlatform::UNKOWN);
PyImage::PyImage() {}
PyImage::PyImage(std::shared_ptr<BaseImage>& image) : image_(image) {}

PyImage::PyImage(const py::array& numpy_array, ImageFormat format) {
  // 从numpy数组创建图像
  uint32_t width = numpy_array.shape(1);
  uint32_t height = numpy_array.shape(0);
  uint32_t channel = numpy_array.shape(2);

  uint8_t* data = (uint8_t*)numpy_array.data();
  if ((format == ImageFormat::RGB_PACKED ||
       format == ImageFormat::BGR_PACKED) &&
      channel != 3) {
    throw std::invalid_argument(
        "RGB_PACKED or BGR_PACKED format must have 3 "
        "channels");
  }
  if (format == ImageFormat::GRAY && channel != 1) {
    throw std::invalid_argument("GRAY format must have 1 channel");
  }
  if (channel != 1 && channel != 3) {
    throw std::invalid_argument("Unsupported image format,src_channel:" +
                                std::to_string(channel));
  }
  TDLDataType tdl_data_type = py_to_tdl_data_type(numpy_array.dtype());
  if (tdl_data_type == TDLDataType::UNKOWN) {
    throw std::invalid_argument("Unsupported data type,src_data_type");
  }
  image_ =
      ImageFactory::createImage(width, height, format, tdl_data_type, true);
  int32_t ret = image_->copyFromBuffer(
      data,
      width * height * channel * CommonUtils::getDataTypeSize(tdl_data_type));
  if (ret != 0) {
    throw std::invalid_argument("copy from buffer failed,ret:" +
                                std::to_string(ret));
  }
}

PyImage PyImage::fromNumpy(const py::array& numpy_array, ImageFormat format) {
  return PyImage(numpy_array, format);
}

std::tuple<int, int> PyImage::getSize() const {
  return std::make_tuple(image_->getWidth(), image_->getHeight());
}

ImageFormat PyImage::getFormat() const { return image_->getImageFormat(); }

// py::array PyImage::numpy() const {
//   // only surpport RGB packed and RGB planar
//   if (image_->getImageFormat() != ImageFormat::RGB_PACKED &&
//       image_->getImageFormat() != ImageFormat::BGR_PACKED &&
//       image_->getImageFormat() != ImageFormat::RGB_PLANAR &&
//       image_->getImageFormat() != ImageFormat::BGR_PLANAR) {
//     throw std::invalid_argument("only surpport RGB packed and RGB planar");
//   }
//   std::vector<ssize_t> shape;
//   std::vector<ssize_t> strides;
//   if (image_->isPlanar()) {
//     shape = {3, image_->getHeight(), image_->getWidth()};
//     strides = {image_->getWidth() * 3, image_->getWidth(), 1};
//   } else {
//     shape = {image_->getHeight(), image_->getWidth(), 3};
//     strides = {image_->getWidth() * 3, 3, 1};
//   }
//   return py::array(image_->getImageFormat(), image_->getImageType(),
//                    image_->getImageData());
// }
PyImage read(const std::string& path) {
  std::shared_ptr<BaseImage> image = ImageFactory::readImage(path);
  if (image == nullptr) {
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg), "read image %s failed", path.c_str());
    throw std::invalid_argument(err_msg);
  }
  return PyImage(image);
}
void write(const PyImage& image, const std::string& path) {
  int32_t ret = ImageFactory::writeImage(path, image.getImage());
  if (ret != 0) {
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg), "write image %s failed,ret:%d",
             path.c_str(), ret);
    throw std::invalid_argument(err_msg);
  }
}

PyImage resize(const PyImage& src, int width, int height) {
  if (gPreprocessor == nullptr) {
    gPreprocessor =
        PreprocessorFactory::createPreprocessor(InferencePlatform::AUTOMATIC);
  }

  const std::shared_ptr<BaseImage> src_image = src.getImage();

  std::shared_ptr<BaseImage> dst_image =
      gPreprocessor->resize(src_image, width, height);
  if (dst_image == nullptr) {
    throw std::invalid_argument("resize image failed");
  }
  return PyImage(dst_image);
}

PyImage crop(const PyImage& src, const std::tuple<int, int, int, int>& roi) {
  if (gPreprocessor == nullptr) {
    gPreprocessor =
        PreprocessorFactory::createPreprocessor(InferencePlatform::AUTOMATIC);
  }
  const std::shared_ptr<BaseImage> src_image = src.getImage();
  std::shared_ptr<BaseImage> dst_image =
      gPreprocessor->crop(src_image, std::get<0>(roi), std::get<1>(roi),
                          std::get<2>(roi), std::get<3>(roi));
  if (dst_image == nullptr) {
    throw std::invalid_argument("crop image failed");
  }
  return PyImage(dst_image);
}

PyImage cropResize(const PyImage& src,
                   const std::tuple<int, int, int, int>& roi, int width,
                   int height) {
  if (gPreprocessor == nullptr) {
    gPreprocessor =
        PreprocessorFactory::createPreprocessor(InferencePlatform::AUTOMATIC);
  }
  const std::shared_ptr<BaseImage> src_image = src.getImage();
  PreprocessParams params;
  memset(&params, 0, sizeof(PreprocessParams));
  params.dst_image_format = src_image->getImageFormat();
  params.dst_pixdata_type = src_image->getPixDataType();
  params.dst_width = width;
  params.dst_height = height;
  params.crop_x = std::get<0>(roi);
  params.crop_y = std::get<1>(roi);
  params.crop_width = std::get<2>(roi);
  params.crop_height = std::get<3>(roi);
  std::shared_ptr<BaseImage> dst_image =
      gPreprocessor->preprocess(src_image, params, nullptr);
  if (dst_image == nullptr) {
    throw std::invalid_argument("crop resize image failed");
  }
  return PyImage(dst_image);
}

PyImage align_face(const PyImage& image,
                   const std::vector<float>& src_landmark_xy,
                   const std::vector<float>& dst_landmark_xy, int num_points) {
  std::shared_ptr<BaseImage> image_ptr = image.getImage();

  // 验证关键点数据有效性
  if (src_landmark_xy.size() < 2 ||
      src_landmark_xy.size() != static_cast<size_t>(num_points * 2)) {
    throw std::invalid_argument("Invalid number of landmark points");
  }

  // 如果未提供目标关键点，则使用默认值
  std::vector<float> default_dst_landmarks;
  const float* actual_dst_landmark = dst_landmark_xy.data();

  if (dst_landmark_xy.empty()) {
    // 使用默认的5点人脸对齐模板（归一化坐标）
    default_dst_landmarks = {
        38.2946f, 51.6963f,  // 左眼
        73.5318f, 51.5014f,  // 右眼
        56.0252f, 71.7366f,  // 鼻子
        41.5493f, 92.3655f,  // 左嘴角
        70.7299f, 99.3655f   // 右嘴角
    };
    actual_dst_landmark = default_dst_landmarks.data();
    num_points = 5;  // 确保点数匹配
  }

  // 调用底层人脸对齐函数
  std::shared_ptr<BaseImage> aligned_image =
      ImageFactory::alignFace(image_ptr, src_landmark_xy.data(),
                              actual_dst_landmark, num_points, nullptr);

  return PyImage(aligned_image);
}
}  // namespace pytdl