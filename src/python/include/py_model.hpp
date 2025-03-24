#ifndef PYTHON_MODEL_HPP_
#define PYTHON_MODEL_HPP_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image/base_image.hpp"
#include "model/base_model.hpp"
#include "py_image.hpp"
#include "tdl_model_factory.hpp"
namespace py = pybind11;
namespace pytdl {

class PyModel {
 public:
  PyModel(ModelType model_type, const std::string& model_path,
          const int device_id = 0);

  virtual ~PyModel() = default;

 protected:
  std::shared_ptr<BaseModel> model_;
};
class PyObejectDetector : public PyModel {
 public:
  // 构造函数
  PyObejectDetector(ModelType model_type, const std::string& model_path,
                    const int device_id = 0);
  py::list inference(const PyImage& image, py::dict parameters = py::dict());
};
class PyFaceDetector : public PyObejectDetector {
 public:
  PyFaceDetector(ModelType model_type, const std::string& model_path,
                 const int device_id = 0);
  py::list inference(const PyImage& image, py::dict parameters = py::dict());
};

class PyClassifier : public PyModel {
 public:
  PyClassifier(ModelType model_type, const std::string& model_path,
               const int device_id = 0);
  py::dict inference(const PyImage& image, py::dict parameters = py::dict());
};

class PyFeatureExtractor : public PyModel {
 public:
  PyFeatureExtractor(ModelType model_type, const std::string&
  model_path,
                     const int device_id = 0);
  py::array_t<float> inference(const PyImage& image,
                               py::dict parameters = py::dict());
};

class PyAttributeExtractor : public PyModel {
 public:
  PyAttributeExtractor(ModelType model_type, const std::string& model_path,
                       const int device_id = 0);
  py::dict inference(const PyImage& image, py::dict parameters = py::dict());
};

class PyKeyPointDetector : public PyModel {
 public:
  PyKeyPointDetector(ModelType model_type, const std::string& model_path,
                     const int device_id = 0);
  py::dict inference(const PyImage& image, py::dict parameters = py::dict());
};

class PyInstanceSegmentation : public PyModel {
 public:
  PyInstanceSegmentation(ModelType model_type, const std::string& model_path,
                 const int device_id = 0);
  py::dict inference(const PyImage& image, py::dict parameters = py::dict());
};

class PyFaceLandmark : public PyModel {
 public:
  PyFaceLandmark(ModelType model_type, const std::string& model_path,
                 const int device_id = 0);
  py::dict inference(const PyImage& image, py::dict parameters = py::dict());
};
class PySemanticSegmentation : public PyModel {
 public:
  PySemanticSegmentation(ModelType model_type, const std::string& model_path,
                 const int device_id = 0);
  py::dict inference(const PyImage& image, py::dict parameters = py::dict());
};

class PyLaneDetection : public PyModel {
 public:
  PyLaneDetection(ModelType model_type, const std::string& model_path,
                 const int device_id = 0);
  py::list inference(const PyImage& image, py::dict parameters = py::dict());
};

}  // namespace pytdl
#endif
