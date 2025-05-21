#include "py_image.hpp"
#include "py_llm.hpp"
#include "py_model.hpp"
using namespace pytdl;

// 明确指定函数类型，解决重载问题
PyImage (*read_func)(const std::string&) = &read;
void (*write_func)(const PyImage&, const std::string&) = &write;
PyImage (*resize_func)(const PyImage&, int, int) = &resize;
PyImage (*crop_func)(const PyImage&,
                     const std::tuple<int, int, int, int>&) = &crop;
PyImage (*crop_resize_func)(const PyImage&,
                            const std::tuple<int, int, int, int>&, int,
                            int) = &cropResize;
PyImage (*alignFace_func)(const PyImage& image,
                          const std::vector<float>& src_landmark_xy,
                          const std::vector<float>& dst_landmark_xy,
                          int num_points) = &alignFace;

// pybind11绑定实现
PYBIND11_MODULE(tdl, m) {
  m.doc() = "tdl sdk module python binding";

  // 图像模块
  py::module image = m.def_submodule("image", "image module");
  // 绑定枚举
  py::enum_<ImageFormat>(image, "ImageFormat")
      .value("RGB_PLANAR", ImageFormat::RGB_PLANAR)
      .value("BGR_PLANAR", ImageFormat::BGR_PLANAR)
      .value("RGB_PACKED", ImageFormat::RGB_PACKED)
      .value("BGR_PACKED", ImageFormat::BGR_PACKED)
      .value("GRAY", ImageFormat::GRAY)
      .value("YUV420SP_UV", ImageFormat::YUV420SP_UV)
      .value("YUV420SP_VU", ImageFormat::YUV420SP_VU)
      //   .value("NV12", ImageFormat::YUV420SP_UV)
      //   .value("NV21", ImageFormat::YUV420SP_VU)
      .value("YUV420P_UV", ImageFormat::YUV420P_UV)
      .value("YUV420P_VU", ImageFormat::YUV420P_VU)
      .value("YUV422P_UV", ImageFormat::YUV422P_UV)
      .value("YUV422P_VU", ImageFormat::YUV422P_VU)
      .value("YUV422SP_UV", ImageFormat::YUV422SP_UV)
      .value("YUV422SP_VU", ImageFormat::YUV422SP_VU)
      .export_values();

  py::enum_<TDLDataType>(image, "TDLDataType")
      .value("UINT8", TDLDataType::UINT8)
      .value("INT8", TDLDataType::INT8)
      .value("UINT16", TDLDataType::UINT16)
      .value("INT16", TDLDataType::INT16)
      .value("UINT32", TDLDataType::UINT32)
      .value("INT32", TDLDataType::INT32)
      .value("FP32", TDLDataType::FP32)
      .export_values();

  // 绑定图像类
  py::class_<PyImage>(image, "Image")
      .def(py::init<>())
      .def_static("from_numpy", &PyImage::fromNumpy, py::arg("numpy_array"),
                  py::arg("format") = ImageFormat::BGR_PACKED)
      .def("get_size", &PyImage::getSize)
      .def("get_format", &PyImage::getFormat);

  // 绑定模块函数，使用明确类型的函数指针
  image.def("write", write_func, py::arg("image"), py::arg("path"));
  image.def("read", read_func, py::arg("path"));
  image.def("resize", resize_func, py::arg("src"), py::arg("width"),
            py::arg("height"));
  image.def("crop", crop_func, py::arg("src"), py::arg("roi"));
  image.def("crop_resize", crop_resize_func, py::arg("src"), py::arg("roi"),
            py::arg("width"), py::arg("height"));
  image.def("alignFace", alignFace_func, py::arg("image"),
            py::arg("src_landmark_xy"), py::arg("dst_landmark_xy"),
            py::arg("num_points"));

  // 神经网络模块
  py::module nn = m.def_submodule("nn", "Neural network algorithms module");
  py::enum_<ModelType> model_type_enum(nn, "ModelType");
#define X(name, comment) model_type_enum.value(#name, ModelType::name);
  // 直接用 MODEL_TYPE_LIST 把所有 name 都展开一次
  MODEL_TYPE_LIST
#undef X
  model_type_enum.export_values();

  py::class_<PyModel>(nn, "Model")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("getPreprocessParameters", &PyModel::getPreprocessParameters)
      .def("setPreprocessParameters", &PyModel::setPreprocessParameters);

  py::class_<PyObjectDetector>(nn, "ObjectDetector")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyObjectDetector::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyObjectDetector::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyFaceDetector, PyModel>(nn, "FaceDetector")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyFaceDetector::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyFaceDetector::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyFaceLandmark, PyModel>(nn, "FaceLandmark")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyFaceLandmark::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyFaceLandmark::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyClassifier, PyModel>(nn, "Classifier")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def(
          "inference",
          py::overload_cast<const PyImage&, py::dict>(&PyClassifier::inference),
          py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyClassifier::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyKeyPointDetector, PyModel>(nn, "KeyPointDetector")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyKeyPointDetector::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyKeyPointDetector::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PySemanticSegmentation, PyModel>(nn, "SemanticSegmentation")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PySemanticSegmentation::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PySemanticSegmentation::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyInstanceSegmentation, PyModel>(nn, "InstanceSegmentation")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyInstanceSegmentation::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyInstanceSegmentation::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyLaneDetection, PyModel>(nn, "LaneDetection")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyLaneDetection::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyLaneDetection::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyAttributeExtractor, PyModel>(nn, "AttributeExtractor")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyAttributeExtractor::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyAttributeExtractor::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyFeatureExtractor, PyModel>(nn, "FeatureExtractor")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyFeatureExtractor::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyFeatureExtractor::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::class_<PyCharacterRecognitor, PyModel>(nn, "CharacterRecognitor")
      .def(py::init<ModelType, std::string, int>(), py::arg("model_type"),
           py::arg("model_path"), py::arg("device_id") = 0)
      .def("inference",
           py::overload_cast<const PyImage&, py::dict>(
               &PyCharacterRecognitor::inference),
           py::arg("image"), py::arg("parameters") = py::dict())
      .def("inference",
           py::overload_cast<
               const py::array_t<unsigned char, py::array::c_style>&, py::dict>(
               &PyCharacterRecognitor::inference),
           py::arg("input") = py::array_t<unsigned char>(),
           py::arg("parameters") = py::dict());

  py::module llm = m.def_submodule("llm", "LLM module");
  llm.def("fetch_video", &pytdl::fetch_video, py::arg("video_path"),
          py::arg("desired_fps") = 2.0, py::arg("desired_nframes") = 0,
          py::arg("max_video_sec") = 0);
  llm.def("test_fetch_video_ts", &pytdl::test_fetch_video_ts,
          py::arg("video_path"), py::arg("desired_fps") = 2.0,
          py::arg("desired_nframes") = 0, py::arg("max_video_sec") = 0);
  //   注册Qwen类
  py::class_<pytdl::PyQwen>(llm, "Qwen")
      .def(py::init<>())
      .def("model_open", &pytdl::PyQwen::modelOpen, py::arg("model_path"))
      .def("model_close", &pytdl::PyQwen::modelClose)
      .def("inference_first", &pytdl::PyQwen::inferenceFirst,
           py::arg("input_tokens"))
      .def("inference_next", &pytdl::PyQwen::inferenceNext)
      .def("inference_generate", &pytdl::PyQwen::inferenceGenerate,
           py::arg("input_tokens"), py::arg("eos_token"))
      .def("get_infer_param", &pytdl::PyQwen::getInferParam)
      .def("__enter__", [](pytdl::PyQwen& self) { return &self; })
      .def("__exit__", [](pytdl::PyQwen& self, py::object, py::object,
                          py::object) { self.modelClose(); });

  // 注册Qwen2VL类
  py::class_<pytdl::PyQwen2VL>(llm, "Qwen2VL")
      .def(py::init<>())
      .def("init", &pytdl::PyQwen2VL::init, py::arg("dev_id"),
           py::arg("model_path"))
      .def("deinit", &pytdl::PyQwen2VL::deinit)
      .def("forward_first", &pytdl::PyQwen2VL::forward_first, py::arg("tokens"),
           py::arg("position_ids"), py::arg("pixel_values"), py::arg("posids"),
           py::arg("attnmask"), py::arg("img_offset"), py::arg("pixel_num"))
      .def("forward_next", &pytdl::PyQwen2VL::forward_next)
      .def("set_generation_mode", &pytdl::PyQwen2VL::set_generation_mode,
           py::arg("mode"))
      .def("get_generation_mode", &pytdl::PyQwen2VL::get_generation_mode)
      .def_readwrite("generation_mode", &pytdl::PyQwen2VL::generation_mode)
      .def_readwrite("SEQLEN", &pytdl::PyQwen2VL::SEQLEN)
      .def_readwrite("token_length", &pytdl::PyQwen2VL::token_length)
      .def_readwrite("HIDDEN_SIZE", &pytdl::PyQwen2VL::HIDDEN_SIZE)
      .def_readwrite("NUM_LAYERS", &pytdl::PyQwen2VL::NUM_LAYERS)
      .def_readwrite("MAX_POS", &pytdl::PyQwen2VL::MAX_POS)
      .def_readwrite("MAX_PIXELS", &pytdl::PyQwen2VL::MAX_PIXELS)
      .def_readwrite("VIT_DIMS", &pytdl::PyQwen2VL::VIT_DIMS)
      .def("__enter__", [](pytdl::PyQwen2VL& self) { return &self; })
      .def("__exit__", [](pytdl::PyQwen2VL& self, py::object, py::object,
                          py::object) { self.deinit(); });
}
