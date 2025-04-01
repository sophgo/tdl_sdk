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

  // 神经网络模块
  py::module nn = m.def_submodule("nn", "Neural network algorithms module");
  py::enum_<ModelType>(nn, "ModelType")
      // face detection model
      .value("MBV2_DET_PERSON", ModelType::MBV2_DET_PERSON)
      .value("YOLOV5_DET_COCO80", ModelType::YOLOV5_DET_COCO80)
      .value("YOLOV6_DET_COCO80", ModelType::YOLOV6_DET_COCO80)
      .value("YOLOV8_DET_COCO80", ModelType::YOLOV8_DET_COCO80)
      .value("YOLOV10_DET_COCO80", ModelType::YOLOV10_DET_COCO80)
      .value("YOLOV8N_DET_HAND", ModelType::YOLOV8N_DET_HAND)
      .value("YOLOV8N_DET_PET_PERSON",
             ModelType::YOLOV8N_DET_PET_PERSON)  // 0:cat,1:dog,2:person
      .value(
          "YOLOV8N_DET_PERSON_VEHICLE",
          ModelType::
              YOLOV8N_DET_PERSON_VEHICLE)  // 0:car,1:bus,2:truck,3:rider with
                                           // motorcycle,4:person,5:bike,6:motorcycle
      .value("YOLOV8N_DET_HAND_FACE_PERSON",
             ModelType::YOLOV8N_DET_HAND_FACE_PERSON)  // 0:hand,1:face,2:person
      .value("YOLOV8N_DET_HEAD_PERSON",
             ModelType::YOLOV8N_DET_HEAD_PERSON)  // 0:person,1:head
      .value("YOLOV8N_DET_HEAD_HARDHAT",
             ModelType::YOLOV8N_DET_HEAD_HARDHAT)  // 0:head,1:hardhat
      .value("YOLOV8N_DET_FIRE_SMOKE",
             ModelType::YOLOV8N_DET_FIRE_SMOKE)                // 0:fire,1:smoke
      .value("YOLOV8N_DET_FIRE", ModelType::YOLOV8N_DET_FIRE)  // 0:fire
      .value("YOLOV8N_DET_HEAD_SHOULDER",
             ModelType::YOLOV8N_DET_HEAD_SHOULDER)  // 0:head shoulder
      .value("YOLOV8N_DET_LICENSE_PLATE",
             ModelType::YOLOV8N_DET_LICENSE_PLATE)  // 0:license plate
      .value(
          "YOLOV8N_DET_TRAFFIC_LIGHT",
          ModelType::
              YOLOV8N_DET_TRAFFIC_LIGHT)  // 0:red,1:yellow,2:green,3:off,4:wait
                                          // on
      .value("YOLOV8N_DET_MONITOR_PERSON",
             ModelType::YOLOV8N_DET_MONITOR_PERSON)  // 0:person

      // face detection model
      .value("SCRFD_DET_FACE", ModelType::SCRFD_DET_FACE)  // 0:face + landm
      .value("RETINA_DET_FACE", ModelType::RETINA_DET_FACE)
      .value("RETINA_DET_FACE_IR", ModelType::RETINA_DET_FACE_IR)
      .value("KEYPOINT_FACE_V2",
             ModelType::KEYPOINT_FACE_V2)  // 5 landmarks + blurness score
      .value("CLS_ATTRIBUTE_FACE",
             ModelType::CLS_ATTRIBUTE_FACE)  // age,gener,glass,mask
      .value("FEATURE_BMFACER34",
             ModelType::FEATURE_BMFACER34)  // resnet34 512 dim feature
      .value("FEATURE_BMFACER50",
             ModelType::FEATURE_BMFACER50)  // resnet50 512 dim feature

      // image classification models
      .value("CLS_MASK", ModelType::CLS_MASK)                // 0:mask,1:no mask
      .value("CLS_RGBLIVENESS", ModelType::CLS_RGBLIVENESS)  // 0:fake,1:live
      .value("CLS_ISP_SCENE", ModelType::CLS_ISP_SCENE)
      .value("CLS_HAND_GESTURE",
             ModelType::CLS_HAND_GESTURE)  // 0:fist,1:five,2:none,3:two
      .value(
          "CLS_KEYPOINT_HAND_GESTURE",
          ModelType::
              CLS_KEYPOINT_HAND_GESTURE)  // 0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two

      // sound classification models
      .value("CLS_SOUND_BABAY_CRY",
             ModelType::CLS_SOUND_BABAY_CRY)  // 0:background,1:cry
      .value(
          "CLS_SOUND_COMMAND",
          ModelType::CLS_SOUND_COMMAND)  // 0:background,1:command1,2:command2
                                         // ...

      // image keypoint models
      .value("KEYPOINT_LICENSE_PLATE", ModelType::KEYPOINT_LICENSE_PLATE)
      .value("KEYPOINT_HAND", ModelType::KEYPOINT_HAND)
      .value(
          "KEYPOINT_YOLOV8POSE_PERSON17",
          ModelType::KEYPOINT_YOLOV8POSE_PERSON17)  // 17 keypoints for person
      .value("KEYPOINT_SIMCC_PERSON17", ModelType::KEYPOINT_SIMCC_PERSON17)

      // lane detection models
      .value("LSTR_DET_LANE", ModelType::LSTR_DET_LANE)

      // license plate recognition models
      .value("RECOGNITION_LICENSE_PLATE", ModelType::RECOGNITION_LICENSE_PLATE)

      // image segmentation models
      .value("YOLOV8_SEG_COCO80", ModelType::YOLOV8_SEG_COCO80)
      .value(
          "TOPFORMER_SEG_PERSON_FACE_VEHICLE",
          ModelType::
              TOPFORMER_SEG_PERSON_FACE_VEHICLE)  // 0:background,1:person,2:face,3:vehicle,4:license
                                                  // plate
      .value(
          "TOPFORMER_SEG_MOTION",
          ModelType::TOPFORMER_SEG_MOTION)  // 0:static,2:transsition 3:motion

      // CLIP models
      .value("CLIP_FEATURE_IMG", ModelType::CLIP_FEATURE_IMG)
      .value("CLIP_FEATURE_TEXT", ModelType::CLIP_FEATURE_TEXT)
      .export_values();
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

  py::class_<PyFaceDetector>(nn, "FaceDetector")
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

  py::class_<PyFaceLandmark>(nn, "FaceLandmark")
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

  py::class_<PyClassifier>(nn, "Classifier")
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

  py::class_<PyKeyPointDetector>(nn, "KeyPointDetector")
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

  py::class_<PySemanticSegmentation>(nn, "SemanticSegmentation")
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

  py::class_<PyInstanceSegmentation>(nn, "InstanceSegmentation")
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

  py::class_<PyLaneDetection>(nn, "LaneDetection")
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

  py::class_<PyAttributeExtractor>(nn, "AttributeExtractor")
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

  py::class_<PyFeatureExtractor>(nn, "FeatureExtractor")
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

  py::class_<PyCharacterRecognitor>(nn, "CharacterRecognitor")
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

  // 注册Qwen类
  py::class_<PyQwen>(llm, "Qwen")
      .def(py::init<>())
      .def("model_open", &PyQwen::modelOpen, py::arg("model_path"))
      .def("model_close", &PyQwen::modelClose)
      .def("inference_first", &PyQwen::inferenceFirst, py::arg("input_tokens"))
      .def("inference_next", &PyQwen::inferenceNext)
      .def("inference_generate", &PyQwen::inferenceGenerate,
           py::arg("input_tokens"), py::arg("eos_token"))
      .def("get_infer_param", &PyQwen::getInferParam)
      .def("__enter__", [](PyQwen& self) { return &self; })
      .def("__exit__", [](PyQwen& self, py::object, py::object, py::object) {
        self.modelClose();
      });
}