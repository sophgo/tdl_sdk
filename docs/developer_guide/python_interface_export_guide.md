# python接口导出指南

简单来说，将python格式的输入转换为c++的输入格式，调用c++模块，然后将c++格式的输出转换为python的输出格式

- python接口导出主要涉及以下三个文件，建议先学习已经导出的功能是如何实现的

  - ./tdl_sdk/src/python/src/py_interface.cpp

  - ./tdl_sdk/src/python/src/py_model.cpp

  - ./tdl_sdk/src/python/include/py_model.hpp

## 一、新增模型属于已存在的功能类

  当前TDL SDK已经集成目标检测、人脸检测、分类、关键点检测、语义分割、实例分割、车道线检测、属性提取、特征提取、字符提取,这些python功能类。

- 如何新增模型属于已存在的功能类 （搬自tdl_sdk接口开发规范第五章）
  - **在./tdl_sdk/src/python/include/py_model.hpp头文件，完成python绑定层接口定义。** 以目标检测为例，需要声明构造函数和两个inference接口（第一个用于处理内部image类处理的输入，第二个用于处理使用OpenCV处理的输入）
  
    ```cpp
    class PyObjectDetector : public PyModel {
    public:
      // 构造函数
      PyObjectDetector(ModelType model_type, const std::string& model_path,
                        const int device_id = 0);
      py::list inference(const PyImage& image, py::dict parameters = py::dict());
      py::list inference(const py::array_t<unsigned char, py::array::c_style> &input, 
                     py::dict parameters = py::dict());
    };
    ```
  
  - **在./tdl_sdk/src/python/src/py_model.cpp中完成python绑定层接口的实现。**
  
    ```cpp
    PyObjectDetector::PyObjectDetector(ModelType model_type,
                                   const std::string& model_path,
                                   const int device_id)
    : PyModel(model_type, model_path, device_id) {}

    py::list PyObjectDetector::inference(const PyImage& image,
                                        py::dict parameters) {
      std::vector<std::shared_ptr<BaseImage>> images;
      images.push_back(image.getImage());
      std::map<std::string, float> parameters_map;
      for (auto& item : parameters) {
       parameters_map[item.first.cast<std::string>()] = item.second.cast<float>();
     }
      std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
      model_->inference(images, out_datas, parameters_map);
      std::shared_ptr<ModelOutputInfo> output_info = out_datas[0];
      if (output_info->getType() != ModelOutputType::OBJECT_DETECTION) {
        throw std::runtime_error("Model output type is not OBJECT_DETECTION");
      }
      std::shared_ptr<ModelBoxInfo> box_info =
          std::dynamic_pointer_cast<ModelBoxInfo>(output_info);
      if (!box_info) {
        throw std::runtime_error("Failed to cast to ModelBoxInfo");
      }
      py::list bboxes;
      for (auto& box : box_info->bboxes) {
        py::dict box_dict;
        box_dict[py::str("class_id")] = box.class_id;
        box_dict[py::str("class_name")] = object_type_to_string(box.object_type);
        box_dict[py::str("x1")] = box.x1;
        box_dict[py::str("y1")] = box.y1;
        box_dict[py::str("x2")] = box.x2;
        box_dict[py::str("y2")] = box.y2;
        box_dict[py::str("score")] = box.score;
        bboxes.append(box_dict);
      }
      return bboxes;
    }
    py::list PyObjectDetector::inference(
        const py::array_t<unsigned char, py::array::c_style>& input,
        py::dict parameters) {
      PyImage image = PyImage::fromNumpy(input);
      return inference(image, parameters);  
    }
    ```

  - **在./tdl_sdk/src/python/src/py_interface.cpp的ModelType中定义模型ID即可。**
  
    ```cpp
     py::module nn = m.def_submodule("nn", "Neural network algorithms module");
      py::enum_<ModelType>(nn, "ModelType")
          .value("MBV2_DET_PERSON", ModelType::MBV2_DET_PERSON)
          .export_values();
    ```

  - **在./tdl_sdk/src/python/src/py_interface.cpp中python和C++接口绑定。**
  
    ```cpp
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
    ```
  
## 二、新增一个功能类
  
  以tracker模块为例，已有头文件定义tracker模块的类与函数./tdl_sdk/include/components/tracker_types.hpp

- 如何新增一个模块

  - **在./tdl_sdk/src/python/src/py_interface.cpp中完成子模块的定义。** 绑定枚举类型、函数类、工厂类等，*基于相关头文件中的定义*；

      py::enum_ 绑定c++枚举类型

      py::class_ 绑定c++类，支持：

      构造函数 def(py::init<>())

      成员函数 def("method_name", &Class::method)

      成员变量 def_readwrite("var_name", &Class::var)

      静态方法 def_static("static_method", &Class::static_method)

    ```cpp
      py::module tracker = m.def_submodule("tracker", "Tracker module");

      // 绑定 TrackStatus 枚举
      py::enum_<TrackStatus>(tracker, "TrackStatus")
          .value("NEW", TrackStatus::NEW)
          .value("TRACKED", TrackStatus::TRACKED)
          .value("LOST", TrackStatus::LOST)
          .value("REMOVED", TrackStatus::REMOVED)
          .export_values();

      // 绑定 TrackerType 枚举
      py::enum_<TrackerType>(tracker, "TrackerType")
          .value("TDL_MOT_SORT", TrackerType::TDL_MOT_SORT)
          .export_values();

      // 绑定 TrackerConfig 类
      py::class_<TrackerConfig>(tracker, "TrackerConfig")
          .def(py::init<>())
          .def_readwrite("max_unmatched_times", &TrackerConfig::max_unmatched_times_)
          .def_readwrite("track_confirmed_frames", &TrackerConfig::track_confirmed_frames_)
          .def_readwrite("track_init_score_thresh", &TrackerConfig::track_init_score_thresh_)
          .def_readwrite("high_score_thresh", &TrackerConfig::high_score_thresh_)
          .def_readwrite("high_score_iou_dist_thresh", &TrackerConfig::high_score_iou_dist_thresh_)
          .def_readwrite("low_score_iou_dist_thresh", &TrackerConfig::low_score_iou_dist_thresh_);

      // 绑定 TrackerInfo 类
      py::class_<TrackerInfo>(tracker, "TrackerInfo")
          .def(py::init<>())
          .def_readwrite("box_info", &TrackerInfo::box_info_)
          .def_readwrite("status", &TrackerInfo::status_)
          .def_readwrite("obj_idx", &TrackerInfo::obj_idx_)
          .def_readwrite("track_id", &TrackerInfo::track_id_)
          .def_readwrite("velocity_x", &TrackerInfo::velocity_x_)
          .def_readwrite("velocity_y", &TrackerInfo::velocity_y_);

      // 绑定 PyTracker 类
      py::class_<PyTracker>(tracker, "Tracker")
          .def(py::init<TrackerType>(), py::arg("type"))
          .def("set_pair_config", &PyTracker::setPairConfig, py::arg("object_pair_config"))
          .def("set_track_config", &PyTracker::setTrackConfig, py::arg("track_config"))
          .def("get_track_config", &PyTracker::getTrackConfig)
          .def("track", &PyTracker::track, py::arg("boxes"), py::arg("frame_id"), py::arg("frame_time"))
          .def("set_img_size", &PyTracker::setImgSize, py::arg("width"), py::arg("height"));

      py::class_<TrackerFactory>(m, "TrackerFactory")
          .def_static("createTracker", &TrackerFactory::createTracker);
    ```

  - **在./tdl_sdk/src/python/include/py_model.hpp头文件，完成python绑定层接口定义。**
  
    *主要是写一个构造函数将c++中相关功能以python可调用的方式进行封装*

    以tracker模块为例，构造函数创建PyTracker类的实例，根据传入的type参数（类型为TrackerType的枚举值，第一步已经绑定过），通过TrackerFactory创建实际的跟踪器对象（第一步已绑定），并存储在成员变量tracker_中
  
    ```cpp
      class PyTracker {
      public:
        PyTracker(TrackerType type);
        void setPairConfig(const std::map<TDLObjectType, TDLObjectType>& object_pair_config);
        void setTrackConfig(const TrackerConfig& track_config);
        TrackerConfig getTrackConfig();
        py::list track(const py::list& boxes, uint64_t frame_id, float frame_time);
        void setImgSize(int width, int height);

      private:
        std::shared_ptr<Tracker> tracker_;
      };
    ```

  - **在./tdl_sdk/src/python/src/py_model.cpp中完成python绑定层接口的实现。**

    - 以tracker模块为例，根据第二步的构造函数**初始化跟踪器**
  
    ```cpp
      PyTracker::PyTracker(TrackerType type) {
      tracker_ = TrackerFactory::createTracker(type);
      }
      void PyTracker::setPairConfig(const std::map<TDLObjectType, TDLObjectType>& object_pair_config) {
      tracker_->setPairConfig(object_pair_config);
      }
      void PyTracker::setTrackConfig(const TrackerConfig& track_config) {
        tracker_->setTrackConfig(track_config);
      }
      TrackerConfig PyTracker::getTrackConfig() {
        return tracker_->getTrackConfig();
      }
    ```

    - **明确模块中函数的输入**，例如在tracker模块中，输入为目标检测框信息与帧id

      *将python的输入转换为c++*，例如python传来的目标框列表boxes转换为c++中的objectBoxInfo结构体列表
  
    ```cpp
        py::list PyTracker::track(const py::list& boxes, uint64_t frame_id, float frame_time) {
        static std::unordered_map<size_t, ObjectBoxInfo> prev_boxes;
        std::vector<ObjectBoxInfo> box_vec;
        for (size_t i = 0; i < boxes.size(); ++i) {
          py::dict box_dict = boxes[i].cast<py::dict>();
          ObjectBoxInfo box_info;
          box_info.x1 = box_dict["x1"].cast<float>();
          box_info.y1 = box_dict["y1"].cast<float>();
          box_info.x2 = box_dict["x2"].cast<float>();
          box_info.y2 = box_dict["y2"].cast<float>();
          box_info.class_id = box_dict["class_id"].cast<int>();
          box_info.score = box_dict["score"].cast<float>();
          box_info.object_type = static_cast<TDLObjectType>(box_info.class_id);
          box_vec.push_back(box_info);
        }
    ```

    - **调用c++逻辑**

      以tracker模块为例，调用底层跟踪器的track方法进行目标跟踪（具体track方法在mot.cpp文件）

    ```cpp
        std::vector<TrackerInfo> trackers;
        tracker_->track(box_vec, frame_id, trackers);
    ```

    - **将c++的结果转换为python格式**

      tracker模块中额外添加了速度判断逻辑，根据具体需要具体分析

    ```cpp
        py::list result;
        for (auto& tracker : trackers) {
          py::dict tracker_dict;
          py::dict box_dict;
          box_dict["x1"] = tracker.box_info_.x1;
          box_dict["y1"] = tracker.box_info_.y1;
          box_dict["x2"] = tracker.box_info_.x2;
          box_dict["y2"] = tracker.box_info_.y2;
          box_dict["class_id"] = tracker.box_info_.class_id;
          box_dict["score"] = tracker.box_info_.score;
          tracker_dict["box_info"] = box_dict;
          tracker_dict["status"] = static_cast<int>(tracker.status_);
          tracker_dict["obj_idx"] = tracker.obj_idx_;
          tracker_dict["track_id"] = tracker.track_id_;
          float velocity_x = std::isinf(tracker.velocity_x_) ? 0.0f : tracker.velocity_x_;
          float velocity_y = std::isinf(tracker.velocity_y_) ? 0.0f : tracker.velocity_y_;
          tracker_dict["velocity_x"] = velocity_x;
          tracker_dict["velocity_y"] = velocity_y;
          result.append(tracker_dict);
          prev_boxes[tracker.track_id_] = tracker.box_info_;
        }
        return result;
    ```

## 完成模型或功能类的新增后

- 根据编译指南在盒子上编译

    [编译指南](https://github.com/sophgo/tdl_sdk/blob/master/docs/getting_started/build.md)

- 编译成功后，设计测试样例，测试功能是否正常
