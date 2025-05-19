# 集成新模型

借助tdl_sdk框架，可以使新模型集成变得更加简单，框架可以帮助完成如下工作：

* 图像预处理
  * 用户只需配置预处理参数，框架内会自动调用预处理设备实现预处理
* 模型推理，无需编写任何推理相关代码
* 内存管理，框架内会自动管理内存，用户无需担心内存泄漏问题

还具有如下优势：

* 支持多平台，用户无需关心平台差异
* 性能、资源优化
* 硬件资源充分利用

## 现有模型类添加新模型文件

现有的模型已经在tdl_sdk中实现，只是需要在模型工厂中添加对应的新模型ID方便调用，操作步骤如下：

* 在[tdl_model_list.h](../../include/nn/tdl_model_list.h)中添加新模型的model_id
* 在[tdl_model_factory.cpp](../../src/components/nn/tdl_model_factory.cpp)中添加新模型的创建函数
* 在[model_factory.json](../../configs/model/model_factory.json)中添加新模型的配置信息

## 集成新的模型类型

* 在src/componets/nn目录下
  * 根据新模型的任务类型，选择合适的文件夹，假如没有匹配的，就新建一个该任务类型的文件夹
  * 在文件夹内添加新模型的头文件和源文件
    * 头文件派生自[base_model.hpp](../../include/framework/model/base_model.hpp)
    * 源文件实现[outputParse](../../include/framework/model/base_model.hpp)函数
* 创建新的模型ID
  * 在[tdl_model_list.h](../../include/nn/tdl_model_list.h)中添加新模型的model_id
* 更新模型工厂
  * 在[tdl_model_factory.cpp](../../src/components/nn/tdl_model_factory.cpp)中添加新模型的创建函数
  * 在[model_factory.json](../../configs/model/model_factory.json)中添加新模型的配置信息
* 编译
  * 参考[编译自有模型](run_private_model.md)
