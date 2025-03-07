# code structure

## 总体结构

```
tdl_sdk/
  ├── CMakeLists.txt                 # 根 CMake 构建脚本
  ├── cmake/                           # CMake 相关模块
  │   ├── opencv.cmake               # OpenCV 依赖查找
  │   ├── middleware.cmake            # 中间件依赖查找
  │   ├── mlir.cmake                 # MLIR 依赖查找
  │   ├── thirdparty.cmake            # 第三方库依赖查找
  ├── docs/                          # 文档目录
  │   ├── README.md                  # 项目说明和快速上手指南
  │   ├── index.md                 # 文档首页，目录索引，可作为在线文档的首页入口
  │   ├── getting_started/         # 入门指南，帮助用户快速上手
  │   │   ├── installation.md      # 安装说明：依赖项安装、环境配置、SDK安装等
  │   │   ├── build.md             # 构建指南：使用 CMake 编译项目的步骤说明
  │   │   └── quick_start.md       # 快速开始：最小示例、运行示例程序的步骤
  │   ├── user_guide/              # 用户手册，面向最终用户的使用说明
  │   │   ├── usage.md             # 应用使用说明：如何调用 API、运行插件及示例程序
  │   │   ├── configuration.md     # 配置说明：如何通过外部配置文件指定模型信息等
  │   │   └── examples.md          # 常见用例：展示典型使用场景和操作步骤
  │   ├── developer_guide/         # 开发者文档，面向内部和第三方开发者
  │   │   ├── code_structure.md    # 代码结构说明：项目目录和模块划分的详细介绍
  │   │   ├── coding_conventions.md# 代码风格和约定：推荐的编程规范和最佳实践
  │   │   └── contribution.md      # 贡献指南：如何提交代码、报告问题及参与社区讨论
  │   ├── design_docs/             # 架构设计文档，记录系统整体设计与决策
  │   │   ├── architecture_overview.md # 系统架构概览：整体设计思想、模块交互关系等
  │   │   ├── module_design.md     # 模块详细设计：各模块的接口、实现思路和扩展说明
  │   │   └── api_design.md        # API 与插件接口设计：接口规范、数据结构说明等
  │   ├── api_reference/           # API 参考手册（可自动生成，也可手工编写）
  │   │   └── index.md             # API 文档入口：按照模块或命名空间组织的接口说明
  │   └── tutorials/               # 教程，提供实践操作的分步指南
  │       └── tutorial1.md         # 教程1：例如如何使用 SDK 与 OpenCV 进行图像处理
  ├── include/                       # **对外导出的头文件**
  │   └── tdl_sdk/                   # 项目主头文件目录
  │       ├── framework/             # **框架层 API**
  │       │   ├── model/
  │       │   │   ├── base_model.hpp         # 抽象基类 
  │       │   │   └── llm_model.hpp          # LLM 模型 
  │       │   ├── preprocess/
  │       │   │   └── base_preprocess.hpp    # 预处理基类 
  │       │   ├── tensor/
  │       │   │   └── base_tensor.hpp        # 张量类 
  │       │   ├── memory/
  │       │   │   └── base_memory_pool.hpp   # 内存池 
  │       │   ├── image/
  │       │   │   └── base_image.hpp         # 图像类 
  │       │   └── common/                    # 公共工具 
  │       ├── app/                    # app 
  |       └── c_apis/                     # **应用 API**
  ├── src/                           # **内部实现**
  │   ├── framework/                 # **框架层**
  │   │   ├── model/                 # 具体模型实现 
  │   │   ├── preprocess/             # 预处理实现 
  │   │   ├── tensor/                 # 张量处理实现 
  │   │   ├── memory/                 # 内存管理 
  │   │   ├── image/                  # 图像处理 
  │   │   ├── common/                 # 公共工具 
  │   │   └── utils/                  # 内部工具类 
  │   ├── components/                 # **插件模块 **
  │   │   ├── nn/                 # **模型插件**
  │   │   │   ├── object_detection/   # 目标检测模型
  │   │   │   ├── face_detection/     # 人脸检测模型
  │   │   │   └── plate_detection/    # 车牌检测模型
  │   │   ├── tracker/                # 跟踪算法
  │   │   ├── capture/                # 目标选优
  │   │   ├── drawer/                 # 绘图
  │   │   └── camera/                 # 解码
  │   ├── pipeline/                  # **流式并行框架**
  │   ├── app/                       # **应用层**
  │   ├── c_apis/                    # **C API 封装**
  │   └── python/                    # **Python 绑定**
  ├── regression/                     # **单元测试**
  │   ├── CMakeLists.txt              # Google Test 构建
  │   ├── reg_objdet.cpp              # 目标检测测试
  │   ├── reg_face.cpp                # 人脸检测测试
  │   └── reg_ocr.cpp                 # OCR 测试
  ├── examples/                        # **示例代码**
  │   ├── cpp/                         # C++ 示例
  │   ├── c/                           # C 示例
  │   └── python/                      # Python 示例
  │       ├── demo_objdet.py
  │       ├── demo_face.py
  │       └── demo_ocr.py
  ├── third_party/                     # **第三方库**
  ├── scripts/                         # 脚本
  └── README.md                        # 项目说明


```

## 模块

### framework

为实现跨平台模型推理的统一框架，基于此框架部署的模型，可以在多种硬件平台运行。该框架包括的模块：

* image
  * 图像类的抽象封装
* tensor
  * 张量类的抽象封装
* memory
  * 内存池类的抽象封装
* model
  * 模型类的抽象封装
* preprocess
  * 预处理类的抽象封装
* common
  * 公共定义

### components

具体算法相关的组件，包括：

* nn
  * 各种神经网络模型类
* tracker
  * 跟踪算法
* capture
  * 目标选优
* drawer
  * 绘图
* camera
  * 解码

### pipeline

用于将多个组件串联起来，形成一个完整的流程。

### preprocess

### tensor

## layer

## operator
