# code structure

## 总体结构

```
tdl_sdk/
  ├── CMakeLists.txt                 # 根 CMake 构建脚本
  ├── build_tdl_sdk.sh               # SDK 构建脚本
  ├── clang-format.sh                # 代码格式化脚本
  ├── clang-tidy.sh                  # 代码静态分析脚本
  ├── .clang-format                  # Clang 格式化配置
  ├── .clang-tidy                    # Clang 静态分析配置
  ├── cmake/                         # CMake 相关模块
  │   ├── opencv.cmake               # OpenCV 依赖查找
  │   ├── middleware.cmake           # 中间件依赖查找
  │   ├── mlir.cmake                 # MLIR 依赖查找
  │   └── thirdparty.cmake           # 第三方库依赖查找
  ├── docs/                          # 文档目录
  │   ├── README.md                  # 文档说明
  │   ├── LICENSE                    # 许可证文件
  │   ├── getting_started/           # 入门指南
  │   ├── developer_guide/           # 开发者文档
  │   ├── api_reference/             # API 参考手册
  │   └── images/                    # 文档图片资源
  ├── include/                       # **对外导出的头文件**
  │   ├── framework/                 # **框架层 API**
  │   │   ├── model/                 # 模型相关接口
  │   │   ├── preprocess/            # 预处理相关接口
  │   │   ├── tensor/                # 张量相关接口
  │   │   ├── memory/                # 内存管理相关接口
  │   │   ├── image/                 # 图像处理相关接口
  │   │   └── common/                # 公共定义和工具
  │   ├── components/                # **组件 API**
  │   ├── nn/                        # **神经网络模型 API**
  │   └── c_apis/                    # **C 语言 API**
  ├── src/                           # **内部实现**
  │   ├── framework/                 # **框架层实现**
  │   │   ├── model/                 # 模型实现
  │   │   ├── preprocess/            # 预处理实现
  │   │   ├── tensor/                # 张量处理实现
  │   │   ├── memory/                # 内存管理实现
  │   │   ├── image/                 # 图像处理实现
  │   │   └── common/                # 公共工具实现
  │   ├── components/                # **组件实现**
  │   │   ├── nn/                    # 神经网络模型实现
  │   │   ├── tracker/               # 目标跟踪实现
  │   │   ├── capture/               # 目标选优实现
  │   │   ├── drawer/                # 绘图实现
  │   │   └── camera/                # 相机和解码实现
  │   ├── c_apis/                    # **C API 封装**
  │   └── python/                    # **Python 绑定**
  ├── sample/                        # **示例代码**
  │   ├── cpp/                       # C++ 示例
  │   ├── c/                         # C 示例
  │   └── python/                    # Python 示例
  ├── regression/                    # **单元测试**
  ├── evaluation/                    # **性能评估**
  ├── tool/                          # **工具集**
  ├── toolchain/                     # **工具链**
  ├── scripts/                       # **脚本**
  └── README.md                      # 项目说明
```

## 模块说明

### framework

为实现跨平台模型推理的统一框架，基于此框架部署的模型，可以在多种硬件平台运行。该框架包括的模块：

* **image**
  * 图像类的抽象封装，支持多种图像格式和数据类型
  * 提供图像读取、转换、处理等基础功能
  * 支持 OpenCV、VPSS 等多种后端实现

* **tensor**
  * 张量类的抽象封装，用于表示神经网络模型的输入输出
  * 支持多种数据类型和内存布局
  * 提供张量操作和转换功能

* **memory**
  * 内存池类的抽象封装，用于高效内存管理
  * 支持多种内存类型（系统内存、设备内存等）
  * 提供内存分配、释放和复用机制

* **model**
  * 模型类的抽象封装，用于加载和运行神经网络模型
  * 支持多种模型格式（ONNX、TensorFlow、PyTorch等）
  * 提供模型推理和优化功能

* **preprocess**
  * 预处理类的抽象封装，用于图像预处理
  * 支持多种预处理操作（缩放、裁剪、归一化等）
  * 提供 OpenCV、VPSS 等多种后端实现

* **common**
  * 公共定义和工具，包括错误码、日志、配置等
  * 提供跨模块使用的通用功能

### components

具体算法相关的组件，包括：

* **nn**
  * 各种神经网络模型实现，如目标检测、人脸检测、车牌识别等
  * 提供模型加载、推理和结果解析功能

* **tracker**
  * 目标跟踪算法实现
  * 支持多种跟踪算法（KCF、SORT、DeepSORT等）

* **capture**
  * 目标选优算法实现
  * 用于从多个目标中选择最优目标

* **drawer**
  * 绘图功能实现
  * 支持绘制检测框、关键点、文本等

* **camera**
  * 相机和解码功能实现
  * 支持多种相机接口和解码格式

### c_apis

C 语言 API 封装，提供跨语言调用支持：

* 提供与 C++ API 功能对应的 C 接口
* 支持 C 语言应用程序集成
* 提供内存管理和错误处理机制

### python

Python 绑定，提供 Python 语言调用支持：

* 使用 pybind11 实现 C++ 到 Python 的绑定
* 提供与 C++ API 功能对应的 Python 接口
* 支持 NumPy 数组与张量的互转

### sample

示例代码，展示 SDK 的使用方法：

* **cpp**
  * C++ 语言示例，展示框架层和组件层的使用方法
  * 包括模型加载、图像处理、目标检测等示例

* **c**
  * C 语言示例，展示 C API 的使用方法
  * 包括模型加载、图像处理、目标检测等示例

* **python**
  * Python 语言示例，展示 Python 绑定的使用方法
  * 包括模型加载、图像处理、目标检测等示例

### regression

单元测试，确保 SDK 功能的正确性：

* 使用 Google Test 框架
* 包括框架层、组件层、C API 和 Python 绑定的测试
* 提供自动化测试和回归测试功能

### evaluation

性能评估，用于评估 SDK 的性能：

* 提供性能测试和基准测试功能
* 支持多种性能指标（吞吐量、延迟、内存使用等）
* 提供性能分析和优化建议

### tool

工具集，提供开发和调试支持：

* 模型转换工具
* 性能分析工具
* 调试和日志工具

### toolchain

工具链，提供编译和构建支持：

* 交叉编译工具链
* 依赖库和头文件
* 构建脚本和配置

### scripts

脚本，提供自动化支持：

* 构建脚本
* 测试脚本
* 部署脚本
