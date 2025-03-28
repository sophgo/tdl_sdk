# TDL_SDK 公版深度学习软件包（Turnkey Deep Learning SDK）

TDL_SDK 是全自研的基于算能Middleware及TPU SDK的通用深度学习软件包，
方便用户快速部署模型并测试，旨在为用户提供高效、便捷的模型部署解决方案。

![TDL SDK系统框架](docs/api_reference/source/Design_Overview/media/Design002.png)
<p align="center"><b>图 1: TDL SDK 框架</b></p>

支持算能科技发布的多种平台和芯片架构。
包括但不限于：
- CV180X
- CV181X
- CV186AH/BM1688
- BM1684/BM1684X

当前已完成对

- V410 SDK
- BM1688 & CV186AH SDK

等平台的SDK支持，只需将tdl_sdk置在对应版本的项目根路径中便可使用，后续还会新增更多平台的支持。

## 文档链接

为了帮助用户快速上手 TDL_SDK，我们提供了以下指南文档：

| 文档名称       | 描述                                   |
|----------------|----------------------------------------|
| [TDL SDK编译指南](getting_started/build.md) | 详细介绍如何配置环境并编译 TDL_SDK。 |
| [TDL SDK运行指南](getting_started/run.md)   | 提供运行示例和测试模型的步骤说明。   |
| [TDL SDK开发指南](docs/api_reference/source/index.rst) | 提供 TDL SDK 的详细开发文档和 API 参考。 |

## 模型获取

算能TPU SDK支持两种模型文件格式：

- *.bmodel：基于libsophon中的bmruntime推理的模型结构；
- *.cvimodel：基于cviruntime推理的模型结构

我们已为大家准备好了经过优化的模型文件存放于 [tdl_models](https://github.com/sophgo/tdl_models) 仓库中，可直接拿来使用。

``` shell
git clone https://github.com/sophgo/tdl_models.git
```

## 测试TDL_SDK

TDL_SDK为每一种任务场景及模型都提供了三种风格的sample：

- c
- cpp
- python

用户可以根据自己的喜好选择运行的sample。

## 版本发布

| 版本号   | 发布时间       | 更新内容                                   |
|----------|----------------|--------------------------------------------|
| v2.0     | 2025-03-31     | 初始版本发布，支持 V410 SDK 和 BM1688 SDK。 |
