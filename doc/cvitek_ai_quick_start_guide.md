![一張含有 文字美工圖案 的圖片 自動產生的描述](./assets/2c8a4d93bc5177fa7ebc73a864153354.png)
# CVITEK AISDK软件开发参考
>Version: 1.1.0
>
>SDK Version: 3.0.0
>
>Release date: 2022-6-15


© 2020 北京晶视智能科技有限公司

本文件所含信息归<u>北京晶视智能科技有限公司</u>所有。

未经授权，严禁全部或部分复制或披露该等信息。

## 版本纪录

| 版本  | 日期      | 修订说明         | 修订人              |
|-------|-----------|------------------|---------------------|
| 1.0.0 | 2021/6/30 | 初稿             | AI Application Team |
| 1.0.1 | 2022/2/11 | 新增API 范例说明 | AI Application Team |
| 1.1.0 | 2022/6/15 | 新增API          | AI Application Team |


## 法律声明

本数据手册包含北京晶视智能科技有限公司（下称“晶视智能”）的保密信息。未经授权，禁止使用或披露本数据手册中包含的信息。如您未经授权披露全部或部分保密信息，导致晶视智能遭受任何损失或损害，您应对因之产生的损失/损害承担责任。

本文件内信息如有更改，恕不另行通知。晶视智能不对使用或依赖本文件所含信息承担任何责任。

本数据手册和本文件所含的所有信息均按“原样”提供，无任何明示、暗示、法定或其他形式的保证。晶视智能特别声明未做任何适销性、非侵权性和特定用途适用性的默示保证，亦对本数据手册所使用、包含或提供的任何第三方的软件不提供任何保证；用户同意仅向该第三方寻求与此相关的任何保证索赔。此外，晶视智能亦不对任何其根据用户规格或符合特定标准或公开讨论而制作的可交付成果承担责任。
[toc]

# 1. 功能概述

## 1.1. 目的

CVITek所提供的AI集成算法，用以缩短应用程序开发所需的时间。此架构实现了AI所需算法包含其前后处理，提供统一且便捷的编程界面。目前AI SDK包含物件侦测, 人脸辨识, 人脸识别, 行人重识别, 追踪, 语意分割, 车牌辨识, 车牌检测, 双目活体识别, 姿态检测等算法。

# 2. 设计概述

## 2.1. 系统架构

下图为AI SDK系统架构图；AI SDK架构在CVITek的Middleware及TPU SDK上。主要分为三大模块：Core，Service，Application。Core主要提供算法相关接口，封装复杂的底层操作及算法细节。令使用者可以直接使用VI或VPSS取得的Video Frame Buffer进行模型推理。AI SDK内部会对模型进行相应的前后处理，并完成推理。Service提供算法相关辅助API，例如：绘图, 特征比对, 区域入侵判定等功能。Application封装应用逻辑，目前包含人脸抓拍的应用逻辑。

![](assets/e0d5602295abc5cd77c4258be726ab4b.png)

Figure 1.

这三个模块分别放在三个Library中:

| 模块        | 静态库             | 动态库                   |
|-------------|--------------------|--------------------------|
| Core        | libcvai_core.so    | libcvai_core-static.a    |
| Service     | libcvai_service.so | libcvai_service-static.a |
| Application | libcvai_app.so     | libcvai_app-static.a     |

## 2.2. 档案结构

AI SDK档案结构如下：

| 目录名称 | 说明               |
|----------|--------------------|
| include  | AI SDK headers     |
| sample   | 范例代码           |
| doc      | Markdown格式文档   |
| lib      | AI SDK静态和动态库 |
| bin      | AI SDK sample      |

# 3. API参考

## 3.1. 句柄
```C++
typedef void \*cviai_handle_t;
typedef void \*cviai_service_handle_t;
```
**[说明]**
AI SDK的句柄，不同模块之间有各自的句柄，但是创建cviai_service_handle_t模块时会需要使用到cviai_handle_t作为输入。

## 3.2. CVI_AI_Core

### 3.2.1. Common

#### 3.2.1.1. CVI_AI_CreateHandle
```
CVI_S32 CVI_AI_CreateHandle(cviai_handle_t \*handle);
```
**[说明]**

创建使用AI SDK 句柄。AI SDK会自动创建VPSS Group。

**[参数]**

|       | 资料型态         | 参数名称 | 说明         |
|-------|------------------|----------|--------------|
| 进/出 | cviai_handle_t\* | handle   | 输入句柄指标 |

#### 3.2.1.2. CVI_AI_CreateHandle2
```shell
CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t \*handle, const VPSS_GRP vpssGroupId, const CVI_U8 vpssDev);
```
**[说明]**

创建使用AI SDK句柄，并使用指定的VPSS Group ID及Dev ID创建VPSS。

**[参数]**

|    | 资料型态         | 参数名称    | 说明               |
|----|------------------|-------------|--------------------|
| 出 | cviai_handle_t\* | handle      | 输入句柄指标       |
| 进 | VPSS_GRP         | vpssGroupId | VPSS使用的group id |
| 进 | CVI_U8           | vpssDev     | VPSS Device id     |

#### 3.2.1.3. CVI_AI_DestroyHandle
```shell
CVI_S32 CVI_AI_DestroyHandle(cviai_handle_t handle);
```
**[说明]**

销毁创造的句柄cviai_handle_t。同时销毁所有开启的模型

**[参数]**

|    | 资料型态       | 参数名称 | 说明     |
|----|----------------|----------|----------|
| 进 | cviai_handle_t | handle   | 输入句柄 |

#### 3.2.1.4. CVI_AI_GetModelPath
```C++
const char \*CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, char \*\*filepath);
```
**[说明]**

取得已经设置的内建支援模型函式的模型路径。使用完毕需要自行释放filepath之变量。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明   |
|----|--------------------------|----------|--------|
| 进 | cviai_handle_t           | handle   | 句柄   |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型ID |

**[输出]**

|    | 资料型态 | 说明         |
|----|----------|--------------|
| 出 | char\*   | 模型路径指标 |

#### 3.2.1.5. CVI_AI_OpenModel
```C++
CVI_S32 CVI_AI_OpenModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const char \*filepath);
```
**[说明]**

开启并初始化模型。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明             |
|----|--------------------------|----------|------------------|
| 进 | cviai_handle_t           | handle   | 句柄             |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型 index       |
| 进 | const char \*            | filepath | cvimodel模型路径 |

#### 3.2.1.6. CVI_AI_SetSkipVpssPreprocess
```C++
CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, bool skip);
```
**[说明]**

指定model不进行预处理。AI SDK在默认情况下会使用内部创建的VPSS进行模型的预处理(skip = false)。当skip为true时，AI SDK将不会对该模型进行预处理。模型输入必须由外部进行预处理后，再输入模型。通常用于VI直接Binding VPSS且只使用单一模型的状况。可以使用CVI_AI_GetVpssChnConfig来取得模型的VPSS预处理参数。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明           |
|----|--------------------------|----------|----------------|
| 进 | cviai_handle_t           | handle   | 句柄           |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型ID         |
| 进 | bool                     | skip     | 是否跳过前处理 |

#### 3.2.1.7. CVI_AI_GetSkipVpssPreprocess
```C++
CVI_S32 CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, bool \*skip);
```
**[说明]**

询问模型是否会在AI SDK内进行预处理。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明           |
|----|--------------------------|----------|----------------|
| 进 | cviai_handle_t           | handle   | 句柄           |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型ID         |
| 出 | bool\*                   | skip     | 是否跳过前处理 |

#### 3.2.1.8. CVI_AI_SetVpssThread
```C++
CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const uint32_t thread);
```
**[说明]**

设置特定模型使用的线程id。在AI SDK内，一个Vpss Thread代表一组Vpss Group设置。默认使用Thread 0为模型使用的Vpss Group。当在多线程上各自使用同一个AI SDK Handle来进行模型推理时，必须使用此API指定不同的Vpss Thread来避免Race Condition。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明   |
|----|--------------------------|----------|--------|
| 进 | cviai_handle_t           | handle   | 句柄   |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型ID |
| 进 | uint32_t                 | thread   | 线程id |

#### 3.2.1.9. CVI_AI_SetVpssThread2
```C++
CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const uint32_t thread, const VPSS_GRP vpssGroupId);
```
**[说明]**

同CVI_AI_SetVpssThread。可以指定Vpss Group ID。

**[参数]**

|    | 资料型态                 | 参数名称    | 说明          |
|----|--------------------------|-------------|---------------|
| 进 | cviai_handle_t           | handle      | 句柄          |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model       | 模型ID        |
| 进 | uint32_t                 | thread      | 线程id        |
| 进 | VPSS_GRP                 | vpssGroupId | VPSS Group id |

#### 3.2.1.10. CVI_AI_GetVpssThread
```
CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, uint32_t \*thread);
```
**[说明]**

取得模型使用的thread id。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明       |
|----|--------------------------|----------|------------|
| 进 | cviai_handle_t           | handle   | 句柄       |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型ID     |
| 出 | uint32_t\*               | thread   | VPSS线程id |

#### 3.2.1.11. CVI_S32 CVI_AI_GetVpssGrpIds
```C++
CVI_S32 CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP \*\*groups, uint32_t \*num);
```
**[说明]**

取得句柄内全部使用到的Vpss group id，使用完毕后groups要自行释放。

**[参数]**

|    | 资料型态       | 参数名称 | 说明         |
|----|----------------|----------|--------------|
| 进 | cviai_handle_t | handle   | 句柄         |
| 出 | VPSS_GRP \*\*  | groups   | 空指标的参考 |
| 出 | uint32_t\*     | num      | groups的长度 |

#### 3.2.1.12. CVI_AI_SetVpssTimeout
```C++
CVI_S32 CVI_AI_SetVpssTimeout(cviai_handle_t handle, uint32_t timeout);
```
**[说明]**

设置AI SDK等待VPSS硬件超时的时间，预设为100ms。此设置适用于所有AI SDK内的VPSS Thread。

**[参数]**

|    | 资料型态       | 参数名称 | 说明     |
|----|----------------|----------|----------|
| 进 | cviai_handle_t | handle   | 句柄     |
| 进 | uint32_t       | timeout  | 超时时间 |

#### 3.2.1.13. CVI_AI_SetVBPool

CVI_S32 CVI_AI_SetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL pool_id);

**[说明]**

指定VBPool给AI SDK内部VPSS。指定后，AI SDK内部VPSS会直接从此Pool拿取内存。若不用此API指定Pool，默认由系统自动分配。

**[参数]**

|    | 资料型态       | 参数名称 | 说明                                                                 |
|----|----------------|----------|----------------------------------------------------------------------|
| 进 | cviai_handle_t | handle   | 句柄                                                                 |
| 进 | uint32_t       | thread   | VPSS线程id                                                           |
| 进 | VB_POOL        | pool_id  | VB Pool Id。若设置为INVALID_POOLID，表示不指定Pool，由系统自动分配。 |

#### 3.2.1.14. CVI_AI_GetVBPool

CVI_S32 CVI_AI_SetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL \*pool_id);

**[说明]**

取得指定VPSS使用的VBPool Id。若未使用CVI_AI_SetVBPool指定Pool，则会得到INVALID_POOLID。

**[参数]**

|    | 资料型态       | 参数名称 | 说明                   |
|----|----------------|----------|------------------------|
| 进 | cviai_handle_t | handle   | 句柄                   |
| 进 | uint32_t       | thread   | VPSS线程id             |
| 出 | VB_POOL \*     | pool_id  | 目前使用的VB Pool Id。 |

#### 3.2.1.15. CVI_AI_CloseAllModel

CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle);

**[说明]**

卸载所有在句柄中已经加载的模型。

**[参数]**

|    | 资料型态       | 参数名称 | 说明 |
|----|----------------|----------|------|
| 进 | cviai_handle_t | handle   | 句柄 |

#### 3.2.1.16. CVI_AI_CloseModel
```C++
CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model);
```
**[说明]**

卸载特定在句柄中已经加载的模型。

**[参数]**

|    | 资料型态                 | 参数名称 | 说明      |
|----|--------------------------|----------|-----------|
| 进 | cviai_handle_t           | handle   | 句柄      |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model    | 模型index |

#### 3.2.1.17. CVI_AI_Dequantize

```C++
CVI_S32 CVI_AI_Dequantize(const int8_t \*quantizedData, float \*data, const uint32_t bufferSize, const float dequantizeThreshold);
```
**[说明]**

Dequantize int8数值到Float。

**[参数]**

|    | 资料型态        | 参数名称            | 说明          |
|----|-----------------|---------------------|---------------|
| 进 | const int8_t \* | quantizedData       | Int8数据      |
| 出 | float \*        | data                | Float输出数据 |
| 进 | const uint32_t  | bufferSize          | Int8数据数量  |
| 进 | const float     | dequantizeThreshold | 量化阀值      |

#### 3.2.1.18. CVI_AI_ObjectNMS

CVI_S32 CVI_AI_ObjectNMS(const cvai_object_t \*obj, cvai_object_t \*objNMS, const float threshold, const char method);
**说明**
对cviai_object_t内的bbox做Non-Maximum Suppression算法。
**[参数]**

|    | 资料型态               | 参数名称  | 说明                                                         |
|----|------------------------|-----------|--------------------------------------------------------------|
| 进 | const cvai_object_t \* | obj       | 欲进行NMS的Object Meta                                       |
| 出 | cvai_object_t \*       | objNMS    | NMS后的结果                                                  |
| 进 | const float            | threshold | IOU threshold                                                |
| 进 | const char             | method    | ‘u’: Intersection over Union ‘m’: Intersection over min area |

#### 3.2.1.19. CVI_AI_FaceNMS

CVI_S32 CVI_AI_ObjectNMS(const cvai_face_t \*face, cvai_face_t \*faceNMS, const float threshold, const char method);

**[说明]**

对cvai_face_t内的bbox做Non-Maximum Suppression算法。

**[参数]**

|    | 资料型态             | 参数名称  | 说明                                                         |
|----|----------------------|-----------|--------------------------------------------------------------|
| 进 | const cvai_face_t \* | face      | 欲进行NMS的face meta                                         |
| 出 | cvai_face_t \*       | faceNMS   | NMS后的结果                                                  |
| 进 | const float          | threshold | IOU threshold                                                |
| 进 | const char           | method    | ‘u’: Intersection over Union ‘m’: Intersection over min area |

#### 3.2.1.20. CVI_AI_FaceAlignment

CVI_S32 CVI_S32 CVI_AI_FaceAlignment(VIDEO_FRAME_INFO_S \*inFrame, const uint32_t metaWidth, const uint32_t metaHeight, const cvai_face_info_t \*info, VIDEO_FRAME_INFO_S \*outFrame, const bool enableGDC);

**[说明]**

对inFrame影像face进行Face Alignment，采用InsightFace Alignment参数。

**[参数]**

|    | 资料型态                  | 参数名称   | 说明                       |
|----|---------------------------|------------|----------------------------|
| 进 | VIDEO_FRAME_INFO_S \*     | inFrame    | 输入影像                   |
| 进 | const uint32_t metaWidth  | metaWidth  | Info中frame的宽度          |
| 进 | const uint32_t metaHeight | metaHeight | Info中frame的高度          |
| 进 | const cvai_face_info_t \* | info       | Face info                  |
| 出 | VIDEO_FRAME_INFO_S \*     | outFrame   | Face Alignment后的人脸影像 |
| 进 | const bool                | enableGDC  | 是否使用GDC硬件            |

#### 3.2.1.21. CVI_AI_CropImage

CVI_S32 CVI_S32 CVI_AI_CropImage(VIDEO_FRAME_INFO_S \*srcFrame, cvai_image_t \*dst, cvai_bbox_t \*bbox, bool cvtRGB888);

**[说明]**

从srcFrame影像中撷取bbox指定范围影像。

**[参数]**

|    | 资料型态              | 参数名称  | 说明                               |
|----|-----------------------|-----------|------------------------------------|
| 进 | VIDEO_FRAME_INFO_S \* | srcFrame  | 输入影像，目前仅支持RGB Packed格式 |
| 出 | cvai_image_t \*       | dst       | 输出影像                           |
| 进 | cvai_bbox_t \*        | bbox      | Bounding box                       |
| 进 | bool                  | cvtRGB888 | 是否转换成RGB888格式输出           |

#### 3.2.1.22. CVI_AI_CropImage_Face
```C++
CVI_S32 CVI_S32 CVI_AI_CropImage_Face(VIDEO_FRAME_INFO_S \*srcFrame, cvai_image_t \*dst, cvai_face_info_t \*face_info, bool align);
```
**[说明]**

从srcFrame影像中撷取face bbox指定范围影像。

**[参数]**

|    | 资料型态              | 参数名称  | 说明                                                    |
|----|-----------------------|-----------|---------------------------------------------------------|
| 进 | VIDEO_FRAME_INFO_S \* | srcFrame  | 输入影像，目前仅支持RGB Packed格式                      |
| 出 | cvai_image_t \*       | dst       | 输出影像                                                |
| 进 | cvai_face_info_t \*   | face_info | 指定的face info                                         |
| 进 | bool                  | align     | 是否进行face alignmen。采用InsightFace Alignment参数。  |
| 进 | bool                  | cvtRGB888 | 是否转换成RGB888格式输出                                |

#### 3.2.1.23. CVI_AI_SoftMax
```C++
CVI_S32 CVI_AI_SoftMax(const float \*inputBuffer, float \*outputBuffer, const uint32_t bufferSize);
```
**[说明]**

对inputBuffer计算Softmax。

**[参数]**

|    | 资料型态       | 参数名称     | 说明                |
|----|----------------|--------------|---------------------|
| 进 | const float \* | inputBuffer  | 欲进行softmax的缓冲 |
| 出 | const float \* | outputBuffer | Softmax后的结果     |
| 进 | const uint32_t | bufferSize   | 缓冲大小            |

#### 3.2.1.24. CVI_AI_GetVpssChnConfig
```C++
CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const CVI_U32 frameWidth, const CVI_U32 frameHeight, const CVI_U32 idx, cvai_vpssconfig_t \*chnConfig);
```
**[说明]**

取得在模型预处理使用的VPSS参数。

**[参数]**

|    | 资料型态                 | 参数名称    | 说明             |
|----|--------------------------|-------------|------------------|
| 进 | cviai_handle_t           | handle      | 句柄             |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model       | 模型id           |
| 进 | CVI_U32                  | frameWidth  | 输入影像宽       |
| 进 | CVI_U32                  | frameHeight | 输入影像高       |
| 进 | CVI_U32                  | idx         | 模型的输入index  |
| 出 | cvai_vpssconfig_t\*      | chnConfig   | 回传的参数设定值 |

#### 3.2.1.25. CVI_AI_Free
```C++
CVI_A_Free(X)
```
**[说明]**

释放模型结果产生的数据结构。某些巨树结构中包含malloc出来的子项，因此需要做释放。

**[参数]**

以下为支援的输入变量

-   `cvai_feature_t`
-   `cvai_pts_t`
-   `cvai_tracker_t`
-   `cvai_face_info_t`
-   `cvai_face_t`
-   `cvai_object_info_t`
-   `cvai_object_t`

#### 3.2.1.26. CVI_AI_CopyInfo
```C++
CVI_A_CopyInfo(IN, OUT)
```
**[说明]**

泛型拷贝cviai结构API。malloc内部的指标空间并做完整复制。

**[参数]**

|    | 资料型态                                                    | 参数名称 | 说明     |
|----|-------------------------------------------------------------|----------|----------|
| 进 | 支持型态： cvai_face_info_t cvai_object_info_t cvai_image_t | IN       | 复制来源 |
| 出 | 支持型态： cvai_face_info_t cvai_object_info_t cvai_image_t | OUT      | 复制目的 |

#### 3.2.1.27. CVI_AI_RescaleMetaCenter

**[说明]**

将结构内的坐标还原到与输入影像相同之大小，适用于padding影像为上下左右，

**[参数]**

以下为支援的输入变量

-   `cvai_face_t`
-   `cvai_object_t`

#### 3.2.1.28. CVI_AI_RescaleMetaRB

**[说明]**

将结构内的坐标还原到与输入影像相同之大小，适用于padding影像为右下，

**[参数]**

以下为支援的输入变量

-   cvai_face_t
-   cvai_object \_t

#### 3.2.1.29. getFeatureTypeSize

getFeatureTypeSize(feature_type_e type);

**[说明]**

取得特征值的单位大小。

**[参数]**

|      | 资料型态       | 参数名称 | 说明                 |
|------|----------------|----------|----------------------|
| 进   | feature_type_e | type     | 单位                 |
| 回传 | int            | X        | 单位为byte之单位大小 |

#### 3.2.1.30. CVI_AI_SetModelThreshold
```C++
CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, float threshold);
```
**[说明]**

设置模型阀值，目前仅支持针对Detection类型的模型进行设置。

**[参数]**

|    | 资料型态                 | 参数名称  | 说明           |
|----|--------------------------|-----------|----------------|
| 进 | cviai_handle_t           | handle    | 句柄           |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model     | 模型index      |
| 进 | float                    | threshold | 阀值(0.0\~1.0) |

#### 3.2.1.31. CVI_AI_GetModelThreshold
```C++
CVI_S32 CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, float \*threshold);
```
**[说明]**

取出模型阀值，目前仅支持Detection类型模型。

**[参数]**

|    | 资料型态                 | 参数名称  | 说明      |
|----|--------------------------|-----------|-----------|
| 进 | cviai_handle_t           | handle    | 句柄      |
| 进 | CVI_AI_SUPPORTED_MODEL_E | model     | 模型index |
| 出 | float \*                 | threshold | 阀值      |

### 3.2.2. 物件侦测

#### 3.2.2.1. CVI_AI_MobileDetV2_Vehicle

CVI_S32 CVI_AI_MobileDetV2_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

**[说明]**

使用MobilDetV2-Vehicle模型进行推理，此模型可侦测Car, Motorcycle, Truck三个类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.2. CVI_AI_MobileDetV2_Pedestrian

CVI_S32 CVI_AI_MobileDetV2_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

**[说明]**

使用MobilDetV2-Pedestrian系列模型进行推理，此模型可侦测person类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.3. CVI_AI_MobileDetV2_Person_Vehicle
```C++
CVI_S32 CVI_AI_MobileDetV2_Person_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);
```
**[说明]**

使用MobilDetV2-Person-Vehicle模型进行推理，此模型可侦测人车非类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.4. CVI_AI_MobileDetV2_Person_Pets
```C++
CVI_S32 CVI_AI_MobileDetV2_Person_Pets(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);
```
**[说明]**

使用MobilDetV2-Lite-Person-Pets模型进行推理，此模型可侦测person, cat, dog等类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.5. CVI_AI_MobileDetV2_COCO80

CVI_S32 CVI_AI_MobileDetV2_COCO80(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

**[说明]**

使用MobilDetV2 COCO80系列模型进行推理，此模型可侦测标准COCO dataset的 80个类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.6. CVI_AI_Yolov3

CVI_S32 CVI_AI_Yolov3 (cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

**[说明]**

使用YoloV3模型进行推理，此模型可侦测COCO 80个类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.7. CVI_AI_YoloX

CVI_S32 CVI_AI_YoloX(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

**[说明]**

使用YoloX模型进行推理，此模型可侦测COCO 80个类别。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.2.8. CVI_AI_SelectDetectClass

CVI_S32 CVI_AI_SelectDetectClass(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, uint32_t num_classes, ...)

**[说明]**

过滤Object Detection模型输出结果, 保留列举的类别或群组。类别为不定参数，数量根据num_classes而定。详细类别及群组Index可参考cvai_obj_class_id_e及cvai_obj_det_group_type_e。目前仅支持MobileDetV2, YoloX系列模型。

**[参数]**

|    | 资料型态                                        | 参数名称    | 说明                     |
|----|-------------------------------------------------|-------------|--------------------------|
| 进 | cviai_handle_t                                  | handle      | 句柄                     |
| 进 | CVI_AI_SUPPORTED_MODEL_E                        | model       | 模型Index                |
| 进 | uint32_t                                        | num_classes | 保留的类别个数           |
| 进 | cvai_obj_class_id_e或 cvai_obj_det_group_type_e | …           | 保留的Class ID或Group ID |

#### 3.2.2.9. CVI_AI_ThermalPerson

CVI_S32 CVI_AI_ThermalPerson(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

**[说明]**

热显影像人型。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_object_t \*      | faces    | 侦测到的人形 |

### 3.2.3. 人脸侦测

#### 3.2.3.1. CVI_AI_RetinaFace
```C++
CVI_S32 CVI_AI_RetinaFace(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);
```
**[说明]**

使用`RetinaFace`模型侦测人脸。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.3.2. CVI_AI_RetinaFace_IR
```C++
CVI_S32 CVI_AI_RetinaFace_IR(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);
```
**[说明]**

使用`RetinaFace`模型在IR影像中侦测人脸。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入IR影像   |
| 出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.3.3. CVI_AI_RetinaFace_Hardhat

CVI_S32 CVI_AI_RetinaFace_Hardhat(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

**[说明]**

使用RetinaFace模型侦测戴安全帽人脸。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入IR影像   |
| 出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.3.4. CVI_AI_ThermalFace

CVI_S32 CVI_AI_ThermalFace(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

****[说明]****

热显影像人脸侦测。

****[参数]****

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.3.5. CVI_AI_FaceQuality
```C++
CVI_S32 CVI_AI_FaceQuality(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces, bool \*skip);
```
**[说明]**

判断传入的faces结构中的人脸质量评估并同时侦测人脸角度。质量受人脸清晰程度与是否遮档影响。人脸质量分数为 faces-\>info[i].face_quality，人脸角度放在 faces-\>info[i].head_pose中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明                                                               |
|-------|-----------------------|----------|--------------------------------------------------------------------|
| 进    | cviai_handle_t        | handle   | 句柄                                                               |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像                                                           |
| 进/出 | cvai_face_t \*        | face     | 侦测到的人脸                                                       |
| 进    | bool \*               | skip     | Bool array, 指定哪个人脸需要做face quality。NULL表示全部人脸都做。 |

#### 3.2.3.6. CVI_AI_FaceMaskDetection

CVI_S32 CVI_AI_FaceMaskDetection(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

****[说明]****

侦测戴口罩人脸。人脸分数存放在faces-\>info[i].bbox.score，戴口罩人脸分数存放在faces-\>info[i].mask_score。

**[参数]**

|    | 资料型态              | 参数名称 | 说明         |
|----|-----------------------|----------|--------------|
| 进 | cviai_handle_t        | handle   | 句柄         |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.3.7. CVI_AI_MaskClassification

CVI_S32 CVI_AI_MaskClassification(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*face);

****[说明]****

判断传入的faces中的所有人脸是否为戴口罩人脸。呼叫此接口前，必须先执行一次人脸侦测。戴口罩人脸分数存放在faces-\>info[i].mask_score。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 进/出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

### 3.2.4. 人脸识别

#### 3.2.4.1. CVI_AI_FaceRecognition

CVI_S32 CVI_AI_FaceRecognition(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

****[说明]****

抽取人脸特征。此接口会针对face中所有人脸进行特征抽取。并放在faces-\>info[i].feature中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 进/出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.4.2. CVI_AI_FaceRecognitionOne

CVI_S32 CVI_AI_FaceRecognitionOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces, int face_idx);

****[说明]****

抽取人脸特征。此接口仅会针对指定的face index进行特征抽取。并放在faces-\>info[index].feature中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明                                         |
|-------|-----------------------|----------|----------------------------------------------|
| 进    | cviai_handle_t        | handle   | 句柄                                         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像                                     |
| 进/出 | cvai_face_t \*        | faces    | 侦测到的人脸                                 |
| 进    | int                   | face_idx | 欲进行特征抽取的face index。-1表示全部抽取。 |

#### 3.2.4.3. CVI_AI_FaceAttribute

CVI_S32 CVI_AI_FaceAttribute(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

****[说明]****

抽取人脸特征及人脸属性。此接口会针对face中所有人脸进行特征抽取及人脸属性。人脸属性包含：性别, 表情, 年龄及种族，结果分别放在faces-\>info[i].feature, faces-\>info[i].age, faces-\>info[i].emotion, faces-\>info[i].gender, faces-\>info[i].race。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 进/出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

#### 3.2.4.4. CVI_AI_FaceAttributeOne

CVI_S32 CVI_AI_FaceAttributeOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces, int face_idx);

****[说明]****

抽取人脸特征。此接口仅会针对指定的face index进行特征抽取。人脸属性包含：性别, 表情, 年龄及种族，结果分别放在faces-\>info[i].feature, faces-\>info[i].age, faces-\>info[i].emotion, faces-\>info[i].gender, faces-\>info[i].race。

**[参数]**

|       | 资料型态              | 参数名称 | 说明                                         |
|-------|-----------------------|----------|----------------------------------------------|
| 进    | cviai_handle_t        | handle   | 句柄                                         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像                                     |
| 进/出 | cvai_face_t \*        | faces    | 侦测到的人脸                                 |
| 进    | int                   | face_idx | 欲进行特征抽取的face index。-1表示全部抽取。 |

#### 3.2.4.5. CVI_AI_MaskFaceRecognition

CVI_S32 CVI_AI_MaskFaceRecognition(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

****[说明]****

抽取戴口罩人脸特征。此接口会针对face中所有人脸进行特征抽取。并放在faces-\>info[i].feature中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 进/出 | cvai_face_t \*        | faces    | 侦测到的人脸 |

### 3.2.5. 行人识别

#### 3.2.5.1. CVI_AI_OSNet

CVI_S32 CVI_AI_OSNet(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

****[说明]****

使用person-reid模型抽取行人特征。此接口会针对obj中所有的Person类别物件进行特征抽取。并放在obj-\>info[i].feature中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 进/出 | cvai_object_t \*      | obj      | 侦测到的物件 |

#### 3.2.5.2. CVI_AI_OSNetOne

CVI_S32 CVI_AI_OSNetOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj, int obj_idx);

****[说明]****

使用person-reid模型抽取行人特征。此接口仅会针对指定的obj物件进行特征抽取。并放在obj-\>info[i].feature中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明                                        |
|-------|-----------------------|----------|---------------------------------------------|
| 进    | cviai_handle_t        | handle   | 句柄                                        |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像                                    |
| 进/出 | cvai_object_t \*      | obj      | 侦测到的物件                                |
| 进    | int                   | obj_idx  | 欲进行特征抽取的物件index。-1表示全部抽取。 |

### 3.2.6. 物件追踪

#### 3.2.6.1. CVI_AI_DeepSORT_Init

CVI_S32 CVI_AI_DeepSORT_Init(const cviai_handle_t handle, bool use_specific_counter);

****[说明]****

初始化DeepSORT算法。

**[参数]**

|    | 资料型态       | 参数名称             | 说明                         |
|----|----------------|----------------------|------------------------------|
| 进 | cviai_handle_t | handle               | 句柄                         |
| 进 | bool           | use_specific_counter | 是否每一个物件类别各自分配id |

#### 3.2.6.2. CVI_AI_DeepSORT_GetDefaultConfig

CVI_S32 CVI_AI_DeepSORT_GetDefaultConfig(cvai_deepsort_config_t \*ds_conf);

****[说明]****

取得DeepSORT默认参数。

**[参数]**

|       | 资料型态                  | 参数名称 | 说明         |
|-------|---------------------------|----------|--------------|
| 进/出 | cvai_deepsort_config_t \* | ds_conf  | DeepSORT参数 |

#### 3.2.6.3. CVI_AI_DeepSORT_SetConfig

CVI_S32 CVI_AI_DeepSORT_SetConfig(const cviai_handle_t handle , cvai_deepsort_config_t \*ds_conf, int cviai_obj_type, bool show_config);

****[说明]****

设置DeepSORT参数。

**[参数]**

|       | 资料型态                  | 参数名称       | 说明                                                             |
|-------|---------------------------|----------------|------------------------------------------------------------------|
| 进    | cviai_handle_t            | handle         | 句柄                                                             |
| 进/出 | cvai_deepsort_config_t \* | ds_conf        | DeepSORT参数                                                     |
| 进    | int                       | cviai_obj_type | -1表示此为默认设置。非-1值表示针对cviai_obj_type的类别设置参数。 |
| 进    | bool                      | show_config    | 显示设定                                                         |

#### 3.2.6.4. CVI_AI_DeepSORT_GetConfig

CVI_S32 CVI_AI_DeepSORT_GetConfig(const cviai_handle_t handle , cvai_deepsort_config_t \*ds_conf, int cviai_obj_type);

****[说明]****

询问DeepSORT设置的参数。

**[参数]**

|    | 资料型态                  | 参数名称       | 说明                                                             |
|----|---------------------------|----------------|------------------------------------------------------------------|
| 进 | cviai_handle_t            | handle         | AI SDK句柄                                                       |
| 出 | cvai_deepsort_config_t \* | ds_conf        | DeepSORT参数                                                     |
| 进 | int                       | cviai_obj_type | -1表示取得默认参数。非-1值表示针对cviai_obj_type的类别设置的参数 |

#### 3.2.6.5. CVI_AI_DeepSORT_CleanCounter

CVI_S32 CVI_AI_DeepSORT_CleanCounter(const cviai_handle_t handle);

****[说明]****

清除DeepSORT 纪录的ID counter。

**[参数]**

|    | 资料型态       | 参数名称 | 说明 |
|----|----------------|----------|------|
| 进 | cviai_handle_t | handle   | 句柄 |

#### 3.2.6.6. CVI_AI_DeepSORT_Obj

CVI_S32 CVI_AI_DeepSORT_Obj(const cviai_handle_t handle, cvai_object_t \*obj, cvai_tracker_t \*tracker_t, bool use_reid);

****[说明]****

追踪物件，更新Tracker状态。此接口会赋予每个Object一个Unique ID。可从obj-\>info[i].unique_id取得。tracker_t会纪录DeepSORT对每个object的追踪状态及目前的预测Bounding Box。若欲使用物件外观特征进行追踪，需将use_reid设置true, 并在追踪之前使用CVI_AI_OSNet进行特征抽取。目前特征抽取只支持人型。

**[参数]**

|       | 资料型态          | 参数名称  | 说明                         |
|-------|-------------------|-----------|------------------------------|
| 进    | cviai_handle_t    | handle    | 句柄                         |
| 进/出 | cvai_object_t \*  | obj       | 欲进行追踪的物件             |
| 出    | cvai_tracker_t \* | tracker_t | 物件的追踪状态               |
| 进    | bool              | use_reid  | 是否使用物件外观特征进行追踪 |

#### 3.2.6.7. CVI_AI_DeepSORT_Face

CVI_S32 CVI_AI_DeepSORT_Face(const cviai_handle_t handle, cvai_face_t \*face, cvai_tracker_t \*tracker_t, bool use_reid);

****[说明]****

追踪人脸，更新Tracker状态。此接口会赋予每个人脸一个Unique ID。可从face-\>info[i].unique_id取得。tracker_t会纪录DeepSORT对每个人脸的追踪状态及目前的预测Bounding Box。若欲使用人脸特征进行追踪，use_reid须设置为true。并在追踪之前调用CVI_AI_FaceRecognition计算人脸特征。

**[参数]**

|       | 资料型态          | 参数名称  | 说明                                        |
|-------|-------------------|-----------|---------------------------------------------|
| 进    | cviai_handle_t    | handle    | 句柄                                        |
| 进/出 | cvai_face_t \*    | face      | 欲进行追踪的人脸                            |
| 出    | cvai_tracker_t \* | tracker_t | 人脸的追踪状态                              |
| 进    | bool              | use_reid  | 是否使用外观特征进行追踪。目前仅能设置false |

### 3.2.7. 运动侦测

#### 3.2.7.1. CVI_AI_Set_MotionDetection_Background

CVI_S32 CVI_AI_Set_MotionDetection_Background(const cviai_handle_t handle,

VIDEO_FRAME_INFO_S \*frame);

****[说明]****

设置Motion Detection背景，第一次运行此接口时会对Motion Detection进行初始化，后续再调用次接口仅会更新背景。AI SDK中Motion Detection使用帧差法。

**[参数]**

|    | 资料型态              | 参数名称 | 说明 |
|----|-----------------------|----------|------|
| 进 | cviai_handle_t        | handle   | 句柄 |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 背景 |

#### 3.2.7.2. CVI_AI_MotionDetection

CVI_S32 CVI_AI_MotionDetection (const cviai_handle_t handle,

VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*objects, uint32_t threshold, double min_area);

****[说明]****

使用帧差法侦测物件。侦测结果会存放在objects内。

**[参数]**

|    | 资料型态              | 参数名称  | 说明                                               |
|----|-----------------------|-----------|----------------------------------------------------|
| 进 | cviai_handle_t        | handle    | 句柄                                               |
| 进 | VIDEO_FRAME_INFO_S \* | frame     | 影像                                               |
| 出 | cvai_object_t \*      | object    | 运动侦测结果                                       |
| 进 | uint32_t              | threshold | 帧差法阀值，须为0-255                              |
| 进 | double                | min_area  | 最小物件面积(Pixels)，过滤掉小于此数值面积的物件。 |

### 3.2.8. 车牌辨识

#### 3.2.8.1. CVI_AI_LicensePlateDetection

CVI_S32 CVI_AI_LicensePlateDetection(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*vehicle_meta);

****[说明]****

车牌侦测。呼叫此API之前，必须先执行一次车辆侦测。此算法会在已侦测到的物件上进行车牌侦测。车牌位置会放在 obj-\>info[i].vehicle_properity-\>license_pts中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明               |
|-------|-----------------------|----------|--------------------|
| 进    | cviai_handle_t        | handle   | 句柄               |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 影像               |
| 进/出 | cvai_object_t \*      | obj      | 物件(车辆)侦测结果 |

#### 3.2.8.2. CVI_AI_LicensePlateRecognition_TW

CVI_AI_LicensePlateRecognition_TW(const cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

****[说明]****

对传入的obj中所有车辆进行车牌辨识(台湾)。呼叫此API之前，必须先调用CVI_AI_LicensePlateDetection执行一次车牌侦测。车牌号码储存在obj-\>info[i].vehicle_properity-\>license_char 。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 影像         |
| 进/出 | cvai_object_t \*      | obj      | 车牌侦测结果 |

#### 3.2.8.3. CVI_AI_LicensePlateRecognition_CN

CVI_AI_LicensePlateRecognition_CN(const cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

****[说明]****

对传入的obj中所有车辆进行车牌辨识(大陆)。呼叫此API之前，必须先调用CVI_AI_LicensePlateDetection执行一次车牌侦测。车牌号码储存在obj-\>info[i].vehicle_properity-\>license_char 。

**[参数]**

|       | 资料型态              | 参数名称 | 说明         |
|-------|-----------------------|----------|--------------|
| 进    | cviai_handle_t        | handle   | 句柄         |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 影像         |
| 进/出 | cvai_object_t \*      | obj      | 车牌侦测结果 |

### 3.2.9. 篡改侦测

#### 3.2.9.1. CVI_AI_TamperDetection

CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, float \*moving_score)

****[说明]****

摄影机篡改侦测。此算法基于高斯模型建立背景模型，并用去背法算出差值当作篡改分数(moving_score)。

**[参数]**

|    | 资料型态              | 参数名称      | 说明     |
|----|-----------------------|---------------|----------|
| 进 | cviai_handle_t        | handle        | 句柄     |
| 进 | VIDEO_FRAME_INFO_S \* | frame         | 影像     |
| 出 | float \*              | moveing_score | 篡改分数 |

### 3.2.10. 活体识别

#### 3.2.10.1. CVI_AI_Liveness

CVI_S32 CVI_AI_Liveness (const cviai_handle_t handle, VIDEO_FRAME_INFO_S \* rgbFrame, VIDEO_FRAME_INFO_S \* irFrame, , cvai_face_t \* rgb_faces, cvai_face_t \* ir_faces)

****[说明]****

RGB, IR双目活体识别。判断rgb_faces和ir_faces中的人脸是否为活体。活体分数置于 rgb_face -\>info[i].liveness_score 中。

**[参数]**

|       | 资料型态              | 参数名称 | 说明                      |
|-------|-----------------------|----------|---------------------------|
| 进    | cviai_handle_t        | handle   | 句柄                      |
| 进    | VIDEO_FRAME_INFO_S \* | rgbFrame | RGB影像                   |
| 进    | VIDEO_FRAME_INFO_S \* | irFrame  | IR影像                    |
| 进/出 | cvai_face_t \*        | rgb_meta | 侦测到的RGB人脸/ 活体分数 |
| 进    | cvai_face_t \*        | ir_meta  | 侦测到的IR人脸            |

### 3.2.11. 姿态检测

#### 3.2.11.1. CVI_AI_AlphaPose

CVI_S32 CVI_AI_AlphaPose (cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_object_t \*obj);

****[说明]****

使用alphapose模型进行推理，预测17个骨骼关键点。检测结果置于 obj-\>info[i].pedestrian_properity-\>pose_17。

**[参数]**

|        | 资料型态              | 参数名称 | 说明                             |
|--------|-----------------------|----------|----------------------------------|
| 进     | cviai_handle_t        | handle   | 句柄                             |
| 进     | VIDEO_FRAME_INFO_S \* | frame    | 输入影像                         |
| 进/出  | cvai_object_t \*      | obj      | 侦测到的人 /  17个骨骼关键点坐标 |

### 3.2.12. 语义分割

#### 3.2.12.1. CVI_AI_DeeplabV3

CVI_S32 CVI_AI_DeeplabV3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, VIDEO_FRAME_INFO_S \*out_frame, cvai_class_filter_t \*filter);

****[说明]****

使用DeepLab V3模型进行语义分割。

**[参数]**

|    | 资料型态              | 参数名称  | 说明       |
|----|-----------------------|-----------|------------|
| 进 | cviai_handle_t        | handle    | 句柄       |
| 进 | VIDEO_FRAME_INFO_S \* | frame     | 输入影像   |
| 出 | VIDEO_FRAME_INFO_S \* | out_frame | 输出影像   |
| 进 | cvai_class_filter_t\* | filter    | 保留的类别 |

### 3.2.13. 跌倒检测

#### 3.2.13.1. CVI_AI_Fall

CVI_S32 CVI_AI_Fall (cviai_handle_t handle, cvai_object_t \*obj);

****[说明]****

使用物件侦测与姿态检测之结果，判断跌倒状态。在运行此API前需要先调用CVI_AI_AlphaPose取得人体关键点。跌倒检测结果置于 obj-\>info[i].pedestrian_properity-\>fall 。

**[参数]**

|        | 资料型态              | 参数名称 | 说明         |
|--------|-----------------------|----------|--------------|
| 进     | cviai_handle_t        | handle   | 句柄         |
| 进     | VIDEO_FRAME_INFO_S \* | frame    | 输入影像     |
| 进/出  | cvai_object_t \*      | obj      | 跌倒状态结果 |

### 3.2.14. 驾驶疲劳检测

#### 3.2.14.1. CVI_AI_FaceLandmarker

CVI_S32 CVI_AI_FaceLandmarker (cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

****[说明]****

需先使用人脸检测，产生出106个人脸特征点检测的结果，将结果放入face-\>dms[i].landmarks_106 并且更新5个人脸特征点 face-\>dms[i].landmarks_5。

**[参数]**

|       | 资料型态              | 参数名称 | 说明     |
|-------|-----------------------|----------|----------|
| 进    | cviai_handle_t        | handle   | 句柄     |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像 |
| 进/出 | cvai_face_t \*        | face     | 人脸     |

#### 3.2.14.2. CVI_AI_EyeClassification

CVI_S32 CVI_AI_EyeClassification (cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

**[说明]**

需先使用人脸检测以及人脸特征点检测的结果，判断眼睛闭合状态，将结果放入face-\>dms[i].reye_score/ face-\>dms[i].leye_score。

**[参数]**

|       | 资料型态              | 参数名称 | 说明     |
|-------|-----------------------|----------|----------|
| 进    | cviai_handle_t        | handle   | 句柄     |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像 |
| 进/出 | cvai_face_t \*        | face     | 人脸     |

#### 3.2.14.3. CVI_AI_YawnClassification

CVI_S32 CVI_AI_YawnClassification (cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

**[说明]**

根据人脸检测和人脸关键点结果进行打哈欠检测。必须先调用CVI_FaceRecognition取得人脸检测和人脸关键点结果。打哈欠结果会放入face-\>dms[i].yawn_score 。分数为0.0\~1.0间。

**[参数]**

|       | 资料型态              | 参数名称 | 说明     |
|-------|-----------------------|----------|----------|
| 进    | cviai_handle_t        | handle   | 句柄     |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像 |
| 进/出 | cvai_face_t \*        | face     | 人脸     |

#### 3.2.14.4. CVI_AI_IncarObjectDetection

CVI_S32 CVI_AI_IncarObjectDetection(cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, cvai_face_t \*faces);

**[说明]**

使用物件侦测检测物件（水杯／马克杯／电话）是否出现在驾驶周边，将判断结果输出成object格式 ，放入到`face->dms[i].dms.dms_od`。

**[参数]**

|       | 资料型态              | 参数名称 | 说明     |
|-------|-----------------------|----------|----------|
| 进    | cviai_handle_t        | handle   | 句柄     |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像 |
| 进/出 | cvai_face_t \*        | face     | 人脸     |

### 3.2.15. 声音分类

#### 3.2.15.1. CVI_AI_SoundClassification
```C++
CVI_S32 CVI_AI_SoundClassification (cviai_handle_t handle, VIDEO_FRAME_INFO_S \*frame, int \*index);
```
**[说明]**

判断frame中音讯属于哪个类别。并将各类别分数排序后输出。

**[参数]**

|       | 资料型态              | 参数名称 | 说明           |
|-------|-----------------------|----------|----------------|
| 进    | cviai_handle_t        | handle   | 句柄           |
| 进    | VIDEO_FRAME_INFO_S \* | frame    | 输入影像       |
| 进/出 | int \*                | index    | 每个类别的分数 |

## 3.3. CVIAI_Service

### 3.3.1. CVI_AI_Service_CreateHandle
```C++
CVI_S32 CVI_AI_Service_CreateHandle (cviai_service_handle_t \*handle, cvai_handle ai_handle);
```
**[说明]**

创建Service句柄

**[参数]**

|    | 资料型态                  | 参数名称  | 说明            |
|----|---------------------------|-----------|-----------------|
| 进 | cviai_service_handle_t \* | handle    | 句柄            |
| 进 | cviai_handle_t            | ai_handle | cviai_core 句柄 |

### 3.3.2. CVI_AI_Service_DestroyHandle
```C++
CVI_S32 CVI_AI_Service_DestroyHandle (cviai_service_handle_t \*handle);
```
**[说明]**

销毁Service句柄

**[参数]**

|    | 资料型态                  | 参数名称 | 说明 |
|----|---------------------------|----------|------|
| 进 | cviai_service_handle_t \* | handle   | 句柄 |

### 3.3.3. CVI_AI_Service_Polygon_SetTarget
```C++
CVI_S32 CVI_AI_Service_Polygon_SetTarget(cviai_service_handle_t handle, const cvai_pts_t \*pts);
```
**[说明]**

设定区域入侵范围。pts为凸多边形点坐标，顺序需为顺直针或逆时针。调用CVI_AI_Service_Polygon_Intersect判断一个bounding box是否侵入已划定范围。

**[参数]**

|    | 资料型态                  | 参数名称 | 说明       |
|----|---------------------------|----------|------------|
| 进 | cviai_service_handle_t \* | handle   | 句柄       |
| 进 | cvai_pts_t \*             | pts      | 凸多边形点 |

### 3.3.4. CVI_AI_Service_Polygon_Intersect
```C++
CVI_S32 CVI_AI_Service_Polygon_Intersect(cviai_service_handle_t handle, const cvai_bbox_t \*bbox, bool \*has_intersect);
```
**[说明]**

根据CVI_AI_Service_Polygon_SetTarget所设定区域入侵范围。判断给定之gounding box侵入范围。

**[参数]**

|    | 资料型态                  | 参数名称      | 说明         |
|----|---------------------------|---------------|--------------|
| 进 | cviai_service_handle_t \* | handle        | 句柄         |
| 进 | cvai_bbox_t \*            | bbox          | Bounding box |
| 出 | bool                      | has_intersect | 是否入侵     |

### 3.3.5. CVI_AI_Service_RegisterFeatureArray
```C++
CVI_S32 CVI_AI_Service_RegisterFeatureArray(cviai_service_handle_t handle, const cvai_service_feature_array_t featureArray, const cvai_service_feature_matching_e method);
```
**[说明]**

注册特征库，将featureArray中所含特征进行预计算并搬入内存中。

**[参数]**

|    | 资料型态                              | 参数名称     | 说明                               |
|----|---------------------------------------|--------------|------------------------------------|
| 进 | cviai_service_handle_t \*             | handle       | 句柄                               |
| 进 | const cvai_service_feature_array_t    | featureArray | 特征阵列结构                       |
| 进 | const cvai_service_feature_matching_e | method       | 比对方法，目前仅支援COS_SIMILARITY |

### 3.3.6. CVI_AI_Service_CalcualteSimilarity
```C++
CVI_S32 CVI_AI_Service_CalculateSimilarity(cviai_service_handle_t handle, const cvai_feature_t \*feature_rhs, const cvai_feature_t \*feature_lhs, float \*score);
```
**[说明]**

使用CPU计算两个特征之Cosine Similarity。其计算公式如下：

其中n 为特征长度。目前仅支援INT8特征

**[参数]**

|    | 资料型态                  | 参数名称    | 说明       |
|----|---------------------------|-------------|------------|
| 进 | cviai_service_handle_t \* | handle      | 句柄       |
| 进 | const cvai_feature_t \*   | feature_rhs | 第一个特征 |
| 进 | const cvai_feature_t \*   | feature_lhs | 第二个特征 |
| 出 | float \*                  | score       | 相似度     |

### 3.3.7. CVI_AI_Service_ObjectInfoMatching

CVI_S32 CVI_AI_Service_ObjectInfoMatching(cviai_service_handle_t handle, const cvai_object_info_t \*object_info, const uint32_t topk, float threshold, uint32_t \*indices, float \*sims, uint32_t \*size);

**[说明]**

计算object_info中的物件特征和已注册之物件特征库之Cosine Similarity。并取出大于threshold的Top-K个相似度。其计算公式如下：

其中n 为特征长度。若特征库数量少于1000笔会以CPU进行计算，否则会以启动TPU进行计算。注册特征需要调用CVI_AI_Service_RegisterFeatureArray。目前仅支援INT8特征

**[参数]**

|    | 资料型态                    | 参数名称    | 说明                                   |
|----|-----------------------------|-------------|----------------------------------------|
| 进 | cviai_service_handle_t \*   | handle      | 句柄                                   |
| 进 | const cvai_object_info_t \* | object_info | 物件Info                               |
| 进 | const uint32_t              | topk        | 取topk个相似度                         |
| 出 | float                       | threshold   | 相似度阀值，高于此阀值之相似度才会取出 |
| 出 | uint32_t \*                 | indices     | 符合条件之相似度在库内的Index          |
| 出 | float \*                    | sims        | 符合条件之相似度                       |
| 出 | uint32_t \*                 | size        | 最终取出的相似度个数                   |

### 3.3.8. CVI_AI_Service_FaceInfoMatching
```C++
CVI_S32 CVI_AI_Service_FaceInfoMatching(cviai_service_handle_t handle, const cvai_face_info_t \*face_info, const uint32_t topk, float threshold, uint32_t \*indices, float \*sims, uint32_t \*size);
```
**[说明]**

计算face_info中的人脸特征和已注册之人脸特征库之Cosine Similarity。并取出大于threshold的Top-K个相似度。其计算公式如下：

其中n 为特征长度。若特征库数量少于1000笔会以CPU进行计算，否则会以启动TPU进行计算。注册特征需要调用CVI_AI_Service_RegisterFeatureArray。目前仅支援INT8特征

**[参数]**

|    | 资料型态                  | 参数名称  | 说明                                   |
|----|---------------------------|-----------|----------------------------------------|
| 进 | cviai_service_handle_t \* | handle    | 句柄                                   |
| 进 | const cvai_face_info_t \* | face_info | Face info                              |
| 进 | const uint32_t            | topk      | 取topk个相似度                         |
| 出 | float                     | threshold | 相似度阀值，高于此阀值之相似度才会取出 |
| 出 | uint32_t \*               | indices   | 符合条件之相似度在库内的Index          |
| 出 | float \*                  | sims      | 符合条件之相似度                       |
| 出 | uint32_t \*               | size      | 最终取出的相似度个数                   |

### 3.3.9. CVI_AI_Service_RawMatching
```C++
CVI_S32 CVI_AI_Service_RawMatching(cviai_service_handle_t handle, const void \*feature,

const feature_type_e type, const uint32_t topk, float threshold, uint32_t \*indices, float \*scores, uint32_t \*size)
```
**[说明]**

计算特征和已注册之特征库之Cosine Similarity。并取出大于threshold的Top-K个相似度。其计算公式如下：

其中n 为特征长度。若特征库数量少于1000笔会以CPU进行计算，否则会以启动TPU进行计算。注册特征需要调用CVI_AI_Service_RegisterFeatureArray。和CVI_AI_Service_FaceInfoMatching及CVI_AI_Service_ObjectInfoMatching不同的是，此API直接使用特征阵列进行比对，不需传入cvai_face_info_t或cvai_object_info_t。此API限制特征类型需要和特征库之特征类型相同。目前仅支援INT8特征

**[参数]**

|    | 资料型态                  | 参数名称  | 说明                                   |
|----|---------------------------|-----------|----------------------------------------|
| 进 | cviai_service_handle_t \* | handle    | 句柄                                   |
| 进 | const void \*             | feature   | 特征阵列                               |
| 进 | const feature_type_e      | type      | 特征类型，目前仅支援TYPE_INT8          |
| 进 | const uint32_t            | topk      | 取topk个相似度                         |
| 出 | float                     | threshold | 相似度阀值，高于此阀值之相似度才会取出 |
| 出 | uint32_t \*               | indices   | 符合条件之相似度在库内的Index          |
| 出 | float \*                  | scores    | 符合条件之相似度                       |
| 出 | uint32_t \*               | size      | 最终取出的相似度个数                   |

### 3.3.10. CVI_AI_Service_FaceAngle

CVI_S32 CVI_AI_Service_FaceAngle(const cvai_pts_t \*pts, cvai_head_pose_t \*hp);

**[说明]**

计算单个人脸姿态

**[参数]**

|    | 资料型态         | 参数名称 | 说明          |
|----|------------------|----------|---------------|
| 进 | cvai_pts_t \*    | pts      | 人脸 landmark |
| 出 | cvai_head_pose_t | hp       | 人脸姿态      |

### 3.3.11. CVI_AI_Service_FaceAngleForAll

CVI_S32 CVI_AI_Service_FaceAngleForAll(const cvai_face_t \*meta);

**[说明]**

计算多个人脸姿态

**[参数]**

|       | 资料型态       | 参数名称 | 说明     |
|-------|----------------|----------|----------|
| 进/出 | cvai_face_t \* | meta     | 人脸资料 |

### 3.3.12. CVI_AI_Service_FaceDigitalZoom

CVI_S32 CVI_AI_Service_FaceDigitalZoom(

cviai_service_handle_t handle,

const VIDEO_FRAME_INFO_S \*inFrame,

const cvai_face_t \*meta,

const float face_skip_ratio,

const float trans_ratio,

const float padding_ratio,

VIDEO_FRAME_INFO_S \*outFrame);

**[说明]**

将人脸侦测结果之人脸进行放大(zoom in)

**[参数]**

|    | 资料型态               | 参数名称        | 说明                 |
|----|------------------------|-----------------|----------------------|
| 进 | cviai_service_handle_t | handle          | 句柄                 |
| 进 | VIDEO_FRAME_INFO_S \*  | inFrame         | 输入影像             |
| 进 | cvai_face_t \*         | meta            | 人脸资料             |
| 进 | float                  | face_skip_ratio | 忽略比率             |
| 进 | float                  | trans_ratio     | 放大比率             |
| 进 | float                  | padding_ratio   | 扩展bounding box比例 |
| 出 | VIDEO_FRAME_INFO_S \*  | outFrame        | 输出影像             |

### 3.3.13. CVI_AI_Service_FaceDrawPts

CVI_S32 CVI_AI_Service_FaceDrawPts(cvai_pts_t \*pts, VIDEO_FRAME_INFO_S \*frame);

**[说明]**

绘制人脸 landmark

**[参数]**

|       | 资料型态           | 参数名称 | 说明          |
|-------|--------------------|----------|---------------|
| 进    | cvai_pts_t \*      | pts      | 人脸 landmark |
| 进/出 | VIDEO_FRAME_INFO_S | hp       | 输入/输出影像 |

### 3.3.14. CVI_AI_Service_FaceDrawRect

CVI_S32 CVI_AI_Service_FaceDrawRect(cviai_service_handle_t handle, const cvai_face_t \*meta, VIDEO_FRAME_INFO_S \*frame, const bool drawText, cvai_service_brush_t brush);

**[说明]**

绘制人脸方框

**[参数]**

|       | 资料型态               | 参数名称 | 说明             |
|-------|------------------------|----------|------------------|
| 进    | cviai_service_handle_t | handle   | 句柄             |
| 进    | cvai_face_t \*         | meta     | 人脸资料         |
| 进/出 | VIDEO_FRAME_INFO_S \*  | frame    | 输入/输出影像    |
| 进    | bool                   | drawText | 是否绘制人脸名字 |
| 进    | cvai_service_brush_t   | brush    | 颜色             |

### 3.3.15. CVI_AI_Service_ObjectDigitalZoom

CVI_S32 CVI_AI_Service_ObjectDigitalZoom(cviai_service_handle_t handle,

const VIDEO_FRAME_INFO_S \*inFrame, const cvai_object_t \*meta, const float obj_skip_ratio, const float trans_ratio, const float padding_ratio, VIDEO_FRAME_INFO_S \*outFrame);

**[说明]**

将物件侦测结果之物件进行放大(zoom in)

**[参数]**

|    | 资料型态               | 参数名称       | 说明                 |
|----|------------------------|----------------|----------------------|
| 进 | cviai_service_handle_t | handle         | 句柄                 |
| 进 | VIDEO_FRAME_INFO_S \*  | inFrame        | 输入影像             |
| 进 | cvai_object_t \*       | meta           | 物件资料             |
| 进 | float                  | obj_skip_ratio | 忽略比率             |
| 进 | float                  | trans_ratio    | 放大比率             |
| 进 | float                  | padding_ratio  | 扩展bounding box比例 |
| 出 | VIDEO_FRAME_INFO_S \*  | outFrame       | 输出影像             |

### 3.3.16. CVI_AI_Service_ObjectDitgitalZoomExt

CVI_S32 CVI_AI_Service_ObjectDigitalZoomExt(cviai_service_handle_t handle, const VIDEO_FRAME_INFO_S \*inFrame, const cvai_object_t \*meta, const float obj_skip_ratio, const float trans_ratio, const float pad_ratio_left, const float pad_ratio_right, const float pad_ratio_top, const float pad_ratio_bottom, VIDEO_FRAME_INFO_S \*outFrame);

**[说明]**

将物件侦测结果之物件进行放大(zoom in)

**[参数]**

|    | 资料型态               | 参数名称         | 说明       |
|----|------------------------|------------------|------------|
| 进 | cviai_service_handle_t | handle           | 句柄       |
| 进 | VIDEO_FRAME_INFO_S \*  | inFrame          | 输入影像   |
| 进 | cvai_object_t \*       | meta             | 物件资料   |
| 进 | float                  | obj_skip_ratio   | 忽略比率   |
| 进 | float                  | trans_ratio      | 放大比率   |
| 进 | float                  | pad_ratio_left   | 扩张率(左) |
| 进 | float                  | pad_ratio_right  | 扩张率(右) |
| 进 | float                  | pad_ratio_top    | 扩张率(上) |
| 进 | float                  | pad_ratio_bottom | 扩张率(下) |
| 出 | VIDEO_FRAME_INFO_S \*  | outFrame         | 输出影像   |

### 3.3.17. CVI_AI_Service_ObjectDrawPose

CVI_S32 CVI_AI_Service_ObjectDrawPose(const cvai_object_t \*meta, VIDEO_FRAME_INFO_S \*frame);

**[说明]**

绘制姿态侦测之17个骨骼点

**[参数]**

|    | 资料型态              | 参数名称 | 说明           |
|----|-----------------------|----------|----------------|
| 进 | cvai_object_t \*      | meta     | 骨骼点侦测结果 |
| 进 | VIDEO_FRAME_INFO_S \* | frame    | 输入影像       |
### 3.3.18. CVI_AI_Service_ObjectDrawRect

CVI_S32 CVI_AI_Service_ObjectDrawRect(cviai_service_handle_t handle, const cvai_object_t \*meta, VIDEO_FRAME_INFO_S \*frame, const bool drawText);

**[说明]**

绘制物件侦测框

**[参数]**

|       | 资料型态               | 参数名称 | 说明             |
|-------|------------------------|----------|------------------|
| 进    | cviai_service_handle_t | handle   | 句柄             |
| 进    | cvai_object_t \*       | meta     | 物件侦测结果     |
| 进/出 | VIDEO_FRAME_INFO_S \*  | frame    | 输入/输出 影像   |
| 进    | bool                   | drawText | 是否绘制类别文字 |

### 3.3.19. CVI_AI_Service_ObjectWriteText

CVI_S32 CVI_AI_Service_ObjectWriteText(char \*name, int x, int y, VIDEO_FRAME_INFO_S \*frame, float r,

float g, float b)

**[说明]**

绘制指定文字

**[参数]**

|       | 资料型态              | 参数名称 | 说明                  |
|-------|-----------------------|----------|-----------------------|
| 进    | char\*                | name     | 绘制的文字            |
| 进    | int                   | x        | 绘制的x坐标           |
| 进    | int                   | y        | 绘制的y坐标           |
| 进/出 | VIDEO_FRAME_INFO_S \* | frame    | 输入/输出 影像        |
| 进    | float                 | r        | 绘制颜色 r channel值  |
| 进    | float                 | g        | 绘制颜色 g channel值  |
| 进    | float                 | b        | 绘制颜色 b channel值  |

### 3.3.20. CVI_AI_Service_Incar_ObjectDrawRect

CVI_S32 CVI_AI_Service_ObjectWriteText(cviai_service_handle_t handle, const cvai_dms_od_t \*meta, VIDEO_FRAME_INFO_S \*frame, const bool drawText, IVE_COLOR_S color)

**[说明]**

绘制指定文字

**[参数]**

|       | 资料型态               | 参数名称 | 说明             |
|-------|------------------------|----------|------------------|
| 进    | cviai_service_handle_t | handle   | 句柄             |
| 进    | cvai_dms_od_t\*        | meta     | 物件侦测结果     |
| 进/出 | VIDEO_FRAME_INFO_S \*  | frame    | 输入/输出 影像   |
| 进    | const bool             | drawText | 是否绘制类别文字 |
| 进    | IVE_COLOR_S            | color    | 绘制颜色         |

# 4. 应用(APP)

## 4.1. 目的

CVITek AI application，APP，是基于AI SDK，并针对不同客户端应用，所设计的solution。APP整合AI SDK，提供客户更方便的操作API。APP code为open source，可以作为客户端开发的参考。

[编译]

1.  下载AI SDK与其依赖之SDK：MW、TPU、IVE。
2.  移动至AI SDK的module/app目录
3.  执行以下指令：

    make MW_PATH=\<MW_SDK\> TPU_PATH=\<TPU_SDK\> IVE_PATH=\<IVE_SDK\>

    make install

    make clean

编译完成的lib会放在AI SDK的tmp_install目录下

## 4.2. API

### 4.2.1. 句柄

typedef struct {

cviai_handle_t ai_handle;

IVE_HANDLE ive_handle;

face_capture_t \*face_cpt_info;

} cviai_app_context_t;

typedef cviai_app_context \*cviai_app_handle_t;

**[说明]**

cviai_app_handle_t为cviai_app_context的指标型态，其中包含ai handle、ive handle与其他应用之资料结构。

#### 4.2.1.1. CVI_AI_APP_CreateHandle

CVI_S32 CVI_AI_APP_CreateHandle(cviai_app_handle_t \*handle, cviai_handle_t ai_handle, IVE_HANDLE ive_handle);

**[说明]**

创建使用APP所需的指标。需输入ai handle与ive handle。

**[参数]**

|    | 资料型态             | 参数名称   | 说明         |
|----|----------------------|------------|--------------|
| 出 | cviai_app_handle_t\* | handle     | 输入句柄指标 |
| 进 | cviai_handle_t       | ai_handle  | AI句柄       |
| 进 | IVE_HANDLE           | ive_handle | IVE句柄      |

#### 4.2.1.2. CVI_AI_APP_DestroyHandle

CVI_S32 CVI_AI_APP_DestroyHandle(cviai_app_handle_t handle);

**[说明]**

销毁创造的句柄cviai_app_handle_t。只会销毁个别应用程序所使用之资料，不影响ai handle与ive handle。

**[参数]**

|    | 资料型态           | 参数名称 | 说明         |
|----|--------------------|----------|--------------|
| 进 | cviai_app_handle_t | handle   | 输入句柄指标 |

### 4.2.2. 人脸抓拍

人脸抓拍 (Face Capture) 结合人脸侦测、多物件追踪、人脸质量检测，功能为抓拍 (或撷取) 影像中不同人的脸部照片。抓拍条件可利用设定档来调整，例如：抓拍时间点、人脸质量检测算法、人脸角度阀值…。

[设定档]

| 参数名称                  | 预设值 | 说明                                                                                                                     |
|---------------------------|--------|--------------------------------------------------------------------------------------------------------------------------|
| Miss_Time_Limit           | 40     | 人脸遗失时间限制。当APP连续无法追踪到某个face，会判定此 face已离开。[单位：frame]                                        |
| Threshold_Size_Min        | 32     | 最小/最大可接受人脸大小，如果face bbox的任一边小于/大于此阀值，quality会强制设为0。                                      |
| Threshold_Size_Max        | 512    |                                                                                                                          |
| Quality_Assessment_Method | 0      | 若人脸评估不使用FQNet时，启用内建质量检测算法 0: 基于人脸大小与角度 1: 基于眼睛距离                                      |
| Threshold_Quality         | 0.1    | 人脸质量阀值，若新的face的quality大于此阀值，且比当前撷取之face的quality还高，则会撷取并更新暂存区face资料。             |
| Threshold_Quality_High    | 0.95   | 人脸质量阀值（高），若暂存区某face的quality高于此阀值，则判定此 face 为高质量，后续不会再进行更新。（仅适用于level 2,3） |
| Threshold_Yaw             | 0.25   | 人脸角度阀值，若角度大于此阀值，quality会强制设为0。（一单位为90度）                                                     |
| Threshold_Pitch           | 0.25   |                                                                                                                          |
| Threshold_Roll            | 0.25   |                                                                                                                          |
| FAST_Mode_Interval        | 10     | FAST模式抓拍间隔。[单位：frame]                                                                                          |
| FAST_Mode_Capture_Num     | 3      | FAST模式抓拍次数。                                                                                                       |
| CYCLE_Mode_Interval       | 20     | CYCLE模式抓拍间隔。[单位：frame]                                                                                         |
| AUTO_Mode_Time_Limit      | 0      | AUTO模式最后输出的时限。[单位：frame]                                                                                    |
| AUTO_Mode_Fast_Cap        | 1      | AUTO模式是否输出进行快速抓拍1次。                                                                                        |
| Capture_Aligned_Face      | 0      | 抓拍/撷取人脸是否进行校正。                                                                                              |

[人脸质量检测算法]

| \# | 算法               | 计算方式                                                                                                                                                                                                                                                                                 |
|----|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0  | 基于人脸大小与角度 | Face Area Score 定义标准人脸大小A_base = 112 \* 112 计算侦测到的人脸面积A_face = 长 \* 宽 计算MIN(1.0, A_face/A_base) 作为分数 Face Pose Score 分别计算人脸角度 yaw, pitch, roll 并取其绝对值 计算1 - (yaw + pitch + roll) / 3作为分数 Face Quality = Face Area Score \* Face Pose Score |
| 1  | 基于眼睛距离       | 定义标准瞳距 D = 80 计算双眼距离 d 计算MIN(1.0, d/D) 当作分数                                                                                                                                                                                                                            |

#### 4.2.2.1. CVI_AI_APP_FaceCapture_Init

CVI_AI_APP_FaceCapture_Init(const cviai_app_handle_t handle, uint32_t buffer_size);

**[说明]**

初始化人脸抓拍数据结构。

**[参数]**

|    | 资料型态           | 参数名称    | 说明           |
|----|--------------------|-------------|----------------|
| 进 | cviai_app_handle_t | handle      | 输入句柄指标   |
| 进 | uint32_t           | buffer_size | 人脸暂存区大小 |

#### 4.2.2.2. CVI_AI_APP_FaceCapture_QuickSetUp

CVI_S32 CVI_AI_APP_FaceCapture_QuickSetUp(const cviai_app_handle_t handle, const char \*fd_model_path, const char \*fq_model_path);

**[说明]**

快速设定人脸抓拍。

**[参数]**

|    | 资料型态           | 参数名称      | 说明                 |
|----|--------------------|---------------|----------------------|
| 进 | cviai_app_handle_t | handle        | 输入句柄指标         |
| 进 | char\*             | fd_model_path | 人脸侦测模型路径     |
| 进 | char\*             | fq_model_path | 人脸质量检测模型路径 |

#### 4.2.2.3. CVI_AI_APP_FaceCapture_GetDefaultConfig

CVI_S32 CVI_AI_APP_FaceCapture_GetDefaultConfig(face_capture_config_t \*cfg);

**[说明]**

取得人脸抓拍预设参数。

**[参数]**

|    | 资料型态                | 参数名称 | 说明         |
|----|-------------------------|----------|--------------|
| 出 | face_capture_config_t\* | cfg      | 人脸抓拍参数 |

#### 4.2.2.4. CVI_AI_APP_FaceCapture_SetConfig

CVI_S32 CVI_AI_APP_FaceCapture_SetConfig(const cviai_app_handle_t handle, face_capture_config_t \*cfg);

**[说明]**

设定人脸抓拍参数。

**[参数]**

|    | 资料型态                | 参数名称 | 说明         |
|----|-------------------------|----------|--------------|
| 进 | cviai_app_handle_t      | handle   | 输入句柄指标 |
| 进 | face_capture_config_t\* | cfg      | 人脸抓拍参数 |

#### 4.2.2.5. CVI_AI_APP_FaceCapture_SetMode

CVI_S32 CVI_AI_APP_FaceCapture_SetMode(const cviai_app_handle_t handle, capture_mode_e mode);

**[说明]**

设定人脸抓拍模式。

**[参数]**

|    | 资料型态           | 参数名称 | 说明         |
|----|--------------------|----------|--------------|
| 进 | cviai_app_handle_t | handle   | 输入句柄指标 |
| 进 | capture_mode_e     | mode     | 人脸抓拍模式 |

#### 4.2.2.6. CVI_AI_APP_FaceCapture_Run

CVI_S32 CVI_AI_APP_FaceCapture_Run(const cviai_app_handle_t handle, VIDEO_FRAME_INFO_S \*frame);

**[说明]**

执行人脸抓拍。

**[参数]**

|    | 资料型态             | 参数名称 | 说明         |
|----|----------------------|----------|--------------|
| 进 | cviai_app_handle_t   | handle   | 输入句柄指标 |
| 进 | VIDEO_FRAME_INFO_S\* | frame    | 输入影像     |

#### 4.2.2.7. CVI_AI_APP_FaceCapture_CleanAll

CVI_S32 CVI_AI_APP_FaceCapture_CleanAll(const cviai_app_handle_t handle);

**[说明]**

清除所有人脸抓拍暂存区之数据资料。

**[参数]**

|    | 资料型态           | 参数名称 | 说明         |
|----|--------------------|----------|--------------|
| 进 | cviai_app_handle_t | handle   | 输入句柄指标 |

### 4.2.3. 人型抓拍

人型抓拍 (Face Capture) 结合人型侦测、多物件追踪、人脸质量检测，功能为抓拍 (或撷取) 影像中不同人的脸部照片。抓拍条件可利用设定档来调整，例如：抓拍时间点、人脸质量检测算法、人脸角度阀值…。

[设定档]

| 参数名称                  | 预设值 | 说明                                                                                                                     |
|---------------------------|--------|--------------------------------------------------------------------------------------------------------------------------|
| Miss_Time_Limit           | 40     | 人脸遗失时间限制。当APP连续无法追踪到某个face，会判定此 face已离开。[单位：frame]                                        |
| Threshold_Size_Min        | 32     | 最小/最大可接受人脸大小，如果face bbox的任一边小于/大于此阀值，quality会强制设为0。                                      |
| Threshold_Size_Max        | 512    |                                                                                                                          |
| Quality_Assessment_Method | 0      | 若人脸评估不使用FQNet时，启用内建质量检测算法 0: 基于人脸大小与角度 1: 基于眼睛距离                                      |
| Threshold_Quality         | 0.1    | 人脸质量阀值，若新的face的quality大于此阀值，且比当前撷取之face的quality还高，则会撷取并更新暂存区face资料。             |
| Threshold_Quality_High    | 0.95   | 人脸质量阀值（高），若暂存区某face的quality高于此阀值，则判定此 face 为高质量，后续不会再进行更新。（仅适用于level 2,3） |
| Threshold_Yaw             | 0.25   | 人脸角度阀值，若角度大于此阀值，quality会强制设为0。（一单位为90度）                                                     |
| Threshold_Pitch           | 0.25   |                                                                                                                          |
| Threshold_Roll            | 0.25   |                                                                                                                          |
| FAST_Mode_Interval        | 10     | FAST模式抓拍间隔。[单位：frame]                                                                                          |
| FAST_Mode_Capture_Num     | 3      | FAST模式抓拍次数。                                                                                                       |
| CYCLE_Mode_Interval       | 20     | CYCLE模式抓拍间隔。[单位：frame]                                                                                         |
| AUTO_Mode_Time_Limit      | 0      | AUTO模式最后输出的时限。[单位：frame]                                                                                    |
| AUTO_Mode_Fast_Cap        | 1      | AUTO模式是否输出进行快速抓拍1次。                                                                                        |
| Capture_Aligned_Face      | 0      | 抓拍/撷取人脸是否进行校正。                                                                                              |

[人脸质量检测算法]

| \# | 算法               | 计算方式                                                                                                                                                                                                                                                                                 |
|----|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0  | 基于人脸大小与角度 | Face Area Score 定义标准人脸大小A_base = 112 \* 112 计算侦测到的人脸面积A_face = 长 \* 宽 计算MIN(1.0, A_face/A_base) 作为分数 Face Pose Score 分别计算人脸角度 yaw, pitch, roll 并取其绝对值 计算1 - (yaw + pitch + roll) / 3作为分数 Face Quality = Face Area Score \* Face Pose Score |
| 1  | 基于眼睛距离       | 定义标准瞳距 D = 80 计算双眼距离 d 计算MIN(1.0, d/D) 当作分数                                                                                                                                                                                                                            |

#### 4.2.3.1. CVI_AI_APP_PersonCapture_Init

CVI_AI_APP_PersonCapture_Init(const cviai_app_handle_t handle, uint32_t buffer_size);

**[说明]**

初始化人形抓拍数据结构。

**[参数]**

|    | 资料型态           | 参数名称    | 说明           |
|----|--------------------|-------------|----------------|
| 进 | cviai_app_handle_t | handle      | 输入句柄指标   |
| 进 | uint32_t           | buffer_size | 人脸暂存区大小 |

#### 4.2.3.2. CVI_AI_APP_PersonCapture_QuickSetUp

CVI_S32 CVI_AI_APP_PersonCapture_QuickSetUp(const cviai_app_handle_t handle,

const char \*od_model_name,

const char \*od_model_path,

const char \*reid_model_path);

**[说明]**

快速设定人型抓拍。

**[参数]**

|    | 资料型态           | 参数名称        | 说明             |
|----|--------------------|-----------------|------------------|
| 进 | cviai_app_handle_t | handle          | 输入句柄指标     |
| 进 | const char \*      | od_model_name   | 人型侦测模型名称 |
| 进 | const char \*      | od_model_path   | 人型侦测模型路径 |
| 进 | const char \*      | reid_model_path | ReID模型路径     |

#### 4.2.3.3. CVI_AI_APP_FaceCapture_GetDefaultConfig

CVI_S32 CVI_AI_APP_PersonCapture_GetDefaultConfig(person_capture_config_t \*cfg);

**[说明]**

取得人型抓拍预设参数。

**[参数]**

|    | 资料型态                  | 参数名称 | 说明         |
|----|---------------------------|----------|--------------|
| 出 | person_capture_config_t\* | cfg      | 人型抓拍参数 |

#### 4.2.3.4. CVI_AI_APP_PersonCapture_SetConfig

CVI_S32 CVI_AI_APP_PersonCapture_SetConfig(const cviai_app_handle_t handle, person_capture_config_t \*cfg);

**[说明]**

设定人型抓拍参数。

**[参数]**

|    | 资料型态                  | 参数名称 | 说明         |
|----|---------------------------|----------|--------------|
| 进 | cviai_app_handle_t        | handle   | 输入句柄指标 |
| 进 | person_capture_config_t\* | cfg      | 人型抓拍参数 |

#### 4.2.3.5. CVI_AI_APP_PersonCapture_SetMode

CVI_S32 CVI_AI_APP_PersonCapture_SetMode(const cviai_app_handle_t handle, capture_mode_e mode);

**[说明]**

设定人型抓拍模式。

**[参数]**

|    | 资料型态           | 参数名称 | 说明         |
|----|--------------------|----------|--------------|
| 进 | cviai_app_handle_t | handle   | 输入句柄指标 |
| 进 | capture_mode_e     | mode     | 人型抓拍模式 |

#### 4.2.3.6. CVI_AI_APP_PersonCapture_Run

CVI_S32 CVI_AI_APP_PersonCapture_Run(const cviai_app_handle_t handle, VIDEO_FRAME_INFO_S \*frame);

**[说明]**

执行人型抓拍。

**[参数]**

|    | 资料型态             | 参数名称 | 说明         |
|----|----------------------|----------|--------------|
| 进 | cviai_app_handle_t   | handle   | 输入句柄指标 |
| 进 | VIDEO_FRAME_INFO_S\* | frame    | 输入影像     |

#### 4.2.3.7. CVI_AI_APP_PersonCapture_CleanAll

CVI_S32 CVI_AI_APP_PersonCapture_ClanAll(const cviai_app_handle_t handle);

**[说明]**

清除所有人型抓拍暂存区之数据资料。

**[参数]**

|    | 资料型态           | 参数名称 | 说明         |
|----|--------------------|----------|--------------|
| 进 | cviai_app_handle_t | handle   | 输入句柄指标 |

# 5. 数据类型

## 5.1. CVI_AI_Core

### 5.1.1. CVI_AI_SUPPORTED_MODEL_E

**[说明]**

此enum定义AI SDK中所有Deep Learning Model。下表为每个模型Id和其模型功能说明。

| 模型ID                                         | 说明               |
|------------------------------------------------|--------------------|
| CVI_AI_SUPPORTED_MODEL_RETINAFACE              | 人脸侦测           |
| CVI_AI_SUPPORTED_MODEL_THERMALFACE             | 热显人脸侦测       |
| CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE           | 人脸属性和人脸辨识 |
| CVI_AI_SUPPORTED_MODEL_FACERECOGNITION         | 人脸辨识           |
| CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION     | 戴口罩人脸辨识     |
| CVI_AI_SUPPORTED_MODEL_FACEQUALITY             | 人脸质量           |
| CVI_AI_SUPPORTED_MODEL_LIVENESS                | 双目活体辨识       |
| CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION      | 人脸口罩辨识       |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE     | 交通工具侦测       |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN  | 行人侦测           |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS | 猫狗及人型侦测     |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80      | 80类物件侦测       |
| CVI_AI_SUPPORTED_MODEL_YOLOV3                  | 80类物件侦测       |
| CVI_AI_SUPPORTED_MODEL_OSNET                   | 行人重识别         |
| CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION     | 声音辨识           |
| CVI_AI_SUPPORTED_MODEL_WPODNET                 | 车牌侦测           |
| CVI_AI_SUPPORTED_MODEL_LPRNET_TW               | 台湾地区车牌辨识   |
| CVI_AI_SUPPORTED_MODEL_LPRNET_CN               | 大陆地区车牌辨识   |
| CVI_AI_SUPPORTED_MODEL_DEEPLABV3               | 语意分割           |
| CVI_AI_SUPPORTED_MODEL_ALPHAPOSE               | 人体关键点侦测     |
| CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION       | 闭眼辨识           |
| CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION      | 打哈欠辨识         |
| CVI_AI_SUPPORTED_MODEL_FACELANDMARKER          | 人脸关键点侦测     |
| CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION    | 车内物件辨识       |
| CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION     | 抽菸辨识           |
| CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION       | 口罩人脸侦测       |

下表为每个模型Id对应的模型档案及推理使用的function：

| 模型ID                                            | Inference Function                               | 模型档案                                                                                                                                                                                                       |
|---------------------------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CVI_AI_SUPPORTED_MODEL_RETINAFACE                 | CVI_AI_RetinaFace                                | retinaface_mnet0.25_342_608.cvimodel retinaface_mnet0.25_608_342.cvimodel retinaface_mnet0.25_608.cvimodel                                                                                                     |
| CVI_AI_SUPPORTED_MODEL_THERMALFACE                | CVI_AI_ThermalFace                               | thermalfd-v1.cvimodel                                                                                                                                                                                          |
| CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE              | CVI_AI_FaceAttribute CVI_AI_FaceAttributeOne     | cviface-v3-attribute.cvimodel                                                                                                                                                                                  |
| CVI_AI_SUPPORTED_MODEL_FACERECOGNITION            | CVI_AI_FaceRecognition CVI_AI_FaceRecognitionOne | cviface-v4.cvimodel cviface-v5-m.cvimodel cviface-v5-s.cvimodel                                                                                                                                                |
| CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION        | CVI_AI_MaskFaceRecognition                       | masked-fr-v1-m.cvimodel                                                                                                                                                                                        |
| CVI_AI_SUPPORTED_MODEL_FACEQUALITY                | CVI_AI_FaceQuality                               | fqnet-v5_shufflenetv2-softmax.cvimodel                                                                                                                                                                         |
| CVI_AI_SUPPORTED_MODEL_LIVENESS                   | CVI_AI_Liveness                                  | liveness-rgb-ir.cvimodel                                                                                                                                                                                       |
| CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION         | CVI_AI_MaskClassification                        | mask_classifier.cvimodel                                                                                                                                                                                       |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE        | CVI_AI_MobileDetV2_Vehicle                       | mobiledetv2-vehicle-d0-ls.cvimodel                                                                                                                                                                             |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN     | CVI_AI_MobileDetV2_Pedestrian                    | mobiledetv2-pedestrian-d0-ls-384.cvimodel mobiledetv2-pedestrian-d0-ls-640.cvimodel mobiledetv2-pedestrian-d0-ls-768.cvimodel mobiledetv2-pedestrian-d1-ls.cvimodel mobiledetv2-pedestrian-d1-ls-1024.cvimodel |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE | CVI_AI_MobileDetV2_Person_Vehicle                | mobiledetv2-person-vehicle-ls-768.cvimodel mobiledetv2-person-vehicle-ls.cvimodel                                                                                                                              |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS    | CVI_AI_MobileDetV2_Person_Pets                   | mobiledetv2-lite-person-pets-ls.cvimodel                                                                                                                                                                       |
| CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80         | CVI_AI_MobileDetV2_COCO80                        | mobiledetv2-d0-ls.cvimodel mobiledetv2-d1-ls.cvimodel mobiledetv2-d2-ls.cvimodel                                                                                                                               |
| CVI_AI_SUPPORTED_MODEL_YOLOV3                     | CVI_AI_Yolov3                                    | yolo_v3_416.cvimodel                                                                                                                                                                                           |
| CVI_AI_SUPPORTED_MODEL_OSNET                      | CVI_AI_OSNet CVI_AI_OSNetOne                     | person-reid-v1.cvimodel                                                                                                                                                                                        |
| CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION        | CVI_AI_SoundClassification                       | es_classification.cvimodel soundcmd_bf16.cvimodel                                                                                                                                                              |
| CVI_AI_SUPPORTED_MODEL_WPODNET                    | CVI_AI_LicensePlateDetection                     | wpodnet_v0_bf16.cvimodel                                                                                                                                                                                       |
| CVI_AI_SUPPORTED_MODEL_LPRNET_TW                  | CVI_AI_LicensePlateRecognition_TW                | lprnet_v0_tw_bf16.cvimodel                                                                                                                                                                                     |
| CVI_AI_SUPPORTED_MODEL_LPRNET_CN                  | CVI_AI_LicensePlateRecognition_CN                | lprnet_v1_cn_bf16.cvimodel                                                                                                                                                                                     |
| CVI_AI_SUPPORTED_MODEL_DEEPLABV3                  | CVI_AI_DeeplabV3                                 | deeplabv3_mobilenetv2_640x360.cvimodel                                                                                                                                                                         |
| CVI_AI_SUPPORTED_MODEL_ALPHAPOSE                  | CVI_AI_AlphaPose                                 | alphapose.cvimodel                                                                                                                                                                                             |
| CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION          | CVI_AI_EyeClassification                         | eye_v1_bf16.cvimodel                                                                                                                                                                                           |
| CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION         | CVI_AI_YawnClassification                        | yawn_v1_bf16.cvimodel                                                                                                                                                                                          |
| CVI_AI_SUPPORTED_MODEL_FACELANDMARKER             | CVI_AI_FaceLandmarker                            | face_landmark_bf16.cvimodel                                                                                                                                                                                    |
| CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION       | CVI_AI_IncarObjectDetection                      | incar_od_v0_bf16.cvimodel                                                                                                                                                                                      |
| CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION        | CVI_AI_SmokeClassification                       | N/A                                                                                                                                                                                                            |
| CVI_AI_SUPPORTED_MODEL_FDMASKDETECTION            | CVI_AI_FaceMaskDetection                         | yolox_RetinafaceMask_mosaic1_lrelu_wmp_addinoc25_occlude_432_768_int8.cvimodel                                                                                                                                 |
| CVI_AI_SUPPORTED_MODEL_YOLOX                      | CVI_AI_YoloX                                     | yolox_nano.cvimodel yolox_tiny.cvimodel                                                                                                                                                                        |

### 5.1.2. cvai_obj_class_id_e

**[说明]**

此enum定义物件侦测类别。每一类别归属于一个类别群组。

| 类别                           | 类别群组                    |
|--------------------------------|-----------------------------|
| CVI_AI_DET_TYPE_PERSON         | CVI_AI_DET_GROUP_PERSON     |
| CVI_AI_DET_TYPE_BICYCLE        | CVI_AI_DET_GROUP_VEHICLE    |
| CVI_AI_DET_TYPE_CAR            |                             |
| CVI_AI_DET_TYPE_MOTORBIKE      |                             |
| CVI_AI_DET_TYPE_AEROPLANE      |                             |
| CVI_AI_DET_TYPE_BUS            |                             |
| CVI_AI_DET_TYPE_TRAIN          |                             |
| CVI_AI_DET_TYPE_TRUCK          |                             |
| CVI_AI_DET_TYPE_BOAT           |                             |
| CVI_AI_DET_TYPE_TRAFFIC_LIGHT  | CVI_AI_DET_GROUP_OUTDOOR    |
| CVI_AI_DET_TYPE_FIRE_HYDRANT   |                             |
| CVI_AI_DET_TYPE_STREET_SIGN    |                             |
| CVI_AI_DET_TYPE_STOP_SIGN      |                             |
| CVI_AI_DET_TYPE_PARKING_METER  |                             |
| CVI_AI_DET_TYPE_BENCH          |                             |
| CVI_AI_DET_TYPE_BIRD           | CVI_AI_DET_GROUP_ANIMAL     |
| CVI_AI_DET_TYPE_CAT            |                             |
| CVI_AI_DET_TYPE_DOG            |                             |
| CVI_AI_DET_TYPE_HORSE          |                             |
| CVI_AI_DET_TYPE_SHEEP          |                             |
| CVI_AI_DET_TYPE_COW            |                             |
| CVI_AI_DET_TYPE_ELEPHANT       |                             |
| CVI_AI_DET_TYPE_BEAR           |                             |
| CVI_AI_DET_TYPE_ZEBRA          |                             |
| CVI_AI_DET_TYPE_GIRAFFE        |                             |
| CVI_AI_DET_TYPE_HAT            | CVI_AI_DET_GROUP_ACCESSORY  |
| CVI_AI_DET_TYPE_BACKPACK       |                             |
| CVI_AI_DET_TYPE_UMBRELLA       |                             |
| CVI_AI_DET_TYPE_SHOE           |                             |
| CVI_AI_DET_TYPE_EYE_GLASSES    |                             |
| CVI_AI_DET_TYPE_HANDBAG        |                             |
| CVI_AI_DET_TYPE_TIE            |                             |
| CVI_AI_DET_TYPE_SUITCASE       |                             |
| CVI_AI_DET_TYPE_FRISBEE        | CVI_AI_DET_GROUP_SPORTS     |
| CVI_AI_DET_TYPE_SKIS           |                             |
| CVI_AI_DET_TYPE_SNOWBOARD      |                             |
| CVI_AI_DET_TYPE_SPORTS_BALL    |                             |
| CVI_AI_DET_TYPE_KITE           |                             |
| CVI_AI_DET_TYPE_BASEBALL_BAT   |                             |
| CVI_AI_DET_TYPE_BASEBALL_GLOVE |                             |
| CVI_AI_DET_TYPE_SKATEBOARD     |                             |
| CVI_AI_DET_TYPE_SURFBOARD      |                             |
| CVI_AI_DET_TYPE_TENNIS_RACKET  |                             |
| CVI_AI_DET_TYPE_BOTTLE         | CVI_AI_DET_GROUP_KITCHEN    |
| CVI_AI_DET_TYPE_PLATE          |                             |
| CVI_AI_DET_TYPE_WINE_GLASS     |                             |
| CVI_AI_DET_TYPE_CUP            |                             |
| CVI_AI_DET_TYPE_FORK           |                             |
| CVI_AI_DET_TYPE_KNIFE          |                             |
| CVI_AI_DET_TYPE_SPOON          |                             |
| CVI_AI_DET_TYPE_BOWL           |                             |
| CVI_AI_DET_TYPE_BANANA         | CVI_AI_DET_GROUP_FOOD       |
| CVI_AI_DET_TYPE_APPLE          |                             |
| CVI_AI_DET_TYPE_SANDWICH       |                             |
| CVI_AI_DET_TYPE_ORANGE         |                             |
| CVI_AI_DET_TYPE_BROCCOLI       |                             |
| CVI_AI_DET_TYPE_CARROT         |                             |
| CVI_AI_DET_TYPE_HOT_DOG        |                             |
| CVI_AI_DET_TYPE_PIZZA          |                             |
| CVI_AI_DET_TYPE_DONUT          |                             |
| CVI_AI_DET_TYPE_CAKE           |                             |
| CVI_AI_DET_TYPE_CHAIR          | CVI_AI_DET_GROUP_FURNITURE  |
| CVI_AI_DET_TYPE_SOFA           |                             |
| CVI_AI_DET_TYPE_POTTED_PLANT   |                             |
| CVI_AI_DET_TYPE_BED            |                             |
| CVI_AI_DET_TYPE_MIRROR         |                             |
| CVI_AI_DET_TYPE_DINING_TABLE   |                             |
| CVI_AI_DET_TYPE_WINDOW         |                             |
| CVI_AI_DET_TYPE_DESK           |                             |
| CVI_AI_DET_TYPE_TOILET         |                             |
| CVI_AI_DET_TYPE_DOOR           |                             |
| CVI_AI_DET_TYPE_TV_MONITOR     | CVI_AI_DET_GROUP_ELECTRONIC |
| CVI_AI_DET_TYPE_LAPTOP         |                             |
| CVI_AI_DET_TYPE_MOUSE          |                             |
| CVI_AI_DET_TYPE_REMOTE         |                             |
| CVI_AI_DET_TYPE_KEYBOARD       |                             |
| CVI_AI_DET_TYPE_CELL_PHONE     |                             |
| CVI_AI_DET_TYPE_MICROWAVE      | CVI_AI_DET_GROUP_APPLIANCE  |
| CVI_AI_DET_TYPE_OVEN           |                             |
| CVI_AI_DET_TYPE_TOASTER        |                             |
| CVI_AI_DET_TYPE_SINK           |                             |
| CVI_AI_DET_TYPE_REFRIGERATOR   |                             |
| CVI_AI_DET_TYPE_BLENDER        |                             |
| CVI_AI_DET_TYPE_BOOK           | CVI_AI_DET_GROUP_INDOOR     |
| CVI_AI_DET_TYPE_CLOCK          |                             |
| CVI_AI_DET_TYPE_VASE           |                             |
| CVI_AI_DET_TYPE_SCISSORS       |                             |
| CVI_AI_DET_TYPE_TEDDY_BEAR     |                             |
| CVI_AI_DET_TYPE_HAIR_DRIER     |                             |
| CVI_AI_DET_TYPE_TOOTHBRUSH     |                             |
| CVI_AI_DET_TYPE_HAIR_BRUSH     |                             |

### 5.1.3. cvai_obj_det_group_type_e

**[说明]**

此enum定义物件类别群组。

| 类别群组                    | 叙述     |
|-----------------------------|----------|
| CVI_AI_DET_GROUP_ALL        | 全部类别 |
| CVI_AI_DET_GROUP_PERSON     | 人形     |
| CVI_AI_DET_GROUP_VEHICLE    | 交通工具 |
| CVI_AI_DET_GROUP_OUTDOOR    | 户外     |
| CVI_AI_DET_GROUP_ANIMAL     | 动物     |
| CVI_AI_DET_GROUP_ACCESSORY  | 配件     |
| CVI_AI_DET_GROUP_SPORTS     | 运动     |
| CVI_AI_DET_GROUP_KITCHEN    | 厨房     |
| CVI_AI_DET_GROUP_FOOD       | 食物     |
| CVI_AI_DET_GROUP_FURNITURE  | 家具     |
| CVI_AI_DET_GROUP_ELECTRONIC | 电子设备 |
| CVI_AI_DET_GROUP_APPLIANCE  | 器具     |
| CVI_AI_DET_GROUP_INDOOR     | 室内用品 |

### 5.1.4. feature_type_e

[enum]

| 数值 | 名称        | 叙述             |
|------|-------------|------------------|
| 0    | TYPE_INT8   | int8_t特征类型   |
| 1    | TYPE_UINT8  | uint8_t特征类型  |
| 2    | TYPE_INT16  | int16_t特征类型  |
| 3    | TYPE_UINT16 | uint16_t特征类型 |
| 4    | TYPE_INT32  | int32_t特征类型  |
| 5    | TYPE_UINT32 | uint32_t特征类型 |
| 6    | TYPE_BF16   | bf16特征类型     |
| 7    | TYPE_FLOAT  | float特征类型    |

### 5.1.5. meta_rescale_type_e

[enum]

| 数值 | 名称             | 叙述              |
|------|------------------|-------------------|
| 0    | RESCALE_UNKNOWN  | 未知              |
| 1    | RESCALE_NOASPECT | 不依比例直接调整  |
| 2    | RESCALE_CENTER   | 在四周进行padding |
| 3    | RESCALE_RB       | 在右下进行padding |

### 5.1.6. cvai_bbox_t

| 类型  | 参数名称 | 叙述                    |
|-------|----------|-------------------------|
| float | x1       | 侦测框左上点坐标之 x 值 |
| float | y1       | 侦测框左上点坐标之 y 值 |
| float | x2       | 侦测框右下点坐标之 x 值 |
| float | y2       | 侦测框右下点坐标之 y 值 |
| float | score    | 侦测框之信心程度        |

### 5.1.7. cvai_feature_t

| 类型           | 参数名称 | 叙述     |
|----------------|----------|----------|
| int8_t\*       | ptr      | 位址     |
| uint32_t       | size     | 特征维度 |
| feature_type_e | type     | 特征型态 |

### 5.1.8. cvai_pts_t

| 类型     | 参数名称 | 叙述       |
|----------|----------|------------|
| float\*  | x        | 坐标x      |
| float\*  | y        | 坐标y      |
| uint32_t | size     | 坐标点个数 |

### 5.1.9. cvai_4_pts_t

| 类型  | 参数名称 | 叙述               |
|-------|----------|--------------------|
| float | x[4]     | 4个坐标点之x坐标值 |
| float | y[4]     | 4个坐标点之y坐标值 |

### 5.1.10. cvai_vpssconfig_t

| 类型              | 参数名称  | 叙述         |
|-------------------|-----------|--------------|
| VPSS_SCALE_COEF_E | chn_coeff | Rescale方式  |
| VPSS_CHN_ATTR_S   | chn_attr  | VPSS属性资料 |

### 5.1.11. cvai_tracker_t

| 类型                   | 参数名称 | 叙述         |
|------------------------|----------|--------------|
| uint32_t               | size     | 追踪讯息数量 |
| cvai_tracker_info_t \* | info     | 追踪讯息结构 |

### 5.1.12. cvai_tracker_info_t

| 类型                  | 参数名称 | 叙述                   |
|-----------------------|----------|------------------------|
| cvai_trk_state_type_t | state    | 追踪状态               |
| cvai_bbox_t           | bbox     | 追踪预测之Bounding Box |

### 5.1.13. cvai_trk_state_type_t

[enum]

| 数值 | 名称                 | 叙述             |
|------|----------------------|------------------|
| 0    | CVI_TRACKER_NEW      | 追踪状态为新增   |
| 1    | CVI_TRACKER_UNSTABLE | 追踪状态为不稳定 |
| 2    | CVI_TRACKER_STABLE   | 追踪状态为稳定   |

### 5.1.14. cvai_deepsort_config_t

| 类型                         | 参数名称                                | 叙述                                   |
|------------------------------|-----------------------------------------|----------------------------------------|
| float                        | max_distance_iou                        | 进行BBox匹配时最大IOU距离              |
| float                        | max_distance_consine                    | 进行Feature匹配时最大consine距离       |
| int                          | max_unmatched_times \_for_bbox_matching | 参与BBox匹配的目标最大未匹配次数之数量 |
| cvai_kalman_filter_config_t  | kfilter_conf                            | Kalman Filter设定                      |
| cvai_kalman_tracker_config_t | ktracker_conf                           | Kalman Tracker 设定                    |

### 5.1.15. cvai_kalman_filter_config_t

| 类型     | 参数名称    | 叙述                   |
|----------|-------------|------------------------|
| float[8] | Q_std_alpha | Process Noise 参数     |
| float[8] | Q_std_beta  | Process Noise 参数     |
| int[8]   | Q_std_x_idx | Process Noise 参数     |
| float[4] | R_std_alpha | Measurement Noise 参数 |
| float[4] | R_std_beta  | Measurement Noise 参数 |
| int[4]   | R_std_x_idx | Measurement Noise 参数 |

**[说明]**

对于追踪目标运动状态X

Process Nose (运动偏差), Q, 其中

Measurement Nose (量测偏差), R, 同理运动偏差公式

### 5.1.16. cvai_kalman_tracker_config_t

| 类型     | 参数名称                | 叙述                          |
|----------|-------------------------|-------------------------------|
| int      | max_unmatched_num       | 追踪目标最大遗失数            |
| int      | accreditation_threshold | 追踪状态转为稳定之阀值        |
| int      | feature_budget_size     | 保存追踪目标feature之最大数量 |
| int      | feature_update_interval | 更新feature之时间间距         |
| float[8] | P_std_alpha             | Initial Covariance 参数       |
| float[8] | P_std_beta              | Initial Covariance 参数       |
| int[8]   | P_std_x_idx             | Initial Covariance 参数       |

**[说明]**

Initial Covariance (初始运动状态偏差), P, 同理运动偏差公式

### 5.1.17. cvai_liveness_ir_position_e

[enum]

| 数值 | 名称              | 叙述                |
|------|-------------------|---------------------|
| 0    | LIVENESS_IR_LEFT  | IR镜头在RGB镜头左侧 |
| 1    | LIVENESS_IR_RIGHT | IR镜头在RGB镜头右侧 |

### 5.1.18. cvai_head_pose_t

| 类型     | 参数名称               | 叙述           |
|----------|------------------------|----------------|
| float    | yaw                    | 偏摆角         |
| float    | pitch                  | 俯仰角         |
| float    | roll                   | 翻滚角         |
| float[3] | facialUnitNormalVector | 脸部之面向方位 |

### 5.1.19. cvai_face_info_t

| 类型                | 参数名称       | 叙述             |
|---------------------|----------------|------------------|
| char                | name[128]      | 人脸名           |
| uint64_t            | unique_id      | 人脸ID           |
| cvai_bbox_t         | bbox           | 人脸侦测框       |
| cvai_pts_t          | pts            | 人脸特征点       |
| cvai_feature_t      | feature        | 人脸特征         |
| cvai_face_emotion_e | emotion        | 表情             |
| cvai_face_gender_e  | gender         | 性别             |
| cvai_face_race_e    | race           | 种族             |
| float               | age            | 年龄             |
| float               | liveness_score | 活体机率值       |
| float               | mask_score     | 人脸戴口罩机率值 |
| float               | face_quality   | 人脸质量         |
| cvai_head_pose_t    | head_pose      | 人脸角度信息     |

### 5.1.20. cvai_face_t

| 类型               | 参数名称 | 叙述         |
|--------------------|----------|--------------|
| uint32_t           | size     | 人脸个数     |
| uint32_t           | width    | 原始图片之宽 |
| uint32_t           | height   | 原始图片之高 |
| cvai_face_info_t\* | info     | 人脸综合信息 |

### 5.1.21. cvai_pose17_meta_t

| 类型  | 参数名称  | 叙述                       |
|-------|-----------|----------------------------|
| float | x[17]     | 17个骨骼关键点的x坐标      |
| float | y[17]     | 17个骨骼关键点的y坐标      |
| float | score[17] | 17个骨骼关键点的预测信心值 |

### 5.1.22. cvai_vehicle_meta

| 类型         | 参数名称     | 叙述             |
|--------------|--------------|------------------|
| cvai_4_pts_t | license_pts  | 车牌4个角坐标    |
| cvai_bbox_t  | license_bbox | 车牌Bounding Box |
| char[255]    | license_char | 车牌号码         |

**[说明]**

车牌4个角坐标依序为左上、右上、右下至左下。

### 5.1.23. cvai_class_filter_t

| 类型       | 参数名称              | 叙述               |
|------------|-----------------------|--------------------|
| uint32_t\* | preserved_class_ids   | 要保留的类别id     |
| uint32_t   | num_preserved_classes | 要保留的类别id个数 |

### 5.1.24. cvai_dms_t

| 类型             | 参数名称      | 叙述                            |
|------------------|---------------|---------------------------------|
| float            | reye_score    | 右眼开合分数                    |
| float            | leye_score    | 左眼开合分数                    |
| float            | yawn_score    | 嘴巴闭合分数                    |
| float            | phone_score   | 讲电话分数                      |
| float            | smoke_score   | 抽烟分数                        |
| cvai_pts_t       | landmarks_106 | 106个特征点                     |
| cvai_pts_t       | landmarks_5   | 5个特征点                       |
| cvai_head_pose_t | head_pose     | 透过106个特征点算出来的人脸角度 |
| cvai_dms_od_t    | dms_od        | 车内的物件侦测结果              |

### 5.1.25. cvai_dms_od_t

| 类型                 | 参数名称     | 叙述          |
|----------------------|--------------|---------------|
| uint32_t             | size         | 有几个物件    |
| uint32_t             | width        | 宽度          |
| uint32_t             | height       | 长度          |
| meta_rescale_type_e  | rescale_type | rescale的形态 |
| cvai_dms_od_info_t\* | info         | 物件的资讯    |

### 5.1.26. cvai_dms_od_info_t

| 类型        | 参数名称 | 叙述             |
|-------------|----------|------------------|
| char[128]   | name     | 物体名称         |
| int         | classes  | 物体类别         |
| cvai_bbox_t | bbox     | 物体Bounding Box |

### 5.1.27. cvai_face_emotion_e

**[说明]**

人脸表情Enmu

| 表情             | 叙述 |
|------------------|------|
| EMOTION_UNKNOWN  | 未知 |
| EMOTION_HAPPY    | 高兴 |
| EMOTION_SURPRISE | 惊讶 |
| EMOTION_FEAR     | 恐惧 |
| EMOTION_DISGUST  | 厌恶 |
| EMOTION_SAD      | 伤心 |
| EMOTION_ANGER    | 生气 |
| EMOTION_NEUTRAL  | 自然 |

### 5.1.28. cvai_face_race_e

| 种族           | 叙述     |
|----------------|----------|
| RACE_UNKNOWN   | 未知     |
| RACE_CAUCASIAN | 高加索人 |
| RACE_BLACK     | 黑人     |
| RACE_ASIAN     | 亚洲人   |

### 5.1.29. cvai_pedestrian_meta

| 类型               | 参数名称 | 叙述         |
|--------------------|----------|--------------|
| cvai_pose17_meta_t | pose17   | 人体17关键点 |
| bool               | fall     | 受否跌倒     |

### 5.1.30. cvai_object_info_t

| 类型                 | 参数名称            | 叙述         |
|----------------------|---------------------|--------------|
| char                 | name                | 物件类别名   |
| uint64_t             | unique_id           | Unique id    |
| cvai_box_t           | bbox                | Bounding box |
| cvai_feature_t       | feature             | 物件特征     |
| int                  | classes             | 类别ID       |
| cvai_vehicle_meta    | vehicle_property    | 车辆属性     |
| cvai_pedestrian_meta | pedestrian_property | 行人属性     |

### 5.1.31. cvai_object_t

| 类型                  | 参数名称     | 叙述                       |
|-----------------------|--------------|----------------------------|
| uint32_t              | size         | info所含物件个数           |
| uint32_t              | width        | 原始图片之宽               |
| uint32_t              | height       | 原始图片之高               |
| meta_rescale_type_e   | rescale_type | 模型前处理采用的resize方式 |
| cvai_object_info_t \* | info         | 物件信息                   |

### 5.1.32. CVI_AI_Service

### 5.1.33. cvai_service_feature_matching_e

**[说明]**

特征比对计算方法，目前仅支援Cosine Similarity。

[定义]

| 名称           | 叙述              |
|----------------|-------------------|
| COS_SIMILARITY | Cosine similarity |

### 5.1.34. cvai_service_feature_array_t

**[说明]**

特征阵列，此结构包含了特征阵列指标, 长度, 特征个数, 及特征类型等信息。在注册特征库时需要传入此结构。

[定义]

| 类型           | 参数名称       | 叙述         |
|----------------|----------------|--------------|
| int8_t \*      | ptr            | 特征阵列指标 |
| uint32_t       | feature_length | 单一特征长度 |
| uint32_t       | data_num       | 特征个数     |
| feature_type_e | type           | 特征类型     |

### 5.1.35. cvai_service_brush_t

**[说明]**

绘图笔刷结构，可指定欲使用之RGB及笔刷大小。

[定义]

| 类型            | 参数名称 | 叙述          |
|-----------------|----------|---------------|
| Inner structure | color    | 欲使用的RGB值 |
| uint32_t        | size     | 笔刷大小      |

# 6. 范例程序

## 6.1. 编译范例程序

1.  下载AI SDK与其依赖之SDK：MW、TPU、CVI Tracer、IVE(182x/183x系列使用 TPU-IVE SDK，181x系列使用IVE SDK)并解压缩。
    1.  182x/183x系列下载以下SDK
    2.  181x系列下载以下SDK
1.  移动至AI SDK的sample目录
2.  执行make指令：
    1.  **CV183x 64bit glibc**
    2.  **CV182x 32bit uclibc**
    3.  **CR182x riscv-64 musl**
3.  执行make install

    编译完成的可执行档会放在AI SDK的tmp_install目录下

## 6.2. 物件侦测

### 6.2.1. sample_vi_od

**[简述]**

侦测物件并显示侦测框于画面中透过RTSP输出。可用VLC开启rtsp://[ip]/h264

**[用法]**
```
./sample_vi_od \<model_name\> \<model_path\> \<threshold\>
```
参数说明：

| 参数       | 说明                                                                                                                                                                                                               |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_name | 模型名称，可接受参数：mobiledetv2-lite, mobiledetv2-lite-person-pets, mobiledetv2-d0, mobiledetv2-d1, mobiledetv2-d2, mobiledetv2-vehicle-d0, mobiledetv2-pedestrian-d0, mobiledetv2-person-vehicle, yolov3, yolox |
| model_path | 模型档案路径。                                                                                                                                                                                                     |
| threshold  | 物件侦测分数阀值，范围为0.0\~1.0。                                                                                                                                                                                 |

**[范例]**

以下范例为使用mobiledetv2-d0-ls.cvimodel模型进行侦测，threshold为0.5。并使用RTSP进行串流。
```
./sample_vi_od mobiledetv2-coco80 mobiledetv2-d0-ls.cvimodel 0.5
```
**[输出]**

Console显示找到的物件数量。Video output显示侦测框。

### 6.2.2. sample_vi_obj_counting

**[简述]**

流量统计，物件计数器，以人、汽车为例。

**[用法]**
```
./sample_vi_obj_counting

\<detection_model_name\>

\<detection_model_path\>

\<reid_model_path\>

\<use_stable_counter\>
```
参数说明：

| 参数                 | 说明                                                                                  |
|----------------------|---------------------------------------------------------------------------------------|
| detection_model_name | 物件侦测模型名称，可接受参数： mobiledetv2-d0, mobiledetv2-d1, mobiledetv2-d2, yolov3 |
| detection_model_path | 物件侦测模型档案路径。                                                                |
| reid_model_path      | ReID模型档案路径                                                                      |
| use_stable_counter   | 是否只考虑稳定追踪状态之物件                                                          |

**[范例]**
```
./sample_vi_obj_counting

mobiledetv2-coco80

model/mobiledetv2-d0-ls.cvimodel

model/person-reid-v1.cvimodel
```
1

**[输出]**

Console显示人、汽车个别流量

## 6.3. 物件追踪

### 6.3.1. sample_vi_object_tracking

**[简述]**

人型追踪。包含物件侦测、多物件追踪算法。

**[用法]**
```
./sample_vi_people_tracking

\<detection_model_name\>

\<detection_model_path\>

\<reid_model_path\>
```
参数说明：

| 参数                 | 说明                                                                                                                 |
|----------------------|----------------------------------------------------------------------------------------------------------------------|
| detection_model_name | 物件侦测模型名称，可接受参数： mobiledetv2-d0, mobiledetv2-d1, mobiledetv2-d2, mobiledetv2-pedestrian, yolov3, yolox |
| detection_model_path | 物件侦测模型路径。                                                                                                   |
| reid_model_path      | ReID模型路径                                                                                                         |

**[范例]**
```
./sample_vi_object_tracking

mobiledetv2-pedestrian

model/mobiledetv2-pedestrian-d0.cvimodel

model/person-reid-v1.cvimodel
```
**[输出]**

Video output标示人型bounding box与ID。

## 6.4. 人脸侦测

### 6.4.1. sample_vi_fd

**[简述]**

人脸侦测。

**[用法]**
```
./sample_vi_fd

\<retina_model_path\>
```
**[范例]**
```
./sample_vi_fd retinaface_mnet0.25_608_342.cvimodel
```
参数说明：

| 参数              | 说明                 |
|-------------------|----------------------|
| retina_model_path | RetinaFace模型路径。 |

### 6.4.2. sample_vi_fdmask

**[简述]**

人脸侦测。

**[用法]**

./sample_vi_fdmask \<model_path\>

**[范例]**
```
./sample_vi_fdmask yolox_RetinafaceMask_Im_342_768_int8.cvimodel
```
参数说明：

| 参数       | 说明             |
|------------|------------------|
| model_path | fdmask模型路径。 |

**[输出]**

Console显示找到人脸的数量。Video output显示侦测结果，并标注戴口罩分数。

## 6.5. 运动侦测

### 6.5.1. sample_vi_md

**[简述]**

此范例会匡出和背景不同的物件，每10000张frame会更新一次背景。

**[用法]**
```
./sample_vi_md \<threshold\> \<min_area\>
```
参数说明：

| 参数      | 说明                                                            |
|-----------|-----------------------------------------------------------------|
| threshold | 物件侦测分数阀值，范围为0\~255。                                |
| min_area  | 最小物件面积，如希望侦测大于100x100物件，可以指定min_area=10000 |

**[范例]**

以下范例为threshold=30, 最小侦测物件为100x100, 并透过RTSP串流。
```
./sample_vi_md 30 10000
```
**[输出]**

Console显示找到的物件数量。Video output显示侦测框。

## 6.6. 车牌辨识

### 6.6.1. sample_vi_lpdr

**[简述]**

车牌辨识。包含车辆侦测、车牌侦测与车牌识别算法。

**[用法]**
```shell
./sample_vi_lpdr \
<detection model name> \
<detection_model_path> \
<license_plate_detection_model_path> \
<license_plate_recognition_model_path> \
<license_format (tw/cn)>
```
参数说明：

| 参数                                 | 说明                                                                                                                                             |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| detection_model_name                 | 物件侦测模型名称，可接受参数： mobiledetv2-d0, mobiledetv2-d1, mobiledetv2-d2, mobiledetv2-vehicle, mobiledetv2-person-vehicle-d0, yolov3, yolox |
| detection_model_path                 | 物件侦测模型路径。                                                                                                                               |
| license_plate_detection_model_path   | 车牌侦测模型路径                                                                                                                                 |
| license_plate_recognition_model_path | 车牌识别模型路径                                                                                                                                 |
| license_format                       | 车牌格式。tw: 台湾地区 / cn: 大陆地区                                                                                                            |

**[范例]**
```
./sample_vi_lpdr
mobiledetv2-vehicle
model/mobiledetv2-vehicle-d0-ls.cvimodel
model/wpodnet_v0_bf16.cvimodel
model/lprnet_v0_tw_bf16.cvimodel
cn
```
**[输出]**

`Console`显示找到车辆的数量、个别车辆是否有车牌、车牌号码。

## 6.7. 入侵侦测

### 6.7.1. sample_vi_intrusion_det

**[简述]**

入侵侦测。

**[用法]**
```
./sample_vi_intrusion_det

\<object_detection_model_name\>

\<object_detection_model_path\>

\<threshold (optional): threshold for detection model\>
```
参数说明：

| 参数                        | 说明                                                                                                                                                                                               |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| object_detection_model_name | 物件侦测模型名称，可接受参数：mobiledetv2-lite, mobiledetv2-lite-person-pets, mobiledetv2-d0, mobiledetv2-d1, mobiledetv2-d2, mobiledetv2-pedestrian-d0, mobiledetv2-person-vehicle, yolov3, yolox |
| object_detection_model_path | 物件侦测模型路径。                                                                                                                                                                                 |
| threshold                   | 物件侦测分数阀值，范围为0.0\~1.0。                                                                                                                                                                 |

**[范例]**

./sample_vi_intrusion_det mobiledetv2-d0 model/mobiledetv2-d0-ls.cvimodel 0.5

**[输出]**

Video output显示物件bounding box是否有入侵预先定义之区域。

#### 6.7.1.1. 环境音检测

#### 6.7.1.2. sample_aud_esc

**[简述]**

从MIC读入音讯，进行环境音侦测。可侦测类别列表如下：

| 类别Id | 类别           | 说明        |
|--------|----------------|-------------|
| 0      | Sneezing       | 打喷嚏      |
| 1      |                |             |
| 2      | Clapping       | 拍手        |
| 3      | Baby Cry       | 婴儿哭声    |
| 4      | Glass breaking | 玻璃破碎声  |
| 5      | Office         | 办公室/背景 |

**[用法]**
```
./sample_aud_esc
\<esc_model_path\>
\<record 0 or 1\>
\<output file path\>
```
参数说明：

| 参数             | 说明                                      |
|------------------|-------------------------------------------|
| esc_model_path   | 环境音侦测模型档案路径，可接受参数。      |
| record           | 是否录制MIC声音。0表示不录制，1表示录制。 |
| output file path | 若record为1，需指定路径输出档案位置       |

**[范例]**
```
./sample_read_esc esc_classification.cvimodel 0
```
**[输出]**

Console中显示侦测到的环境音类别。

## 6.8. APP

### 6.8.1. 人脸抓拍

#### 6.8.1.1. sample_app_face_capture

**[简述]**

人脸抓拍。人脸侦测并进行追踪算法，依据不同模式，抓拍/撷取人脸的照片

**[用法]**
```shelll
./sample_app_face_capture

\<face_detection_model_path\>

\<face_quality_model_path\>

\<config_path\>

\<mode, 0: face, 1: interval, 2: leave, 3: intelligent\>

\<use FQNet (0/1)\>

\<tracking buffer size\>

\<FD threshold\>

\<write image (0/1)\>

\<video output, 0: disable, 1: output to panel, 2: output through rtsp\>
```

参数说明：

| 参数                      | 说明                                           |
|---------------------------|------------------------------------------------|
| face_detection_model_path | 人脸侦测模型路径。                             |
| face_quality_model_path   | 人脸质量评估模型路径。                         |
| config_path               | 设定档路径                                     |
| mode                      | 模式，详细功能参阅下方**[说明]**栏。               |
| use FQNet                 | 是否启用人脸质量评估模型。                     |
| tracking buffer size      | APP追踪的人脸暂存区大小。                      |
| FD threshold              | 人脸侦测分数阀值，范围为0.0\~1.0。             |
| write image               | 是否将人脸影像输出至档案。                     |
| video_output              | 0表示不输出 1表示输出到Panel 2表示透过Rtsp输出 |

**[范例]**
```shell
./sample_app_face_capture
model/retinaface_mnet0.25_342_608.cvimodel
model/fqnet-v5_shufflenetv2-softmax.cvimodel
config/cfg_app_face_capture.txt
0 0 10 0.5 1 2
```
**[输出]**

若启用write image，APP会将face image写入images/ 这个资料夹中。

![一張含有 地板, 室內, 個人, 直立的 的圖片 自動產生的描述](assets/16096ab904544532efe9b45623f71a65.jpeg)

绿框表示Face Quality高于门槛，蓝框则为低于门槛。框框上方数字格式为 [user ID] quality，客户能观察quality变化，并针对不同场景来设置适合的threshold。

**[说明]**

模式说明：

1.  快速 (`fast`)
    -   当face进入画面后，以间隔 10 `frames` (可设定) 连续抓拍3次 (可设定)，并纪录每个间隔中质量最佳的face。（输出起始快照3张）
2.  间隔 (`interval`)
    -   当face进入画面后，以间隔20 `frames` (可设定) 抓拍人脸。（间隔输出）
3.  离开 (`leave`)
    -   当`face`进入画面后，每一张`frame`更新最佳的`face`，当face`离开画面40 `frames` (可设定) 后，系统会判定此人已离开。（输出离开前最佳`face`）
4.  智能 (`intelligent`)
    -   当`face`进入画面后，每一张 `frame`更新最佳的`face`，当`face`离开画面40 `frames` (可设定) 后，系统会判定此人已离开。（输出初始快照与离开前最佳`face`）

# 7. 错误码

| 错误代码   | 定义                          | 叙述                     |
|------------|-------------------------------|--------------------------|
| 0xFFFFFFFF | CVIAI_FAILURE                 | API 调用失败             |
| 0xC0010101 | CVIAI_ERR_INVALID_MODEL_PATH  | 不正确的模型路径         |
| 0xC0010102 | CVIAI_ERR_OPEN_MODEL          | 开启模型失败             |
| 0xC0010103 | CVIAI_ERR_CLOSE_MODEL         | 关闭模型失败             |
| 0xC0010104 | CVIAI_ERR_GET_VPSS_CHN_CONFIG | 取得VPSS CHN设置失败     |
| 0xC0010105 | CVIAI_ERR_INFERENCE           | 模型推理失败             |
| 0xC0010106 | CVIAI_ERR_INVALID_ARGS        | 不正确的参数             |
| 0xC0010107 | CVIAI_ERR_INIT_VPSS           | 初始化VPSS失败           |
| 0xC0010108 | CVIAI_ERR_VPSS_SEND_FRAME     | 送Frame到VPSS时失败      |
| 0xC0010109 | CVIAI_ERR_VPSS_GET_FRAME      | 从VPSS取得Frame失败      |
| 0xC001010A | CVIAI_ERR_MODEL_INITIALIZED   | 模型未开启               |
| 0xC001010B | CVIAI_ERR_NOT_YET_INITIALIZED | 功能未初始化             |
| 0xC001010C | CVIAI_ERR_NOT_YET_IMPLEMENTED | 功能尚未实作             |
| 0xC001010D | CVIAI_ERR_ALLOC_ION_FAIL      | 分配ION内存失败          |
| 0xC0010201 | CVIAI_ERR_MD_OPERATION_FAILED | 运行Motion Detection失败 |
