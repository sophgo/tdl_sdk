.. vim: syntax=rst

API参考
================

句柄
~~~~~~~~~~~~~~~

【语法】

.. code-block:: c
  
  typedef void *cviai_handle_t;
  typedef void *cviai_service_handle_t;

【描述】

AI SDK的句柄，不同模块之间有各自的句柄，但是创建 ``cviai_service_handle_t`` 模块时会需要使用到 ``cviai_handle_t`` 作为输入。

CVI_AI_Core
~~~~~~~~~~~~~~~

Common
^^^^^^^^^

CVI_AI_CreateHandle
-------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_CreateHandle(cviai_handle_t *handle);

【描述】

创建使用AI SDK 句柄。AI SDK会自动创建VPSS Group。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入/输出
     - cviai_handle_t\*
     - handle
     - 输入句柄指针  


CVI_AI_CreateHandle2
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_CreateHandle2(cviai_handle_t *handle, const VPSS_GRP vpssGroupId, const CVI_U8 vpssDev);

【描述】

创建使用AI SDK句柄，并使用指定的VPSS Group ID及Dev ID创建VPSS。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输出
     - cviai_handle_t\*
     - handle
     - 输入句柄指针 

   * - 输入
     - VPSS_GRP
     - vpssGroupId
     - VPSS使用的group id       

   * - 输入
     - CVI_U8  
     - vpssDev
     - VPSS Device id 


CVI_AI_DestroyHandle
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DestroyHandle(cviai_handle_t handle);

【描述】

销毁创造的句柄cviai_handle_t。同时销毁所有开启的模型

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 输入句柄        


CVI_AI_GetModelPath
-------------------

.. code-block:: c
  
  const char *CVI_AI_GetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, char **filepath);

【描述】

取得内部已经设置支持的模型的模型路径。使用完毕需要自行释放filepath之变量。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入
     - cviai_handle_t     
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E   
     - model  
     - 模型ID       


【输出】

.. list-table::
   :widths: 33 33 33
   :header-rows: 1


   * -
     - 数据型态
     - 说明

   * - 输出
     - char\*       
     - 模型路径指针     


CVI_AI_OpenModel
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_OpenModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const char *filepath);

【描述】

开启并初始化模型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入
     - cviai_handle_t      
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E
     - model  
     - 模型 index   

   * - 输入
     - const char\*       
     - filepath
     - cvimodel模型路径 


CVI_AI_SetSkipVpssPreprocess
----------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, bool skip);

【描述】

指定model不进行预处理。

AI SDK在默认情况下会使用内部创建的VPSS进行模型的预处理(skip = false)。

当skip为true时，AI SDK将不会对该模型进行预处理。

模型输入必须由外部进行预处理后，再输入模型。

通常用于VI直接Binding VPSS且只使用单一模型的状况。

可以使用 `CVI_AI_GetVpssChnConfig`_ 来取得模型的VPSS预处理参数。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入
     - cviai_handle_t     
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E   
     - model  
     - 模型ID       

   * - 输入
     - bool   
     - skip   
     - 是否跳过前处理   


CVI_AI_GetSkipVpssPreprocess
----------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_GetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, bool *skip);

【描述】

询问模型是否会在AI SDK内进行预处理。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入
     - cviai_handle_t     
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E   
     - model  
     - 模型ID       

   * - 输出
     - bool\* 
     - skip   
     - 是否跳过前处理   


CVI_AI_SetVpssThread
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const uint32_t thread);

【描述】

设置特定模型使用的线程id。在AI SDK内，一个Vpss Thread代表一组Vpss Group设置。

默认使用Thread 0为模型使用的Vpss Group。

当在多线程上各自使用同一个AI SDK Handle来进行模型推理时，必须使用此API指定不同的Vpss Thread来避免Race Condition。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入
     - cviai_handle_t     
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E   
     - model  
     - 模型ID       

   * - 输入
     - uint32_t 
     - thread 
     - 线程id       


CVI_AI_SetVpssThread2
---------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const uint32_t thread, const VPSS_GRP vpssGroupId);

【描述】

同CVI_AI_SetVpssThread。可以指定Vpss Group ID。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t     
     - handle
     - 句柄       

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E   
     - model 
     - 模型ID     

   * - 输入
     - uint32_t 
     - thread
     - 线程id     

   * - 输入
     - VPSS_GRP 
     - vpssGroupId
     - VPSS Group id  


CVI_AI_GetVpssThread
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, uint32_t *thread);

【描述】

取得模型使用的thread id。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明
  

   * - 输入
     - cviai_handle_t     
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E   
     - model  
     - 模型ID       

   * - 输出
     - uint32_t\*         
     - thread 
     - VPSS线程id   


【语法】

.. code-block:: c

CVI_S32 CVI_AI_GetVpssGrpIds
----------------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_GetVpssGrpIds(cviai_handle_t handle, VPSS_GRP **groups, uint32_t *num);

【描述】

取得句柄内全部使用到的Vpss group id，使用完毕后groups要自行释放。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄         

   * - 输出
     - VPSS_GRP \*\* 
     - groups   
     - 空指针的参考 

   * - 输出
     - uint32_t\*
     - num  
     - groups的长度 


CVI_AI_SetVpssTimeout
---------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetVpssTimeout(cviai_handle_t handle, uint32_t timeout);

【描述】

设置AI SDK等待VPSS硬件超时的时间，预设为100ms。

此设置适用于所有AI SDK内的VPSS Thread。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄        

   * - 输入
     - uint32_t  
     - timeout   
     - 超时时间    


CVI_AI_SetVBPool
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL pool_id);

【描述】

指定VBPool给AI SDK内部VPSS。指定后，AI SDK内部VPSS会直接从此Pool拿取内存。

若不用此API指定Pool，默认由系统自动分配。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄        

   * - 输入
     - uint32_t  
     - thread
     - VPSS线程id  

   * - 输入
     - VB_POOL   
     - pool_id   
     - VB Pool Id。若设置为INVALID_POOLID， 表示不指定Pool，由系统自动分配。


CVI_AI_GetVBPool
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetVBPool(cviai_handle_t handle, uint32_t thread, VB_POOL *pool_id);

【描述】

取得指定VPSS使用的VBPool Id。若未使用 `CVI_AI_SetVBPool`_ 指定Pool，则会得到INVALID_POOLID。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄        

   * - 输入
     - uint32_t  
     - thread
     - VPSS线程id  

   * - 输出
     - VB_POOL\*
     - pool_id   
     - 目前使用的VB Pool Id。  


CVI_AI_CloseAllModel
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle);

【描述】

卸除所有在句柄中已经加载的模型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄        


CVI_AI_CloseModel
-----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model);

【描述】

卸除特定在句柄中已经加载的模型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄  

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E
     - model   
     - 模型index         


CVI_AI_Dequantize
-----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Dequantize(const int8_t *quantizedData, float *data, const uint32_t bufferSize, const float dequantizeThreshold);

【描述】

Dequantize int8数值到Float。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const int8_t\*  
     - quantizedData 
     - Int8数据 

   * - 输出
     - float\*
     - data      
     - Float输出数据

   * - 输入
     - const uint32_t   
     - bufferSize
     - Int8数据数量 

   * - 输入
     - const float  
     - dequantizeThreshold
     - 量化阀值 


CVI_AI_ObjectNMS
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_ObjectNMS(const cvai_object_t *obj, cvai_object_t *objNMS, const float threshold, const char method);

【描述】

对cviai_object_t内的bbox做Non-Maximum Suppression算法。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const cvai_object_t\*
     - obj 
     - 想进行NMS的Object Meta

   * - 输出
     - cvai_object_t\*  
     - objNMS
     - NMS后的结果

   * - 输入
     - const float       
     - threshold
     - IOU threshold 

   * - 输入
     - const char        
     - method  
     - 'u': Intersection over Union
      
       'm': Intersection over min area


CVI_AI_FaceNMS
--------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_ObjectNMS(const cvai_face_t *face, cvai_face_t *faceNMS, const float threshold, const char method);

【描述】

对 cviai_object_t 内的bbox做Non-Maximum Suppression算法。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const cvai_face_t\*
     - face
     - 想进行NMS的face meta  

   * - 输出
     - cvai_face_t\*
     - faceNMS 
     - NMS后的结果   

   * - 输入
     - const float       
     - threshold
     - IOU threshold 

   * - 输入
     - const char        
     - method  
     - 'u': Intersection over Union

       'm': Intersection over min area


CVI_AI_FaceAlignment
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceAlignment(VIDEO_FRAME_INFO_S *inFrame, const uint32_t metaWidth, const uint32_t metaHeight, const cvai_face_info_t *info, VIDEO_FRAME_INFO_S *outFrame, const bool enableGDC);

【描述】

对inFrame图像face进行Face Alignment，采用InsightFace Alignment参数。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - inFrame   
     - 输入图像     

   * - 输入
     - const uint32_t metaWidth 
     - metaWidth 
     - Info中frame的宽度

   * - 输入
     - const uint32_t metaHeight
     - metaHeight
     - Info中frame的高度

   * - 输入
     - const cvai_face_info_t\*
     - info  
     - Face info    

   * - 输出
     - VIDEO_FRAME_INFO_S\*
     - outFrame  
     - Face Alignment后的人脸图像

   * - 输入
     - const bool       
     - enableGDC 
     - 是否使用GDC硬件  


CVI_AI_CropImage
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_CropImage(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst, cvai_bbox_t *bbox, bool cvtRGB888);

【描述】

从srcFrame图像中截取bbox指定区域图像。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - srcFrame 
     - 输入图像，目前仅支持RGB Packed格式   

   * - 输出
     - cvai_image_t\*
     - dst  
     - 输出图像     

   * - 输入
     - cvai_bbox_t\*
     - bbox 
     - Bounding box 

   * - 输入
     - bool 
     - cvtRGB888
     - 是否转换成RGB888格式输出 


CVI_AI_CropImage_Face
---------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_CropImage_Face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst, cvai_face_info_t *face_info, bool align);

【描述】

从srcFrame图像中截取face bbox指定范围图像。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - srcFrame 
     - 输入图像，目前仅支持RGB Packed格式   

   * - 输出
     - cvai_image_t\*
     - dst  
     - 输出图像     

   * - 输入
     - cvai_face_info_t\*
     - face_info
     - 指定的face info  

   * - 输入
     - bool 
     - align
     - 是否进行facealig nmen。采用InsightFace Alignment参数。  

   * - 输入
     - bool 
     - cvtRGB888
     - 是否转换成RGB888格式输出 


CVI_AI_SoftMax
--------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SoftMax(const float *inputBuffer, float *outputBuffer, const uint32_t bufferSize);

【描述】

对inputBuffer计算Softmax。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const float\*
     - inputBuffer
     - 想进行softmax的缓冲  

   * - 输出
     - const float\*
     - outputBuffer
     - Softmax后的结果  

   * - 输入
     - const uint32_t   
     - bufferSize
     - 缓冲大小     


CVI_AI_GetVpssChnConfig
-----------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, const CVI_U32 frameWidth, const CVI_U32 frameHeight, const CVI_U32 idx, cvai_vpssconfig_t *chnConfig);

【描述】

取得在模型预处理使用的VPSS参数。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E
     - model   
     - 模型id 

   * - 输入
     - CVI_U32        
     - fr   ameWidth
     - 输入图像宽       

   * - 输入
     - CVI_U32        
     - fra  meHeight
     - 输入图像高       

   * - 输入
     - CVI_U32        
     - idx 
     - 模型的输入index  

   * - 输出
     - cvai_vpssconfig_t\*
     - chnConfig
     - 回传的参数设定值 


CVI_AI_Free
-----------

.. code-block:: none
  
  CVI_A_Free(X)

【描述】

释放模型结果产生的数据结构。某些数据结构中包含malloc出来的子项，因此需要做释放。

【参数】

以下为支持的输入变量

-  `cvai_feature_t <6_Data_Types.html#cvai-feature-t>`__

-  `cvai_pts_t <6_Data_Types.html#cvai-pts-t>`__

-  `cvai_tracker_t <6_Data_Types.html#cvai-tracker-t>`__

-  `cvai_face_info_t <6_Data_Types.html#cvai-face-info-t>`__

-  `cvai_face_t <6_Data_Types.html#cvai-face-t>`__

-  `cvai_object_info_t <6_Data_Types.html#cvai-object-info-t>`__

-  `cvai_object_t <6_Data_Types.html#cvai-object-t>`__

CVI_AI_CopyInfo
---------------

.. code-block:: none
  
  CVI_A_CopyInfo(IN, OUT)

【描述】

泛型拷贝cviai结构API。malloc内部的指针空间并做完整复制。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - 支持型态：    
    
       cvai_face_info_t  
       
       cvai_object_info_t  
       
       cvai_image_t
     - IN
     - 复制来源

   * - 输出
     - 支持型态：    
     
       cvai_face_info_t  
       
       cvai_object_info_t  
       
       cvai_image_t
     - OUT
     - 复制目的


CVI_AI_RescaleMetaCenter
------------------------

【描述】

将结构内的坐标还原到与输入图像相同之大小，适用于padding图像为上下左右，

【参数】

以下为支持的输入变量

-  `cvai_face_t <6_Data_Types.html#cvai-face-t>`__

-  `cvai_object_t <6_Data_Types.html#cvai-object-t>`__

CVI_AI_RescaleMetaRB
--------------------

【描述】

将结构内的坐标还原到与输入图像相同之大小，适用于padding图像为右下，

【参数】

以下为支持的输入变量

-  `cvai_face_t <6_Data_Types.html#cvai-face-t>`__

-  `cvai_object_t <6_Data_Types.html#cvai-object-t>`__

getFeatureTypeSize
------------------

.. code-block:: none
  
  getFeatureTypeSize(feature_type_e type);

【描述】

取得特征值的单位大小。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - feature_type_e
     - type  
     - 单位        

   * - 回传 
     - int   
     - X 
     - 单位为byte之单位大小


CVI_AI_SetModelThreshold
------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, float threshold);

【描述】

设置模型阀值，目前仅支持针对Detection类型的模型进行设置。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t      
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E
     - model   
     - 模型index    

   * - 输入
     - float   
     - threshold
     - 阀值(0.0~1.0)


CVI_AI_GetModelThreshold
------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_GetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, float *threshold);

【描述】

取出模型阀值，目前仅支持Detection类型模型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t      
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E
     - model   
     - 模型index    

   * - 输出
     - float\*
     - threshold
     - 阀值         


对象侦测
^^^^^^^^^^^^^^^^^^^

CVI_AI_MobileDetV2_Vehicle
--------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MobileDetV2_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用MobilDetV2-Vehicle模型进行推理，此模型可侦测Car, Motorcycle, Truck三个类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_object_t\*
     - obj  
     - 侦测到的对象  


CVI_AI_MobileDetV2_Pedestrian
-----------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MobileDetV2_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用MobilDetV2-Pedestrian系列模型进行推理，此模型可侦测person类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_object_t\*
     - obj  
     - 侦测到的对象  


CVI_AI_MobileDetV2_Person_Vehicle
---------------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MobileDetV2_Person_Vehicle(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用MobilDetV2-Person-Vehicle模型进行推理，此模型可侦测人车非类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_object_t\*
     - obj  
     - 侦测到的对象  


CVI_AI_MobileDetV2_Person_Pets
------------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MobileDetV2_Person_Pets(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用MobilDetV2-Lite-Person-Pets模型进行推理，此模型可侦测person, cat, dog等类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输出
     - cvai_object_t\*
     - obj 
     - 侦测到的对象   


CVI_AI_MobileDetV2_COCO80
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MobileDetV2_COCO80(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用MobilDetV2 COCO80系列模型进行推理，此模型可侦测标准COCO dataset的 80个类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输出
     - cvai_object_t\*
     - obj 
     - 侦测到的对象   


CVI_AI_Yolov3
-------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Yolov3 (cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用YoloV3模型进行推理，此模型可侦测COCO 80个类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输出
     - cvai_object_t\*
     - obj 
     - 侦测到的对象   


CVI_AI_YoloX
------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_YoloX(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用YoloX模型进行推理，此模型可侦测COCO 80个类别。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输出
     - cvai_object_t\*
     - obj 
     - 侦测到的对象   


CVI_AI_SelectDetectClass
------------------------

【语法】

.. code-block:: none

  CVI_S32 CVI_AI_SelectDetectClass(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E model, uint32_t num_classes, ...)

【描述】

过滤Object Detection模型输出结果, 保留列举的类别或群组。

类别为不定参数，数量根据num_classes而定。

详细类别及群组Index可参考 `cvai_obj_class_id_e <6_Data_Types.html#cvai-obj-class-id-e>`__ 及 `cvai_obj_det_group_type_e <6_Data_Types.html#cvai-obj-group-type-e>`__。

目前仅支持MobileDetV2, YoloX系列模型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄         

   * - 输入
     - CVI_AI_SUPPORTED_MODEL_E 
     - model 
     - 模型Index    

   * - 输入
     - uint32_t         
     - n  um_classes
     - 保留的类别个数   

   * - 输入
     - cvai_obj_class_id_e或  cvai_obj_det_group_type_e
     - 说明
     - 留的Class ID或Group ID


CVI_AI_ThermalPerson
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_ThermalPerson(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

热显图像人型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_object_t\*
     - faces
     - 侦测到的人形  


人脸侦测
^^^^^^^^^^^^^^^^^^^^^

CVI_AI_RetinaFace
-----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_RetinaFace(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

使用RetinaFace模型侦测人脸。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_face_t\*
     - faces
     - 侦测到的人脸  


CVI_AI_RetinaFace_IR
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_RetinaFace_IR(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

使用RetinaFace模型在IR图像中侦测人脸。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入IR图像

   * - 输出
     - cvai_face_t\*
     - faces
     - 侦测到的人脸  


CVI_AI_RetinaFace_Hardhat
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_RetinaFace_Hardhat(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

使用RetinaFace模型侦测戴安全帽人脸。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入IR图像

   * - 输出
     - cvai_face_t\*
     - faces
     - 侦测到的人脸  


CVI_AI_ThermalFace
------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_ThermalFace(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

热显图像人脸侦测。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_face_t\*
     - faces
     - 侦测到的人脸  


CVI_AI_FaceQuality
------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceQuality(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces, bool *skip);

【描述】

判断传入的faces结构中的人脸质量评估并同时侦测人脸角度。质量受人脸清晰程度与是否遮挡影响。人脸质量分数为 faces->info[i].face_quality，人脸角度放在 faces->info[i].head_pose中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - face         
     - 侦测到的人脸  

     

   * - 输入
     - bool\*  
     - skip 
     - Bool array,    
     
       指定哪个人脸需要做face quality。NULL  表示全部人脸都做。


CVI_AI_FaceMaskDetection
------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceMaskDetection(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

侦测戴口罩人脸。人脸分数存放在faces->info[i].bbox.score，戴口罩人脸分数存放在faces->info[i].mask_score。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - cvai_face_t\*
     - faces
     - 侦测到的人脸  


CVI_AI_MaskClassification
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MaskClassification(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);

【描述】

判断传入的faces中的所有人脸是否为戴口罩人脸。呼叫此接口前，必须先执行一次人脸侦测。戴口罩人脸分数存放在faces->info[i].mask_score。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - faces        
     - 侦测到的人脸  

     


人脸识别
^^^^^^^^^^^^^^^^^^

CVI_AI_FaceRecognition
----------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceRecognition(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

抽取人脸特征。此接口会针对face中所有人脸进行特征抽取。并放在faces->info[i].feature中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入/输出
     - cvai_face_t\*
     - faces          
     - 侦测到的人脸   




CVI_AI_FaceRecognitionOne
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceRecognitionOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces, int face_idx);

【描述】

抽取人脸特征。此接口仅会针对指定的face index进行特征抽取。并放在faces->info[index].feature中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入/输出
     - cvai_face_t\*
     - faces          
     - 侦测到的人脸   



   * - 输入
     - int  
     - face_idx
     - 想进行特征抽取的face index。-1表示全部抽取。


CVI_AI_FaceAttribute
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceAttribute(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

抽取人脸特征及人脸属性。此接口会针对face中所有人脸进行特征抽取及人脸属性。

人脸属性包含：性别, 表情, 年龄及种族，
结果分别放在faces->info[i].feature, faces->info[i].age, faces->info[i].emotion,
faces->info[i].gender, faces->info[i].race。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入/输出
     - cvai_face_t\*
     - faces          
     - 侦测到的人脸   




CVI_AI_FaceAttributeOne
-----------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceAttributeOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces, int face_idx);

【描述】

抽取人脸特征。此接口仅会针对指定的face index进行特征抽取。

人脸属性包含：性别, 表情, 年龄及种族，
结果分别放在faces->info[i].feature, faces->info[i].age, faces->info[i].emotion,
faces->info[i].gender, faces->info[i].race。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入/输出
     - cvai_face_t\*
     - faces         
     - 侦测到的人脸   

   * - 输入
     - int  
     - face_idx
     - 想进行特征抽取的face    index。-1表示全部抽取。


CVI_AI_MaskFaceRecognition
--------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MaskFaceRecognition(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

抽取戴口罩人脸特征。此接口会针对face中所有人脸进行特征抽取。并放在faces->info[i].feature中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入/输出
     - cvai_face_t\*
     - faces   

       
     - 侦测到的人脸   




行人识别
^^^^^^^^^^^^^^^^^^^^^

CVI_AI_OSNet
------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_OSNet(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用person-reid模型抽取行人特征。此接口会针对obj中所有的Person类别对象进行特征抽取。并放在obj->info[i].feature中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入
     - cvai_object_t\*
     - obj 
     - 侦测到的对象   


CVI_AI_OSNetOne
---------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_OSNetOne(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj, int obj_idx);

【描述】

使用person-reid模型抽取行人特征。此接口仅会针对指定的obj对象进行特征抽取。并放在obj->info[i].feature中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 输入图像       

   * - 输入
     - cvai_object_t\*
     - obj 
     - 侦测到的对象   

   * - 输入
     - int
     - obj_idx 
     - 想进行特征抽取的对象    index。-1表示全部抽取。


对象追踪
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CVI_AI_DeepSORT_Init
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_Init(const cviai_handle_t handle, bool use_specific_counter);

【描述】

初始化DeepSORT算法。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄        

   * - 输入
     - bool         
     - use_specific_counter
     - 是否每一个对象类别各自分配id


CVI_AI_DeepSORT_GetDefaultConfig
--------------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_GetDefaultConfig(cvai_deepsort_config_t *ds_conf);

【描述】

取得DeepSORT默认参数。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cvai_deepsort_config_t\* 
     - ds_conf 
     - DeepSORT参数   


CVI_AI_DeepSORT_SetConfig
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_SetConfig(const cviai_handle_t handle , cvai_deepsort_config_t *ds_conf, int cviai_obj_type, bool show_config);

【描述】

设置DeepSORT参数。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄

   * - 输入
     - cvai_deepsort_config_t\*
     - ds_conf
     - DeepSORT参数  

   * - 输入
     - int 
     - cvi ai_obj_type
     - -1表示此为默认设置。   
     
       非-1值表示针对cviai_ob j_type的类别设置参数。

   * - 输入
     - bool
     - show_config
     - 显示设置      


CVI_AI_DeepSORT_GetConfig
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_GetConfig(const cviai_handle_t handle , cvai_deepsort_config_t *ds_conf, int cviai_obj_type);

【描述】

询问DeepSORT设置的参数。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - AI SDK句柄    

   * - 输出
     - cvai_deepsort_config_t\*
     - ds_conf
     - DeepSORT参数  

   * - 输入
     - int 
     - cvi ai_obj_type
     - -1表示取得默认参数。   
     
       非-1值表示针对cviai_ob j_type的类别设置的参数


CVI_AI_DeepSORT_CleanCounter
----------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_CleanCounter(const cviai_handle_t handle);

【描述】

清除DeepSORT 纪录的ID counter。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t
     - handle
     - 句柄


CVI_AI_DeepSORT_Obj
-------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_Obj(const cviai_handle_t handle, cvai_object_t *obj, cvai_tracker_t *tracker_t, bool use_reid);

【描述】

追踪对象，更新Tracker状态。

此接口会赋予每个Object一个Unique ID。

可从obj->info[i].unique_id取得。tracker_t会纪录DeepSORT对每个object的追踪状态及目前的预测Bounding Box。

若想使用对象外观特征进行追踪，需将use_reid设置true, 并在追踪之前使用CVI_AI_OSNet进行特征抽取。

目前特征抽取只支持人型。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - cvai_object_t\*
     - obj 
     - 想进行追踪的对象   

   * - 输出
     - cvai_tracker_t\*
     - t    racker_t
     - 对象的追踪状态 

   * - 输入
     - bool 
     - use_reid
     - 是否使用对象外观特征进行追踪 


CVI_AI_DeepSORT_Face
--------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeepSORT_Face(const cviai_handle_t handle, cvai_face_t *face, cvai_tracker_t *tracker_t, bool use_reid);

【描述】

追踪人脸，更新Tracker状态。

此接口会赋予每个人脸一个Unique ID。可从face->info[i].unique_id取得。

tracker_t会纪录DeepSORT对每个人脸的追踪状态及目前的预测Bounding Box。

若想使用人脸特征进行追踪，use_reid须设置为true。

并在追踪之前调用 `CVI_AI_FaceRecognition`_ 计算人脸特征。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - cvai_face_t\*
     - face
     - 想进行追踪的人脸   

   * - 输出
     - cvai_tracker_t\*
     - tracker_t
     - 人脸的追踪状态 

   * - 输入
     - bool 
     - use_reid
     - 是否使用外观特征进行追踪。目前仅能设置false


运动侦测
^^^^^^^^^^^^^^^^^^^^^

CVI_AI_Set_MotionDetection_Background
-------------------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Set_MotionDetection_Background(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame);

【描述】

设置Motion Detection背景，
第一次运行此接口时会对Motion Detection进行初始化，
后续再调用次接口仅会更新背景。

AI SDK中Motion Detection使用帧差法。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 背景 


CVI_AI_MotionDetection
----------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_MotionDetection (const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *objects, uint32_t threshold, double min_area);

【描述】

使用帧差法侦测对象。侦测结果会存放在objects内。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t   
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 图像 

   * - 输出
     - cvai_object_t\*
     - object  
     - 运动侦测结果   

   * - 输入
     - uint32_t         
     - threshold
     - 帧差法阀值，须为0-255  

   * - 输入
     - double 
     - min_area
     - 最小对象面积(Pixels)，过滤掉  小于此数值面积的物件。 


车牌识别
^^^^^^^^^^^^^^^^^^^^

CVI_AI_LicensePlateDetection
----------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_LicensePlateDetection(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_meta);

【描述】

车牌侦测。呼叫此API之前，必须先执行一次车辆侦测。

此算法会在已侦测到的对象上进行车牌侦测。

车牌位置会放在 obj->info[i].vehicle_properity->license_pts中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 图像 

   * - 输入
     - cvai_object_t\*
     - obj 
     - 对象(车辆)侦测结果 


CVI_AI_LicensePlateRecognition_TW
---------------------------------

.. code-block:: none
  
  CVI_AI_LicensePlateRecognition_TW(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

对传入的obj中所有车辆进行车牌识别(台湾)。

呼叫此API之前，必须先调用CVI_AI_LicensePlateDetection执行一次车牌侦测。

车牌号码储存在obj->info[i].vehicle_properity->license_char 。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄 

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 图像 

   * - 输入
     - cvai_object_t\*
     - obj 
     - 车牌侦测结果   


CVI_AI_LicensePlateRecognition_CN
---------------------------------

.. code-block:: none
  
  CVI_AI_LicensePlateRecognition_CN(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

对传入的obj中所有车辆进行车牌识别(大陆)。

呼叫此API之前，必须先调用CVI_AI_LicensePlateDetection执行一次车牌侦测。

车牌号码储存在obj->info[i].vehicle_properity->license_char 。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t  
     - handle
     - 句柄         

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 图像         

   * - 输入/输出
     - vai_object_t\*
     - bj  
     - 牌侦测结果 


篡改侦测
^^^^^^^^^^^^^^^^^^^^^^^^^^

CVI_AI_TamperDetection
----------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_TamperDetection(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, float *moving_score);

【描述】

摄影机篡改侦测。此算法基于高斯模型建立背景模型，并用去背法算出差值当作篡改分数(moving_score)。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t 
     - handle
     - 句柄         

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame   
     - 图像         

   * - 输出
     - float\*
     - moveing_score
     - 篡改分数     


活体识别
^^^^^^^^^^^^^^^^^^^^^^

CVI_AI_Liveness
---------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Liveness (const cviai_handle_t handle, VIDEO_FRAME_INFO_S *rgbFrame, VIDEO_FRAME_INFO_S *irFrame, , cvai_face_t *rgb_faces, cvai_face_t *ir_faces);

【描述】

RGB, IR双目活体识别。

判断rgb_faces和ir_faces中的人脸是否为活体。

活体分数置于 rgb_face ->info[i].liveness_score 中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t  
     - handle
     - 句柄        

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - rgbFrame   
     - RGB图像     

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - irFrame
     - IR图像      

   * - 输入/输出
     - cvai_face_t\*
     - rgb_meta   


     - 侦测到的RGB人脸/

       活体分数    

   * - 输入
     - cvai_face_t\*
     - ir_meta
     - 侦测到的IR人脸  


姿态检测
^^^^^^^^^^^^^^

CVI_AI_AlphaPose
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_AlphaPose (cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

【描述】

使用alphapose模型进行推理，预测17个骨骼关键点。

检测结果置于 obj->info[i].pedestrian_properity->pose_17。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_object_t\*
     - obj          
     - 侦测到的人 / 

       17个骨骼关键点坐标


语义分割
^^^^^^^^^^^^^^^^^^

CVI_AI_DeeplabV3
----------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_DeeplabV3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *out_frame, cvai_class_filter_t *filter);

【描述】

使用DeepLab V3模型进行语义分割。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输出
     - VIDEO_FRAME_INFO_S\*
     - out_frame
     - 输出图像  

   * - 输入
     - cvai_class_filter_t\*
     - filter   
     - 保留的类别


跌倒检测
^^^^^^^^^^^^^^^^^^^^^^^^

CVI_AI_Fall
-----------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Fall (cviai_handle_t handle, cvai_object_t *obj);

【描述】

使用对象侦测与姿态检测之结果，判断跌倒状态。

在运行此API前需要先调用 `CVI_AI_AlphaPose`_ 取得人体关键点。

跌倒检测结果置于 obj->info[i].pedestrian_properity->fall 。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_object_t\*
     - obj          
     - 跌倒状态结果  

     


驾驶疲劳检测
^^^^^^^^^^^^^^^^^^^^^^

CVI_AI_FaceLandmarker
---------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_FaceLandmarker (cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

需先使用人脸检测，产生出106个人脸特征点检测的结果，将结果放入face->dms[i].landmarks_106 并且更新5个人脸特征点 face->dms[i].landmarks_5。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - face         
     - 人脸      

     


CVI_AI_EyeClassification
------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_EyeClassification (cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

需先使用人脸检测以及人脸特征点检测的结果，判断眼睛闭合状态，将结果放入face->dms[i].reye_score/ face->dms[i].leye_score。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - face         
     - 人脸      

     


CVI_AI_YawnClassification
-------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_YawnClassification (cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

根据人脸检测和人脸关键点结果进行打哈欠检测。必须先调用CVI_FaceRecognition取得人脸检测和人脸关键点结果。打哈欠结果会放入face->dms[i].yawn_score 。分数为0.0~1.0间。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - face         
     - 人脸      

     


CVI_AI_IncarObjectDetection
---------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_IncarObjectDetection(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

【描述】

使用对象侦测检测对象（水杯／马克杯／电话）是否出现在驾驶周边，将判断结果输出成object格式 ，放入到face->dms[i].dms.dms_od。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - face         
     - 人脸      

     


声音分类
^^^^^^^^^^^^^^^^^^^^^^

CVI_AI_SoundClassification
--------------------------

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_SoundClassification (cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, int *index);

【描述】

判断frame中音讯属于哪个类别。并将各类别分数排序后输出。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_handle_t       
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入图像  

   * - 输入
     - int\*
     - index        
     - 每个类别的分数

     


CVIAI_Service
~~~~~~~~~~~~~~~

CVI_AI_Service_CreateHandle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_CreateHandle (cviai_service_handle_t *handle, cvai_handle ai_handle);

【描述】

创建Service句柄

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄      

   * - 输入
     - cviai_handle_t       
     - ai_handle
     - cviai_core 句柄   


CVI_AI_Service_DestroyHandle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_DestroyHandle (cviai_service_handle_t *handle);

【描述】

销毁Service句柄

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄      


CVI_AI_Service_Polygon_SetTarget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_Polygon_SetTarget(cviai_service_handle_t handle, const cvai_pts_t *pts);

【描述】

设定区域入侵范围。pts为凸多边形点坐标，顺序需为顺直针或逆时针。

调用 `CVI_AI_Service_Polygon_Intersect`_ 判断一个bounding box是否侵入已划定范围。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄      

   * - 输入
     - cvai_pts_t\*
     - pts  
     - 凸多边形点


CVI_AI_Service_Polygon_Intersect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_Polygon_Intersect(cviai_service_handle_t handle, const cvai_bbox_t *bbox, bool *has_intersect);

【描述】

根据CVI_AI_Service_Polygon_SetTarget所设定区域入侵范围。判断给定之gounding box侵入范围。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄      

   * - 输入
     - cvai_bbox_t\*
     - bbox   
     - Bounding box  

   * - 输出
     - bool   
     - ha  s_intersect
     - 是否入侵  


CVI_AI_Service_RegisterFeatureArray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_RegisterFeatureArray(cviai_service_handle_t handle, const cvai_service_feature_array_t featureArray, const cvai_service_feature_matching_e method);

【描述】

注册特征库，将featureArray中所含特征进行预计算并搬入内存中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄     

   * - 输入
     - const     cvai_service_feature_array_t
     - fe atureArray
     - 特征数组结构 

   * - 输入
     - const     cvai_service_feature_matching_e
     - method
     - 比        对方法，目前仅支  持COS_SIMILARITY 


CVI_AI_Service_CalcualteSimilarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_CalculateSimilarity(cviai_service_handle_t handle, const cvai_feature_t *feature_rhs, const cvai_feature_t *feature_lhs, float *score);

【描述】

使用CPU计算两个特征之Cosine Similarity。其计算公式如下：

.. math:: sim(\theta) = \frac{A \bullet B}{\left\| A \right\| \bullet \left\| B \right\|} = \frac{\sum_{i = 1}^{n}{A_{i}B_{i}}}{\sqrt{\sum_{i = 1}^{n}A_{i}^{2}}\sqrt{\sum_{i = 1}^{n}B_{i}^{2}}}

其中n 为特征长度。目前仅支持INT8特征

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄     

   * - 输入
     - const cvai_feature_t\*
     - feature_rhs
     - 第一个特征   

   * - 输入
     - const cvai_feature_t\*
     - feature_lhs
     - 第二个特征   

   * - 输出
     - float\*
     - score  
     - 相似度   


CVI_AI_Service_ObjectInfoMatching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectInfoMatching(cviai_service_handle_t handle, const cvai_object_info_t *object_info, const uint32_t topk, float threshold, uint32_t *indices, float *sims, uint32_t *size);

【描述】

计算object_info中的对象特征和已注册之对象特征库之Cosine Similarity。并取出大于threshold的Top-K个相似度。其计算公式如下：

.. math:: sim(\theta) = \frac{A \bullet B}{\left\| A \right\| \bullet \left\| B \right\|} = \frac{\sum_{i = 1}^{n}{A_{i}B_{i}}}{\sqrt{\sum_{i = 1}^{n}A_{i}^{2}}\sqrt{\sum_{i = 1}^{n}B_{i}^{2}}}

其中n 为特征长度。若特征库数量少于1000笔会以CPU进行计算，否则会以启动TPU进行计算。注册特征需要调用CVI_AI_Service_RegisterFeatureArray。目前仅支持INT8特征

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄     

   * - 输入
     - const cvai_object_info_t\*
     - object_info
     - 物件Info 

   * - 输入
     - const uint32_t     
     - topk   
     - 取topk个相似度   

   * - 输出
     - float  
     - threshold  
     - 相似度    阀值，高于此阀值  之相似度才会取出 

   * - 输出
     - uint32_t\*
     - indices
     - 符合条件之相  似度在库内的Index

   * - 输出
     - float\*
     - sims   
     - 符合条件之相似度 

   * - 输出
     - uint32_t\*
     - size   
     - 最终      取出的相似度个数 


CVI_AI_Service_FaceInfoMatching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_FaceInfoMatching(cviai_service_handle_t handle, const cvai_face_info_t *face_info, const uint32_t topk, float threshold, uint32_t *indices, float *sims, uint32_t *size);

【描述】

计算face_info中的人脸特征和已注册之人脸特征库之Cosine Similarity。并取出大于threshold的Top-K个相似度。其计算公式如下：

.. math:: sim(\theta) = \frac{A \bullet B}{\left\| A \right\| \bullet \left\| B \right\|} = \frac{\sum_{i = 1}^{n}{A_{i}B_{i}}}{\sqrt{\sum_{i = 1}^{n}A_{i}^{2}}\sqrt{\sum_{i = 1}^{n}B_{i}^{2}}}

其中n 为特征长度。若特征库数量少于1000笔会以CPU进行计算，否则会以启动TPU进行计算。注册特征需要调用CVI_AI_Service_RegisterFeatureArray。目前仅支持INT8特征

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄     

   * - 输入
     - const cvai_face_info_t\*
     - face_info  
     - Face info

   * - 输入
     - const uint32_t     
     - topk   
     - 取topk个相似度   

   * - 输出
     - float  
     - threshold  
     - 相似度    阀值，高于此阀值  之相似度才会取出 

   * - 输出
     - uint32_t\*
     - indices
     - 符合条件之相  似度在库内的Index

   * - 输出
     - float\*
     - sims   
     - 符合条件之相似度 

   * - 输出
     - uint32_t\*
     - size   
     - 最终      取出的相似度个数 


CVI_AI_Service_RawMatching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_RawMatching(cviai_service_handle_t handle, const void *feature, const feature_type_e type, const uint32_t topk, float threshold, uint32_t *indices, float *scores, uint32_t *size);

【描述】

计算特征和已注册之特征库之Cosine Similarity。并取出大于threshold的Top-K个相似度。其计算公式如下：

.. math:: sim(\theta) = \frac{A \bullet B}{\left\| A \right\| \bullet \left\| B \right\|} = \frac{\sum_{i = 1}^{n}{A_{i}B_{i}}}{\sqrt{\sum_{i = 1}^{n}A_{i}^{2}}\sqrt{\sum_{i = 1}^{n}B_{i}^{2}}}

其中n 为特征长度。若特征库数量少于1000笔会以CPU进行计算，否则会以启动TPU进行计算。注册特征需要调用CVI_AI_Service_RegisterFeatureArray。和CVI_AI_Service_FaceInfoMatching及CVI_AI_Service_ObjectInfoM
atching不同的是，此API直接使用特征数组进行比对，不需传入cvai_face_info_t或cvai_object_info_t。此API限制特征类型需要和特征库之特征类型相同。目前仅支持INT8特征

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t\*
     - handle
     - 句柄     

   * - 输入
     - const void *      
     - feature
     - 特征数组 

   * - 输入
     - const feature_type_e   
     - type   
     - 特征类型，目  前仅支持TYPE_INT8

   * - 输入
     - const uint32_t     
     - topk   
     - 取topk个相似度   

   * - 输出
     - float  
     - threshold  
     - 相似度    阀值，高于此阀值  之相似度才会取出 

   * - 输出
     - uint32_t\*
     - indices
     - 符合条件之相  似度在库内的Index

   * - 输出
     - float\*
     - scores 
     - 符合条件之相似度 

   * - 输出
     - uint32_t\*
     - size   
     - 最终取出的相似度个数 


CVI_AI_Service_FaceAngle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_FaceAngle(const cvai_pts_t *pts, cvai_head_pose_t *hp);

【描述】

计算单个人脸姿态

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cvai_pts_t\*
     - pts  
     - 人脸 landmark 

   * - 输出
     - cvai_head_pose_t     
     - hp   
     - 人脸姿态  


CVI_AI_Service_FaceAngleForAll
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_FaceAngleForAll(const cvai_face_t *meta);

【描述】

计算多个人脸姿态

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cvai_face_t\*
     - meta         
     - 人脸资料  

     


CVI_AI_Service_FaceDigitalZoom
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_FaceDigitalZoom(

  cviai_service_handle_t handle,

  const VIDEO_FRAME_INFO_S *inFrame,

  const cvai_face_t *meta,

  const float face_skip_ratio,

  const float trans_ratio,

  const float padding_ratio,

  VIDEO_FRAME_INFO_S *outFrame);

【描述】

将人脸侦测结果之人脸进行放大(zoom in)

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t  
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - inFrame
     - 输入图像  

   * - 输入
     - cvai_face_t\*
     - meta   
     - 人脸资料  

   * - 输入
     - float 
     - face_skip_ratio
     - 忽略比率  

   * - 输入
     - float 
     - trans_ratio
     - 放大比率  

   * - 输入
     - float 
     - padding_ratio  
     - 扩展bounding   box比例   

   * - 输出
     - VIDEO_FRAME_INFO_S\*
     - outFrame   
     - 输出图像  


CVI_AI_Service_FaceDrawPts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_FaceDrawPts(cvai_pts_t *pts, VIDEO_FRAME_INFO_S *frame);

【描述】

绘制人脸 landmark

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cvai_pts_t\*
     - pts  
     - 人脸 landmark 

   * - 输入
     - VIDEO_FRAME_INFO_S 
     - hp   
     - 输入/输出图像 


CVI_AI_Service_FaceDrawRect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_FaceDrawRect(cviai_service_handle_t handle, const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame, const bool drawText, cvai_service_brush_t brush);

【描述】

绘制人脸方框

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t 
     - handle
     - 句柄      

   * - 输入
     - cvai_face_t\*
     - meta 
     - 人脸资料  

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame
     - 输入/输出图像 

   * - 输入
     - bool   
     - drawText 
     - 是否绘制人脸名字  

   * - 输入
     - cvai_service_brush_t   
     - brush
     - 颜色      


CVI_AI_Service_ObjectDigitalZoom
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectDigitalZoom(cviai_service_handle_t handle, const VIDEO_FRAME_INFO_S *inFrame, const cvai_object_t *meta, const float obj_skip_ratio, const float trans_ratio, const float padding_ratio, VIDEO_FRAME_INFO_S *outFrame);

【描述】

将对象侦测结果之对象进行放大(zoom in)

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t  
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - inFrame
     - 输入图像  

   * - 输入
     - cvai_object_t\*
     - meta   
     - 对象数据  

   * - 输入
     - float 
     - obj_skip_ratio 
     - 忽略比率  

   * - 输入
     - float 
     - trans_ratio
     - 放大比率  

   * - 输入
     - float 
     - padding_ratio  
     - 扩展bounding   box比例   

   * - 输出
     - VIDEO_FRAME_INFO_S\*
     - outFrame   
     - 输出图像  


CVI_AI_Service_ObjectDitgitalZoomExt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectDigitalZoomExt(cviai_service_handle_t handle, const VIDEO_FRAME_INFO_S *inFrame, const cvai_object_t *meta, 
  
  const float obj_skip_ratio, const float trans_ratio, const float pad_ratio_left, const float pad_ratio_right, const float pad_ratio_top, 
  
  const float pad_ratio_bottom, VIDEO_FRAME_INFO_S *outFrame);

【描述】

将对象侦测结果之对象进行放大(zoom in)

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t
     - handle
     - 句柄      

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - inFrame 
     - 输入图像  

   * - 输入
     - cvai_object_t\*
     - meta    
     - 对象数据  

   * - 输入
     - float         
     - obj_skip_ratio  
     - 忽略比率  

   * - 输入
     - float         
     - trans_ratio 
     - 放大比率  

   * - 输入
     - float         
     - pad_ratio_left  
     - 扩张率(左)

   * - 输入
     - float         
     - pad_ratio_right 
     - 扩张率(右)

   * - 输入
     - float         
     - pad_ratio_top   
     - 扩张率(上)

   * - 输入
     - float         
     - pad_ratio_bottom
     - 扩张率(下)

   * - 输出
     - VIDEO_FRAME_INFO_S\*
     - outFrame
     - 输出图像  


CVI_AI_Service_ObjectDrawPose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectDrawPose(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame);

【描述】

绘制姿态侦测之17个骨骼点

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cvai_object_t\*
     - meta   
     - 骨骼点侦测结果

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame  
     - 输入图像  


CVI_AI_Service_ObjectDrawRect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectDrawRect(cviai_service_handle_t handle, const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame, const bool drawText);

【描述】

绘制对象侦测框

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t
     - handle
     - 句柄      

   * - 输入
     - cvai_object_t\*
     - meta  
     - 对象侦测结果  

   * - 输入
     - VIDEO_FRAME_INFO_S\*
     - frame 
     - 输入/输出 图像

   * - 输入
     - bool  
     - drawText  
     - 是否绘制类别文字  


CVI_AI_Service_ObjectWriteText
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectWriteText(char *name, int x, int y, VIDEO_FRAME_INFO_S *frame, float r, float g, float b);

【描述】

绘制指定文字

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - char\* 
     - name   
     - 绘制的文字   

   * - 输入
     - int  
     - x  
     - 绘制的x坐标  

   * - 输入
     - int  
     - y  
     - 绘制的y坐标  

   * - 输入/输出
     - IDEO_FRAME_INFO_S\*
     - rame  
     - 输入/输出 图像   

   * - 输入
     - float
     - r  
     - 绘制颜色 r    channel值

   * - 输入
     - float
     - g  
     - 绘制颜色 g    channel值

   * - 输入
     - float
     - b  
     - 绘制颜色 b    channel值


CVI_AI_Service_Incar_ObjectDrawRect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【语法】

.. code-block:: c

  CVI_S32 CVI_AI_Service_ObjectWriteText(cviai_service_handle_t handle, const cvai_dms_od_t *meta, VIDEO_FRAME_INFO_S *frame, const bool drawText, IVE_COLOR_S color);

【描述】

绘制指定文字

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1


   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - cviai_service_handle_t   
     - handle
     - 句柄     

   * - 输入
     - cvai_dms_od_t\*  
     - meta   
     - 物件侦测结果 

   * - 输入/输出
     - IDEO_FRAME_INFO_S\*
     - rame  
     - 输入/输出 图像   

   * - 输入
     - const bool       
     - drawText   
     - 是否绘制类别文字 

   * - 输入
     - IVE_COLOR_S      
     - color  
     - 绘制颜色 