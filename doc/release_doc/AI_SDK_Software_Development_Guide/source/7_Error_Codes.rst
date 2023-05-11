.. vim: syntax=rst

错误码
================

.. list-table::
   :widths: 1 2 1
   :header-rows: 1


   * - 错误代码
     - 宏定义
     - 描述

   * - 0xFFFFFFFF
     - CVIAI_FAILURE
     - API 调用失败

   * - 0xC0010101
     - CVIAI_ERR_INVALID_MODEL_PATH
     - 不正确的模型路径   

   * - 0xC0010102
     - CVIAI_ERR_OPEN_MODEL
     - 开启模型失败

   * - 0xC0010103
     - CVIAI_ERR_CLOSE_MODEL
     - 关闭模型失败

   * - 0xC0010104
     - CVIAI_ERR_GET_VPSS_CHN_CONFIG
     - 取得VPSS CHN设置失败

   * - 0xC0010105
     - CVIAI_ERR_INFERENCE
     - 模型推理失败

   * - 0xC0010106
     - CVIAI_ERR_INVALID_ARGS
     - 不正确的参数

   * - 0xC0010107
     - CVIAI_ERR_INIT_VPSS
     - 初始化VPSS失败

   * - 0xC0010108
     - CVIAI_ERR_VPSS_SEND_FRAME
     - 送Frame到VPSS时失败

   * - 0xC0010109
     - CVIAI_ERR_VPSS_GET_FRAME
     - 从VPSS取得Frame失败

   * - 0xC001010A
     - CVIAI_ERR_MODEL_INITIALIZED
     - 模型未开启  

   * - 0xC001010B
     - CVIAI_ERR_NOT_YET_INITIALIZED
     - 功能未初始化

   * - 0xC001010C
     - CVIAI_ERR_NOT_YET_IMPLEMENTED
     - 功能尚未实现

   * - 0xC001010D
     - CVIAI_ERR_ALLOC_ION_FAIL
     - 分配ION内存失败    

   * - 0xC0010201
     - CVIAI_ERR_MD_OPERATION_FAILED
     - 运行Motion Detection失败

