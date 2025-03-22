project(framework)
message(STATUS "REPO_DIR:${REPO_DIR}")
include_directories(
                    ${REPO_DIR}/include/tdl_sdk/framework
                    ${REPO_DIR}/include/tdl_sdk/components
                    ${REPO_DIR}/include/tdl_sdk/app
                    )

if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
   ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR 
   ("${CVI_PLATFORM}" STREQUAL "SOPHON"))
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/bm168x_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/net/bm_llm_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/memory/bm_memory_pool.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/model/llm_model.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/utils/qwen_vl_helper.cpp
              )
else()
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/cvi_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/memory/cvi_memory_pool.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/image/vpss_image.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/vpss_preprocessor.cpp
              )

endif()

if("${CVI_PLATFORM}" STREQUAL "SOPHON")
  set(NET_SRCS ${NET_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/image/vpss_image.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/vpss_preprocessor.cpp
              )
endif()

set(PROJ_SRCS   
                ${CMAKE_CURRENT_SOURCE_DIR}/image/base_image.cpp                            
                ${CMAKE_CURRENT_SOURCE_DIR}/image/opencv_image.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/image/image_factory.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/base_preprocessor.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/opencv_preprocessor.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/tensor/base_tensor.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/net/base_net.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/model/base_model.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/net/net_factory.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/utils/common_utils.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/base_memory_pool.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/cpu_memory_pool.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_pool_factory.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/utils/image_alignment.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/common/model_output_types.cpp
                )


add_library(${PROJECT_NAME} OBJECT ${PROJ_SRCS} ${NET_SRCS})

