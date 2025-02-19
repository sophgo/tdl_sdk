project(framework)

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/framework
                    # ${CMAKE_CURRENT_SOURCE_DIR}/../core/face_detection/retina_face
                    # ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils
                    )

if("${CVI_PLATFORM}" STREQUAL "BM1688")
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/bm168x_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/memory/bm_memory_pool.cpp
              )
elseif("${CVI_PLATFORM}" STREQUAL "SOPHON")
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/bm168x_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/memory/cvi_memory_pool.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/image/vpss_image.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/vpss_preprocessor.cpp)
else()
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/cvi_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/memory/cvi_memory_pool.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/image/vpss_image.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/vpss_preprocessor.cpp
              )

endif()

set(PROJ_SRCS   
                ${CMAKE_CURRENT_SOURCE_DIR}/image/base_image.cpp                
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/bm_memory_pool.cpp                
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
                ${CMAKE_CURRENT_SOURCE_DIR}/../core/cvi_tdl_types_mem.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/utils/image_alignment.cpp
                )


add_library(${PROJECT_NAME} OBJECT ${PROJ_SRCS} ${NET_SRCS})

