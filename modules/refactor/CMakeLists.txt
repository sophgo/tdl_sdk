project(refactor)

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/face_detection/retina_face
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils
                    )

if("${CVI_PLATFORM}" STREQUAL "SOPHON")
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/bm168x_net.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/memory/bm_memory_pool.cpp
              )
  add_definitions(-D__SOPHON__)
else()
  set(NET_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/net/cvi_net.cpp
              # ${CMAKE_CURRENT_SOURCE_DIR}/net/cvi_tensor.cpp
              )

endif()
message(STATUS "CVI_PLATFORMxxx: ${CVI_PLATFORM}")
if(DEFINED __SOPHON__)
    message(STATUS "__SOPHON__ is defined")
else()
    message(STATUS "__SOPHON__ is NOT defined")
endif()
set(PROJ_SRCS   
                ${CMAKE_CURRENT_SOURCE_DIR}/image/base_image.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/cvi_memory_pool.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/bm_memory_pool.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/image/vpss_image.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/image/opencv_image.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/preprocessor/vpss_preprocessor.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/preprocessor/opencv_preprocessor.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/net/base_tensor.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/net/base_net.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/models/base_model.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/net/net_factory.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/utils/common_utils.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/base_memory_pool.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_pool_factory.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/models/scrfd.cpp)


add_library(${PROJECT_NAME} OBJECT ${PROJ_SRCS} ${NET_SRCS})

