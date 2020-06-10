# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

cmake_minimum_required(VERSION 3.2.2)
project(qnn_networks_core)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(SRC
    ${CMAKE_SOURCE_DIR}/3rdparty/tracer/tracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qnnctx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/base_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bmodel_runner.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../face/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_INSTALL_PREFIX}/include
    ${PROJECT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/3rdparty/tracer
)

if(CMAKE_TOOLCHAIN_FILE)
    include_directories("${CMAKE_SOURCE_DIR}/3rdparty/prebuilt/include/")
    include_directories("${CMAKE_SOURCE_DIR}/3rdparty/ramdisk/prebuild/include/")
endif()

link_directories("${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_TOOLCHAIN_FILE)
    link_directories("${CMAKE_SOURCE_DIR}/3rdparty/prebuilt/")
    link_directories("${CMAKE_SOURCE_DIR}/3rdparty/ramdisk/prebuild/lib/")
    link_directories("${CMAKE_SOURCE_DIR}/3rdparty/ramdisk/prebuild/lib/aarch64-linux-gnu/")
endif()

add_library(qnn_networks_core SHARED ${SRC})
target_link_libraries(qnn_networks_core ${BM_LIBS} qnn_networks_utils ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(${FEXT_NAME} PROPERTIES
                      VERSION ${QNN_VERSION_STRING}
                      SOVERSION ${QNN_VERSION_MAJOR})

install(TARGETS qnn_networks_core DESTINATION ${BM_INSTALL_PREFIX}/lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include/${BM_INSTALL_INCLUDENAME})
