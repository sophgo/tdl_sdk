# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

cmake_minimum_required(VERSION 3.2.2)
project(qnn_networks_utils)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(SRC
    ${CMAKE_SOURCE_DIR}/3rdparty/tracer/tracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/file_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/face_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/math_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/string_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blas_npu.cpp
)

if(CMAKE_TOOLCHAIN_FILE)
    enable_language(C ASM)
    set(SRC ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/VectorDot.S)
    set(SRC ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/VectorNorm.S)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/VectorDot.S PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/VectorNorm.S PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
else()
    set(SRC ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/blas_cpu.cpp)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/tracer
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../face/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_INSTALL_PREFIX}/include
    ${PROJECT_BINARY_DIR}
)

add_library(qnn_networks_utils SHARED ${SRC})
target_link_libraries(qnn_networks_utils ${BM_LIBS} pthread ${GLOG_LIBRARIES} ${OPENBLAS_LIBRARIES})
set_target_properties(${FEXT_NAME} PROPERTIES
                      VERSION ${QNN_VERSION_STRING}
                      SOVERSION ${QNN_VERSION_MAJOR})


install(TARGETS qnn_networks_utils DESTINATION ${BM_INSTALL_PREFIX}/lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include/${BM_INSTALL_INCLUDENAME})
