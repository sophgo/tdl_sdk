cmake_minimum_required(VERSION 3.16.3)
project(tdl_utils)

if(${CVI_PLATFORM} STREQUAL "CV181X" OR ${CVI_PLATFORM} STREQUAL "CV184X")
add_subdirectory(
    ${REPO_DIR}/src/components/encoder/rtsp
    ${CMAKE_CURRENT_BINARY_DIR}/rtsp_build
    )
endif()

set(UTILS_INCLUDES
    ${SAMPLE_INCLUDES}
    ${REPO_DIR}/src/c_apis/include
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/components/video_decoder
    ${REPO_DIR}/src/components/video_decoder
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(UTILS_INCLUDES ${UTILS_INCLUDES} PARENT_SCOPE)

file(GLOB UTILS_SOURCES
    *.cpp
)

if(${CVI_PLATFORM} STREQUAL "CV181X" OR ${CVI_PLATFORM} STREQUAL "CV184X")
    list(APPEND UTILS_INCLUDES ${RTSP_INCLUDE})
    list(APPEND UTILS_SOURCES ${REPO_DIR}/src/components/encoder/rtsp/rtsp.cpp)
    list(APPEND UTILS_SOURCES ${REPO_DIR}/src/components/video_decoder/video_decoder_factory.cpp)
    list(APPEND UTILS_SOURCES ${REPO_DIR}/src/components/video_decoder/image_folder/image_folder_decoder.cpp)
    list(APPEND UTILS_SOURCES ${REPO_DIR}/src/components/video_decoder/vi_decoder/vi_decoder.cpp)
endif()

if(NOT ${CVI_PLATFORM} STREQUAL "CV181X" AND NOT ${CVI_PLATFORM} STREQUAL "CV184X")
    list(REMOVE_ITEM UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/rtsp_utils.cpp)
endif()

include_directories(${UTILS_INCLUDES})
# Add shared library
add_library(tdl_utils SHARED ${UTILS_SOURCES}  ${MIDDLEWARE_OBJ})
set_target_properties(tdl_utils PROPERTIES
    ENABLE_EXPORTS ON
    POSITION_INDEPENDENT_CODE ON
)

# Add static library
add_library(tdl_utils-static STATIC ${UTILS_SOURCES} ${RTSP_LIBS_STATIC} ${MIDDLEWARE_OBJ})
set_target_properties(tdl_utils-static PROPERTIES
    OUTPUT_NAME tdl_utils
    ENABLE_EXPORTS ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(tdl_utils-static
    "-Wl,--start-group"
    "-Wl,--whole-archive"
    ${CVI_RTSP_LIBPATH_STATIC}
    ${RTSP_LIBS_STATIC}
    ${MIDDLEWARE_LIBS_STATIC}
    "-Wl,--no-whole-archive"
    ${OPENCV_LIBS_IMCODEC_STATIC}
    ${MLIR_LIBS_STATIC}
    ${IVE_LIBS_STATIC}
    dl rt atomic m
    "-Wl,--end-group"
)

target_link_libraries(tdl_utils
    "-Wl,--start-group"
    ${CVI_RTSP_LIBPATH}
    ${RTSP_LIBS_SHARED}
    ${MLIR_LIBS}
    dl rt atomic m
    "-Wl,--end-group"
)

# Install libraries
install(TARGETS tdl_utils tdl_utils-static
    LIBRARY DESTINATION sample/utils/lib
    ARCHIVE DESTINATION sample/utils/lib
)

# Install headers
file(GLOB HEADER_FILES *.h *.hpp)
install(FILES ${HEADER_FILES}
    DESTINATION sample/utils/include
)
