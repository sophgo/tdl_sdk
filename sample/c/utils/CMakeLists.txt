cmake_minimum_required(VERSION 3.16.3)
project(tdl_utils)

if (${CVI_PLATFORM} STREQUAL "CV181X" OR ${CVI_PLATFORM} STREQUAL "CV184X")
    set(SOURCE_DIR $ENV{TOP_DIR}/cvi_rtsp/install)
    set(CVI_RTSP_LIBPATH ${SOURCE_DIR}/lib/libcvi_rtsp.so)
    if(${CVI_PLATFORM} STREQUAL "CV181X")
        set(OTHER_LIB_DIR ${MLIR_SDK_ROOT}/lib)
    else()
        set(OTHER_LIB_DIR ${TOP_DIR}/cvi_rtsp/prebuilt/lib)
    endif()
    set(CVI_RTSP_LIBPATH_STATIC 
        ${OTHER_LIB_DIR}/libBasicUsageEnvironment.a
        ${OTHER_LIB_DIR}/libgroupsock.a
        ${OTHER_LIB_DIR}/libUsageEnvironment.a
        ${OTHER_LIB_DIR}/libliveMedia.a
        ${SOURCE_DIR}/lib/libcvi_rtsp.a
    )
    set(CVI_RTSP_INCLUDE ${SOURCE_DIR}/include/cvi_rtsp)
endif()

set(UTILS_INCLUDES
    ${SAMPLE_INCLUDES}
    ${CVI_RTSP_INCLUDE}
    ${REPO_DIR}/src/c_apis/include
)

set(UTILS_INCLUDES ${UTILS_INCLUDES} PARENT_SCOPE)

include_directories(${UTILS_INCLUDES})

file(GLOB UTILS_SOURCES *.cpp)

if(NOT ${CVI_PLATFORM} STREQUAL "CV181X" AND NOT ${CVI_PLATFORM} STREQUAL "CV184X")
    list(REMOVE_ITEM UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/rtsp_utils.cpp)
endif()

# Add shared library
add_library(tdl_utils SHARED ${UTILS_SOURCES})
set_target_properties(tdl_utils PROPERTIES 
    ENABLE_EXPORTS ON
    POSITION_INDEPENDENT_CODE ON
)

# Add static library
add_library(tdl_utils-static STATIC ${UTILS_SOURCES})
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
