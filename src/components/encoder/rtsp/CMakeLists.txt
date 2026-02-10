project(rtsp)

if (${CVI_PLATFORM} STREQUAL "CV181X" OR ${CVI_PLATFORM} STREQUAL "CV184X")
    set(RTSP_INCLUDE ${CVI_RTSP_INCLUDE} PARENT_SCOPE)
    set(RTSP_LIBS_SHARED ${CVI_RTSP_LIB_SHARED} PARENT_SCOPE)
    set(RTSP_LIBS_STATIC ${CVI_RTSP_LIB_STATIC} PARENT_SCOPE)
endif()

# Add link directories for ISP and sensor libraries
link_directories(
    ${MLIR_LIBS}
    ${MIDDLEWARE_LIBS} 
    ${OPENCV_LIBRARIES}
)

include_directories(
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${CVI_RTSP_INCLUDE}
    ${REPO_DIR}/include/
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
)

# Install headers
install(FILES ${REPO_DIR}/include/components/encoder/rtsp/rtsp.hpp
    DESTINATION sample/utils/include/
)

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

message(STATUS "SRC_RTSP_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})