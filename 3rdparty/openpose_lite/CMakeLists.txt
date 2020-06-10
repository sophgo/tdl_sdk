project(openpose_lite)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

message(${PROJECT_SOURCE_DIR}/include)

#add_subdirectory(src/openpose/utilities)
#add_subdirectory(src/openpose/core)
#add_subdirectory(src/openpose/net)

set(SOURCES_OP_CORE
    ${PROJECT_SOURCE_DIR}/src/openpose/core/array.cpp
    ${PROJECT_SOURCE_DIR}/src/openpose/core/datum.cpp
    ${PROJECT_SOURCE_DIR}/src/openpose/core/point.cpp
    ${PROJECT_SOURCE_DIR}/src/openpose/core/rectangle.cpp)

set(SOURCES_OP_NET
    ${PROJECT_SOURCE_DIR}/src/openpose/net/nmsBase.cpp
    ${PROJECT_SOURCE_DIR}/src/openpose/net/resizeAndMergeBase.cpp
    ${PROJECT_SOURCE_DIR}/src/openpose/net/bodyPartConnectorBase.cpp)

set(SOURCES_OP_POSE
    ${PROJECT_SOURCE_DIR}/src/openpose/pose/poseParameters.cpp)

set(SOURCES_OP_UTILS
    ${PROJECT_SOURCE_DIR}/src/openpose/utilities/errorAndLog.cpp
    ${PROJECT_SOURCE_DIR}/src/openpose/utilities/openCv.cpp)

add_library(openpose_lite SHARED ${SOURCES_OP_CORE} ${SOURCES_OP_NET} ${SOURCES_OP_POSE} ${SOURCES_OP_UTILS})
target_link_libraries(openpose_lite ${OpenCV_LIBS})
install(TARGETS openpose_lite DESTINATION ${BM_INSTALL_PREFIX}/lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/openpose_lite/include DESTINATION ${BM_INSTALL_PREFIX})