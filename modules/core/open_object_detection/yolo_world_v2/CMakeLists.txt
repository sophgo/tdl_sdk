project(yolo_world_v2)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
add_library(${PROJECT_NAME} OBJECT yolo_world_v2.cpp)
