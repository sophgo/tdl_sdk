project(stereo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
add_library(${PROJECT_NAME} OBJECT stereo.cpp)
