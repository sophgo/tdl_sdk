project(tamper_detection)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
                    ${IVE_INCLUDES})
add_library(${PROJECT_NAME} OBJECT tamper_detection.cpp)
