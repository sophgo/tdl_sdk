project(core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../utils)
add_library(${PROJECT_NAME} OBJECT vpss_engine.cpp core.cpp)
