project(core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../utils
                    ${IVE_INCLUDES})
add_library(${PROJECT_NAME} OBJECT vpss_engine.cpp core.cpp model_debugger.cpp)
