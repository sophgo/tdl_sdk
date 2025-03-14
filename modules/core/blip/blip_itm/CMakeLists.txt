project(blip_itm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
add_library(${PROJECT_NAME} OBJECT blip_itm.cpp)
