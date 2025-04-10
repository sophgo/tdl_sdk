project(blip_vqa)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
add_library(${PROJECT_NAME} OBJECT blip_vqa_venc.cpp blip_vqa_tenc.cpp blip_vqa_tdec.cpp)
