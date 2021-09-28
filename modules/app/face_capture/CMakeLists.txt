project(face_capture)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils)
add_library(${PROJECT_NAME} OBJECT face_capture.c)