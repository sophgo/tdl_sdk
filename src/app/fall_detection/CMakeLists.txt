project(fall_detection_app)

include_directories(
                    ${REPO_DIR}/include
                    ${REPO_DIR}/include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

set(PROJ_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/fall_detection_app.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/fall_detection.cpp)

message(STATUS "build ${PROJECT_NAME}")
add_library(fall_detection_app OBJECT ${PROJ_SRCS})
if(APP_VIDEO_ENABLE)
    target_compile_definitions(fall_detection_app PRIVATE VIDEO_ENABLE)
endif()
