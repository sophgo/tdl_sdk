project(face_capture_app)

include_directories(
                    ${REPO_DIR}/include
                    ${REPO_DIR}/include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

set(PROJ_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/face_capture_app.cpp)

message(STATUS "build ${PROJECT_NAME}")
add_library(face_capture_app OBJECT ${PROJ_SRCS})
if(APP_VIDEO_ENABLE)
    target_compile_definitions(face_capture_app PRIVATE VIDEO_ENABLE)
endif()
