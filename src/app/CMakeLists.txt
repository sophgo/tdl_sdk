project(app)
add_subdirectory(face_capture)
add_subdirectory(face_pet_capture)
add_subdirectory(fall_detection)
add_subdirectory(consumer_counting)

include_directories(${REPO_DIR}/include)
include_directories(${REPO_DIR}/include/framework)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(APP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/app_factory.cpp ${CMAKE_CURRENT_SOURCE_DIR}/app_task.cpp)

add_library(app OBJECT ${APP_SRCS})

if(APP_VIDEO_ENABLE)
    target_compile_definitions(app PRIVATE VIDEO_ENABLE)
endif()