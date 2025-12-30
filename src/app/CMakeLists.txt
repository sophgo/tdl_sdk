project(app)

include_directories(${REPO_DIR}/include)
include_directories(${REPO_DIR}/include/framework)
include_directories(${REPO_DIR}/include/nn)
include_directories(${REPO_DIR}/include/components)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(app OBJECT ${SRC_FRAMWORK_FILES_CUR})

if(APP_VIDEO_ENABLE)
    target_compile_definitions(app PRIVATE VIDEO_ENABLE)
endif()