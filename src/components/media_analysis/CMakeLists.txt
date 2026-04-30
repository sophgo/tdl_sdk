cmake_minimum_required(VERSION 3.16.3)
project(media_analysis)

# Add source files
set(SRC_MEDIA_ANALYSIS_FILES
    media_analysis_server.cpp
    media_analysis_event_manager.cpp
    tasks/face/face_matching_task.cpp
    tasks/image_text/image_text_task.cpp
    tasks/image_analysis/image_analysis_task.cpp
    tasks/identity/identity_browse_task.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks/face
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks/image_text
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks/image_analysis
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks/identity
    ${REPO_DIR}/include/components/media_analysis
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/network
    ${REPO_DIR}/include
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/nn
    ${REPO_DIR}/include/components/network/api_poster
)

# Create object library
add_library(${PROJECT_NAME} OBJECT ${SRC_MEDIA_ANALYSIS_FILES})

# Link libraries (if building shared/static lib, but here it's OBJECT lib)
# Dependencies will be resolved when linking the final executable/library


