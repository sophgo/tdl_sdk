project(audio)

include_directories(
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/components/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

file(GLOB_RECURSE SRC_AUDIO_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${PROJECT_NAME} OBJECT ${SRC_AUDIO_FILES_CUR})
