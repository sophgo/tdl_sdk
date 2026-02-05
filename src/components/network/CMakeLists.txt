project(network)

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${REPO_DIR}/include/components/network/api_poster
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if (NOT ("${CVI_PLATFORM}" STREQUAL "CV180X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV181X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV182X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV183X") AND
    NOT ("${CVI_PLATFORM}" STREQUAL "CV184X"))
  list(REMOVE_ITEM SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/api_poster/voice_chat_client.cpp)
endif()


message(STATUS "SRC_MOT_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})