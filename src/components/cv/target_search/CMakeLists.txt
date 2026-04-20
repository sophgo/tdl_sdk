project(target_search)

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

message(STATUS "SRC_GRABCUT_FILES: ${SRC_FRAMWORK_FILES_CUR}")
add_library(target_search OBJECT ${SRC_FRAMWORK_FILES_CUR})
