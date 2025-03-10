project(c_apis)
include_directories(
                    ${REPO_DIR}/include/tdl_sdk/framework
                    ${REPO_DIR}/include/tdl_sdk/components
                    ${REPO_DIR}/include/tdl_sdk/app
                    ${REPO_DIR}/include/tdl_sdk/nn
                    ${REPO_DIR}/include/tdl_sdk/c_apis
                    ${CMAKE_CURRENT_SOURCE_DIR}/include               
                )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})

