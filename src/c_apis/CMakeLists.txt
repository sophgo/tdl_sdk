project(c_apis)
include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${REPO_DIR}/include/c_apis
                    ${CMAKE_CURRENT_SOURCE_DIR}/include               
                )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})

