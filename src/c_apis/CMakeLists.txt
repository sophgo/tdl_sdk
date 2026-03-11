project(c_apis)
include_directories(${REPO_DIR}/include
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${REPO_DIR}/include/c_apis
                    ${REPO_DIR}/src/app
                    ${REPO_DIR}/src/components/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${FBANK_INCLUDES}
                )

file(GLOB_RECURSE SRC_CORE_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/src/tdl_core.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tdl_utils.cpp)
file(GLOB_RECURSE SRC_EX_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/src/tdl_ex.cpp)

add_library(c_apis_core OBJECT ${SRC_CORE_FILES_CUR})
add_library(c_apis_ex OBJECT ${SRC_EX_FILES_CUR})
