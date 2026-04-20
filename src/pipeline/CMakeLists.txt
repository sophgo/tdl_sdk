project(pipeline)
include_directories(
                    ${REPO_DIR}/include/                   
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/nn                   
                    ${CMAKE_CURRENT_SOURCE_DIR}/include               
                )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})

