project(snapshot)

include_directories(
                    ${REPO_DIR}/include/
                    ${REPO_DIR}/include/framework
                    
                    )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


# message(STATUS "SRC_FRAMWORK_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})
