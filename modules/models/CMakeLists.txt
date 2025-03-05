project(models)

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    )



file(GLOB_RECURSE SRC_FRAMWORK_FILES_SUB ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp)
file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_SUB} ${SRC_FRAMWORK_FILES_CUR})
