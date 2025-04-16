project(matcher)

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

# file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# if(("${CVI_PLATFORM}" STREQUAL "CV180X") OR 
#    ("${CVI_PLATFORM}" STREQUAL "CV181X") OR 
#    ("${CVI_PLATFORM}" STREQUAL "CV182X") OR
#    ("${CVI_PLATFORM}" STREQUAL "CV183X") OR
#    ("${CVI_PLATFORM}" STREQUAL "SOPHON") OR
#    ("${CVI_PLATFORM}" STREQUAL "CMODEL_CVITEK"))
#   list(REMOVE_ITEM SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/bm_matcher.cpp)
# endif()
if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684X"))
    file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})
endif()


# message(STATUS "SRC_FRAMWORK_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")

