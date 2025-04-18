project(matcher)

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684X"))
    add_definitions(-DUSE_BM_MATCHER)
    set(SRC_FRAMWORK_FILES_CUR 
        ${CMAKE_CURRENT_SOURCE_DIR}/base_matcher.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/bm_matcher/bm_matcher.cpp
        )
endif()

if("${CVI_PLATFORM}" STREQUAL "CV181X")
    add_definitions(-DUSE_CVI_MATCHER)
    set(SRC_FRAMWORK_FILES_CUR 
        ${CMAKE_CURRENT_SOURCE_DIR}/base_matcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_matcher/cvi_matcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_matcher/utils/cvimath.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_matcher/utils/cvimath_internal.cpp
        )
endif()


if(NOT SRC_FRAMWORK_FILES_CUR)
    set(SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/base_matcher.cpp)
    message(WARNING "No supported platform found for matcher component, using minimal implementation")
endif()

add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})

# message(STATUS "SRC_FRAMWORK_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")

