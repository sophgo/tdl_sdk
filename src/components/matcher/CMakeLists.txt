project(matcher)

include_directories(
    ${REPO_DIR}/include/framework
    ${REPO_DIR}/include/components
    ${REPO_DIR}/include/nn
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

# 设置基本源文件
set(SRC_FRAMWORK_FILES_CUR 
    ${CMAKE_CURRENT_SOURCE_DIR}/base_matcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_matcher/cpu_matcher.cpp
)

# 根据平台添加特定源文件
if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR ("${CVI_PLATFORM}" STREQUAL "CV184X"))
    list(APPEND SRC_FRAMWORK_FILES_CUR 
        ${CMAKE_CURRENT_SOURCE_DIR}/bm_matcher/bm_matcher.cpp
    )
elseif("${CVI_PLATFORM}" STREQUAL "CV181X")
    list(APPEND SRC_FRAMWORK_FILES_CUR 
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_matcher/cvi_matcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_matcher/utils/cvimath.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_matcher/utils/cvimath_internal.cpp
    )
endif()

add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})

# message(STATUS "SRC_FRAMWORK_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")

