project(motion_detect)

include_directories(
                    ${IVE_INCLUDES}
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/motion_detect.cpp ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)

if("${CVI_PLATFORM}" STREQUAL "CV184X" OR "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
    file(GLOB BM_MOTION_DETECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/bm_motion_detect/*.cpp)
    list(APPEND SRC_FRAMWORK_FILES_CUR ${BM_MOTION_DETECT_FILES})
endif()
if("${CVI_PLATFORM}" STREQUAL "CV181X")
    file(GLOB_RECURSE CVI_MOTION_DETECT_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/cvi_motion_detect/*.cpp)
    list(APPEND SRC_FRAMWORK_FILES_CUR ${CVI_MOTION_DETECT_FILES})
endif()

add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})
target_link_libraries(${PROJECT_NAME} ${IVE_LIBS})