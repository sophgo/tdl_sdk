project(ive)

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

set(SRC_FRAMWORK_FILES_CUR 
    ${CMAKE_CURRENT_SOURCE_DIR}/image_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bm_image_processor/api_tpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bm_image_processor/bm_image_processor.cpp
)


# message(STATUS "SRC_FRAMWORK_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})