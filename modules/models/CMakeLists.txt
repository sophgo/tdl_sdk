project(models)

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    )


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/face_detection SRC_FACE_DETECTION_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/object_detection SRC_OBJECT_DETECTION_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/utils SRC_UTILS_FILES)


set(PROJ_SRCS ${SRC_FACE_DETECTION_FILES} ${SRC_OBJECT_DETECTION_FILES} ${SRC_UTILS_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/tdl_model_factory.cpp)
add_library(${PROJECT_NAME} OBJECT ${PROJ_SRCS})
