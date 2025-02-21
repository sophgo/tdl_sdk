project(models)

include_directories(
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    )


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/face_detection SRC_FACE_DETECTION_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/face_landmark SRC_FACE_LANDMARK_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/object_detection SRC_OBJECT_DETECTION_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/utils SRC_UTILS_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/feature_extract SRC_FEATURE_EXTRACT_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/image_classification SRC_IMAGE_CLASSIFICATION_FILES)
set(PROJ_SRCS ${SRC_FACE_DETECTION_FILES} 
              ${SRC_FACE_LANDMARK_FILES} 
              ${SRC_OBJECT_DETECTION_FILES} 
              ${SRC_FEATURE_EXTRACT_FILES}
              ${SRC_IMAGE_CLASSIFICATION_FILES}
              ${SRC_UTILS_FILES} 
              ${CMAKE_CURRENT_SOURCE_DIR}/tdl_model_factory.cpp)
add_library(${PROJECT_NAME} OBJECT ${PROJ_SRCS})
