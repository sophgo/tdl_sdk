project(face_attribute)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
add_library(${PROJECT_NAME} STATIC face_attribute.cpp)
target_link_libraries(${PROJECT_NAME} ${MLIR_LIBS})