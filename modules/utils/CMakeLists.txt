project(utils)
add_library(${PROJECT_NAME} STATIC core_utils.cpp face_utils.cpp)
target_link_libraries(${PROJECT_NAME} ${MLIR_LIBS} ${OPENCV_LIBS})