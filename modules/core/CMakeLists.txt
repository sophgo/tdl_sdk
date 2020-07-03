project(core)
add_library(${PROJECT_NAME} STATIC core.cpp)
target_link_libraries(${PROJECT_NAME} ${MLIR_LIBS})