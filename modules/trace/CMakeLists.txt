project(cvi_tdl_trace)
add_library(${PROJECT_NAME} OBJECT cvi_tdl_trace.cpp)
target_link_libraries(${PROJECT_NAME} ${PERFETTO_LIBS})
