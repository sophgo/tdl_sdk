project(cviai_trace)
add_library(${PROJECT_NAME} OBJECT cviai_trace.cpp)
target_link_libraries(${PROJECT_NAME} ${PERFETTO_LIBS})
