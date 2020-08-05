project(cviai_frservice)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../utils)

add_library(${PROJECT_NAME} SHARED cviai_frservice.cpp)
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)