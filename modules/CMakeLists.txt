add_subdirectory(core)
add_subdirectory(service)
add_subdirectory(frservice)
add_subdirectory(evaluation)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include DESTINATION include/cviai)