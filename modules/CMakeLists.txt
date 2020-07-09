project(cviai)

add_subdirectory(core)
add_subdirectory(utils)
add_subdirectory(face_attribute)

project(cviai)
#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
add_library(${PROJECT_NAME} SHARED $<TARGET_OBJECTS:core>
                                   $<TARGET_OBJECTS:utils>
                                   $<TARGET_OBJECTS:face_attribute>)
target_link_libraries(${PROJECT_NAME} ${MLIR_LIBS} ${OPENCV_LIBS} ${MIDDLEWARE_LIBS})
