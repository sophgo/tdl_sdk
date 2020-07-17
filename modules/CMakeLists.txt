project(cviai)

add_subdirectory(core)
add_subdirectory(utils)
add_subdirectory(face_attribute)
add_subdirectory(face_quality)
add_subdirectory(liveness)
add_subdirectory(retina_face)
add_subdirectory(yolov3)
add_subdirectory(thermal_face_detection)

#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_library(${PROJECT_NAME} SHARED $<TARGET_OBJECTS:core>
                                   $<TARGET_OBJECTS:utils>
                                   $<TARGET_OBJECTS:face_attribute>
				   $<TARGET_OBJECTS:face_quality>
				   $<TARGET_OBJECTS:liveness>
                                   $<TARGET_OBJECTS:retina_face>
				   $<TARGET_OBJECTS:yolov3>
				   $<TARGET_OBJECTS:thermal_face_detection>
				   cviai_types_free.cpp
				   cviai.cpp)
target_link_libraries(${PROJECT_NAME} ${MLIR_LIBS} ${OPENCV_LIBS} ${MIDDLEWARE_LIBS})
