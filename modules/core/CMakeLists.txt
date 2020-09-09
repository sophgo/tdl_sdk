add_subdirectory(core)
add_subdirectory(utils)
add_subdirectory(face_attribute)
add_subdirectory(face_quality)
add_subdirectory(liveness)
add_subdirectory(mask_classification)
add_subdirectory(retina_face)
add_subdirectory(object_detection/yolov3)
add_subdirectory(thermal_face_detection)
add_subdirectory(object_detection/mobiledetv2)
add_subdirectory(mask_face_recognition)
add_subdirectory(tamper_detection)

#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils
                    ${IVE_INCLUDES})
set(CORE_SRCS $<TARGET_OBJECTS:cviai_trace>
              $<TARGET_OBJECTS:core>
              $<TARGET_OBJECTS:utils>
              $<TARGET_OBJECTS:face_attribute>
              $<TARGET_OBJECTS:face_quality>
              $<TARGET_OBJECTS:liveness>
              $<TARGET_OBJECTS:mobiledetv2>
              $<TARGET_OBJECTS:mask_face_recognition>
              $<TARGET_OBJECTS:mask_classification>
              $<TARGET_OBJECTS:retina_face>
              $<TARGET_OBJECTS:yolov3>
              $<TARGET_OBJECTS:thermal_face_detection>
              $<TARGET_OBJECTS:tamper_detection>
              cviai_types_free.cpp
              cviai_rescale_bbox.cpp
              cviai_core.cpp)
set(CORE_DEPS ${PERFETTO_LIBS} ${OPENCV_LIBS_MIN} ${MLIR_LIBS} ${MIDDLEWARE_LIBS} ${IVE_LIBS})

project(cviai_core)
add_library(${PROJECT_NAME} SHARED ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_core-static)
add_library(${PROJECT_NAME} STATIC ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
