add_subdirectory(core)
add_subdirectory(custom)
add_subdirectory(utils)
add_subdirectory(face_attribute)
add_subdirectory(face_quality)
add_subdirectory(liveness)
add_subdirectory(mask_classification)
add_subdirectory(osnet)
add_subdirectory(retina_face)
add_subdirectory(object_detection/yolov3)
add_subdirectory(thermal_face_detection)
add_subdirectory(object_detection/mobiledetv2)
add_subdirectory(mask_face_recognition)
add_subdirectory(tamper_detection)
add_subdirectory(deepsort)
add_subdirectory(es_classification)
add_subdirectory(segmentation)
add_subdirectory(license_plate_recognition)
add_subdirectory(license_plate_detection)
add_subdirectory(alphapose)
add_subdirectory(fall_detection)
add_subdirectory(eye_classification)
add_subdirectory(yawn_classification)
add_subdirectory(face_landmarker)
add_subdirectory(incar_object_detection)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(CORE_SRCS $<TARGET_OBJECTS:cviai_trace>
              $<TARGET_OBJECTS:core>
              $<TARGET_OBJECTS:custom>
              $<TARGET_OBJECTS:utils>
              $<TARGET_OBJECTS:face_attribute>
              $<TARGET_OBJECTS:face_quality>
              $<TARGET_OBJECTS:liveness>
              $<TARGET_OBJECTS:mobiledetv2>
              $<TARGET_OBJECTS:mask_face_recognition>
              $<TARGET_OBJECTS:mask_classification>
              $<TARGET_OBJECTS:osnet>
              $<TARGET_OBJECTS:retina_face>
              $<TARGET_OBJECTS:yolov3>
              $<TARGET_OBJECTS:thermal_face_detection>
              $<TARGET_OBJECTS:tamper_detection>
              $<TARGET_OBJECTS:deepsort>
              $<TARGET_OBJECTS:es_classification>
	          $<TARGET_OBJECTS:segmentation>
	          $<TARGET_OBJECTS:license_plate_recognition>
	          $<TARGET_OBJECTS:license_plate_detection>
	          $<TARGET_OBJECTS:alphapose>
              $<TARGET_OBJECTS:fall_detection>
              $<TARGET_OBJECTS:eye_classification>
              $<TARGET_OBJECTS:yawn_classification>
              $<TARGET_OBJECTS:face_landmarker>
              $<TARGET_OBJECTS:incar_object_detection>
              cviai_types_mem.cpp
              cviai_rescale_bbox.cpp
              cviai_media.cpp
              cviai_utils.cpp
              cviai_custom.cpp
              cviai_core.cpp)
set(CORE_DEPS ${PERFETTO_LIBS} ${OPENCV_LIBS_MIN} ${MLIR_LIBS} ${MIDDLEWARE_LIBS} ${IVE_LIBS})

project(cviai_core)
add_library(${PROJECT_NAME} SHARED ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      VERSION ${MAJOR_VERSION}
                      SOVERSION ${SO_VERSION})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_core-static)
add_library(${PROJECT_NAME} STATIC ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
