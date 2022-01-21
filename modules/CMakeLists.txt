# Modules
project(modules)
# Hide symbols
if("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  add_compile_options(-fvisibility=hidden)
endif()

set(CVIAI_TAG ${LATEST_TAG})
configure_file(version.hpp.in version.hpp @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/cviai_trace
                    ${CMAKE_CURRENT_SOURCE_DIR}/log
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_subdirectory(cviai_trace) # Perfetto tracer comes first
add_subdirectory(core)
add_subdirectory(service)
set(CORES_SRCS $<TARGET_OBJECTS:cviai_core>
               $<TARGET_OBJECTS:cviai_service>
               $<TARGET_OBJECTS:cviai_trace>
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
               $<TARGET_OBJECTS:thermal_person_detection>
               $<TARGET_OBJECTS:tamper_detection>
               $<TARGET_OBJECTS:deepsort>
               $<TARGET_OBJECTS:sound_classification>
	             $<TARGET_OBJECTS:segmentation>
	             $<TARGET_OBJECTS:license_plate_recognition>
	             $<TARGET_OBJECTS:license_plate_detection>
	             $<TARGET_OBJECTS:alphapose>
               $<TARGET_OBJECTS:fall_detection>
               $<TARGET_OBJECTS:eye_classification>
               $<TARGET_OBJECTS:yawn_classification>
               $<TARGET_OBJECTS:face_landmarker>
               $<TARGET_OBJECTS:incar_object_detection>
               $<TARGET_OBJECTS:motion_detection>
               $<TARGET_OBJECTS:smoke_classification>
               $<TARGET_OBJECTS:digital_tracking>
               $<TARGET_OBJECTS:draw_rect>
               $<TARGET_OBJECTS:tracker>
               $<TARGET_OBJECTS:area_detect>
               $<TARGET_OBJECTS:feature_matching>
	             $<TARGET_OBJECTS:face_angle>)
if ("${ENABLE_CVIAI_CV_UTILS}" STREQUAL "ON")
    set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:utils-cv>)
endif()
set(CORES_DEPS ${PERFETTO_LIBS} ${MLIR_LIBS} ${MIDDLEWARE_LIBS} ${IVE_LIBS})
set(CORES_DEPS ${CORES_DEPS} ${OPENCV_LIBS_IMGPROC})
set(CORES_DEPS ${CORES_DEPS} ${OPENCV_LIBS_MIN})

project(cviai_cores)
add_library(${PROJECT_NAME} SHARED ${CORES_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORES_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_cores-static)
add_library(${PROJECT_NAME} STATIC ${CORES_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORES_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

add_subdirectory(app)
add_subdirectory(evaluation)

# Temporarily don't install custom module until complete.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/ DESTINATION include/cviai
        PATTERN "cviai_perfetto.h" EXCLUDE
        PATTERN "core/cviai_types_mem_internal.h" EXCLUDE)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/app/ DESTINATION module/app
        PATTERN "CMakeLists.txt" EXCLUDE)
