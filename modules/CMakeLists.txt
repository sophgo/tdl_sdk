# Modules
project(modules)
# Hide symbols
if("${CMAKE_BUILD_TYPE}" MATCHES "(SDKRelease|Release)")
  add_compile_options(-fvisibility=hidden)
endif()

set(CVI_TDL_TAG ${LATEST_TAG})
configure_file(version.hpp.in version.hpp @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/log
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_app
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_evaluation)

add_subdirectory(app)

add_subdirectory(core)
add_subdirectory(service)
add_subdirectory(test_and_eval)

if(DEFINED NO_OPENCV)
  set(CORES_SRCS $<TARGET_OBJECTS:cvi_tdl_core>
                  $<TARGET_OBJECTS:cvi_tdl_service>
                  $<TARGET_OBJECTS:core>
                  $<TARGET_OBJECTS:custom>
                  $<TARGET_OBJECTS:utils>
                  $<TARGET_OBJECTS:yolov3>
                  $<TARGET_OBJECTS:yolov5>
                  $<TARGET_OBJECTS:yolov6>
                  $<TARGET_OBJECTS:yolov8>
                  $<TARGET_OBJECTS:yolox>
                  $<TARGET_OBJECTS:ppyoloe>
                  $<TARGET_OBJECTS:yolo>
                  $<TARGET_OBJECTS:retina_face>
                  $<TARGET_OBJECTS:digital_tracking>
                  $<TARGET_OBJECTS:draw_rect>
                  $<TARGET_OBJECTS:tracker>
                  $<TARGET_OBJECTS:area_detect>
                  $<TARGET_OBJECTS:feature_matching>
                  $<TARGET_OBJECTS:ivewrapper>
                  $<TARGET_OBJECTS:motion_detection>
                  $<TARGET_OBJECTS:motion_detection>
                  $<TARGET_OBJECTS:mobiledetv2>
                  $<TARGET_OBJECTS:deepsort>
                  $<TARGET_OBJECTS:face_attribute>
                  $<TARGET_OBJECTS:sound_classification>)

  message("module no opencv ,core_srcs:" ${CORES_SRCS} )
  set(CORES_DEPS ${PERFETTO_LIBS} ${MLIR_LIBS} ${MIDDLEWARE_LIBS} ${IVE_LIBS})
else()
    set(CORES_SRCS $<TARGET_OBJECTS:cvi_tdl_core>
      $<TARGET_OBJECTS:cvi_tdl_service>
      $<TARGET_OBJECTS:core>
      $<TARGET_OBJECTS:custom>
      $<TARGET_OBJECTS:utils>
      $<TARGET_OBJECTS:face_attribute>
      $<TARGET_OBJECTS:face_quality>
      $<TARGET_OBJECTS:liveness>
      $<TARGET_OBJECTS:mobiledetv2>
      $<TARGET_OBJECTS:mask_face_recognition>
      $<TARGET_OBJECTS:mask_classification>
      $<TARGET_OBJECTS:hand_classification>
      $<TARGET_OBJECTS:hand_keypoint>
      $<TARGET_OBJECTS:hand_keypoint_classification>
      $<TARGET_OBJECTS:osnet>
      $<TARGET_OBJECTS:retina_face>
      $<TARGET_OBJECTS:yolov3>
      $<TARGET_OBJECTS:yolov5>
      $<TARGET_OBJECTS:yolov6>
      $<TARGET_OBJECTS:yolov8>
      $<TARGET_OBJECTS:yolox>
      $<TARGET_OBJECTS:ppyoloe>
      $<TARGET_OBJECTS:yolo>
      $<TARGET_OBJECTS:yolov8_pose>
      $<TARGET_OBJECTS:thermal_face_detection>
      $<TARGET_OBJECTS:thermal_person_detection>
      $<TARGET_OBJECTS:deepsort>
      $<TARGET_OBJECTS:sound_classification>
      $<TARGET_OBJECTS:segmentation>
      $<TARGET_OBJECTS:license_plate_recognition>
      $<TARGET_OBJECTS:license_plate_recognitionv2>
      $<TARGET_OBJECTS:license_plate_detection>
      $<TARGET_OBJECTS:alphapose>
      $<TARGET_OBJECTS:fall_detection>
      $<TARGET_OBJECTS:eye_classification>
      $<TARGET_OBJECTS:yawn_classification>
      $<TARGET_OBJECTS:face_landmarker>
      $<TARGET_OBJECTS:incar_object_detection>
      $<TARGET_OBJECTS:smoke_classification>
      $<TARGET_OBJECTS:digital_tracking>
      $<TARGET_OBJECTS:draw_rect>
      $<TARGET_OBJECTS:tracker>
      $<TARGET_OBJECTS:area_detect>
      $<TARGET_OBJECTS:face_angle>
      $<TARGET_OBJECTS:ivewrapper>
      $<TARGET_OBJECTS:motion_detection>
      $<TARGET_OBJECTS:tamper_detection>
      $<TARGET_OBJECTS:face_mask_detection>
      $<TARGET_OBJECTS:simcc>
      $<TARGET_OBJECTS:hrnet>
      $<TARGET_OBJECTS:clip>
      $<TARGET_OBJECTS:image_classification>
      $<TARGET_OBJECTS:raw_image_classification>
      $<TARGET_OBJECTS:lane_detection>)

    if(NOT ${CVI_PLATFORM} STREQUAL "CV186X")
    set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:feature_matching>)
    endif()

    if ("${ENABLE_CVI_TDL_CV_UTILS}" STREQUAL "ON")
    set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:utils-cv>)
    endif()
    set(CORES_DEPS ${PERFETTO_LIBS} ${MLIR_LIBS} ${MIDDLEWARE_LIBS} ${IVE_LIBS})
endif()

project(cvi_tdl)
add_library(${PROJECT_NAME} SHARED ${CORES_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORES_DEPS} ${OPENCV_LIBS_IMCODEC})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cvi_tdl-static)
add_library(${PROJECT_NAME} STATIC ${CORES_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORES_DEPS} ${OPENCV_LIBS_IMCODEC_STATIC})
SET_TARGET_PROPERTIES(cvi_tdl-static PROPERTIES OUTPUT_NAME "cvi_tdl")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cvi_tdl_app/ DESTINATION sample/cvi_tdl_app PATTERN "CMakeLists.txt" EXCLUDE)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_and_eval/ DESTINATION sample PATTERN "CMakeLists.txt" EXCLUDE)
# Temporarily don't install custom module until complete.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_app DESTINATION include/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/ DESTINATION include/cvi_tdl
        PATTERN "cvi_tdl_app" EXCLUDE
        PATTERN "cvi_tdl_experimental.h" EXCLUDE
        PATTERN "cvi_tdl_log.hpp" EXCLUDE)
