if ("${ENABLE_CVIAI_CV_UTILS}" STREQUAL "ON")
  add_compile_definitions(ENABLE_CVIAI_CV_UTILS)
endif()
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
add_subdirectory(thermal_person_detection)
add_subdirectory(object_detection/mobiledetv2)
add_subdirectory(mask_face_recognition)
add_subdirectory(deepsort)
add_subdirectory(sound_classification)
add_subdirectory(segmentation)
add_subdirectory(license_plate_recognition)
add_subdirectory(license_plate_detection)
add_subdirectory(alphapose)
add_subdirectory(fall_detection)
add_subdirectory(eye_classification)
add_subdirectory(yawn_classification)
add_subdirectory(face_landmarker)
add_subdirectory(incar_object_detection)
add_subdirectory(smoke_classification)

if ("${USE_IVE}" STREQUAL "ON")
  add_subdirectory(tamper_detection)
  add_subdirectory(motion_detection)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils)

set(CORE_SRCS cviai_types_mem.cpp
              cviai_rescale_bbox.cpp
              cviai_utils.cpp
              cviai_custom.cpp
              cviai_core.cpp)

project(cviai_core)
add_library(${PROJECT_NAME} OBJECT ${CORE_SRCS})
