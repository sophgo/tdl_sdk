# Hide symbols
# if("${CMAKE_BUILD_TYPE}" MATCHES "(SDKRelease|Release)")
#   add_compile_options(-fvisibility=hidden)
# endif()


# tdl lib

add_subdirectory(framework)
add_subdirectory(components)
add_subdirectory(c_apis)
add_subdirectory(pipeline)
add_subdirectory(app)

set(CORES_SRCS
            #    $<TARGET_OBJECTS:components>
               $<TARGET_OBJECTS:framework>
               $<TARGET_OBJECTS:tracker>
               $<TARGET_OBJECTS:matcher>
               $<TARGET_OBJECTS:snapshot>
               $<TARGET_OBJECTS:video_decoder>
               $<TARGET_OBJECTS:nn>               
               $<TARGET_OBJECTS:c_apis>
               $<TARGET_OBJECTS:pipeline>
               $<TARGET_OBJECTS:face_capture_app>
               $<TARGET_OBJECTS:face_pet_capture_app>
               $<TARGET_OBJECTS:fall_detection_app>
               $<TARGET_OBJECTS:app>
               $<TARGET_OBJECTS:intrusion_detect>
               $<TARGET_OBJECTS:occlusion_detect>
               )

if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR ("${CVI_PLATFORM}" STREQUAL "BM1684"))
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:llm>)
endif()
if("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:ive>)
endif()

if("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X" OR "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:motion_detect>)
endif()

# shared library
add_library(tdl SHARED ${CORES_SRCS})
target_link_libraries(tdl ${MLIR_LIBS} ${OPENCV_LIBS_IMCODEC} ${IVE_LIBS})

# static library
add_library(tdl-static STATIC ${CORES_SRCS})
target_link_libraries(tdl-static ${MLIR_LIBS_STATIC} ${OPENCV_LIBS_IMCODEC_STATIC})

if(EXISTS ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends)
    add_subdirectory(python)
endif()

install(TARGETS tdl tdl-static DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/envsetup.sh DESTINATION scripts)