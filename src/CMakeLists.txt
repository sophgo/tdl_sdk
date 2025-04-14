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
               $<TARGET_OBJECTS:video_decoder>
               $<TARGET_OBJECTS:nn>               
               $<TARGET_OBJECTS:c_apis>
               $<TARGET_OBJECTS:pipeline>
               $<TARGET_OBJECTS:face_capture_app>
               $<TARGET_OBJECTS:app>

               )

if(TARGET camera)
    set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:camera>)
endif()

if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR ("${CVI_PLATFORM}" STREQUAL "BM1684"))
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:llm>)
endif()
set(CORES_DEPS ${MLIR_LIBS} )

project(cvi_tdl)
add_library(${PROJECT_NAME} SHARED ${CORES_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORES_DEPS} ${OPENCV_LIBS_IMCODEC})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cvi_tdl-static)
add_library(${PROJECT_NAME} STATIC ${CORES_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORES_DEPS} ${OPENCV_LIBS_IMCODEC_STATIC})
# SET_TARGET_PROPERTIES(cvi_tdl-static PROPERTIES OUTPUT_NAME "cvi_tdl")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

if(EXISTS ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends)
    add_subdirectory(python)
endif()



install(TARGETS cvi_tdl DESTINATION lib)

#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cvi_tdl_app/ DESTINATION sample/cvi_tdl_app PATTERN "CMakeLists.txt" EXCLUDE)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_and_eval/ DESTINATION sample PATTERN "CMakeLists.txt" EXCLUDE)
# Temporarily don't install custom module until complete.
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_tdl_app DESTINATION include/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/envsetup.sh DESTINATION scripts)