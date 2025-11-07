# Hide symbols
# if("${CMAKE_BUILD_TYPE}" MATCHES "(SDKRelease|Release)")
#   add_compile_options(-fvisibility=hidden)
# endif()


# tdl lib

add_subdirectory(framework)
add_subdirectory(components)
add_subdirectory(c_apis)
if(NOT DEFINED ENABLE_APP_PIPELINE OR ENABLE_APP_PIPELINE)
   add_subdirectory(pipeline)
   add_subdirectory(app)
endif()

set(CORES_SRCS
               $<TARGET_OBJECTS:framework>
               $<TARGET_OBJECTS:tracker>
               $<TARGET_OBJECTS:matcher>
               $<TARGET_OBJECTS:snapshot>
               $<TARGET_OBJECTS:nn>
               $<TARGET_OBJECTS:c_apis_core>
               $<TARGET_OBJECTS:intrusion_detect>
               $<TARGET_OBJECTS:occlusion_detect>
               $<TARGET_OBJECTS:target_search>
               $<TARGET_OBJECTS:encoder>
               )

# 条件添加 pipeline 和 app（满足任一条件即添加）
if(NOT DEFINED ENABLE_APP_PIPELINE OR ENABLE_APP_PIPELINE)
    list(APPEND CORES_SRCS 
        $<TARGET_OBJECTS:pipeline>
        $<TARGET_OBJECTS:app>
    )
endif()

if(NOT "${CVI_PLATFORM}" STREQUAL "CMODEL_CV181X" AND NOT "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
   set(EX_SRCS $<TARGET_OBJECTS:c_apis_ex> $<TARGET_OBJECTS:network>)
endif()

if(APP_VIDEO_ENABLE)
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:video_decoder>)
else()
   set(EX_SRCS ${EX_SRCS} $<TARGET_OBJECTS:video_decoder>)
endif()


if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR ("${CVI_PLATFORM}" STREQUAL "BM1684"))
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:llm> )
endif()

if("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:ive>)
endif()

if("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X"  OR "${CVI_PLATFORM}" STREQUAL "SOPHON" OR "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
   set(CORES_SRCS ${CORES_SRCS} $<TARGET_OBJECTS:motion_detect>)
endif()


# shared library
add_library(tdl_core SHARED ${CORES_SRCS})

if(NOT DEFINED ENABLE_APP_PIPELINE OR ENABLE_APP_PIPELINE)
   target_link_libraries(tdl_core
      ${MLIR_LIBS}
      ${OPENCV_LIBS_IMCODEC}
      ${IVE_LIBS}
      ${ZLIB_LIBRARY}
      kaldi-native-fbank-core
   )
else()
   target_link_libraries(tdl_core
   ${MLIR_LIBS}
   ${OPENCV_LIBS_IMCODEC}
   ${IVE_LIBS}
   ${ZLIB_LIBRARY}
   )
endif()

# if("${CVI_PLATFORM}" STREQUAL "SOPHON" OR "${CVI_PLATFORM}" STREQUAL "BM1688")
#    target_link_libraries(tdl_core
#       ${ZLIB_LIBRARY}
#    )
# endif()

add_library(tdl_ex SHARED ${EX_SRCS})
target_link_libraries(tdl_ex ${LIBWEBSOCKETS_LIBS}
                      ${OPENSSL_LIBRARY}
                      ${ZLIB_LIBRARY}
                      ${CURL_LIBRARY})

# static library
add_library(tdl_core-static STATIC ${CORES_SRCS})
if(NOT DEFINED ENABLE_SPEECH_RECOGNITION OR ENABLE_SPEECH_RECOGNITION)
   target_link_libraries(tdl_core-static ${MLIR_LIBS_STATIC} ${OPENCV_LIBS_IMCODEC_STATIC} kaldi-native-fbank-core)
else()
   target_link_libraries(tdl_core-static ${MLIR_LIBS_STATIC} ${OPENCV_LIBS_IMCODEC_STATIC})
endif()

add_library(tdl_ex-static STATIC ${EX_SRCS})
target_link_libraries(tdl_ex-static ${LIBWEBSOCKETS_LIBS_STATIC} ${CURL_LIBRARY_STATIC} ${OPENSSL_LIBRARY_STATIC} ${ZLIB_LIBRARY_STATIC} )

if(EXISTS ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends)
    add_subdirectory(python)
endif()

install(TARGETS tdl_core tdl_core-static tdl_ex tdl_ex-static DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/envsetup.sh DESTINATION scripts)