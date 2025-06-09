project(video_decoder)

include_directories(
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )


if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
   ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR 
   ("${CVI_PLATFORM}" STREQUAL "BM1684"))
  set(DECODER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/opencv_decoder/opencv_decoder.cpp)
  
elseif(${CVI_PLATFORM} STREQUAL "CV180X" OR
       ${CVI_PLATFORM} STREQUAL "CV181X" OR
       ${CVI_PLATFORM} STREQUAL "CV182X" OR
       ${CVI_PLATFORM} STREQUAL "CV183X" OR
       ${CVI_PLATFORM} STREQUAL "CV184X" OR
       ${CVI_PLATFORM} STREQUAL "SOPHON")
  set(DECODER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/vi_decoder/vi_decoder.cpp)
else()
  # add other
endif()



set(PROJ_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/video_decoder_factory.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/image_folder/image_folder_decoder.cpp)


add_library(${PROJECT_NAME} OBJECT ${PROJ_SRCS} ${DECODER_SRCS})

