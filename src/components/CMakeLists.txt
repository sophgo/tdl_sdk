

add_subdirectory(nn)
add_subdirectory(snapshot)
add_subdirectory(tracker)
add_subdirectory(matcher)
add_subdirectory(encoder)
add_subdirectory(video_decoder)
add_subdirectory(cv/intrusion_detect)
add_subdirectory(cv/occlusion_detect)
if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684"))
  add_subdirectory(llm)
endif()
if("${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X" OR
   "${CVI_PLATFORM}" STREQUAL "CV184X")
   add_subdirectory(ive)
endif()
if ("${CVI_PLATFORM}" STREQUAL "CV181X" OR
    "${CVI_PLATFORM}" STREQUAL "CV184X" OR
    "${CVI_PLATFORM}" STREQUAL "CMODEL_CV184X")
  add_subdirectory(cv/motion_detect)
endif()