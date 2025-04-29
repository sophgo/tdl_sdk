

add_subdirectory(nn)
add_subdirectory(capture)

add_subdirectory(tracker)
add_subdirectory(matcher)
add_subdirectory(video_decoder)
if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684"))
  add_subdirectory(llm)
endif()