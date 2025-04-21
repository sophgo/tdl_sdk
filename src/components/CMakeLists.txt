

add_subdirectory(nn)
add_subdirectory(capture)

add_subdirectory(tracker)
add_subdirectory(matcher)
if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684X") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684"))
  add_subdirectory(llm)
endif()