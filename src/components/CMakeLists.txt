

add_subdirectory(nn)

add_subdirectory(tracker)
if(("${CVI_PLATFORM}" STREQUAL "BM1688") OR 
    ("${CVI_PLATFORM}" STREQUAL "BM1684X"))
  add_subdirectory(llm)
endif()



