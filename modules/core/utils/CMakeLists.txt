project(utils)

if(DEFINED SIMPLY_MODEL)
set(UTILS_SRC core_utils.cpp
              rescale_utils.cpp
              ccl.cpp
              object_utils.cpp)
else()
set(UTILS_SRC core_utils.cpp
              rescale_utils.cpp
              demangle.cpp
              object_utils.cpp
              ccl.cpp
              profiler.cpp
              img_warp.cpp)
endif()

if(NOT DEFINED SIMPLY_MODEL)
if(NOT DEFINED NO_OPENCV)
  set(UTILS_SRC ${UTILS_SRC} face_utils.cpp image_utils.cpp neon_utils.cpp)
endif()
endif()
add_library(${PROJECT_NAME} OBJECT ${UTILS_SRC})

if(NOT DEFINED SIMPLY_MODEL)
if ((NOT DEFINED NO_OPENCV) AND ("${ENABLE_CVI_TDL_CV_UTILS}" STREQUAL "ON"))
message("add cv-utils")  
add_subdirectory(cv)
endif()
endif()
