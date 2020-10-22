project(thirdparty_fetchcontent)
include(FetchContent)
if (ENABLE_PERFETTO)
  FetchContent_Declare(
    cvi_perfetto
    GIT_REPOSITORY http://10.58.65.3:8480/yangwen.huang/cvi_perfetto.git
    GIT_TAG        origin/master
  )
  FetchContent_MakeAvailable(cvi_perfetto)
  add_subdirectory(${cvi_perfetto_SOURCE_DIR}/sdk)
  include_directories(${cvi_perfetto_SOURCE_DIR}/sdk)
  add_definitions(-DENABLE_TRACE)
  message("Content downloaded to ${cvi_perfetto_SOURCE_DIR}")
endif()

FetchContent_Declare(
  libeigen
  URL http://10.58.65.3:8480/yangwen.huang/eigen/uploads/120926fc4d6334e14e6f214190ab77f0/eigen-3.3.7.tar.gz
)
FetchContent_MakeAvailable(libeigen)
include_directories(${libeigen_SOURCE_DIR}/include/eigen3)
message("Content downloaded to ${libeigen_SOURCE_DIR}")

FetchContent_Declare(
  neon2sse
  GIT_REPOSITORY  https://github.com/intel/ARM_NEON_2_x86_SSE.git
  GIT_TAG origin/master
)
FetchContent_GetProperties(neon2sse)
if(NOT neon2sse_POPULATED)
  FetchContent_Populate(neon2sse)
endif()
include_directories(${neon2sse_SOURCE_DIR})
message("Content downloaded to ${neon2sse_SOURCE_DIR}")

include(FetchContent)
FetchContent_Declare(
  nlohmannjson
  URL      https://github.com/nlohmann/json/releases/download/v3.9.1/json.hpp
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(nlohmannjson)
include_directories(${nlohmannjson_SOURCE_DIR})
message("Content downloaded to ${nlohmannjson_SOURCE_DIR}")
