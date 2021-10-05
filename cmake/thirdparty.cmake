project(thirdparty_fetchcontent)
include(FetchContent)
if (ENABLE_PERFETTO)
  if (NOT SYSTRACE_FALLBACK)
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
endif()

FetchContent_Declare(
  libeigen
  URL http://10.58.65.3:8480/yangwen.huang/eigen/uploads/120926fc4d6334e14e6f214190ab77f0/eigen-3.3.7.tar.gz
)
FetchContent_MakeAvailable(libeigen)
include_directories(${libeigen_SOURCE_DIR}/include/eigen3)
message("Content downloaded to ${libeigen_SOURCE_DIR}")


include(FetchContent)
FetchContent_Declare(
  nlohmannjson
  URL      http://10.58.65.3:8480/jiaji.huang/json/uploads/b3b2d33c7be62b502a21e11e58ff1e9b/json.hpp
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(nlohmannjson)
include_directories(${nlohmannjson_SOURCE_DIR})
message("Content downloaded to ${nlohmannjson_SOURCE_DIR}")
