include(FetchContent)
FetchContent_Declare(
  cvi_perfetto
  GIT_REPOSITORY http://10.58.65.3:8480/yangwen.huang/cvi_perfetto.git
  GIT_TAG        origin/master
)

FetchContent_MakeAvailable(cvi_perfetto)
message("Content downloaded to ${cvi_perfetto_SOURCE_DIR}")
add_subdirectory(${cvi_perfetto_SOURCE_DIR}/sdk)
include_directories(${cvi_perfetto_SOURCE_DIR}/sdk)
