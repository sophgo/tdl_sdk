project(thirdparty_fetchcontent)
include(FetchContent)
if (ENABLE_PERFETTO)
  if (NOT SYSTRACE_FALLBACK)
    FetchContent_Declare(
      cvi_perfetto
      GIT_REPOSITORY ssh://10.240.0.84:29418/cvi_perfetto
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
  GIT_REPOSITORY ssh://10.240.0.84:29418/eigen
  GIT_TAG origin/master
)
FetchContent_MakeAvailable(libeigen)
include_directories(${libeigen_SOURCE_DIR}/include/eigen3)
message("Content downloaded to ${libeigen_SOURCE_DIR}")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY ssh://10.240.0.84:29418/googletest
  GIT_TAG  e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
)
FetchContent_MakeAvailable(googletest)
include_directories(${googletest_SOURCE_DIR}/googletest/include/gtest)
message("Content downloaded to ${googletest_SOURCE_DIR}")

include(FetchContent)
FetchContent_Declare(
  nlohmannjson
  GIT_REPOSITORY ssh://10.240.0.84:29418/nlohmannjson
  GIT_TAG origin/master
)
FetchContent_MakeAvailable(nlohmannjson)
include_directories(${nlohmannjson_SOURCE_DIR})
message("Content downloaded to ${nlohmannjson_SOURCE_DIR}")

FetchContent_Declare(
  stb
  GIT_REPOSITORY ssh://10.240.0.84:29418/stb
  GIT_TAG origin/master
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Populate(stb)
endif()
include_directories(${stb_SOURCE_DIR})

install(DIRECTORY  ${stb_SOURCE_DIR}/ DESTINATION include/stb
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".git" EXCLUDE
    PATTERN ".github" EXCLUDE
    PATTERN "data" EXCLUDE
    PATTERN "deprecated" EXCLUDE
    PATTERN "docs" EXCLUDE
    PATTERN "tests" EXCLUDE
    PATTERN "tools" EXCLUDE)

message("Content downloaded to ${stb_SOURCE_DIR}")
