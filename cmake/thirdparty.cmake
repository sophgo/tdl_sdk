project(thirdparty_fetchcontent)
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

get_filename_component(_deps "../_deps" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${_deps})

if (ENABLE_PERFETTO)
  if (NOT SYSTRACE_FALLBACK)
    if (NOT EXISTS "${FETCHCONTENT_BASE_DIR}/cvi_perfetto-src")
      FetchContent_Declare(
        cvi_perfetto
        GIT_REPOSITORY ssh://10.240.0.84:29418/cvi_perfetto
        GIT_TAG        origin/master
      )
    FetchContent_MakeAvailable(cvi_perfetto)
    endif()
    add_subdirectory(${cvi_perfetto_SOURCE_DIR}/sdk)
    include_directories(${cvi_perfetto_SOURCE_DIR}/sdk)
    add_definitions(-DENABLE_TRACE)
    message("Content downloaded to ${cvi_perfetto_SOURCE_DIR}")
  endif()
endif()

if (NOT IS_DIRECTORY  "${FETCHCONTENT_BASE_DIR}/libeigen-src")
  FetchContent_Declare(
    libeigen
    GIT_REPOSITORY ssh://10.240.0.84:29418/eigen
    GIT_TAG origin/master
  )
  FetchContent_MakeAvailable(libeigen)
  message("Content downloaded to ${libeigen_SOURCE_DIR}")
endif()
include_directories(${FETCHCONTENT_BASE_DIR}/libeigen-src/include/eigen3)

if (NOT IS_DIRECTORY "${FETCHCONTENT_BASE_DIR}/googletest-src")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY ssh://10.240.0.84:29418/googletest
    GIT_TAG  e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
  )
  FetchContent_MakeAvailable(googletest)
  message("Content downloaded to ${googletest_SOURCE_DIR}")
else()
  project(googletest)
    add_subdirectory(${FETCHCONTENT_BASE_DIR}/googletest-src/)
endif()
include_directories(${FETCHCONTENT_BASE_DIR}/googletest-src/googletest/include/gtest)

if(NOT IS_DIRECTORY "${FETCHCONTENT_BASE_DIR}/nlohmannjson-src")
  FetchContent_Declare(
    nlohmannjson
    GIT_REPOSITORY ssh://10.240.0.84:29418/nlohmannjson
    GIT_TAG origin/master
  )
  FetchContent_MakeAvailable(nlohmannjson)
  message("Content downloaded to ${nlohmannjson_SOURCE_DIR}")
endif()
include_directories(${FETCHCONTENT_BASE_DIR}/nlohmannjson-src)


if(NOT IS_DIRECTORY "${FETCHCONTENT_BASE_DIR}/stb-src")
  FetchContent_Declare(
    stb
    GIT_REPOSITORY ssh://10.240.0.84:29418/stb
    GIT_TAG origin/master
  )
  FetchContent_MakeAvailable(stb)
  message("Content downloaded to ${stb_SOURCE_DIR}")
endif()
set(stb_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/stb-src)
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