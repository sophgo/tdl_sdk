project(thirdparty_fetchcontent)

if (ENABLE_PERFETTO)
  if (NOT SYSTRACE_FALLBACK)
    if (NOT EXISTS "${BUILD_DOWNLOAD_DIR}/cvi_perfetto-src")
    FetchContent_Declare(
      cvi_perfetto
      GIT_REPOSITORY ssh://${REPO_USER}${DL_SERVER_IP}:29418/cvitek/cvi_perfetto
      GIT_TAG        origin/master
    )
    FetchContent_MakeAvailable(cvi_perfetto)
    else()
    set(cvi_perfetto_SOURCE_DIR ${BUILD_DOWNLOAD_DIR}/cvi_perfetto-src)
    endif()
    
    set(CVI_PERFETTO_BIN_PERFETTO ${cvi_perfetto_SOURCE_DIR}/bin/${ARCH}/perfetto)
    set(CVI_PERFETTO_BIN_TRACED ${cvi_perfetto_SOURCE_DIR}/bin/${ARCH}/traced)
    set(CVI_PERFETTO_BIN_TRACED_PROBES ${cvi_perfetto_SOURCE_DIR}/bin/${ARCH}/traced_probes)
    set(CVI_PERFETTO_CONFIG ${cvi_perfetto_SOURCE_DIR}/config)
    set(CVI_PERFETTO_SCRIPT ${cvi_perfetto_SOURCE_DIR}/start_daemon.sh)
    
    install(FILES ${CVI_PERFETTO_BIN_PERFETTO} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin/perfetto)
    install(FILES ${CVI_PERFETTO_BIN_TRACED} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin/perfetto)
    install(FILES ${CVI_PERFETTO_BIN_TRACED_PROBES} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin/perfetto)
    install(DIRECTORY ${CVI_PERFETTO_CONFIG} DESTINATION bin/perfetto)
    install(FILES ${CVI_PERFETTO_SCRIPT} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE DESTINATION bin/perfetto)
    
    add_subdirectory(${cvi_perfetto_SOURCE_DIR}/sdk)
    include_directories(${cvi_perfetto_SOURCE_DIR}/sdk)
    add_definitions(-DENABLE_TRACE)
    message("Content downloaded to ${cvi_perfetto_SOURCE_DIR}")
  endif()
endif()

if (NOT IS_DIRECTORY  "${BUILD_DOWNLOAD_DIR}/libeigen-src")
FetchContent_Declare(
  libeigen
  GIT_REPOSITORY ssh://${REPO_USER}${DL_SERVER_IP}:29418/cvitek/eigen
  GIT_TAG origin/master
)
FetchContent_MakeAvailable(libeigen)
message("Content downloaded to ${libeigen_SOURCE_DIR}")
endif()
include_directories(${BUILD_DOWNLOAD_DIR}/libeigen-src/include/eigen3)

if (NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/googletest-src")
FetchContent_Declare(
  googletest
  GIT_REPOSITORY ssh://${REPO_USER}${DL_SERVER_IP}:29418/cvitek/googletest
  GIT_TAG  e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
)
FetchContent_MakeAvailable(googletest)
message("Content downloaded to ${googletest_SOURCE_DIR}")
else()
  project(googletest)
    add_subdirectory(${BUILD_DOWNLOAD_DIR}/googletest-src/)
endif()
include_directories(${BUILD_DOWNLOAD_DIR}/googletest-src/googletest/include/gtest)

if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/nlohmannjson-src")
FetchContent_Declare(
  nlohmannjson
  GIT_REPOSITORY ssh://${REPO_USER}${DL_SERVER_IP}:29418/cvitek/nlohmannjson
  GIT_TAG origin/master
)
FetchContent_MakeAvailable(nlohmannjson)
message("Content downloaded to ${nlohmannjson_SOURCE_DIR}")
endif()
include_directories(${BUILD_DOWNLOAD_DIR}/nlohmannjson-src)

if (BUILD_WEB_VIEW)
  if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/sophapp-src")
    FetchContent_Declare(
      sophapp
      GIT_REPOSITORY ssh://${REPO_USER}${DL_SERVER_IP}:29418/cvitek/sophapp
      GIT_TAG origin/ipcamera
    )
    FetchContent_MakeAvailable(sophapp)
    message("Content downloaded to ${sophapp_SOURCE_DIR}")
  endif()
  set(sophapp_SOURCE_DIR ${BUILD_DOWNLOAD_DIR}/sophapp-src)
endif()

if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/stb-src")
FetchContent_Declare(
  stb
  GIT_REPOSITORY ssh://${REPO_USER}${DL_SERVER_IP}:29418/cvitek/stb
  GIT_TAG origin/master
)
  FetchContent_MakeAvailable(stb)
  message("Content downloaded to ${stb_SOURCE_DIR}")
endif()
set(stb_SOURCE_DIR ${BUILD_DOWNLOAD_DIR}/stb-src)
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
