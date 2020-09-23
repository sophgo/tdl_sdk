# Perfetto tracer
add_subdirectory(cviai_trace)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cviai_trace)

# Modules
# Log header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/log)
add_subdirectory(core)
add_subdirectory(service)
add_subdirectory(frservice)
add_subdirectory(objservice)
add_subdirectory(evaluation)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/ DESTINATION include/cviai
        PATTERN "cviai_perfetto.h" EXCLUDE
        PATTERN "core/cviai_types_mem_internal.h" EXCLUDE)
