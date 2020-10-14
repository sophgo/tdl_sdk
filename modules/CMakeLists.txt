# Modules
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cviai_trace
                    ${CMAKE_CURRENT_SOURCE_DIR}/log
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_subdirectory(cviai_trace) # Perfetto tracer comes first
add_subdirectory(core)
add_subdirectory(service)
add_subdirectory(frservice)
add_subdirectory(objservice)
add_subdirectory(evaluation)

# Temporarily don't install custom module until complete.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/ DESTINATION include/cviai
        PATTERN "cviai_perfetto.h" EXCLUDE
        PATTERN "core/cviai_custom.h" EXCLUDE
        PATTERN "core/cviai_types_mem_internal.h" EXCLUDE)
