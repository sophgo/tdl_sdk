cmake_minimum_required(VERSION 3.16.3)
project(sample_c_camera)

set(CAMERA_INCLUDES
    ${UTILS_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(${CAMERA_INCLUDES})

file(GLOB CAMERA_SRC_FILES *.c)

if(BUILD_SHARED)
    set(CAMERA_LIBS tdl tdl_utils ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES})
else()
    set(CAMERA_STATIC_LIBS
        tdl-static
        tdl_utils-static
        ${MIDDLEWARE_LIBS_STATIC}
        ${OPENCV_LIBS_IMCODEC_STATIC}
        ${MLIR_LIBS_STATIC}
        ${IVE_LIBS_STATIC}
    )
    
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--start-group")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--whole-archive")
    set(CAMERA_LIBS ${CAMERA_STATIC_LIBS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-whole-archive")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--end-group")
endif()

foreach(fname ${CAMERA_SRC_FILES})
    get_filename_component(name ${fname} NAME_WE)
    add_executable(${name} ${fname})
    target_link_libraries(${name} ${CAMERA_LIBS} dl rt atomic m)
    
    if(NOT BUILD_SHARED)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CROSS_COMPILE}strip $<TARGET_FILE:${name}>
            COMMENT "Stripping binary for ${name}")
    endif()
    
    install(TARGETS ${name} DESTINATION bin/c)
endforeach()
