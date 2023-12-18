if(USE_TPU_IVE)
    if("${TPU_IVE_SDK_ROOT}" STREQUAL "")
        message(FATAL_ERROR "Missing ${TPU_IVE_SDK_ROOT}.")
    elseif(EXISTS "${TPU_IVE_SDK_ROOT}")
        message("-- Found TPU_IVE_SDK_ROOT (directory: ${TPU_IVE_SDK_ROOT})")
    else()
        message(FATAL_ERROR "${TPU_IVE_SDK_ROOT} is not a valid folder.")
    endif()

    set(IVE_INCLUDES ${TPU_IVE_SDK_ROOT}/include/)
    set(IVE_LIBS     ${TPU_IVE_SDK_ROOT}/lib/libcvi_ive_tpu.so)

    add_definitions(-DUSE_TPU_IVE)

    if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    install(DIRECTORY ${TPU_IVE_SDK_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/sample/3rd/include/ive)
    install(FILES ${IVE_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/sample/3rd/lib)
    endif()
else()
    # Use standalone IVE hardware
    set(IVE_INCLUDES ${MIDDLEWARE_SDK_ROOT}/include/)
    set(IVE_LIBS     ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_ive.so)
endif()