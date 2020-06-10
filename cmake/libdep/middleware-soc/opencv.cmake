# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if(CMAKE_TOOLCHAIN_FILE)
    if("${LIBDEP_OPENCV_DIR}" STREQUAL "")
        extract_package("${LIBDEP_MIDDLEWARE_DIR}/opencv_3.4.0_${BM_NETWORKS_TARGET_BASENAME}.tar.gz" "${LIBDEP_MIDDLEWARE_DIR}" "opencv" "")

        if (NOT EXISTS "${LIBDEP_MIDDLEWARE_DIR}/opencv")
            execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${TAR_NAME}
            WORKING_DIRECTORY ${LIBDEP_MIDDLEWARE_DIR}/
            )
        endif()
        set(LIBDEP_OPENCV_DIR ${LIBDEP_MIDDLEWARE_DIR}/opencv)
    endif()

    set(OpenCV_INCLUDE_DIRS
        ${LIBDEP_OPENCV_DIR}/include
    )
    include_directories(
        ${OpenCV_INCLUDE_DIRS}
    )

    set(OpenCV_LIBS
        ${Bmvid_LIBS}
        ${FFMPEG_LIBS}
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_core.so
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_imgproc.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_calib3d.so
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_imgcodecs.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_stitching.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_superres.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_dnn.so
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_ml.so
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_videoio.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_features2d.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_objdetect.so
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_video.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_flann.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_photo.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_videostab.so
        ${LIBDEP_OPENCV_DIR}/lib/libopencv_highgui.so
        #${LIBDEP_OPENCV_DIR}/lib/libopencv_shape.so
    )

    install(DIRECTORY ${LIBDEP_OPENCV_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include)
    install(DIRECTORY ${LIBDEP_OPENCV_DIR}/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)
    set(OpenCV_VERSION "3.4.0") # prebuilt version
else()
    find_package(OpenCV 3.4.0 REQUIRED)
    if("${OpenCV_VERSION}" STREQUAL "3.4.0")
        message("OpenCV version: ${OpenCV_VERSION} found")
        # No need to define since it's in /usr/local/lib
        set(LIBDEP_OPENCV_DIR "")
    else()
        message(WARNING "OpenCV version ${OpenCV_VERSION} not match with bmopencv")
    endif()
endif()
