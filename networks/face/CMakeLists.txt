# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(qnn_networks_face)
include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/openpose_lite/include
    ${CMAKE_SOURCE_DIR}/3rdparty/tracer
    ${LIBLINEAR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/face
)

option(BUILDNET_FACEPOSE "Build Facepose" ON)
option(BUILDNET_REID "Build ReID" ON)
option(BUILDNET_MOBILENETSSD "Build MobileNet + SSD" ON)
option(BUILDNET_SSH "Build ssh" ON)
option(BUILDNET_FACE_ATTRIBUTE "Build FaceAttribute" ON)
option(BUILDNET_MTCNN "Build mtcnn" ON)
option(BUILDNET_ANTIFACESPOOFING "Build Anti FaceSpoofing" ON)
option(BUILDNET_OPENPOSE "Build Openpose" ON)
option(BUILDNET_YOLO "Build Yolo" ON)

message("======================================================")
message("[Facenet summary]")
message("    BUILD_FDNET_ALL:           ${BUILD_FDNET_ALL}")
message("")
message("[BUILDNET]")
message("    BUILDNET_FACEPOSE:         ${BUILDNET_FACEPOSE}")
message("    BUILDNET_MOBILENETSSD:     ${BUILDNET_MOBILENETSSD}")
message("    BUILDNET_REID:             ${BUILDNET_REID}")
message("    BUILDNET_SSH:          	${BUILDNET_SSH}")
message("    BUILDNET_FACE_ATTRIBUTE:   ${BUILDNET_FACE_ATTRIBUTE}")
message("    BUILDNET_MTCNN:            ${BUILDNET_MTCNN}")
message("    BUILDNET_ANTIFACESPOOFING: ${BUILDNET_ANTIFACESPOOFING}")
message("    BUILDNET_OPENPOSE:         ${BUILDNET_OPENPOSE}")
message("    BUILDNET_YOLO:             ${BUILDNET_YOLO}")
message("======================================================")

set(FACE_COMMON_SRC
    ${CMAKE_SOURCE_DIR}/3rdparty/tracer/tracer.cpp
    src/face.cpp
    src/face_align.cpp
)

if(BUILDNET_FACE_ATTRIBUTE)
    add_definitions(-DFACE_ATTRIBUTE_112x112)
endif()

if(BUILD_FDNET_ALL)
    if(BUILDNET_FACEPOSE)
        set(FDNET_SRC ${FDNET_SRC} src/facepose.cpp)
    endif()
    if(BUILDNET_MOBILENETSSD)
        set(FDNET_SRC ${FDNET_SRC} src/mobilenet_ssd.cpp)
    endif()
    if(BUILDNET_REID)
        set(FDNET_SRC ${FDNET_SRC} src/re_id.cpp)
    endif()
    if(BUILDNET_SSH)
        set(SSH_SRC
            ${FDNET_SRC}
            src/mtcnn_helperfunc.cpp
            src/generate_anchors.cpp
            src/proposal_layer_nocaffe.cpp
            src/net_ssh.cpp
        )
        set(FDNET_SRC ${FDNET_SRC} ${SSH_SRC})
    endif()
    if(BUILDNET_FACE_ATTRIBUTE)
        set(FDNET_SRC
            ${FDNET_SRC}
            src/face_attribute.cpp
            )
    endif()
    if(BUILDNET_MTCNN)
        set(FDNET_SRC ${FDNET_SRC} src/mtcnn_helperfunc.cpp src/mtcnn.cpp)
    endif()
    if(BUILDNET_ANTIFACESPOOFING)
        set(FDNET_SRC
            ${FDNET_SRC}
            src/anti_facespoofing.cpp
            src/anti_facespoofingcv.cpp
            src/anti_facespoofingdepth.cpp
            src/anti_facespoofingpatch.cpp)
    endif()
    if(BUILDNET_OPENPOSE)
        set(FDNET_SRC
            ${FDNET_SRC}
            src/openpose.cpp)
        set(OPENPOSE_LINK_LIBS openpose_lite)
    endif()
    if(BUILDNET_YOLO)
        set(FDNET_SRC
            ${FDNET_SRC}
            src/yolo.cpp)
    endif()
    set_library_ext(NAME qnn_facenet TYPE SHARED SRC ${FDNET_SRC} DEP qnn_networks_utils qnn_networks_core ${OPENPOSE_LINK_LIBS} ${OpenCV_LIBS} DEF ${FDNET_DEF_VAR} OPTION true INSTALL true)

else()
    set(FDNET_DEP qnn_networks_utils qnn_networks_core)
    if(BUILDNET_FACEPOSE)
        set(FDNET_DEP ${FDNET_DEP} qnn_facepose)
    endif()
    if(BUILDNET_MOBILENETSSD)
        set(FDNET_DEP ${FDNET_DEP} qnn_mobilenet_ssd)
    endif()
    if(BUILDNET_REID)
        set(FDNET_DEP ${FDNET_DEP} qnn_reid)
    endif()
    if(BUILDNET_SSH)
        set(FDNET_DEP ${FDNET_DEP} qnn_ssh)
    endif()
    if(BUILDNET_FACE_ATTRIBUTE)
        set(FDNET_DEP ${FDNET_DEP} face_attribute)
    endif()
    if(BUILDNET_MTCNN)
        set(FDNET_DEP ${FDNET_DEP} qnn_mtcnn)
    endif()
    if(BUILDNET_ANTIFACESPOOFING)
        set(FDNET_DEP ${FDNET_SRC} qnn_antifacespoofing)
    endif()
    if(BUILDNET_OPENPOSE)
        set(FDNET_DEP ${FDNET_SRC} qnn_openpose)
    endif()
    if(BUILDNET_YOLO)
        set(FDNET_DEP ${FDNET_SRC} qnn_yolo)
    endif()

    set(NET_SSH_SRC
        ${FACE_COMMON_SRC}
        src/mtcnn_helperfunc.cpp
        src/generate_anchors.cpp
        src/proposal_layer_nocaffe.cpp
        src/fd_ssh.cpp
        src/ssh.cpp
        src/onet.cpp
    )

    set(FACE_ATTRIBUTE_SRC
        ${FACE_COMMON_SRC}
        src/face_attribute.cpp
    )

    set(FACEPOSE_SRC
        ${FACE_COMMON_SRC}
        src/facepose.cpp
    )

    set(MOBILENET_SRC
        ${FACE_COMMON_SRC}
        src/mobilenet_ssd.cpp
    )

    set(REID_SRC
        ${FACE_COMMON_SRC}
        src/re_id.cpp
    )

    set(MTCNN_SRC
        ${FACE_COMMON_SRC}
        src/mtcnn_helperfunc.cpp
        src/pnet.cpp
        src/rnet.cpp
        src/onet.cpp
        src/mtcnn.cpp
    )

    set(ANTIFACESPOOFING_SRC
        ${MTCNN_SRC}
        ${LIBLINEAR_SRC}
        src/anti_facespoofing.cpp
        src/anti_facespoofing/afs_cv.cpp
        src/anti_facespoofing/afs_classify.cpp
        src/anti_facespoofing/afs_classify_hsv_ycbcr.cpp
        src/anti_facespoofing/afs_canny.cpp
        src/anti_facespoofing/afs_depth.cpp
        src/anti_facespoofing/afs_optflow.cpp
        src/anti_facespoofing/afs_patch.cpp
        src/anti_facespoofing/afs_utils.cpp
    )

    set(OPENPOSE_SRC
        src/openpose.cpp
    )

    set(YOLO_SRC
        src/yolo.cpp
    )

    set_library_ext(NAME qnn_face_attribute TYPE SHARED SRC ${FACE_ATTRIBUTE_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_FACE_ATTRIBUTE} INSTALL true)
    set_library_ext(NAME qnn_ssh TYPE SHARED SRC ${NET_SSH_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_SSH} INSTALL true)
    set_library_ext(NAME qnn_reid TYPE SHARED SRC ${REID_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_REID} INSTALL true)
    set_library_ext(NAME qnn_mobilenet_ssd TYPE SHARED SRC ${MOBILENET_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_MOBILENETSSD} INSTALL true)
    set_library_ext(NAME qnn_facepose TYPE SHARED SRC ${FACEPOSE_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_FACEPOSE} INSTALL true)
    set_library_ext(NAME qnn_mtcnn TYPE SHARED SRC ${MTCNN_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_MTCNN} INSTALL true)
    set_library_ext(NAME qnn_antifacespoofing TYPE SHARED SRC ${ANTIFACESPOOFING_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_MTCNN} INSTALL true)
    set_library_ext(NAME qnn_openpose TYPE SHARED SRC ${OPENPOSE_SRC} DEP qnn_networks_utils qnn_networks_core openpose_lite ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_OPENPOSE} INSTALL true)
    set_library_ext(NAME qnn_yolo TYPE SHARED SRC ${YOLO_SRC} DEP qnn_networks_utils qnn_networks_core ${OpenCV_LIBS} nlohmann_json::nlohmann_json OPTION ${BUILDNET_YOLO} INSTALL true)

    if(BUILDNET_ANTIFACESPOOFING)
        install(FILES ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear/LibLinear.hpp DESTINATION ${BM_INSTALL_PREFIX}/include/)
        install(FILES ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear/linear.h DESTINATION ${BM_INSTALL_PREFIX}/include/)
    endif()
endif()
