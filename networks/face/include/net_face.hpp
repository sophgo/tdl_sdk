// Copyright 2018 Bitmain Inc.
#pragma once
#include "face/anti_facespoofing.hpp"
#include "face/anti_facespoofing/afs_classify.hpp"
#include "face/anti_facespoofing/afs_classify_hsv_ycbcr.hpp"
#include "face/anti_facespoofing/afs_cv.hpp"
#include "face/anti_facespoofing/afs_depth.hpp"
#include "face/anti_facespoofing/afs_patch.hpp"
#include "face/anti_facespoofing/afs_utils.hpp"
#include "face/face.hpp"
#include "face/face_attribute.hpp"
#include "face/face_attribute_stabilizer.hpp"
#include "face/face_types.hpp"
#include "face/facepose.hpp"
#include "face/fd_ssh.hpp"
#include "face/generate_anchors.hpp"
#include "face/mobilenet_ssd.hpp"
#include "face/mtcnn.hpp"
#include "face/onet.hpp"
#include "face/openpose.hpp"
#include "face/pnet.hpp"
#include "face/re_id.hpp"
#include "face/rnet.hpp"
#include "face/ssh.hpp"
#include "face/yolo.hpp"

/** @defgroup NETWORKS_FACE Networks Face
 *  @brief Networks face
 */

/**
 * @defgroup NETWORKS_FD Face Detection
 * \ingroup NETWORKS_FACE
 * @brief Face detection networks
 */

/**
 * @defgroup NETWORKS_ATTR Face Attribute
 * \ingroup NETWORKS_FACE
 * @brief Face attribute networks
 */

/**
 * @defgroup NETWORKS_PR People Recognition
 * @brief People recognition networks
 */

/**
 * @defgroup NETWORKS_OD Object Detection
 * @brief Object detection networks
 */
