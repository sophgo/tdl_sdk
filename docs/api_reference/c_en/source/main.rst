.. vim: syntax=rst

Model List
================

.. list-table::
   :widths: 1 1 

   * - Model Name
     - Description

   * - TDL_MODEL_MBV2_DET_PERSON
     - Human Detection Model

   * - TDL_MODEL_YOLOV5_DET_COCO80
     - YOLOv5 COCO80 Detection Model

   * - TDL_MODEL_YOLOV8_DET_COCO80
     - YOLOv8 COCO80 Detection Model

   * - TDL_MODEL_YOLOV10_DET_COCO80
     - YOLOv10 COCO80 Detection Model

   * - TDL_MODEL_YOLOV8N_DET_HAND
     - Hand Detection Model

   * - TDL_MODEL_YOLOV8N_DET_PET_PERSON
     - Pet and Person Detection Model (0:cat, 1:dog, 2:person)

   * - TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE
     - Person and Vehicle Detection Model (0:car, 1:bus, 2:truck, 3:motorcyclist, 4:person, 5:bicycle, 6:motorcycle)

   * - TDL_MODEL_YOLOV8N_DET_HAND_FACE_PERSON
     - Hand, Face and Person Detection Model (0:hand, 1:face, 2:person)

   * - TDL_MODEL_YOLOV8N_DET_HEAD_PERSON
     - Head Detection Model (0:person, 1:head)

   * - TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT
     - Head and Hardhat Detection Model (0:head, 1:hardhat)

   * - TDL_MODEL_YOLOV8N_DET_FIRE_SMOKE
     - Fire and Smoke Detection Model (0:fire, 1:smoke)

   * - TDL_MODEL_YOLOV8N_DET_FIRE
     - Fire Detection Model (0:fire)

   * - TDL_MODEL_YOLOV8N_DET_HEAD_SHOULDER
     - Head and Shoulder Detection Model (0:head-shoulder)

   * - TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE
     - License Plate Detection Model (0:license plate)

   * - TDL_MODEL_YOLOV8N_DET_TRAFFIC_LIGHT
     - Traffic Light Detection Model (0:red, 1:yellow, 2:green, 3:off, 4:waiting)

   * - TDL_MODEL_SCRFD_DET_FACE
     - Face Detection Model (0:face + keypoints)

   * - TDL_MODEL_RETINA_DET_FACE
     - Face Detection Model

   * - TDL_MODEL_RETINA_DET_FACE_IR
     - Infrared Face Detection Model

   * - TDL_MODEL_KEYPOINT_FACE_V2
     - Face Detection Model with 5 Keypoints and Blur Score

   * - TDL_MODEL_CLS_ATTRIBUTE_FACE
     - Face Attribute Classification Model (age, gender, glasses, mask)

   * - TDL_MODEL_FEATURE_BMFACE_R34
     - ResNet34 512-dimensional Feature Extraction Model

   * - TDL_MODEL_FEATURE_BMFACE_R50
     - ResNet50 512-dimensional Feature Extraction Model

   * - TDL_MODEL_CLS_MASK
     - Mask Detection Model (0:wearing mask, 1:not wearing mask)

   * - TDL_MODEL_CLS_RGBLIVENESS
     - Liveness Detection Model (0:real, 1:fake)

   * - TDL_MODEL_CLS_ISP_SCENE
     - ISP Scene Classification Model

   * - TDL_MODEL_CLS_HAND_GESTURE
     - Hand Gesture Classification Model (0:fist, 1:five fingers, 2:none, 3:two)

   * - TDL_MODEL_CLS_KEYPOINT_HAND_GESTURE
     - Hand Gesture Keypoint Classification Model (0:fist, 1:five fingers, 2:four fingers, 3:none, 4:ok, 5:one, 6:three, 7:three2, 8:two)

   * - TDL_MODEL_CLS_SOUND_BABAY_CRY
     - Baby Cry Sound Classification Model (0:background, 1:crying)

   * - TDL_MODEL_CLS_SOUND_COMMAND
     - Command Sound Classification Model

   * - TDL_MODEL_KEYPOINT_LICENSE_PLATE
     - License Plate Keypoint Detection Model

   * - TDL_MODEL_KEYPOINT_HAND
     - Hand Keypoint Detection Model

   * - TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17
     - Human 17 Keypoint Detection Model

   * - TDL_MODEL_KEYPOINT_SIMCC_PERSON17
     - SIMCC 17 Keypoint Detection Model

   * - TDL_MODEL_LSTR_DET_LANE
     - Lane Detection Model

   * - TDL_MODEL_RECOGNITION_LICENSE_PLATE
     - License Plate Recognition Model

   * - TDL_MODEL_YOLOV8_SEG_COCO80
     - YOLOv8 COCO80 Segmentation Model

   * - TDL_MODEL_SEG_PERSON_FACE_VEHICLE \
       _VEHICLE
     - Person, Face and Vehicle Segmentation Model (0:background, 1:person, 2:face, 3:vehicle, 4:license plate)

   * - TDL_MODEL_SEG_MOTION
     - Motion Segmentation Model (0:static, 2:transition, 3:motion)

   * - TDL_MODEL_IMG_FEATURE_CLIP
     - Image Feature Extraction Model

   * - TDL_MODELTEXT_FEATURE_CLIP
     - Text Feature Extraction Model

Structure Reference
======================

TDLDataTypeE
~~~~~~~~~~~~~~~

【Description】

Data Type Enumeration Class

【Definition】

.. code-block:: c

  typedef enum {
    TDL_TYPE_INT8 = 0, /**< Equals to int8_t. */
    TDL_TYPE_UINT8,    /**< Equals to uint8_t. */
    TDL_TYPE_INT16,    /**< Equals to int16_t. */
    TDL_TYPE_UINT16,   /**< Equals to uint16_t. */
    TDL_TYPE_INT32,    /**< Equals to int32_t. */
    TDL_TYPE_UINT32,   /**< Equals to uint32_t. */
    TDL_TYPE_BF16,     /**< Equals to bf17. */
    TDL_TYPE_FP16,     /**< Equals to fp16. */
    TDL_TYPE_FP32,     /**< Equals to fp32. */
    TDL_TYPE_UNKOWN    /**< Equals to unkown. */
  } TDLDataTypeE;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type Enumeration
     - Description

   * - TDL_TYPE_INT8
     - Signed 8-bit integer

   * - TDL_TYPE_UINT8
     - Unsigned 8-bit integer

   * - TDL_TYPE_INT16
     - Signed 16-bit integer

   * - TDL_TYPE_UINT16
     - Unsigned 16-bit integer

   * - TDL_TYPE_INT32
     - Signed 32-bit integer

   * - TDL_TYPE_UINT32
     - Unsigned 32-bit integer

   * - TDL_TYPE_BF16
     - 16-bit floating point (1 sign bit, 8 exponent bits, 7 mantissa bits)

   * - TDL_TYPE_FP16
     - 16-bit floating point (1 sign bit, 5 exponent bits, 10 mantissa bits)

   * - FTDL_TYPE_FP32
     - 32-bit floating point

TDLBox
~~~~~~~~~~~~~~~

【Description】

Box coordinate data

【Definition】

.. code-block:: c

  typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
  } TDLBox;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - x1
     - x coordinate of top-left corner

   * - y1
     - y coordinate of top-left corner

   * - x2
     - x coordinate of bottom-right corner

   * - y2
     - y coordinate of bottom-right corner


TDLFeature
~~~~~~~~~~~~~~~

【Description】

Feature value data

【Definition】

.. code-block:: c

  typedef struct {
    int8_t *ptr;
    uint32_t size;
    TDLDataTypeE type;
  } TDLFeature;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - ptr
     - Feature value data

   * - size
     - Data size

   * - type
     - Data type


TDLPoints
~~~~~~~~~~~~~~~

【Description】

Coordinate queue data

【Definition】

.. code-block:: c

  typedef struct {
    float *x;
    float *y;
    uint32_t size;
    float score;
  } TDLPoints;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - x
     - x coordinate queue data

   * - y
     - y coordinate queue data

   * - size
     - Size of coordinate queue
  
   * - score
     - Score

TDLLandmarkInfo
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Feature point information

【Definition】

.. code-block:: c

  typedef struct {
    float x;
    float y;
    float score;
  } TDLLandmarkInfo;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - x
     - x coordinate of feature point

   * - y
     - y coordinate of feature point
  
   * - score
     - Score

TDLObjectInfo
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Object detection information

【Definition】

.. code-block:: c

  typedef struct {
    TDLBox box;
    float score;
    int class_id;
    uint32_t landmark_size;
    TDLLandmarkInfo *landmark_properity;
    TDLObjectTypeE obj_type;
  } TDLObjectInfo;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - score
     - Object detection score

   * - class_id
     - Object detection class id
  
   * - landmark_size
     - Size of object detection feature points

   * - TDLLandmarkInfo
     - Object detection feature point information

   * - obj_type
     - Object detection type

TDLObject
~~~~~~~~~~~~~~~

【Description】

Object detection data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;

    TDLObjectInfo *info;
  } TDLObject;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of detected objects

   * - width
     - Width of detection image
  
   * - height
     - Height of detection image

   * - info
     - Object detection information

TDLFaceInfo
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Face information

【Definition】

.. code-block:: c

  typedef struct {
    char name[128];
    float score;
    uint64_t track_id;
    TDLBox box;
    TDLPoints landmarks;
    TDLFeature feature;

    float gender_score;
    float glass_score;
    float age;
    float liveness_score;
    float hardhat_score;
    float mask_score;

    float recog_score;
    float face_quality;
    float pose_score;
    float blurness;
  } TDLFaceInfo;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - name
     - Face name

   * - score
     - Face score
  
   * - track_id
     - Face tracking id

   * - box
     - Face box information

   * - landmarks
     - Face feature points

   * - feature
     - Face feature value
  
   * - gender_score
     - Face gender score

   * - glass_score
     - Whether wearing glasses

   * - age
     - Face age

   * - liveness_score
     - Face liveness score
  
   * - hardhat_score
     - Face hardhat score

   * - recog_score
     - Face recognition score

   * - face_quality
     - Face quality score

   * - pose_score
     - Face pose score
  
   * - blurness
     - Face blur degree

TDLFace
~~~~~~~~~~~~~~~

【Description】

Face data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    TDLFaceInfo *info;
  } TDLFace;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of faces

   * - width
     - Width of face image
  
   * - height
     - Height of face image

   * - info
     - Face information

TDLClassInfo
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Classification information

【Definition】

.. code-block:: c

  typedef struct {
    int32_t class_id;
    float score;
  } TDLClassInfo;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - class_id
     - Classification class

   * - score
     - Classification score
  
TDLClass
~~~~~~~~~~~~~~~

【Description】

Classification data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    TDLClassInfo *info;
  } TDLClass;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of classifications

   * - info
     - Classification information

TDLKeypointInfo
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Keypoint information

【Definition】

.. code-block:: c

  typedef struct {
    float x;
    float y;
    float score;
  } TDLKeypointInfo;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - x
     - x coordinate of keypoint

   * - y
     - y coordinate of keypoint

   * - score
     - Keypoint score

TDLKeypoint
~~~~~~~~~~~~~~~

【Description】

Keypoint data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    TDLKeypointInfo *info;
  } TDLKeypoint;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of keypoints

   * - width
     - Image width
  
   * - height
     - Image height

   * - info
     - Keypoint information

TDLSegmentation
~~~~~~~~~~~~~~~

【Description】

Semantic segmentation data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t output_width;
    uint32_t output_height;
    uint8_t *class_id;
    uint8_t *class_conf;
  } TDLSegmentation;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - width
     - Image width
  
   * - height
     - Image height

   * - output_width
     - Output image width
  
   * - output_height
     - Output image height

   * - class_id
     - Classification class

   * - class_conf
     - Classification coordinate information

TDLInstanceSegInfo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

【Description】

Instance segmentation information

【Definition】

.. code-block:: c

  typedef struct {
    uint8_t *mask;
    float *mask_point;
    uint32_t mask_point_size;
    TDLObjectInfo *obj_info;
  } TDLInstanceSegInfo;

TDLInstanceSeg
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Instance segmentation data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    uint32_t mask_width;
    uint32_t mask_height;
    TDLInstanceSegInfo *info;
  } TDLInstanceSeg;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of instance segmentations

   * - width
     - Image width
  
   * - height
     - Image height

   * - mask_width
     - Mask width
  
   * - mask_height
     - Mask height

   * - info
     - Instance segmentation information

TDLLanePoint
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Lane detection coordinate points

【Definition】

.. code-block:: c

  typedef struct {
    float x[2];
    float y[2];
    float score;
  } TDLLanePoint;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - x
     - x coordinate queue

   * - y
     - y coordinate queue
  
   * - score
     - Lane detection score

TDLLane
~~~~~~~~~~~~~~~

【Description】

Lane detection data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    TDLLanePoint *lane;
    int lane_state;
  } TDLLane;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of lane detections

   * - width
     - Image width
  
   * - height
     - Image height

   * - lane
     - Lane detection coordinate points
  
   * - lane_state
     - Lane state

TDLDepthLogits
~~~~~~~~~~~~~~~~~~~~~~

【Description】

Depth estimation data

【Definition】

.. code-block:: c

  typedef struct {
    int w;
    int h;
    int8_t *int_logits;
  } TDLDepthLogits;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - w
     - Image width
  
   * - h
     - Image height

   * - int_logits
     - Depth estimation information
  
TDLTracker
~~~~~~~~~~~~~~~

【Description】

Tracking data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint64_t id;
    TDLBox bbox;
    int out_num;
  } TDLTracker;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of tracked targets
  
   * - id
     - Tracking target ID

   * - bbox
     - Tracking target bounding box

   * - out_num
     - Number of times target is out of frame

TDLOcr
~~~~~~~~~~~~~~~

【Description】

Text recognition data

【Definition】

.. code-block:: c

  typedef struct {
    uint32_t size;
    char* text_info;
  } TDLOcr;

【Members】

.. list-table::
   :widths: 1 1

   * - Data Type
     - Description

   * - size
     - Number of text recognitions
  
   * - text_info
     - Text recognition information

API Reference
================

Handles
~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c
  
  typedef void *TDLHandle;
  typedef void *TDLImage;

【Description】

TDL SDK handles, TDLHandle is the core operation handle, TDLImage is the image data abstraction handle.

TDL_CreateHandle
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  TDLHandle TDL_CreateHandle(const int32_t tpu_device_id);

【Description】

Create a TDLHandle object.

【Parameters】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - const int32_t
     - tpu_device_id
     - Specified TPU device ID

TDL_DestroyHandle
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_DestroyHandle(TDLHandle handle);

【Description】

Destroy a TDLHandle object.

【Parameters】

.. list-table::
   :widths: 1 2 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object to be destroyed

TDL_WrapVPSSFrame
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  TDLImage TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory);

【Description】

Wrap a VPSS frame as a TDLImageHandle object.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - void\*
     - vpss_frame
     - VPSS frame to be wrapped

   * - Input
     - bool
     - own_memory
     - Whether to own the memory

TDL_ReadImage
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  TDLImage TDL_ReadImage(const char *path);

【Description】

Read an image as a TDLImageHandle object.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - const char\*
     - path
     - Image path

TDL_ReadBin
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  TDLImage TDL_ReadBin(const char *path, int count, TDLDataTypeE data_type);

【Description】

Read file content as a TDLImageHandle object.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - const char\*
     - path
     - Binary file path

   * - Input
     - int
     - count
     - Data count in file

   * - Input
     - TDLDataTypeE
     - data_type
     - Input data type

TDL_DestroyImage
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_DestroyImage(TDLImage image_handle);

【Description】

Destroy a TDLImageHandle object.

【Parameters】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object to be destroyed

TDL_OpenModel
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_OpenModel(TDLHandle handle,
                        const TDLModel model_id,
                        const char *model_path);

【Description】

Load a specified type of model into the TDLHandle object.

【Parameters】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - const char\*
     - model_path
     - Model path

TDL_CloseModel
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_CloseModel(TDLHandle handle,
                         const TDLModel model_id);

【Description】

Unload the specified type of model and release related resources.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

TDL_Detection
~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_Detection(TDLHandle handle,
                        const TDLModel model_id,
                        TDLImage image_handle,
                        TDLObject *object_meta);

【Description】

Execute inference detection with the specified model and return detection result metadata.

【Parameters】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLObject\*
     - object_meta
     - Output detection result metadata

TDL_FaceDetection
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_FaceDetection(TDLHandle handle,
                            const TDLModel model_id,
                            TDLImage image_handle,
                            TDLFace *face_meta);

【Description】

Execute face detection and return face detection result metadata.

【Parameters】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLFace\*
     - face_meta
     - Output face detection result metadata

TDL_FaceAttribute
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_FaceAttribute(TDLHandle handle,
                            const TDLModel model_id,
                            TDLImage image_handle,
                            TDLFace *face_meta);

【Description】

Execute face attribute analysis, requires face detection results for feature analysis.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Input/Output
     - TDLFace\*
     - face_meta
     - Input face detection results, output additional attribute information

TDL_FaceLandmark
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_FaceLandmark(TDLHandle handle,
                           const TDLModel model_id,
                           TDLImage image_handle,
                           TDLFace *face_meta);

【Description】

Execute face keypoint detection, supplementing keypoint coordinates to existing face detection results.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Input/Output
     - TDLFace\*
     - face_meta
     - Input face detection results, output additional keypoint coordinates

TDL_Classfification
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_Classfification(TDLHandle handle,
                              const TDLModel model_id,
                              TDLImage image_handle,
                              TDLClassInfo *class_info);

【Description】

Execute general classification recognition.

【Parameters】

.. list-table::
   :widths: 1 2 1 3
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLClassInfo\*
     - class_info
     - Output classification results

TDL_ObjectClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_ObjectClassification(TDLHandle handle,
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLObject *object_meta,
                                   TDLClass *class_info);

【Description】

Perform fine-grained classification on detected objects.

【Parameters】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Input
     - TDLObject\*
     - object_meta
     - Detected object information

   * - Output
     - TDLClass\*
     - class_info
     - Output object classification results

TDL_KeypointDetection
~~~~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_KeypointDetection(TDLHandle handle,
                                const TDLModel model_id,
                                TDLImage image_handle,
                                TDLKeypoint *keypoint_meta);

【Description】

Perform human/object keypoint detection.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLKeypoint\*
     - keypoint_meta
     - Output keypoint coordinates and confidence

TDL_InstanceSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_InstanceSegmentation(TDLHandle handle, 
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLInstanceSeg *inst_seg_meta);

【Description】

Perform instance segmentation (Instance Segmentation), detecting the pixel-level contours of each separate object in the image.

【Parameters】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLInstanceSeg\*
     - inst_seg_meta
     - Output instance segmentation results (including mask and bbox)

TDL_SemanticSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_SemanticSegmentation(TDLHandle handle,
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLSegmentation *seg_meta);

【Description】

Perform semantic segmentation (Semantic Segmentation), classifying each pixel in the image.

【Parameters】

.. list-table::
   :widths: 1 2 2 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLSegmentation\*
     - seg_meta
     - Output segmentation results (label map)

TDL_FeatureExtraction
~~~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_FeatureExtraction(TDLHandle handle,
                                const TDLModel model_id,
                                TDLImage image_handle,
                                TDLFeature *feature_meta);

【Description】

Extract deep feature vectors from the image.

【Parameters】

.. list-table::
   :widths: 1 2 1 3
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLFeature\*
     - feature_meta
     - Output feature vector

TDL_LaneDetection
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_LaneDetection(TDLHandle handle,
                            const TDLModel model_id,
                            TDLImage image_handle,
                            TDLLane *lane_meta);

【Description】

Detect lane lines and their attributes.

【Parameters】

.. list-table::
   :widths: 1 2 1 3
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLLane\*
     - lane_meta
     - Output lane line coordinates and attributes

TDL_DepthStereo
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_DepthStereo(TDLHandle handle,
                          const TDLModel model_id,
                          TDLImage image_handle,
                          TDLDepthLogits *depth_logist);

【Description】

Depth estimation based on stereo vision, outputting depth confidence map.

【Parameters】

.. list-table::
   :widths: 1 3 2 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLDepthLogits\*
     - depth_logist
     - Output depth confidence data

TDL_Tracking
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_Tracking(TDLHandle handle,
                       const TDLModel model_id,
                       TDLImage image_handle,
                       TDLObject *object_meta,
                       TDLTracker *tracker_meta);


【Description】

Multi-object tracking, associating detected objects across frames.

【Parameters】

.. list-table::
   :widths: 1 3 2 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Input/Output
     - TDLObject\*
     - object_meta
     - Input detection results, output tracking IDs

   * - Output
     - TDLTracker\*
     - tracker_meta
     - Output tracker status information

TDL_CharacterRecognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_CharacterRecognition(TDLHandle handle,
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLOcr *char_meta);

【Description】

Character recognition, supporting text detection and recognition.

【Parameters】

.. list-table::
   :widths: 1 3 2 3
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLOcr\*
     - char_meta
     - Output recognition results (text content and position)

TDL_LoadModelConfig
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_LoadModelConfig(TDLHandle handle,
                             const char *model_config_json_path);

【Description】

Load model configuration information, after loading you can open models using only model IDs.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const char*
     - model_config_json_path
     - Model configuration file path, if NULL, defaults to configs/model/model_config.json

TDL_SetModelDir
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_SetModelDir(TDLHandle handle,
                          const char *model_dir);

【Description】

Set the model directory path.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const char*
     - model_dir
     - Path to tdl_models repository (subfolders for different platforms)

TDL_SetModelThreshold
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_SetModelThreshold(TDLHandle handle,
                                const TDLModel model_id,
                                float threshold);

【Description】

Set the model threshold value.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - float
     - threshold
     - Model threshold value

TDL_IspClassification
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_IspClassification(TDLHandle handle,
                                const TDLModel model_id,
                                TDLImage image_handle,
                                TDLIspMeta *isp_meta,
                                TDLClass *class_info);

【Description】

Execute ISP image classification task.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Input
     - TDLIspMeta*
     - isp_meta
     - Input ISP related data

   * - Output
     - TDLClass*
     - class_info
     - Output classification results

TDL_Keypoint
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_Keypoint(TDLHandle handle,
                        const TDLModel model_id,
                        TDLImage image_handle,
                        TDLKeypoint *keypoint_meta);

【Description】

Execute keypoint detection task.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLKeypoint*
     - keypoint_meta
     - Output detected keypoint coordinates and confidence

TDL_DetectionKeypoint
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_DetectionKeypoint(TDLHandle handle,
                                const TDLModel model_id,
                                TDLImage image_handle,
                                TDLObject *object_meta);

【Description】

Execute keypoint detection task based on object coordinates (performs keypoint detection after cropping based on target coordinates).

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const TDLModel
     - model_id
     - Model type enumeration

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Output
     - TDLObject*
     - object_meta
     - Output detected keypoint coordinates and confidence

TDL_IntrusionDetection
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_IntrusionDetection(TDLHandle handle,
                                 TDLPoints *regions,
                                 TDLBox *box,
                                 bool *is_intrusion);

【Description】

Execute intrusion detection.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - TDLPoints*
     - regions
     - Background region point set array

   * - Input
     - TDLBox*
     - box
     - Detection region bbox

   * - Output
     - bool*
     - is_intrusion
     - Output intrusion detection result

TDL_MotionDetection
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_MotionDetection(TDLHandle handle,
                              TDLImage background,
                              TDLImage detect_image,
                              TDLObject *roi,
                              uint8_t threshold,
                              double min_area,
                              TDLObject *obj_meta);

【Description】

Execute motion detection task.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - TDLImage
     - background
     - Background image

   * - Input
     - TDLImage
     - detect_image
     - Detection image

   * - Input
     - TDLObject*
     - roi
     - Detection region

   * - Input
     - uint8_t
     - threshold
     - Threshold value

   * - Input
     - double
     - min_area
     - Minimum area

   * - Output
     - TDLObject*
     - obj_meta
     - Output detection results

TDL_APP_Init
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_APP_Init(TDLHandle handle,
                        const char *task,
                        const char *config_file,
                        char ***channel_names,
                        uint8_t *channel_size);

【Description】

Initialize APP task.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const char*
     - task
     - APP task name

   * - Input
     - const char*
     - config_file
     - APP json configuration file path

   * - Output
     - char***
     - channel_names
     - Name information for each video stream

   * - Output
     - uint8_t*
     - channel_size
     - Number of video streams

TDL_APP_SetFrame
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_APP_SetFrame(TDLHandle handle,
                           const char *channel_name,
                           TDLImage image_handle,
                           uint64_t frame_id,
                           int buffer_size);

【Description】

Send frame to APP.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const char*
     - channel_name
     - Current channel name

   * - Input
     - TDLImage
     - image_handle
     - TDLImageHandle object

   * - Input
     - uint64_t
     - frame_id
     - Frame ID of current TDLImageHandle object

   * - Input
     - int
     - buffer_size
     - Number of frames cached by inference thread

TDL_APP_Capture
~~~~~~~~~~~~~~~~~~~~~

【Syntax】

.. code-block:: c

  int32_t TDL_APP_Capture(TDLHandle handle,
                          const char *channel_name,
                          TDLCaptureInfo *capture_info);

【Description】

Execute face capture task.

【Parameters】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - Data Type
     - Parameter Name
     - Description

   * - Input
     - TDLHandle
     - handle
     - TDLHandle object

   * - Input
     - const char*
     - channel_name
     - Current channel name

   * - Output
     - TDLCaptureInfo*
     - capture_info
     - Capture results