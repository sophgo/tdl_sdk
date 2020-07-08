#ifndef _CV183X_FACELIB_H_
#define _CV183X_FACELIB_H_

#include <stdint.h>
#include <stdbool.h>

#include "cvi_comm_video.h"
#include "cvi_sys.h"
#include "cvi_face_types.hpp"
#include "cvi_object_types.hpp"
#include "draw.h"

typedef void* cv183x_facelib_handle_t;

typedef struct {
    void *internal_data;
} cv183x_image_t;

typedef enum {
    RGB_FORMAT_888,
    RGB_FORMAT_565,
    BGR_FORMAT_888,
    BGR_FORMAT_565,
} cv183x_rgb_type;

typedef struct {
	float threshold_1v1;
	float threshold_1vN;
	float threshold_facereg;
	float pitch;
	float yaw;
	float roll;
	int face_quality;
	int face_pixel_min;
	int light_sens;
	int move_sens;
	float threshold_liveness;
	bool wdr_en;
} cv183x_facelib_attr_t;

typedef struct {
	bool fd_en;
	bool facereg_en;
	bool face_matching_en;
	bool yolo_en;
	bool config_yolo; //decide whether or not support yolo
	bool config_liveness;//decide whether or not support liveness
	char *model_face_fd;
	char *model_face_extr;
	char *model_face_liveness;
	char *model_yolo3;
	char *model_face_thermal;
	cv183x_facelib_attr_t attr;
} cv183x_facelib_config_t;

enum
{
  PLANE_Y = 0,
  PLANE_U,
  PLANE_V,
  PLANE_NUM
};

/*============================================================================================================*/
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief	Library init
 * @param 	In: config_file, path to config file
 * @param   Out: handle, handle to the library internal data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xFaceLibOpen(const cv183x_facelib_config_t *facelib_config, cv183x_facelib_handle_t *handle);

/**
 * @brief   Library release
 * @param   In: handle, handle to the libdrary internal data
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceLibClose(const cv183x_facelib_handle_t handle);

/**
 * @brief   Get face attribute settings
 * @param   In: handle, handle to the libdrary internal data
 * @param   Out: facelib_attr, pointer of face attribute stting data
 * @return  Ok:0, Error:-1
**/
int Cv183xGetFaceLibAttr(const cv183x_facelib_handle_t handle, cv183x_facelib_attr_t *facelib_attr);

/**
 * @brief   Update face attribute settings
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: facelib_attr, pointer of face attribute stting data
 * @return  Ok:0, Error:-1
**/
int Cv183xUpdateFaceLibAttr(const cv183x_facelib_handle_t handle, const cv183x_facelib_attr_t *facelib_attr);

/**
 * @brief   Detect faces in image
 * @param   In: handle: handle to the libdrary internal data
 * @param   In: image: the image to be detected
 * @param   Out: faces: facd infos cvi_face_t
 * @param   Out: face_count: the num of detected face  
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *image, 
                     cvi_face_t *faces, int *face_count) ;

/**
 * @brief   Detect liveness in image
 * @param   In: handle: handle to the libdrary internal data
 * @param   In: image1: color image
 * @param   In: image2: ir image
 * @param   Out: faces: facd infos cvi_face_t
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceLivenessDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *image1, 
                     VIDEO_FRAME_INFO_S *image2, cvi_face_t *faces);

/**
 * @brief   Extract features in image of given faces
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: image, the image to be processed
 * @param   In: face_info, An array of cvi_face_info_t
 * @param   In: face_count, number of faces
 * @param   Out: feature_list, An array of cvi_face_feature_t. Its size equals to face_count
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRecognize(cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *image, cvi_face_t *face);

/**
 * @brief   Check if quality of the face meet the requirement
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: image, the image to be processed
 * @param   In: face_info, An array of cvi_face_info_t
 * @param   In: checker_id,
 * @param   Out: result: 0, the image passes the quality check, non-zero: the image does not meet the requirement
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceQualityCheck(const cv183x_facelib_handle_t handle, const cv183x_image_t *image, 
						   const cvi_face_info_t *face_info, int checker_id, int *result);

/**
 * @brief   Insert entry into repository with id, name and features of a person
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: name
 * @param   In: features
 * @param   Out: id
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRepoAddIdentity(cv183x_facelib_handle_t handle, cvi_face_t *face_info, cvi_face_id_t *id);
int Cv183xFaceRepoAddIdentity_withFace(cv183x_facelib_handle_t handle, cvi_face_t *face_info,cvi_face_id_t *id,char *image_name);



/**
 * @brief   Get name of a person from repository
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: id
 * @param   Out: name
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRepoGetIdentityName(const cv183x_facelib_handle_t handle, const cvi_face_id_t id,
									cvi_face_name_t *name);


/**
 * @brief   Get name of a person from repository
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: id
 * @param   Out: name
 * @param   Out: feature
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRepoGetIdentityNameFeature(const cv183x_facelib_handle_t handle, const cvi_face_id_t id,
										 cvi_face_name_t *name, cvi_face_feature_t *feature);

/**
 * @brief   Update entry in repository with id, name and features of a person
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: id
 * @param   In: name
 * @param   In: features
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRepoUpdateIdentity(const cv183x_facelib_handle_t handle, const cvi_face_id_t id,
								 const cvi_face_name_t *name, const cvi_face_feature_t *features);

/**
 * @brief   Remove a specific entry in repository
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: id
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRepoRemoveIdentity(const cv183x_facelib_handle_t handle, const cvi_face_id_t id);

/**
 * @brief   Remove all entries in repository
 * @param   In: handle, handle to the libdrary internal data
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceRepoRemoveAllIdentities(const cv183x_facelib_handle_t handle);
int Cv183xFaceRepoShowList(const cv183x_facelib_handle_t handle);

/**
 * @brief   Compute similarity of two given features and check if score is greater than threshold
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: features
 * @param   In: features2
 * @param   Out: match, 1: when matched, 0: when not matched
 * @param   Out: score, a score in [0, 1] indiciating similarity score between the feature and identity
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceVerify1v1(const cv183x_facelib_handle_t handle, const cvi_face_feature_t *features1, 
	const cvi_face_feature_t *features2, int *match, float *score);

/**
 * @brief   Match an extracted face feature against all entries in repositroy, returns the identity with highest score
 * @param   In: handle, handle to the libdrary internal data
 * @param   In: features
 * @param   Out: id, pointer to the matched identity, negative id indicates no matching identity
 * @param   Out: score, a score in [0, 1] indiciating similarity score between the feature and identity
 * @return  Ok:0, Error:-1
**/
int Cv183xFaceIdentify1vN(const cv183x_facelib_handle_t handle, cvi_face_t *face_info, float threshold);

/**
 * @brief   Read image from a file
 * @param   In: image_file, path to the image file
 * @param   Out: image, handle of the image data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xImageRead(const char *image_file, cv183x_image_t *image);

/**
 * @brief   Read image from a file
 * @param   In: image_file, path to the image file
 * @param   In: image, handle of the image data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xImageWrite(const char *image_file, const cv183x_image_t *image);

/**
 * @brief   Create image from given YUV data
 * @param   In: yuv_buffer, yuv data buffer
 * @param   In: width, width of the given image
 * @param   In: height, height of the given image
 * @param   Out: image, handle of the image data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xImageLoadYuvData(const char *yuv_buffer, size_t width, size_t height, cv183x_image_t *image);

/**
 * @brief   Create image from given RGB/BGR data
 * @param   In: rgb_buffer, rgb data buffer
 * @param   In: width, width of the given image
 * @param   In: height, height of the given image
 * @param   In: format, format of RGB data
 * @param   Out: image, handle of the image data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xImageLoadRgbData(const char *rgb_buffer, size_t width, size_t height, 
	cv183x_rgb_type format, cv183x_image_t *image);

/**
 * @brief   Create image from given JPG data
 * @param   In: jpg_buffer, jpg data buffer
 * @param   In: buf_size, length of JPG data
 * @param   Out: image, handle of the image data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xImageLoadJpgData(const char *jpg_buffer, size_t buf_size, cv183x_image_t *image);

/**
 * @brief   Convert image to YUV data
 * @param   In: source_image, handle of the image data
 * @param   Out: yuv_buffer, buffer pointer to yuv data, call free(*yuv_buffer) to release memory
 * @param   Out: img_width
 * @param   Out: img_height
 **/
int Cv183xImageToYuv(const cv183x_image_t *source_image, char **yuv_buffer, size_t *img_width, size_t *img_height);
/**
 * @brief   Convert image to RGB data
 * @param   In: source_image, handle of the image data
 * @param   In: format, type of RGB
 * @param   Out: rgb_buffer, buffer pointer to rgb data, call free(*rgb_buffer) to release memory
 * @param   Out: img_width
 * @param   Out: img_height
 **/
int Cv183xImageToRgb(const cv183x_image_t *source_image, cv183x_rgb_type format, 
	char **rgb_buffer, size_t *img_width, size_t *img_height);
/**
 * @brief   Convert image to JPG data
 * @param   In: source_image, handle of the image data
 * @param   Out: jpg_buffer, buffer pointer to jpg data, call free(*jpg_buffer) to release memory
 * @param   Out: buffer_length
 **/
int Cv183xImageToJpg(const cv183x_image_t *source_image, char **jpg_buffer, size_t *buffer_length);

/**
 * @brief   Release inaternal data of an image from memory
 * @param   In: image: handle of the image data
 * @return  Ok:0, Error:-1
 * @note
**/
int Cv183xImageRelease(cv183x_image_t *image);

/**
 * @brief   Create an image by resizing given source image
 * @param   In: source_image, handle of the image data
 * @param   Out: output_image, handle of the image data
 * @param   In: resize_width
 * @param   In: resize_height
 */
int Cv183xImageResize(const cv183x_image_t *source_image, cv183x_image_t *output_image, 
	size_t resize_width, size_t resize_height);

/**
 * @brief   Create an image by rotating given source image
 * @param   In: source_image, handle of the image data
 * @param   Out: output_image, handle of the image data
 * @param   In: angle, must be one of 90/180/270
 */
int Cv183xImageRotate(cv183x_image_t *source_image, cv183x_image_t *output_image, int angle);



void yolo3_inference(VIDEO_FRAME_INFO_S *frame, cvi_object_meta_t *meta, int det_type);


void Cv183x_face_Encode(VIDEO_FRAME_INFO_S *stOutFrame, cvi_face_t *face,char *capture_img,char *recored_img);
int Cv183xGetVPSSFrame(char* register_img,VIDEO_FRAME_INFO_S *out_fdframe
                        ,VIDEO_FRAME_INFO_S *out_frframe
                        ,VPSS_GRP VpssGrp);
int Cv183x_Jpg2PNG(char *src_jpg,char *dst_png);

int CviThermalFaceLibOpen(const cv183x_facelib_config_t *facelib_config, cv183x_facelib_handle_t *handle);
int CviThermalFaceLibClose(const cv183x_facelib_handle_t handle);
int CviThermalFaceDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *image,
						 cvi_face_t *faces, int *face_count);

#ifdef __cplusplus
}
#endif
#endif
