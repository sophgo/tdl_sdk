#ifndef _CVI_TDL_MEDIA_H_
#define _CVI_TDL_MEDIA_H_
#include "cvi_comm.h"

#define DLL_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup core_media Convert Buffer or File to VIDEO_FRAME_INFO_S
 * \ingroup core_cvitdlcore
 */
/**@{*/

/**
 * @brief Convert given image buffer to VB frame.
 *
 * @param buffer The input image buffer.
 * @param width Input image width.
 * @param height Input image height.
 * @param stride Input image stride.
 * @param inFormat Input image buffer format.
 * @param blk VB block id.
 * @param frame Output read image.
 * @param outFormat Set output format, only supports RGB, BGR, planar.
 * @return int Return CVI_TDL_SUCCESS if read succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_Buffer2VBFrame(const uint8_t *buffer, uint32_t width, uint32_t height,
                                          uint32_t stride, const PIXEL_FORMAT_E inFormat,
                                          VB_BLK *blk, VIDEO_FRAME_INFO_S *frame,
                                          const PIXEL_FORMAT_E outFormat);

/**
 * @brief Read image from given path and return a VIDEO_FRAME_INFO_S allocated from ION.
 *
 * @param filepath GIven image path.
 * @param frame Output read image.
 * @param format Set output format, only supports RGB, BGR, planar.
 * @return int Return CVI_TDL_SUCCESS if read succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ReadImage(const char *filepath, VIDEO_FRAME_INFO_S *frame,
                                     const PIXEL_FORMAT_E format);
DLL_EXPORT CVI_S32 CVI_TDL_ReadImage_Resize(const char *filepath, VIDEO_FRAME_INFO_S *frame,
                                            PIXEL_FORMAT_E format, uint32_t width, uint32_t height);
/**
 * @brief Release image which is read from CVI_TDL_ReadImage.
 *
 * @param filepath GIven image path.
 * @param frame Output read image.
 * @return int Return CVI_TDL_SUCCESS if read succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ReleaseImage(VIDEO_FRAME_INFO_S *frame);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVI_TDL_MEDIA_H_
