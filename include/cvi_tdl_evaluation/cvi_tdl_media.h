#ifndef _CVI_TDL_MEDIA_H_
#define _CVI_TDL_MEDIA_H_
#include <cvi_comm.h>

#define DLL_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef void *imgprocess_t;
DLL_EXPORT CVI_S32 CVI_TDL_Create_ImageProcessor(imgprocess_t *hanlde);

/**
 * @brief Read image from given path and return a VIDEO_FRAME_INFO_S allocated from ION.
 *
 * @param filepath GIven image path.
 * @param frame Output read image.
 * @param format Set output format, only supports RGB, BGR, planar.
 * @return int Return CVI_TDL_SUCCESS if read succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ReadImage(imgprocess_t hanlde, const char *filepath,
                                     VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format);
DLL_EXPORT CVI_S32 CVI_TDL_ReadImage_Resize(imgprocess_t hanlde, const char *filepath,
                                            VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format,
                                            uint32_t width, uint32_t height);
/**
 * @brief Release image which is read from CVI_TDL_ReadImage.
 *
 * @param filepath GIven image path.
 * @param frame Output read image.
 * @return int Return CVI_TDL_SUCCESS if read succeed.
 */
DLL_EXPORT CVI_S32 CVI_TDL_ReleaseImage(imgprocess_t hanlde, VIDEO_FRAME_INFO_S *frame);

#ifdef __cplusplus
}
#endif

#endif  // End of _CVI_TDL_MEDIA_H_
