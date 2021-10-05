#ifndef _CVIAI_CUSTOM_H_
#define _CVIAI_CUSTOM_H_
#include "core/cviai_core.h"

/**
 * \addtogroup core_ai_custom Custom AI Inference Support
 * \ingroup core_ai
 */

/** @typedef preProcessFunc
 * @ingroup core_ai_custom
 * @brief A preprocess function pointer.
 */
typedef void (*preProcessFunc)(VIDEO_FRAME_INFO_S *inFrames, VIDEO_FRAME_INFO_S *outFrames,
                               CVI_U32 numOfFrames);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup core_custom_ai_settings Custom AI Inference Setting Functions
 * \ingroup core_ai_custom
 */
/**@{*/

/**
 * @brief Add a custom AI inference instance to handle.
 *
 * @param handle An AI SDK handle.
 * @param id Returned id of the instance.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_AddInference(cviai_handle_t handle, uint32_t *id);

/**
 * @brief Set the model path for custom networks.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param filepath File path to the cvimodel file.
 * @return int Return CVIAI_SUCCESS if load model succeed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_SetModelPath(cviai_handle_t handle, const uint32_t id,
                                              const char *filepath);

/**
 * @brief Get set model path from custom models.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @return model path.
 */
DLL_EXPORT const char *CVI_AI_Custom_GetModelPath(cviai_handle_t handle, const uint32_t id);

/**
 * @brief Set different vpss thread for custom models. Vpss group id is not thread safe. We
 * recommended to change a thread if the process is not sequential.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @return int Return CVIAI_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_SetVpssThread(cviai_handle_t handle, const uint32_t id,
                                               const uint32_t thread);

/**
 * @brief Set different vpss thread for custom models. Vpss group id is not thread safe. We
 * recommended to change a thread if the process is not sequential. This function requires manually
 * assigning a vpss group id.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param thread The vpss thread index user desired. Note this param will changed if previous index
 * is not used.
 * @param vpssGroupId Assign a vpss group id if a new vpss instance needs to be created.
 * @return int Return CVIAI_SUCCESS if successfully changed.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_SetVpssThread2(cviai_handle_t handle, const uint32_t id,
                                                const uint32_t thread, const VPSS_GRP vpssGroupId);

/**
 * @brief Get the set thread index for given custom model id.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param thread Output thread index.
 * @return int Return CVIAI_SUCCESS.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_GetVpssThread(cviai_handle_t handle, const uint32_t id,
                                               uint32_t *thread);

/**
 * @brief If choose to use VPSS to do scaling and quantization. You'll have to set the factor, mean,
 * and whether the image should keep aspect ratio. The framework will get the quantization threshold
 * from loaded cvimodel If you don't want to use VPSS, you'll have to setup the preprocess function
 * pointer.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param inputIndex The input tensor index.
 * @param factor The scaling factor.
 * @param mean The scaling mean.
 * @param length The length of the fector and mean array. Must be 1 (will duplicate to 3) or 3.
 * @param keepAspectRatio Whether the image should keep aspect ratio when resize to input size of
 * the cvimodel.
 * @return int Return CVIAI_SUCCESS on success.
 *
 * @see CVI_AI_Custom_SetPreprocessFuncPtr
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_SetVpssPreprocessParam(cviai_handle_t handle, const uint32_t id,
                                                        const uint32_t inputIndex,
                                                        const float *factor, const float *mean,
                                                        const uint32_t length,
                                                        const bool keepAspectRatio);

/**
 * @brief This function is similar to CVI_AI_Custom_SetVpssPreprocessParam but you can directly set
 * the quantized facetor and quantized mean by yourself.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param inputIndex The input tensor index.
 * @param qFactor The quantized scaling factor.
 * @param qMean The quantized scaling mean.
 * @param length The length of the fector and mean array. Must be 1 (will duplicate to 3) or 3.
 * @param keepAspectRatio Whether the image should keep aspect ratio when resize to input size of
 * the cvimodel.
 * @return int Return CVIAI_SUCCESS on success.
 *
 * @see CVI_AI_Custom_SetVpssPreprocessParam
 * @see CVI_AI_Custom_SetPreprocessFuncPtr
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_SetVpssPreprocessParamRaw(cviai_handle_t handle, const uint32_t id,
                                                           const uint32_t inputIndex,
                                                           const float *qFactor, const float *qMean,
                                                           const uint32_t length,
                                                           const bool keepAspectRatio);

/**
 * @brief
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param func The preprocessing function pointer.
 * @param use_tensor_input Use tensor input instead of VIDEO_FRAME_S. If set to true, you must
 * implement the function pointer and copy the data to the input tensor by yourself. It'll also
 * ignore use_vpss_sq. Default is false.
 * @param use_vpss_sq Use VPSS to do scaling and quantization. You can set to false if you like to
 * do all of the preprocessing by yourself. If use_tensor_input and use_vpss_sq are both set to
 * false, the VIDEO_FRAME_S will be directly sent into cvimodel. If use_vpss_sq is set to true, this
 * parameter will be ignored. Default is true.
 * @return int Return CVIAI_SUCCESS on success.
 *
 * @see CVI_AI_Custom_SetVpssPreprocessParam
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_SetPreprocessFuncPtr(cviai_handle_t handle, const uint32_t id,
                                                      preProcessFunc func,
                                                      const bool use_tensor_input,
                                                      const bool use_vpss_sq);

/**
 * @brief Close the chosen custom model. Will not delete the instance.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_CloseModel(cviai_handle_t handle, const uint32_t id);

/**@}*/

/**
 * \addtogroup core_custom_ai_inference Custom AI Inference Functions
 * \ingroup core_ai_custom
 */
/**@{*/

/**
 * @brief Get the input tensor size with given tensor name. Once this function is called, you cannot
 * change the settings of the model unless you call CVI_AI_Custom_CloseModel.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param tensorName The input tensor name.
 * @param n The batch size of the tensor.
 * @param c The channel size of the tensor.
 * @param h The height of the tensor.
 * @param w The width of the tensor.
 * @return int Return CVIAI_SUCCESS on success.
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_GetInputTensorNCHW(cviai_handle_t handle, const uint32_t id,
                                                    const char *tensorName, uint32_t *n,
                                                    uint32_t *c, uint32_t *h, uint32_t *w);

/**
 * @brief Do custom model inference. Once this function is called, you cannot change the settings of
 * the model unless you call CVI_AI_Custom_CloseModel.
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param frame Input video frame.
 * @param numOfFrames Number of input frames.
 * @return int Return CVIAI_SUCCESS on success.
 *
 * @see CVI_AI_Custom_CloseModel
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_RunInference(cviai_handle_t handle, const uint32_t id,
                                              VIDEO_FRAME_INFO_S *frame, uint32_t numOfFrames);

/**
 * @brief
 *
 * @param handle An AI SDK handle.
 * @param id Id of the instance.
 * @param tensorName The name of the output tensor.
 * @param tensor The output tensor in int8_t*, may need to be casted before use.
 * @param tensorCount The number of the feature in the tensor.
 * @param unitSize The size for each element of the tensor.
 * @return int Return CVIAI_SUCCESS on success.
 *
 */
DLL_EXPORT CVI_S32 CVI_AI_Custom_GetOutputTensor(cviai_handle_t handle, const uint32_t id,
                                                 const char *tensorName, int8_t **tensor,
                                                 uint32_t *tensorCount, uint16_t *unitSize);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif  // End of _CVIAI_CUSTOM_H_
