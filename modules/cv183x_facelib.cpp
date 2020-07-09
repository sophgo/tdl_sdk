#include <memory>
#include <vector>
#include <string>
#include <unistd.h>

#include <bmruntime.h>
#include <cvimath/cvimath.h>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/time.h>
struct timeval tv1,tv2;

#include "image.hpp"
#include "cv183x_facelib_v0.0.1.h"
#include "cvi_face_utils.hpp"

#include "cvi_retina_face/retina_face.h"
#include "cvi_face_attribute/face_attribute.h"
#include "cvi_feature_matching/cached_repo.hpp"
#include "cvi_feature_matching/face_repo.h"
#include "cvi_liveness/liveness.h"
#include "cvi_yolove3/run_yolo.h"
#include "cvi_thermal_face_detect/thermal_face.h"

using namespace std;
using namespace cv;
typedef CachedRepo<512> Repository;

typedef struct {
    unique_ptr<Repository> repo;
    cv183x_facelib_attr_t attr;
    // TPU instance here.
    bmctx_t bm_ctx;
    cvk_context_t *cvk_ctx;
    bool enable_feature_tpu;
} cv183x_facelib_context_t;

int Cv183xFeatureMatching(const cv183x_facelib_handle_t handle, int8_t *feature, uint32_t *k_index, float *k_value, float *buffer, const int k, const float threshold);

int Cv183xFaceLibOpen(const cv183x_facelib_config_t *facelib_config, cv183x_facelib_handle_t *handle){
    int ret;
    printf("Cv183xFaceLibOpen\n");

    cv183x_facelib_context_t *context = new cv183x_facelib_context_t;

    context->attr = facelib_config->attr;

    if (facelib_config->model_face_fd != NULL) {
        init_network_retina(facelib_config->model_face_fd);
        printf("init_network_retina done\n");
    }

    if (facelib_config->config_liveness && (facelib_config->model_face_liveness != NULL)) {
        init_network_liveness(facelib_config->model_face_liveness);
        printf("init_network_liveness done\n");
    }

    if (facelib_config->model_face_extr != NULL) {
        ret = init_network_face_attribute(facelib_config->model_face_extr);
        printf("init_network_face_attribute ret:%d\n",ret);
    }

    if (facelib_config->config_yolo && (facelib_config->model_yolo3 != NULL)) {
        init_network_yolov3(facelib_config->model_yolo3);
        printf("init_network_yolov3 done\n");
    }

    // ret = bm_init_chip(0, &context->bm_ctx, "cv183x");
    // if (ret != BM_SUCCESS) {
    //   fprintf(stderr, "cvi_init failed, err %d\n", ret);
    //   return CVI_FAILURE;
    // }
    // cvk_reg_info_t req_info;
    // strncpy(req_info.chip_ver_str, "cv1880v2", sizeof(req_info.chip_ver_str) - 1);
    // req_info.cmdbuf_size = 0x10000000;
    // req_info.cmdbuf = static_cast<uint8_t *>(malloc(req_info.cmdbuf_size));
    // context->cvk_ctx = cvikernel_register(&req_info);

   *handle = context;
    return 0;
}

int Cv183xFaceLibClose(cv183x_facelib_handle_t handle) {
    cv183x_facelib_context_t *fctx = static_cast<cv183x_facelib_context_t*>(handle);

    clean_network_retina();
    clean_network_face_attribute();
    clean_network_liveness();

    if (fctx->cvk_ctx) {
      fctx->cvk_ctx->ops->cleanup(fctx->cvk_ctx);
    }
    bm_exit(fctx->bm_ctx);
    delete fctx;
    return 0;
}

static Repository::feature_t convert_feature(const cvi_face_feature_t *feature) {
    Repository::feature_t f(NUM_FACE_FEATURE_DIM);
    for (size_t i = 0; i < NUM_FACE_FEATURE_DIM; i++) {
        f[i] = feature->features[i];
    }
    return f;
}

static void convert_feature(cvi_face_feature_t *dest, const Repository::feature_t &src) {
    for (size_t i = 0; i < min<size_t>(NUM_FACE_FEATURE_DIM, src.size()); i++) {
        dest->features[i] = src[i];
    }
    for (size_t i = src.size(); i < NUM_FACE_FEATURE_DIM; i++) {
        dest->features[i] = 0;
    }
}


int Cv183xGetFaceLibAttr(const cv183x_facelib_handle_t handle, cv183x_facelib_attr_t *facelib_attr) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    *facelib_attr = ctx->attr;
	return 0;
}

int Cv183xUpdateFaceLibAttr(const cv183x_facelib_handle_t handle, const cv183x_facelib_attr_t *facelib_attr) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    ctx->attr = *facelib_attr;
	return 0;
}

int Cv183xFaceDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *stfdFrame,
                     cvi_face_t *faces, int *face_count) {

    retina_face_inference(stfdFrame, faces, face_count);

    return 0;
}

int Cv183xFaceLivenessDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *image1,
                     VIDEO_FRAME_INFO_S *image2, cvi_face_t *faces) {

    liveness_inference(image1,image2,faces);

    return 0;
}

int Cv183xFaceRecognize(cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *stfrFrame, cvi_face_t *face) {

    for (size_t i = 0; i < face->size; i++) {
        face_attribute_inference(stfrFrame, face);
    }

    return 0;
}

int Cv183xFaceVerify1v1(const cv183x_facelib_handle_t handle, const cvi_face_feature_t *features1,
    const cvi_face_feature_t *features2, int *match, float *score) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    int32_t value1 = 0, value2 = 0, value3 = 0;
    for (uint32_t i = 0; i < NUM_FACE_FEATURE_DIM; i++) {
        value1 += (short)features1->features[i] * features1->features[i];
        value2 += (short)features2->features[i] * features2->features[i];
        value3 += (short)features1->features[i] * features2->features[i];
    }
    *score = (float)value3 / (sqrt(value1) * sqrt(value2));

    if (*score >= ctx->attr.threshold_1v1) {
        *match = 1;
    } else {
        *match = 0;
    }
    return 0;
}

int Cv183xFaceIdentify1vN(const cv183x_facelib_handle_t handle, cvi_face_t *meta, float threshold) {
    if (!meta->size) {
        return -1;
    }
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    // cv183x_db_feature_t *dbmeta = ctx->db_feature;
    // if (dbmeta->num == 0) {
    //     printf("Error. DB is empty.\n");
    //     return -1;
    // }
    // for (int i = 0; i < meta->size; ++i) {
    //     uint32_t index[1] = {0};
    //     float score[1] = {0};
    //     int face_id = 0;

        // if (0 > Cv183xFeatureMatching(handle, meta->face_info[i].face_feature, index, score, ctx->db_feature->db_buffer, 1, threshold)) {
        //     continue;
        // }
        // if (index[0] != -1) {
        //     cvi_repo_get_face_by_offset(&(ctx->db_repo), index[0], &face_id, meta->face_info[i].name, sizeof(meta->face_info[i].name));
        //     printf("fd name %s\n",meta->face_info[i].name );
       
        //     return face_id;
        // }

    // }

    return -1;
}


int Cv183xFeatureMatching(const cv183x_facelib_handle_t handle, int8_t *feature,
                          uint32_t *k_index, float *k_value, float *buffer, const int k,
                          const float threshold) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    // cv183x_db_feature_t *dbmeta = ctx->db_feature;
    // if (ctx->enable_feature_tpu) {
    //     printf("Currently unsupported.\n");
    //     return -1;
    // } else {

    //     gettimeofday(&tv1, NULL);
    //     // printf("db number = %d\n",dbmeta->num);


    //     cvm_cpu_i8data_ip_match(feature, dbmeta->raw, dbmeta->raw_unit, k_index,
    //                             k_value, buffer, NUM_FACE_FEATURE_DIM, dbmeta->num, k);
    //     gettimeofday(&tv2, NULL);
    //     #if 1 //def PERF_PROFILING
    //     //printf("fd matching time :%d\n",(tv2.tv_usec-tv1.tv_usec)/1000+(tv2.tv_sec-tv1.tv_sec)*1000);
    //     #endif
    // }
    // for (size_t i = 0; i < k; i++) {
    //   if (k_value[i] < threshold) {
    //       k_index[i] = -1;
    //       //printf("cmp point1 = %f\n",k_value[i]);

    //       k_value[i] = 0;

    //   }
    //   else
    //   {
    //       printf("cmp point = %f\n",k_value[i]);
    //       return 0;
    //   }
      
    // }
    return -1;
}

int Cv183xImageRead(const char *image_file, cv183x_image_t *image) {
    cv::Mat mat;
    //printf("%s,%d .\n", __FUNCTION__, __LINE__);
    //printf("%s .\n", image_file);
    try {
        mat = cv::imread(image_file, cv::IMREAD_COLOR);
        if (mat.empty()) {
            printf("file [%s] is empty.\n", image_file);
            return -2;
        }
    } catch (cv::Exception &e) {
        printf("file [%s] not exist.\n", image_file);
        return -1;
    }

    image_t *img = new image_t(std::move(mat));
    image->internal_data = img;

    return 0;
}

int Cv183xImageWrite(const char *image_file, const cv183x_image_t *image) {
	cv::Mat img = *((image_t*)(image->internal_data));
    printf("%s,%d .\n", __FUNCTION__, __LINE__);
    try {
        cv::imwrite(image_file, img);
    } catch (cv::Exception &e) {
        return -1;
    }
	return 0;
}

int Cv183xImageLoadYuvData(const char *yuv_buffer, size_t width, size_t height, cv183x_image_t *image) {
    cv::Mat mat;
    //printf("%s,%d .\n", __FUNCTION__, __LINE__);
    try {
        cv::Mat input(height, width, CV_8UC3, (void *)yuv_buffer);
        cv::cvtColor(input, mat, cv::COLOR_YUV2BGR);
    } catch (cv::Exception &e) {
        return -1;
    }

    image_t *img = new image_t(std::move(mat));
    image->internal_data = img;

    return 0;
}

static int BgrFormatChannelCount(cv183x_rgb_type format) {
    switch (format) {
    case BGR_FORMAT_565:
    case RGB_FORMAT_565:
        return 2;
    case BGR_FORMAT_888:
    case RGB_FORMAT_888:
        return 3;
    default:
        return 0;
    }
}

static int FormatToBgrConversion(cv183x_rgb_type format) {
    switch (format) {
    case BGR_FORMAT_565:
        return cv::COLOR_BGR5652BGR;
        break;
    case BGR_FORMAT_888:
        return cv::COLOR_COLORCVT_MAX;
        break;
    case RGB_FORMAT_565:
        return cv::COLOR_BGR5652RGB;
        break;
    case RGB_FORMAT_888:
        return cv::COLOR_RGB2BGR;
        break;
    default:
        return -1;
    }
}
static int BgrToFormatConversion(cv183x_rgb_type format) {
    switch (format) {
    case BGR_FORMAT_565:
        return cv::COLOR_BGR2BGR565;
        break;
    case BGR_FORMAT_888:
        return cv::COLOR_COLORCVT_MAX;
        break;
    case RGB_FORMAT_565:
        return cv::COLOR_RGB2BGR565;
        break;
    case RGB_FORMAT_888:
        return cv::COLOR_BGR2RGB;
        break;
    default:
        return -1;
    }
}

int Cv183xImageLoadRgbData(const char *rgb_buffer, size_t width, size_t height, cv183x_rgb_type format, cv183x_image_t *image) {
    cv::Mat mat;
    int channel = BgrFormatChannelCount(format);
    if (channel <= 0) {
        return -1;
    }
    int type = (channel == 2 ? CV_8UC2 : CV_8UC3);
    int conversion = FormatToBgrConversion(format);

    try {
        cv::Mat input(height, width, type, (void *)rgb_buffer);
        if (conversion != cv::COLOR_COLORCVT_MAX) {
            cv::cvtColor(input, mat, conversion);
        } else {
            mat = input;
        }
    } catch (cv::Exception &e) {
        return -1;
    }

    image_t *img = new image_t(std::move(mat));
    image->internal_data = img;

    return 0;
}

int Cv183xImageLoadJpgData(const char *jpg_buffer, size_t buf_size, cv183x_image_t *image) {
    cv::Mat mat;
    try {
        cv::Mat input(buf_size, 1, CV_8UC1, (void *)jpg_buffer);
        mat = cv::imdecode(input, cv::IMREAD_COLOR);
    } catch (cv::Exception &e) {
        return -1;
    }

    image_t *img = new image_t(std::move(mat));
    image->internal_data = img;

    return 0;
}

int Cv183xImageToYuv(const  cv183x_image_t *source_image, char **yuv_buffer, size_t *img_width, size_t *img_height) {
    cv::Mat img = *((image_t*)(source_image->internal_data));

    int buffer_length = img.size[0] * img.size[1] * 3;
    *yuv_buffer = (char*)malloc(buffer_length);
    *img_height = img.size[0];
    *img_width = img.size[1];

    cv::Mat dst(img.size[0], img.size[1], CV_8UC3, (void *)*yuv_buffer);
    try {
        cv::cvtColor(img, dst, cv::COLOR_BGR2YUV);
    } catch (cv::Exception &e) {
        free(yuv_buffer);
        return -1;
    }

    return 0;
}

int Cv183xImageToRgb(const cv183x_image_t *source_image, cv183x_rgb_type format, char **rgb_buffer,
                        size_t *img_width, size_t *img_height) {
    int channel = BgrFormatChannelCount(format);
    if (channel <= 0) {
        return -1;
    }
    cv::Mat img = *((image_t*)(source_image->internal_data));

    int buffer_length = img.size[0] * img.size[1] * channel;
    *rgb_buffer = (char*)malloc(buffer_length);
    *img_height = img.size[0];
    *img_width = img.size[1];

    int conversion = BgrToFormatConversion(format);
    if (conversion >= 0 && conversion != cv::COLOR_COLORCVT_MAX) {
        cv::Mat dst(img.size[0], img.size[1], channel == 2 ? CV_8UC2 : CV_8UC3, (void *)*rgb_buffer);
        try {
            cv::cvtColor(img, dst, conversion);
        } catch (cv::Exception &e) {
            free(rgb_buffer);
            return -1;
        }
    } else {
        memcpy(*rgb_buffer, img.data, buffer_length);
    }

    return 0;
}

int Cv183xImageToJpg(const cv183x_image_t *source_image, char **jpg_buffer, size_t *buffer_length) {
    cv::Mat img = *((image_t*)(source_image->internal_data));
    vector<unsigned char> buffer;

    try {
        cv::imencode(".jpg", img, buffer);
    } catch (cv::Exception &e) {
        return -1;
    }

    *jpg_buffer = (char*)malloc(buffer.size());
    *buffer_length = buffer.size();
    memcpy(*jpg_buffer, buffer.data(), buffer.size());

    return 0;
}

int Cv183xImageRelease(cv183x_image_t *image) {
    //printf("%s,%d .\n", __FUNCTION__, __LINE__);
    //cv::Mat frame = *((image_t*)(image->internal_data));
    //frame.release();
    delete (image_t *)(image->internal_data);
    return 0;
}

int Cv183xImageResize(cv183x_image_t *source_image, cv183x_image_t *output_image, size_t resize_width, size_t resize_height) {
    cv::Mat mat;
    try {
        image_t *img = (image_t*)(source_image->internal_data);
        cv::resize((cv::Mat)(*img), mat, cv::Size(resize_width, resize_height));
    } catch (cv::Exception &e) {
        return -1;
    }

    image_t *img = new image_t(std::move(mat));
    output_image->internal_data = img;

    return 0;
}

int Cv183xImageRotate(cv183x_image_t *source_image, cv183x_image_t *output_image, int angle) {
    if (angle != 90 && angle != 180 && angle != 270) {
        return -1;
    }

    cv::Mat result;
    try {
        cv::Mat img = *((image_t*)(source_image->internal_data));
        cv::Mat transform = cv::getRotationMatrix2D(cv::Point2f(img.size[0] / 2, img.size[1] / 2), angle, 1);
        cv::warpAffine(img, result, transform, {img.size[1], img.size[0]});
    } catch (cv::Exception &e) {
        return -1;
    }

    image_t *img = new image_t(std::move(result));
    output_image->internal_data = img;

    return 0;
}

int Cv183x_Jpg2PNG(char *src_jpg,char *dst_png)
{
    cv::Mat face = imread(src_jpg,1);
    int s32Ret = CVI_SUCCESS;
    if(face.data == NULL)
    {
        printf("invalid file\n");
        return CVI_FAILURE;
    }
    imwrite(dst_png,face);

    return s32Ret;
}

int Cv183xObjDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *stObjDetFrame,
                    cvi_object_meta_t *obj_meta, int det_type) {
    yolov3_inference(stObjDetFrame, obj_meta, det_type);

    return 0;
}

int CviThermalFaceLibOpen(const cv183x_facelib_config_t *facelib_config, cv183x_facelib_handle_t *handle)
{
    cv183x_facelib_context_t *context = new cv183x_facelib_context_t;

    context->attr = facelib_config->attr;

    if (facelib_config->model_face_thermal != NULL) {
        init_network_thermal(facelib_config->model_face_thermal);
        // init_network_thermal("/mnt/data/thermal_face_detection.bf16sigmoid.cvimodel");
    }

    *handle = context;

    return 0;
}

int CviThermalFaceLibClose(const cv183x_facelib_handle_t handle)
{
    clean_network_thermal();

    return 0;
}

int CviThermalFaceDetect(const cv183x_facelib_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                         cvi_face_t *faces, int *face_count)
{
    thermal_face_inference(frame, faces, face_count);

    return 0;
}
