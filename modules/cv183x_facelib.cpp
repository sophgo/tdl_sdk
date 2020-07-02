#include <memory>
#include <vector>
#include <string>
#include <unistd.h>

#include <bmruntime.h>
#include <cvimath/cvimath.h>
#include <db_repo.h>
#include <assert.h>

#include "cvi_buffer.h"
#include "cvi_ae_comm.h"
#include "cvi_awb_comm.h"
#include "cvi_comm_isp.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"
#include "cvi_isp.h"
#include "cvi_ae.h"
#include "cvi_vpss.h"


//#include <opencv2/videoio.hpp>
//#include <opencv2/videoio/videoio_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <sqlite3.h>
//#include "cvi_type.h"

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


typedef struct cv183x_db_feature {
    int8_t *raw;
    float *raw_unit;

    uint64_t raw_gaddr;
    bmmem_t bm_device;
    uint32_t size;
    uint32_t num;
    // Only used in TPU
    int8_t *gemm_result_vaddr;
    uint64_t gemm_result_gaddr;
    float *db_buffer;
} cv183x_db_feature_t;

typedef struct {
    unique_ptr<Repository> repo;
    cvi_db_repo_t db_repo;
    cv183x_facelib_attr_t attr;
    // TPU instance here.
    bmctx_t bm_ctx;
    cvk_context_t *cvk_ctx;
    cv183x_db_feature_t *db_feature;
    bool enable_feature_tpu;
} cv183x_facelib_context_t;

int Cv183xFeatureMatching(const cv183x_facelib_handle_t handle, int8_t *feature, uint32_t *k_index, float *k_value, float *buffer, const int k, const float threshold);

int Cv183xFaceLibOpen(const cv183x_facelib_config_t *facelib_config, cv183x_facelib_handle_t *handle){
    int ret;
    printf("Cv183xFaceLibOpen\n");

    cv183x_facelib_context_t *context = new cv183x_facelib_context_t;

    if (cvi_repo_open(&(context->db_repo), facelib_config->db_repo_path)) {
        return -1;
    }

    context->attr = facelib_config->attr;

    if (facelib_config->model_face_fd != NULL) {
        init_network_retina(facelib_config->model_face_fd);
        printf("init_network_retina done\n");
    }
    
    if (facelib_config->model_face_extr) {
        ret = init_network_face_attribute(facelib_config->model_face_extr);
        printf("init_network_face_attribute ret:%d\n",ret);
    }
    
    
    // if (facelib_config->config_liveness && (facelib_config->model_face_liveness != NULL))
    // {
        init_network_liveness(facelib_config->model_face_liveness);
        printf("init_network_liveness done\n");
    // }

    if (facelib_config->config_yolo && (facelib_config->model_yolo3 != NULL))
    {
        init_network_yolov3(facelib_config->model_yolo3);
        printf("init_network_yolov3 done\n");
    }

    ret = bm_init_chip(0, &context->bm_ctx, "cv1880v2");
    if (ret != BM_SUCCESS) {
      fprintf(stderr, "cvi_init failed, err %d\n", ret);
      return CVI_FAILURE;
    }
    cvk_reg_info_t req_info;
    strncpy(req_info.chip_ver_str, "cv1880v2", sizeof(req_info.chip_ver_str) - 1);
    req_info.cmdbuf_size = 0x10000000;
    req_info.cmdbuf = static_cast<uint8_t *>(malloc(req_info.cmdbuf_size));
    context->cvk_ctx = cvikernel_register(&req_info);

    // Init DB
    context->db_feature = new cv183x_db_feature_t;
    // Init value
    context->db_feature->raw = nullptr;
    context->db_feature->raw_unit = nullptr;
    context->db_feature->db_buffer = nullptr;
    context->db_feature->raw_gaddr = -1;
    context->db_feature->bm_device = NULL;
    context->db_feature->size = 0;
    context->db_feature->num = 0;

   *handle = context;
    return 0;
}

int Cv183xResetVerify(cv183x_facelib_handle_t handle) {
    cv183x_facelib_context_t *context = static_cast<cv183x_facelib_context_t*>(handle);
    if (context->db_feature->bm_device == NULL) {
        if (context->db_feature->raw) {
            free(context->db_feature->raw);
        }
        if (context->db_feature->raw_unit != nullptr) {
            delete[] context->db_feature->raw_unit;
        }
    } else {
        bmmem_device_free(context->bm_ctx, context->db_feature->bm_device);
    }
    if (context->db_feature->db_buffer != nullptr) {
        delete[] context->db_feature->db_buffer;
    }

    context->db_feature->raw = nullptr;
    context->db_feature->raw_unit = nullptr;
    context->db_feature->raw_gaddr = -1;
    context->db_feature->bm_device = NULL;
    context->db_feature->size = 0;
    context->db_feature->num = 0;
    return 0;
}

int Cv183xFaceLibClose(cv183x_facelib_handle_t handle) {
    cv183x_facelib_context_t *fctx = static_cast<cv183x_facelib_context_t*>(handle);

    clean_network_retina();
    clean_network_face_attribute();
    clean_network_liveness();

    Cv183xResetVerify(handle);
    delete fctx->db_feature;

    if (fctx->cvk_ctx) {
      fctx->cvk_ctx->ops->cleanup(fctx->cvk_ctx);
    }
    bm_exit(fctx->bm_ctx);
    delete fctx;
    return 0;
}

int Cv183xLoadIdentify(cv183x_facelib_handle_t handle, bool use_tpu) {
    if (use_tpu) {
        printf("Currently unsupported.\n");
        return -1;
    }
    cv183x_facelib_context_t *context = static_cast<cv183x_facelib_context_t*>(handle);
    context->enable_feature_tpu = use_tpu;
    Cv183xResetVerify(handle);
    cv183x_db_feature_t *dbmeta = context->db_feature;
    // Get from db function.
    uint8_t *from_db = nullptr;
    dbmeta->num = cvi_repo_get_features(&(context->db_repo), 0, 0, &from_db);
    if (dbmeta->num == 0) {
        printf("Error! DB is empty.\n");
        return -1;
    }
    dbmeta->size = NUM_FACE_FEATURE_DIM * dbmeta->num;
    uint32_t data_num;
    if (context->enable_feature_tpu) {
        if (dbmeta->bm_device == NULL) {
            uint32_t total_sz = dbmeta->size + dbmeta->num * sizeof(int32_t);
            dbmeta->bm_device = bmmem_device_alloc_raw(context->bm_ctx, total_sz);
        }
        dbmeta->raw_gaddr = bmmem_device_addr(dbmeta->bm_device);
        dbmeta->raw = (int8_t *)bmmem_device_v_addr(dbmeta->bm_device);
        dbmeta->gemm_result_gaddr = dbmeta->raw_gaddr + dbmeta->size;
        dbmeta->gemm_result_vaddr = dbmeta->raw + dbmeta->size;
        memcpy(dbmeta->raw, (int8_t *)from_db, dbmeta->size);
        free(from_db);
    } else {
        dbmeta->raw = (int8_t *)from_db;
        dbmeta->gemm_result_gaddr = -1;
        dbmeta->gemm_result_vaddr = nullptr;
    }
    dbmeta->raw_unit = new float[dbmeta->num * sizeof(float)];
    cvm_gen_db_i8_unit_length(dbmeta->raw, dbmeta->raw_unit, NUM_FACE_FEATURE_DIM, dbmeta->num);
    if (dbmeta->bm_device != NULL) {
        bmmem_device_flush(context->bm_ctx, dbmeta->bm_device);
    }
    // Create buffer
    dbmeta->db_buffer = new float[dbmeta->num];
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

int Cv183x_person_Info(const cv183x_facelib_handle_t facelib_handle, int face_id, cvi_person_t *person)
{
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(facelib_handle);

    int ret ;
    ret =  cvi_repo_get_person(&(ctx->db_repo), face_id, person);
    printf("face id = %d\n",face_id);
    printf("person name = %s\n",person->image_name);
}

int Cv183xFaceIdentify1vN(const cv183x_facelib_handle_t handle, cvi_face_t *meta, float threshold) {
    if (!meta->size) {
        return -1;
    }
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    cv183x_db_feature_t *dbmeta = ctx->db_feature;
    if (dbmeta->num == 0) {
        printf("Error. DB is empty.\n");
        return -1;
    }
    for (int i = 0; i < meta->size; ++i) {
        uint32_t index[1] = {0};
        float score[1] = {0};
        int face_id = 0;

        if (0 > Cv183xFeatureMatching(handle, meta->face_info[i].face_feature, index, score, ctx->db_feature->db_buffer, 1, threshold)) {
            continue;
        }
        if (index[0] != -1) {
            cvi_repo_get_face_by_offset(&(ctx->db_repo), index[0], &face_id, meta->face_info[i].name, sizeof(meta->face_info[i].name));
            printf("fd name %s\n",meta->face_info[i].name );
       
            return face_id;
        }

    }

    return -1;
}


int Cv183xFeatureMatching(const cv183x_facelib_handle_t handle, int8_t *feature,
                          uint32_t *k_index, float *k_value, float *buffer, const int k,
                          const float threshold) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    cv183x_db_feature_t *dbmeta = ctx->db_feature;
    if (ctx->enable_feature_tpu) {
        printf("Currently unsupported.\n");
        return -1;
    } else {

        gettimeofday(&tv1, NULL);
        // printf("db number = %d\n",dbmeta->num);

        
        cvm_cpu_i8data_ip_match(feature, dbmeta->raw, dbmeta->raw_unit, k_index,
                                k_value, buffer, NUM_FACE_FEATURE_DIM, dbmeta->num, k);
        gettimeofday(&tv2, NULL);
        #if 1 //def PERF_PROFILING
        //printf("fd matching time :%d\n",(tv2.tv_usec-tv1.tv_usec)/1000+(tv2.tv_sec-tv1.tv_sec)*1000);
        #endif
    }
    for (size_t i = 0; i < k; i++) {
      if (k_value[i] < threshold) {
          k_index[i] = -1;
          //printf("cmp point1 = %f\n",k_value[i]);

          k_value[i] = 0;

      }
      else
      {
          printf("cmp point = %f\n",k_value[i]);
          return 0;
      }
      
    }
    return -1;
}

static void set_person_info(cvi_person_t *person,int id,char *name,char *identifier,
                            char *serial,char *ic_card,char *image_name)
{
    person->id = id;
    memset(person, 0, sizeof(cvi_person_t));							

    strncpy(person->name ,name, strlen(name));
    strncpy(person->identifier , identifier, strlen(identifier));
    strncpy(person->serial , serial,strlen(serial));
    strncpy(person->ic_card , ic_card,strlen(ic_card));
    strncpy(person->image_name, image_name,strlen(image_name));
}

int Cv183xFaceRepoAddIdentity_withFace(cv183x_facelib_handle_t handle, cvi_face_t *face_info, cvi_face_id_t *id,char *image_name) 
{
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    cv183x_db_feature_t *dbmeta = ctx->db_feature;

    cvi_person_t person;
    static int ramom_num = 0;

    char *name = face_info->face_info[0].name;

    int face_id = cvi_repo_add_face(&(ctx->db_repo),
            name, (unsigned char *)face_info->face_info->face_feature, sizeof(face_info->face_info->face_feature));

    if (0 > face_id) {
        return -1;
    }

    //id should be unique.

    int tmp = dbmeta->num+1;
    char serial_id[50]="\0";
    sprintf(serial_id,"%d",tmp);

    char identifier[50]="\0";
    sprintf(identifier,"%d",tmp);

    char ic_card[50]="\0";
    sprintf(ic_card,"%d",tmp);

    printf("img name =%s\n",image_name);
    set_person_info(&person,face_id,name,identifier,serial_id,ic_card,image_name);

    int ret;
    ret = cvi_repo_set_person(&(ctx->db_repo), face_id,&person);

    if(ret<0)
    {
        printf("set person info error\n");
        return -1;
    }

    *id = face_id;

    return 0;
}

int Cv183xFaceRepoAddIdentity(cv183x_facelib_handle_t handle, cvi_face_t *face_info, cvi_face_id_t *id) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    char *name = face_info->face_info[0].name;

    int face_id = cvi_repo_add_face(&(ctx->db_repo),
            name, (unsigned char *)face_info->face_info->face_feature, sizeof(face_info->face_info->face_feature));

    if (0 > face_id) {
        return -1;
    }

    *id = face_id;

    return 0;
}

int Cv183xFaceRepoGetIdentityName(const cv183x_facelib_handle_t handle, const cvi_face_id_t id, cvi_face_name_t *name) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    cvi_face_feature_t feature = {0};

    if (0 > cvi_repo_find_face(&(ctx->db_repo), id, name->name, sizeof(name->name), (unsigned char*)feature.features, sizeof(feature.features))) {
        return -1;
    }
    return 0;
}

int Cv183xFaceRepoGetIdentityNameFeature(const cv183x_facelib_handle_t handle, const cvi_face_id_t id,
                                         cvi_face_name_t *name, cvi_face_feature_t *feature) {

    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);
    if (0 > cvi_repo_find_face(&(ctx->db_repo), id, name->name, sizeof(name->name), (unsigned char *)feature->features, sizeof(feature->features))) {
        return -1;
    }

    return 0;
}

int Cv183xFaceRepoUpdateIdentity(const cv183x_facelib_handle_t handle, const cvi_face_id_t id,
                                 const cvi_face_name_t *name, const cvi_face_feature_t *features) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    return cvi_repo_update_face(&(ctx->db_repo), id,
                    name? name->name: NULL,
                    features? (unsigned char*)features->features: NULL,
                    features? sizeof(features->features): 0);
}

int Cv183xFaceRepoRemoveIdentity(const cv183x_facelib_handle_t handle, const cvi_face_id_t id) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    return cvi_repo_delete_face(&(ctx->db_repo), id);
}

int Cv183xFaceRepoRemoveAllIdentities(const cv183x_facelib_handle_t handle) {
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    char path[128] = {0};
    snprintf(path, sizeof(path), "%s", ctx->db_repo.path);

    cvi_repo_close(&(ctx->db_repo));
    unlink(path);

    return cvi_repo_open(&(ctx->db_repo), path);
}

int Cv183xFaceRepoShowList(const cv183x_facelib_handle_t handle)
{
    cv183x_facelib_context_t *ctx = static_cast<cv183x_facelib_context_t*>(handle);

    std::vector<size_t> id_table;
    id_table = ctx->repo->id_list();

    cout << "repo size = " << id_table.size() << endl;
    for (int i=0; i < id_table.size() ; i++) {
        size_t id = id_table[i];
        cout << "id = " << id << " , name = " << ctx->repo->get_name(id).value_or("") << endl;
    }

    return 0;
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


void Cv183x_face_Encode(VIDEO_FRAME_INFO_S *stOutFrame, cvi_face_t *face,char *capture_img,char *recored_img)
{
    cv::Mat rgb_frame(stOutFrame->stVFrame.u32Height, stOutFrame->stVFrame.u32Width, CV_8UC3);
    stOutFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(stOutFrame->stVFrame.u64PhyAddr[0],
                                                           stOutFrame->stVFrame.u32Length[0]);
    char *va_rgb = (char *)stOutFrame->stVFrame.pu8VirAddr[0];
    for (int i = 0; i < rgb_frame.rows; i++) {
        memcpy(rgb_frame.ptr(i, 0), va_rgb + stOutFrame->stVFrame.u32Stride[0] * i, rgb_frame.cols * 3);
    }
    CVI_SYS_Munmap((void *)stOutFrame->stVFrame.pu8VirAddr[0], stOutFrame->stVFrame.u32Length[0]);

    cvi_face_info_t face_info = bbox_rescale(stOutFrame, face, 0);

    int x1 = face_info.bbox.x1;
    int y1 = face_info.bbox.y1;
    int x2 = face_info.bbox.x2;
    int y2 = face_info.bbox.y2;

    // printf("x1 = %d, y1 = %d, x2 = %d, y2 = %d\n",x1,y1,x2,y2);

    int x = x1-5>0?x1-5:0;
    int y = y1-5>0?y1-5:0;
    int width = x2-x1+10;
    int height = y2-y1+10;

    if(x+width>rgb_frame.cols)
        width = rgb_frame.cols-x;
    if(y+height>rgb_frame.rows)
        height = rgb_frame.rows-y;
    // printf("colom = %d, row = %d, x = %d, y = %d, width = %d, height = %d\n",rgb_frame.cols, rgb_frame.rows,x,y,width,height);

    cv::Mat result_image(rgb_frame, cv::Rect(x,y, width, height));
    if(capture_img != NULL)
    {
        imwrite(capture_img,result_image);
    }
    // cv::resize(result_image,result_image,Size(300,300), CV_INTER_LINEAR);
    if(recored_img != NULL)
    {
        imwrite(recored_img,result_image);
    }


}


int Cv183xGetVPSSFrame(char* register_img,VIDEO_FRAME_INFO_S *out_fdframe
                        ,VIDEO_FRAME_INFO_S *out_frframe
                        ,VPSS_GRP VpssGrp)
{
    cv::Mat face = imread(register_img,1);
    int s32Ret = CVI_SUCCESS;
    if(face.data == NULL)
    {
        printf("invalid file\n");
        return CVI_FAILURE;
    }
    VIDEO_FRAME_INFO_S frame;
    int imgWidth = face.cols;
    int imgHeight = face.rows;
    if(imgWidth<32 || imgWidth*imgHeight > 1920*1080 || imgHeight<32)
    {
        printf("img size should be >32*32 && <1920*1080\n");
        return CVI_FAILURE;
    }
	VB_CAL_CONFIG_S stVbCalConfig;
	VB_BLK blk;

	COMMON_GetPicBufferConfig(imgWidth, imgHeight, PIXEL_FORMAT_RGB_888, DATA_BITWIDTH_8
		, COMPRESS_MODE_NONE, DEFAULT_ALIGN, &stVbCalConfig);

	frame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
	frame.stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888;
	frame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
	frame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
	frame.stVFrame.u32Width = imgWidth;
	frame.stVFrame.u32Height = imgHeight;
	frame.stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;
	frame.stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
	frame.stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
	frame.stVFrame.u32TimeRef = 0;
	frame.stVFrame.u64PTS = 0;
	frame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;

	blk = CVI_VB_GetBlock(VB_INVALID_POOLID, stVbCalConfig.u32VBSize);
	if (blk == VB_INVALID_HANDLE) {
		printf("Can't acquire vb block\n");
		return CVI_FAILURE;
	}

	// [WA-01]
	frame.u32PoolId = CVI_VB_Handle2PoolId(blk);
	frame.stVFrame.u32Length[0] = stVbCalConfig.u32MainYSize;
	frame.stVFrame.u64PhyAddr[0] = CVI_VB_Handle2PhysAddr(blk);

    printf("imgWidth:%d imgHeight:%d u32Length[0]:%d stride:%d\n",imgWidth,imgHeight,stVbCalConfig.u32MainYSize,stVbCalConfig.u32MainStride);

	frame.stVFrame.pu8VirAddr[0]
		= (CVI_U8*)CVI_SYS_Mmap(frame.stVFrame.u64PhyAddr[0], frame.stVFrame.u32Length[0]);

    int stride = stVbCalConfig.u32MainStride;
    CVI_U8 *pixels = (CVI_U8 *)frame.stVFrame.pu8VirAddr[0];
    for (int h = 0; h < imgHeight; ++h) {
        memcpy(pixels,face.ptr(h, 0), imgWidth*3);
        pixels += stride;
    }

    VPSS_GRP_ATTR_S stGrpAttr;
    s32Ret = CVI_VPSS_GetGrpAttr(VpssGrp, &stGrpAttr);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetGrpAttr is fail\n");
        goto FAIL;
    }

    stGrpAttr.u32MaxW = imgWidth; //32*32 - 2688*2688 最大放大32倍
    stGrpAttr.u32MaxH = imgHeight;
    s32Ret = CVI_VPSS_SetGrpAttr(VpssGrp, &stGrpAttr);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetGrpAttr is fail\n");
        goto FAIL;

    }

    printf("CVI_VPSS_SendFrame\n");
	CVI_VPSS_SendFrame(VpssGrp, &frame, -1);
    #if 1
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, 0,out_fdframe, 500);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        goto FAIL;

    }
    
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, 1,out_frframe, 500);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn1 failed with %#x\n", s32Ret);
        goto FAIL;
    }

    #else  //dump frame for verify
    VIDEO_FRAME_INFO_S stVideoFrame;
    FILE *fp;
    CVI_U32 plane_offset, u32LumaSize, u32ChromaSize;
    CVI_VOID *vir_addr;
    CVI_VPSS_GetChnFrame(VpssGrp, 0,&stVideoFrame, 500);

    size_t image_size = stVideoFrame.stVFrame.u32Length[0] + stVideoFrame.stVFrame.u32Length[1]
				  + stVideoFrame.stVFrame.u32Length[2];

		u32LumaSize =  stVideoFrame.stVFrame.u32Stride[0] * stVideoFrame.stVFrame.u32Height;
		if (stVideoFrame.stVFrame.enPixelFormat == PIXEL_FORMAT_RGB_888_PLANAR)
			u32ChromaSize = u32LumaSize;
		else if (stVideoFrame.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_422)
			u32ChromaSize =  stVideoFrame.stVFrame.u32Stride[1] * stVideoFrame.stVFrame.u32Height;
		else if (stVideoFrame.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_420)
			u32ChromaSize =  stVideoFrame.stVFrame.u32Stride[1] * stVideoFrame.stVFrame.u32Height / 2;
		else if (stVideoFrame.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_400)
			u32ChromaSize =  0;

		fp = fopen("./dump.bin", "w");
		if (fp == CVI_NULL) {
			fp = fopen("/mnt/data/dump.bin", "wb");
			if (fp == CVI_NULL) {
				CVI_VPSS_ReleaseChnFrame(0, 0, &stVideoFrame);
			}
		}
		printf("width: %d, height: %d, total_buf_length: %d\n",
			   stVideoFrame.stVFrame.u32Width,
			   stVideoFrame.stVFrame.u32Height, image_size);
		vir_addr = CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[0], image_size);
		CVI_SYS_IonInvalidateCache(stVideoFrame.stVFrame.u64PhyAddr[0], vir_addr, image_size);
		plane_offset = 0;
		for (int i = 0; i < 3; i++) {
			if (stVideoFrame.stVFrame.u32Length[i] == 0)
				continue;

			stVideoFrame.stVFrame.pu8VirAddr[i] = (CVI_U8*)vir_addr + plane_offset;
			plane_offset += stVideoFrame.stVFrame.u32Length[i];
			printf("plane(%d): paddr(0x%llx) vaddr(0x%llx) stride(%d)\n",
				   i, stVideoFrame.stVFrame.u64PhyAddr[i],
				   stVideoFrame.stVFrame.pu8VirAddr[i],
				   stVideoFrame.stVFrame.u32Stride[i]);
			fwrite((void *)stVideoFrame.stVFrame.pu8VirAddr[i]
				, (i == 0) ? u32LumaSize : u32ChromaSize, 1, fp);

		}
		CVI_SYS_Munmap(vir_addr, image_size);

		fclose(fp);
    #endif

FAIL:
    CVI_VB_ReleaseBlock(blk);
	CVI_SYS_Munmap(frame.stVFrame.pu8VirAddr[0], frame.stVFrame.u32Length[0]);
	CVI_SYS_Munmap(frame.stVFrame.pu8VirAddr[1], frame.stVFrame.u32Length[1]);
	CVI_SYS_Munmap(frame.stVFrame.pu8VirAddr[2], frame.stVFrame.u32Length[2]);
    return s32Ret;
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
