#ifndef _CVI_FACE_TYPES_H_
#define _CVI_FACE_TYPES_H_

#define FACE_PTS                5
#define MAX_NAME_LEN            128
#define NUM_FACE_FEATURE_DIM    512
#define NUM_EMOTION_FEATURE_DIM 7
#define NUM_GENDER_FEATURE_DIM  2
#define NUM_RACE_FEATURE_DIM    3
#define NUM_AGE_FEATURE_DIM     101

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
	float score;
} cvi_face_detect_rect_t;

typedef struct {
    float x[FACE_PTS];
    float y[FACE_PTS];
} cvi_face_pts_t;

typedef struct {
	int8_t features[NUM_FACE_FEATURE_DIM];
} cvi_face_feature_t;

typedef uint32_t cvi_face_id_t;

typedef struct {
    char name[MAX_NAME_LEN];
} cvi_face_name_t;

typedef struct {
    cvi_face_detect_rect_t bbox;
    cvi_face_pts_t face_pts;
    int8_t face_feature[NUM_FACE_FEATURE_DIM];
	char name[MAX_NAME_LEN];
	float face_liveness;
    char emotion[16];
    char gender[16];
    char race[16];
    float age;
    float mask_score;
} cvi_face_info_t;

typedef struct {
    int size;
	int width;
    int height;
    cvi_face_info_t *face_info;
} cvi_face_t;

typedef enum {
    EMOTION_UNKNOWN = 0,
    EMOTION_HAPPY,
    EMOTION_SURPRISE,
    EMOTION_FEAR,
    EMOTION_DISGUST,
    EMOTION_SAD,
    EMOTION_ANGER,
    EMOTION_NEUTRAL,
} FaceEmotion;

typedef enum {
    GENDER_UNKNOWN = 0,
    GENDER_MALE,
    GENDER_FEMALE,
} FaceGender;

typedef enum { RACE_UNKNOWN = 0, RACE_CAUCASIAN, RACE_BLACK, RACE_ASIAN } FaceRace;

#endif
