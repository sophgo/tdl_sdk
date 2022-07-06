#pragma once
#include <string>
#include <vector>
#include "core/object/cvai_object_types.h"

namespace cviai {
namespace coco_utils {
static const std::vector<std::string> class_names_80 = {
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "potted plant",  "bed",
    "dining table",  "toilet",        "tv monitor",    "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

static const std::vector<std::string> class_names_91 = {
    "person",       "bicycle",      "car",           "motorbike",     "aeroplane",
    "bus",          "train",        "truck",         "boat",          "traffic light",
    "fire hydrant", "street sign",  "stop sign",     "parking meter", "bench",
    "bird",         "cat",          "dog",           "horse",         "sheep",
    "cow",          "elephant",     "bear",          "zebra",         "giraffe",
    "hat",          "backpack",     "umbrella",      "shoe",          "eye glasses",
    "handbag",      "tie",          "suitcase",      "frisbee",       "skis",
    "snowboard",    "sports ball",  "kite",          "baseball bat",  "baseball glove",
    "skateboard",   "surfboard",    "tennis racket", "bottle",        "plate",
    "wine glass",   "cup",          "fork",          "knife",         "spoon",
    "bowl",         "banana",       "apple",         "sandwich",      "orange",
    "broccoli",     "carrot",       "hot dog",       "pizza",         "donut",
    "cake",         "chair",        "sofa",          "potted plant",  "bed",
    "mirror",       "dining table", "window",        "desk",          "toilet",
    "door",         "tv monitor",   "laptop",        "mouse",         "remote",
    "keyboard",     "cell phone",   "microwave",     "oven",          "toaster",
    "sink",         "refrigerator", "blender",       "book",          "clock",
    "vase",         "scissors",     "teddy bear",    "hair drier",    "toothbrush",
    "hair brush"};

static const std::vector<cvai_obj_class_id_e> class_id_map_80_to_91 = {
    CVI_AI_DET_TYPE_PERSON,        CVI_AI_DET_TYPE_BICYCLE,      CVI_AI_DET_TYPE_CAR,
    CVI_AI_DET_TYPE_MOTORBIKE,     CVI_AI_DET_TYPE_AEROPLANE,    CVI_AI_DET_TYPE_BUS,
    CVI_AI_DET_TYPE_TRAIN,         CVI_AI_DET_TYPE_TRUCK,        CVI_AI_DET_TYPE_BOAT,
    CVI_AI_DET_TYPE_TRAFFIC_LIGHT, CVI_AI_DET_TYPE_FIRE_HYDRANT, CVI_AI_DET_TYPE_STOP_SIGN,
    CVI_AI_DET_TYPE_PARKING_METER, CVI_AI_DET_TYPE_BENCH,        CVI_AI_DET_TYPE_BIRD,
    CVI_AI_DET_TYPE_CAT,           CVI_AI_DET_TYPE_DOG,          CVI_AI_DET_TYPE_HORSE,
    CVI_AI_DET_TYPE_SHEEP,         CVI_AI_DET_TYPE_COW,          CVI_AI_DET_TYPE_ELEPHANT,
    CVI_AI_DET_TYPE_BEAR,          CVI_AI_DET_TYPE_ZEBRA,        CVI_AI_DET_TYPE_GIRAFFE,
    CVI_AI_DET_TYPE_BACKPACK,      CVI_AI_DET_TYPE_UMBRELLA,     CVI_AI_DET_TYPE_HANDBAG,
    CVI_AI_DET_TYPE_TIE,           CVI_AI_DET_TYPE_SUITCASE,     CVI_AI_DET_TYPE_FRISBEE,
    CVI_AI_DET_TYPE_SKIS,          CVI_AI_DET_TYPE_SNOWBOARD,    CVI_AI_DET_TYPE_SPORTS_BALL,
    CVI_AI_DET_TYPE_KITE,          CVI_AI_DET_TYPE_BASEBALL_BAT, CVI_AI_DET_TYPE_BASEBALL_GLOVE,
    CVI_AI_DET_TYPE_SKATEBOARD,    CVI_AI_DET_TYPE_SURFBOARD,    CVI_AI_DET_TYPE_TENNIS_RACKET,
    CVI_AI_DET_TYPE_BOTTLE,        CVI_AI_DET_TYPE_WINE_GLASS,   CVI_AI_DET_TYPE_CUP,
    CVI_AI_DET_TYPE_FORK,          CVI_AI_DET_TYPE_KNIFE,        CVI_AI_DET_TYPE_SPOON,
    CVI_AI_DET_TYPE_BOWL,          CVI_AI_DET_TYPE_BANANA,       CVI_AI_DET_TYPE_APPLE,
    CVI_AI_DET_TYPE_SANDWICH,      CVI_AI_DET_TYPE_ORANGE,       CVI_AI_DET_TYPE_BROCCOLI,
    CVI_AI_DET_TYPE_CARROT,        CVI_AI_DET_TYPE_HOT_DOG,      CVI_AI_DET_TYPE_PIZZA,
    CVI_AI_DET_TYPE_DONUT,         CVI_AI_DET_TYPE_CAKE,         CVI_AI_DET_TYPE_CHAIR,
    CVI_AI_DET_TYPE_SOFA,          CVI_AI_DET_TYPE_POTTED_PLANT, CVI_AI_DET_TYPE_BED,
    CVI_AI_DET_TYPE_DINING_TABLE,  CVI_AI_DET_TYPE_TOILET,       CVI_AI_DET_TYPE_TV_MONITOR,
    CVI_AI_DET_TYPE_LAPTOP,        CVI_AI_DET_TYPE_MOUSE,        CVI_AI_DET_TYPE_REMOTE,
    CVI_AI_DET_TYPE_KEYBOARD,      CVI_AI_DET_TYPE_CELL_PHONE,   CVI_AI_DET_TYPE_MICROWAVE,
    CVI_AI_DET_TYPE_OVEN,          CVI_AI_DET_TYPE_TOASTER,      CVI_AI_DET_TYPE_SINK,
    CVI_AI_DET_TYPE_REFRIGERATOR,  CVI_AI_DET_TYPE_BOOK,         CVI_AI_DET_TYPE_CLOCK,
    CVI_AI_DET_TYPE_VASE,          CVI_AI_DET_TYPE_SCISSORS,     CVI_AI_DET_TYPE_TEDDY_BEAR,
    CVI_AI_DET_TYPE_HAIR_DRIER,    CVI_AI_DET_TYPE_TOOTHBRUSH,
};

}  // namespace coco_utils
}  // namespace cviai