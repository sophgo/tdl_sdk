#pragma once
#include <string>
#include <vector>

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

static const std::vector<std::string> class_names_90 = {
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

int map_90_class_id_to_80(int class_id);
}  // namespace coco_utils
}  // namespace cviai