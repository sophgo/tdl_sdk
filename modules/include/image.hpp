#pragma once

#include <cstdint>
#include <string>
#include "memory/bytes.hpp"
#include "memory/encoding.hpp"
#include "generic_image.hpp"

//#ifndef OPENCV_DISABLED
    #include "cvmat_bgr_image.hpp"
    using image_t = cvmat_bgr_image_t;
//#else
//    using image_t = generic_image_t;
//#endif
