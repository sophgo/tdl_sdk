// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file simd_wrapper.h
 * @ingroup NETWORKS_UTILS
 */
#pragma once

#if defined(__aarch64__)
#else
#    include <emmintrin.h>
#    include <immintrin.h>
#endif

#if defined(__aarch64__)
typedef uint8x16_t simd_u8;     /**< unsigned char */
typedef uint8x16x3_t simd_u8_3; /**< unsigned char array of size 3 */
#else
typedef __m128i simd_u8; /**< unsigned char */
typedef union {
    __m128i val[3];
} simd_u8_3; /**< unsigned char array of size 3 */
#endif

#if defined(__aarch64__)
#    define simd_subs_u8(a, b) vqsubq_u8(a, b) /**< Substract saturate */
#else
#    define simd_subs_u8(a, b) _mm_subs_epu8(a, b) /**< Substract saturate */
#endif