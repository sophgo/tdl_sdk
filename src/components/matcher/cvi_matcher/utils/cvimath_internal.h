#ifndef CVIMATH_INTERNAL_H
#define CVIMATH_INTERNAL_H
#include <assert.h>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include "cvikernel.h"
#include "cvimath.h"

// 使用现代C++类型别名
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

using gaddr_t = uint64_t;

typedef uint32_t laddr_t;
typedef uint64_t gaddr_t;
typedef uint32_t ctrl_t;

static inline uint64_t alignUp(uint64_t x, uint64_t n) {
  return (x + n - 1) / n * n;
}

static inline int ceilingFunc(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

static inline int ceilingFuncShift(int numerator, int shift) {
  return (numerator + (1 << shift) - 1) >> shift;
}

static inline int getNumShift(uint64_t num) {
  int n = 0;
  while (!(num & 1)) {
    n++;
    num >>= 1;
  }
  return n;
}

static inline int bitsizeOfFmt(KernelFmt fmt) {
  switch (fmt) {
    case CVK_FMT_F32:
    case CVK_FMT_I32:
      return 32;
    case CVK_FMT_F16:
    case CVK_FMT_I16:
    case CVK_FMT_U16:
    case CVK_FMT_BF16:
      return 16;
    case CVK_FMT_I8:
    case CVK_FMT_U8:
      return 8;
    case CVK_FMT_I4:
      return 4;
    case CVK_FMT_I2:
      return 2;
    case CVK_FMT_I1:
      return 1;
    default:
      assert(0);
      return -1;
  }
}

static inline int bytesizeOfFmt(KernelFmt fmt) { return bitsizeOfFmt(fmt) / 8; }

static inline void tg2TlShape(TensorLmemShape *tl, TensorTgmemShape *tg) {
  tl->n = tg->n;
  tl->c = tg->c;
  tl->h = tg->h;
  tl->w = tg->w;
}

static inline void tl2TgShape(TensorTgmemShape *tg, TensorLmemShape *tl) {
  tg->n = tl->n;
  tg->c = tl->c;
  tg->h = tl->h;
  tg->w = tl->w;
}

/**
 * please refer @example for more details
 */

// public function

/**
 * @brief General Matrix Multiplication
 * that equal \lhs_gaddr * \rhs_gaddr = \dest_gaddr
 *
 * @param cvk_ctx kernel structure
 * @param lhs_gaddr left hand side device memory address
 * @param rhs_gaddr right hand side device memory address
 * @param dest_gaddr destination device memory address
 * @param in_row \lhs_gaddr matrix row
 * @param in_col \lhs_gaddr matrix col
 * @param out_col \dest_gaddr matrix col
 * @param fmt the possible value is \CVK_FMT_BF16 or \CVK_FMT_I8 or \CVK_FMT_U8
 * @example
 *
 * // 1. alloc host memory and put it to device memory
 * // M=in_row K=in_col N=out_col
 * cvk_mg_t *mg_A = _test_put_matrix_g(&ctx, M, K, CVK_FMT_BF16, (uint8_t
 * *)bf16_A); cvk_mg_t *mg_B = _test_put_matrix_g(&ctx, K, N, CVK_FMT_BF16,
 * (uint8_t *)bf16_B); cvk_mg_t *mg_R = _test_put_matrix_g(&ctx, M * 2, N,
 * CVK_FMT_BF16, (uint8_t *)bf16_R);
 *
 * // 2. get device address for gemm
 * gaddr_t gaddr_a = mg_A->start_address;
 * gaddr_t gaddr_b = mg_B->start_address;
 * gaddr_t gaddr_r = mg_R->start_address;
 *
 * // 3. prepare gemm descriptor
 * cvmGemm(cvk_ctx, gaddr_a, gaddr_b, gaddr_r, M, K, N);
 *
 * // 4. submit descriptor
 * test_submit_comp(&ctx, cvk_ctx);
 *
 * // 5. get result from device to host
 * uint16_t *bf16_ref = (uint16_t *)test_get_mg_mem_comp(&ctx, mg_R);
 *
 * @ return slice_num array of {M, N, K}
 */
size_t *cvmGemm(KernelContext *cvk_ctx, gaddr_t bottom_data_gaddr,
                gaddr_t weight_data_gaddr, gaddr_t top_data_gaddr, int in_row,
                int in_col, int out_col, KernelFmt fmt);

/**
 * @brief combine \cvmGemm int8 result to int32
 * the raw output is seperate 32bit result info 4 part with bstride
 * and we need to 'combine' it to human readable
 * for instance, the following is the raw result
 * lsb             31               msb
 * 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x8
 * 0x9 0xa 0xb 0xc 0xd 0xe 0xf 0x0
 * 0x11 0x12 0x13 0x14 0x15 0x16 0x17 0x18
 * 0x19 0x20 0x21 0x22 0x23 0x24 0x25 0x26
 *
 * the value by strategy could be column major:
 * 1. 0x19110901
 * 2. 0x20120a02
 * 3. 0x21130b03
 * and so on
 *
 * @param cvmGemm_strategy return strategy value from \cvmGemm
 * @param cvm_output raw result from \cvmGemm
 * @param [out] i32_R int32 result
 * @param M row of output matrix
 * @param N column of output matrix
 *
 * @return status, 0 means success, other means generates command fail
 */
int cvmCombinGemmI8(size_t *slice_num, uint8_t *i8_C, uint32_t *i32_C, int M,
                    int N);

// mask enum define
enum CVM_MASK_TYPE {
  CVM_MASK_TYPE_GT_0 = 0,  // remain >  0
  CVM_MASK_TYPE_GE_0,      // remain >= 0
  CVM_MASK_TYPE_EQ_0,      // remain  = 0
  CVM_MASK_TYPE_LT_0,      // remain <  0
  CVM_MASK_TYPE_LE_0,      // remain <= 0
  CVM_MASK_MAX
};

#endif  // CVIMATH_INTERNAL_H
