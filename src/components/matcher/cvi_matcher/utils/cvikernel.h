#ifndef CVIKERNEL_H
#define CVIKERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "cviruntime_context.h"

/*
 * Type Definition
 */
typedef enum CVIKERNEL_FMT_E {
  CVK_FMT_F32 = 0,
  CVK_FMT_F16,
  CVK_FMT_I32,
  CVK_FMT_I16,
  CVK_FMT_I8,
  CVK_FMT_I4,
  CVK_FMT_I2,
  CVK_FMT_I1,
  CVK_FMT_U32,
  CVK_FMT_U16,
  CVK_FMT_U8,
  CVK_FMT_BF16,
  CVK_FMT_INVALID
} KernelFmt;

/*
 *  CVI TPU Chip Name Definition
 */
#define CVI_TPU_VERSION_183X "cv183x"
#define CVI_TPU_VERSION_182X "cv182x"
#define CVI_TPU_VERSION_181X "cv181x"
#define CVI_TPU_VERSION_180X "cv180x"

// /*
//  * System information
//  */
// typedef enum CVIKERNEL_HW_FEATURE_E {
//   CVK_HWF_NONE = 0,
//   CVK_HWF_FC_OP1_CONST = 1,     // FC op1 const
//   CVK_HWF_8B_ADD_SUB = 1 << 1,  // 8b add/sub
//   CVK_HWF_MIN_POOL = 1 << 2,    // Min pooling
//   CVK_HWF_M_BRADCAST = 1 << 3,  // Multi broadcast
//   CVK_HWF_QM_LSHIFT = 1 << 4,   // Left shift of quan_m op
//   CVK_HWF_GE = 1 << 5,          // Greater than or equal to
//   CVK_HWF_CMD_PRE_EXE = 1 << 6  // Command pre-execute
// } cvk_hw_feature_t;

typedef struct cvikernel_chip_info {
  uint32_t version;
  uint32_t node_num;
  uint32_t node_shift;
  uint32_t npu_num;
  uint32_t npu_shift;
  uint32_t eu_num;
  uint32_t eu_shift;
  uint32_t lmem_size;
  uint32_t lmem_shift;
  uint32_t lmem_banks;
  uint32_t lmem_bank_size;
  uint64_t lmem_start;
  uint64_t gmem_start;
  uint64_t gmem_size;
  uint64_t features;
} ChipInfo;

/*
 * Fundamental structures for tensor and matrix
 */
typedef struct cvikernel_matrix_lmem_shape {
  uint32_t n, c, w, col;
} MatrixLmemShape;

typedef struct cvikernel_matrix_gmem_shape {
  uint32_t row, col;
} MatrixGmemShape;

typedef struct cvikernel_matrix_lmem_stride {
  uint32_t n, c, h;
} MatrixLmemStride;

typedef struct cvikernel_matrix_tgmem_stride {
  uint32_t row;
} MatrixTGmemStride;

typedef struct cvikernel_tensor_lmem_shape {
  uint32_t n, c, h, w;
} TensorLmemShape;

typedef struct cvikernel_tensor_tgmem_shape {
  uint32_t n, c, h, w;
} TensorTgmemShape;

typedef struct cvikernel_tensor_tgmem_stride {
  uint32_t n, c, h, w;
} TensorTgmemStride;

typedef struct cvikernel_tensor_lmem_stride {
  uint32_t n, c, h, w;
} TensorLmemStride;

// Even though width stride is not in TDMA configuration,
// The strides of all dimensions is enough to calculate correct position in
// global memory, especially in bf16.
// typedef struct cvikernel_tensor_tgmem_stride {
//   uint32_t n, c, h, w;
// } cvk_tg_stride_t;

typedef struct cvikernel_tensor_lmem {
  uint32_t start_address;
  KernelFmt fmt;
  KernelFmt cmprs_fmt;
  TensorLmemShape shape;
  TensorLmemStride stride;
  uint8_t int8_rnd_mode;  // 0 is round to nearset even, 1 is toward zero,
                          // currently used by lut
  uint8_t eu_align;
} TensorLmem;

typedef struct cvikernel_matrix_lmem {
  uint32_t start_address;
  KernelFmt fmt;
  MatrixLmemShape shape;
  MatrixLmemStride stride;
  uint8_t int8_rnd_mode;  // 0 is round to nearset even, 1 is toward zero
  uint8_t eu_align;
} MatrixLmem;

typedef struct cvikernel_tensor_gmem {
  uint8_t base_reg_index;
  uint64_t start_address;
  KernelFmt fmt;
  TensorTgmemShape shape;
  TensorTgmemStride stride;
  uint8_t int8_rnd_mode;  // 0 is round to nearset even, 1 is toward zero
} TensorGmem;

typedef struct cvikernel_compressed_tensor_gmem {
  TensorGmem t;
  uint64_t reserved_size;
  uint8_t bit_length;  // deprecated for zero compress
  uint8_t bias0;
  uint8_t bias1;
  int zero_guard_en;
} CompressedTensorGmem;

typedef struct cvikernel_matrix_gmem {
  uint8_t base_reg_index;
  uint64_t start_address;
  KernelFmt fmt;
  MatrixGmemShape shape;
  MatrixTGmemStride stride;
  uint8_t int8_rnd_mode;  // 0 is round to nearset even, 1 is toward zero
} MatrixGmem;

typedef struct cvikernel_compressed_matrix_gmem {
  MatrixGmem m;
  uint8_t bias0;
  uint8_t bias1;
  int zero_guard_en;
} CompressedMatrixGmem;

/*
 * TDMA Engine APIs: LMEM to LMEM (L2L)
 */
typedef struct {
  uint8_t mv_lut_idx;
  uint8_t mv_lut_base;
  const TensorLmem *src;
  const TensorLmem *dst;
  uint8_t outstanding;  // Concurrent TDMA LD/ST and TDM L2L
  uint16_t layer_id;
} TdmaL2lTensorCopyParam;

typedef struct {
  const TensorLmem *src;
  const TensorLmem *dst;
  int right_shift;
  uint32_t lrn_step;
  uint16_t layer_id;
} TdmaL2lTensorLrnShiftParam;

/*
 * TDMA Engine APIs: LMEM to GMEM (L2G)
 */
typedef struct {
  const TensorLmem *src;
  const TensorGmem *dst;
  uint16_t layer_id;
  uint32_t intra_cmd_paral;  // [0]: disable
                             // [1]: enable TDMA/TIU intra-command parallelism
} TdmaL2gTensorCopyParam;

typedef struct {
  const TensorLmem *src;
  const TensorGmem *dst;
  uint16_t layer_id;
} TdmaL2gTensorCopyNcTransposedParam;

typedef struct {
  const TensorLmem *src;
  const TensorGmem *dst;
  uint16_t layer_id;
} TdmaL2gTensorCopyCwTransposedParam;

typedef struct {
  const TensorLmem *src;
  const CompressedTensorGmem *dst;
  uint16_t layer_id;
  uint32_t intra_cmd_paral;  // [0]: disable
                             // [1]: enable TDMA/TIU intra-command parallelism
} TdmaL2gTensorCopyCompressedParam;

typedef struct {
  uint16_t constant;
  const TensorGmem *dst;
  uint16_t layer_id;
} TdmaL2gTensorFillConstantParam;

typedef struct {
  const MatrixLmem *src;
  const MatrixGmem *dst;
  uint16_t layer_id;
} TdmaL2gMatrixCopyParam;

typedef struct {
  uint32_t src_address;
  uint8_t dst_base_reg_index;
  uint64_t dst_address;
  uint32_t bytes;
  uint16_t layer_id;
} TdmaL2gGeneralCopyParam;

typedef struct {
  uint32_t src_address;
  uint8_t dst_base_reg_index;
  uint64_t dst_address;
  uint32_t src_bytes;
  KernelFmt src_fmt;
  KernelFmt dst_fmt;
  uint16_t layer_id;
} TdmaL2gBf16GeneralCopyParam;

/*
 * TDMA Engine APIs: GMEM to LMEM (G2L)
 */
typedef struct {
  const TensorGmem *src;
  const TensorLmem *dst;
  uint16_t layer_id;
  uint32_t intra_cmd_paral;  // [0]: disable
                             // [1]: enable TDMA/TIU intra-command parallelism
} TdmaG2lTensorCopyParam;

typedef struct {
  const TensorGmem *src;
  const TensorLmem *dst;
  uint16_t layer_id;
} TdmaG2lTensorCopyNcTransposedParam;

typedef struct {
  const TensorGmem *src;
  const TensorLmem *dst;
  uint16_t layer_id;
} TdmaG2lTensorCopyChwRotatedParam;

typedef struct {
  const CompressedTensorGmem *src;
  const TensorLmem *dst;
  uint16_t layer_id;
  uint32_t intra_cmd_paral;  // [0]: disable
                             // [1]: enable TDMA/TIU intra-command parallelism
} TdmaG2lTensorCopyDecompressedParam;

typedef struct {
  uint16_t constant;
  const TensorLmem *dst;
  uint16_t layer_id;
} TdmaG2lTensorFillConstantParam;

typedef struct {
  const CompressedMatrixGmem *src;
  const MatrixLmem *dst;
  uint16_t layer_id;
} TdmaG2lMatrixCopyDecompressedParam;

typedef struct {
  const MatrixLmem *src;
  const CompressedMatrixGmem *dst;
  uint16_t layer_id;
} TdmaL2gMatrixCopyCompressedParam;

typedef struct {
  const MatrixGmem *src;
  const MatrixLmem *dst;
  uint16_t layer_id;
} TdmaG2lMatrixCopyParam;

typedef struct {
  const MatrixGmem *src;
  const MatrixLmem *dst;
  uint16_t layer_id;
} TdmaG2lMatrixCopyRowColTransposedParam;

typedef struct {
  uint8_t src_base_reg_index;
  uint64_t src_address;
  uint32_t dst_address;
  uint32_t bytes;
  uint16_t layer_id;
} TdmaG2lGeneralCopyParam;

typedef struct {
  uint8_t src_base_reg_index;
  uint64_t src_address;
  uint32_t dst_address;
  uint32_t src_bytes;
  KernelFmt src_fmt;
  KernelFmt dst_fmt;
  uint16_t layer_id;
} TdmaG2lBf16GeneralCopyParam;

/*
 * TDMA Engine APIs: GEM to GEM (G2G)
 */
typedef struct {
  const TensorGmem *src;
  const TensorGmem *dst;
  uint16_t layer_id;
} TdmaG2gTensorCopyParam;

/*
 * TIU Engine APIs
 *
 * General rules for tensor arithmetic APIs:
 *
 * 1, All tensors can be either signed or unsigned
 *    if not mentioned otherwise.
 * 2, A tensor @x with both @x_high and @x_low as
 *    parameters can optionally be 8-bit (when @x_high
 *    is NULL) or 16-bit (otherwise).
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a;
  int b_is_const;
  union {
    const TensorLmem *b;
    struct {
      int16_t val;
      int is_signed;
    } b_const;
  };
  uint8_t rshift_bits;
  int relu_enable;
  uint16_t layer_id;
} MulParam;

// Multiplier in quantization down
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a;
  int b_is_const;
  union {
    const TensorLmem *b;
    struct {
      int8_t val;
      int is_signed;
    } b_const;
  };
  uint8_t rshift_bits;
  int relu_enable;
  uint32_t multiplier;
  uint16_t layer_id;
} MulQmParam;

/*
 * @res = @a * @b + @res
 *
 * 1, @res_high must not be NULL since input @res must be 16-bit.
 * 2, If output @res is 8-bit (@res_is_int8 == 1), only @res_low
 *    is used as output tensor.
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a;
  int b_is_const;
  union {
    const TensorLmem *b;
    struct {
      int16_t val;
      int is_signed;
    } b_const;
  };
  int res_is_int8;
  int relu_enable;
  uint8_t lshift_bits;
  uint8_t rshift_bits;
  uint16_t layer_id;
} MacParam;

/*
 * @a and @b must all be 16-bit.
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a_high;
  const TensorLmem *a_low;
  int b_is_const;
  union {
    struct {
      const TensorLmem *high;
      const TensorLmem *low;
    } b;
    struct {
      int16_t val;
      int is_signed;
    } b_const;
  };
  uint8_t rshift_bits;
  int relu_enable;
  uint16_t layer_id;
} AddParam;

/*
 * 1, @a and @b must all be 16-bit.
 * 2, @res must be signed.
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a_high;
  const TensorLmem *a_low;
  const TensorLmem *b_high;
  const TensorLmem *b_low;
  uint8_t rshift_bits;
  uint16_t layer_id;
} SubParam;

/*
 * @a and @b must both be signed or unsigned.
 */
typedef struct {
  const TensorLmem *max;
  const TensorLmem *a;
  int b_is_const;
  union {
    const TensorLmem *b;
    struct {
      int16_t val;
      int is_signed;
    } b_const;
  };
  uint16_t layer_id;
} MaxParam;

/*
 * @a and @b must both be signed or unsigned.
 */
typedef struct {
  const TensorLmem *min;
  const TensorLmem *a;
  int b_is_const;
  union {
    const TensorLmem *b;
    struct {
      int16_t val;
      int is_signed;
    } b_const;
  };
  uint16_t layer_id;
} MinParam;

/*
 * @a and @b must both be signed or unsigned.
 */
typedef struct {
  const TensorLmem *ge;
  const TensorLmem *a;
  int b_is_const;
  union {
    const TensorLmem *b;
    struct {
      int16_t val;
      int is_signed;
    } b_const;
  };
  uint16_t layer_id;
} GeParam;

/*
 * 1, @a must be 16-bit and signed.
 * 2, @res must be 16-bit.
 * 3, @bits must be signed and must range in [-16, 16].
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a_high;
  const TensorLmem *a_low;
  const TensorLmem *bits;
  uint16_t layer_id;
} ArithShiftParam;

typedef struct {
  const TensorLmem *res;
  const TensorLmem *a;
  const TensorLmem *b;
  uint16_t layer_id;
} AndInt8Param;

/*
 * All parameters must be 16-bit.
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a_high;
  const TensorLmem *a_low;
  const TensorLmem *b_high;
  const TensorLmem *b_low;
} AndInt16Param;

typedef struct {
  const TensorLmem *res;
  const TensorLmem *a;
  const TensorLmem *b;
  uint16_t layer_id;
} OrInt8Param;

/*
 * All parameters must be 16-bit.
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a_high;
  const TensorLmem *a_low;
  const TensorLmem *b_high;
  const TensorLmem *b_low;
} OrInt16Param;

typedef struct {
  const TensorLmem *res;
  const TensorLmem *a;
  const TensorLmem *b;
  uint16_t layer_id;
} XorInt8Param;

/*
 * All parameters must be 16-bit.
 */
typedef struct {
  const TensorLmem *res_high;
  const TensorLmem *res_low;
  const TensorLmem *a_high;
  const TensorLmem *a_low;
  const TensorLmem *b_high;
  const TensorLmem *b_low;
} XorInt16Param;

typedef struct {
  const TensorLmem *src;
  const TensorLmem *dst;
  uint16_t layer_id;
} CopyParam;

/*
 * NOTE:
 *   @table is treated logically as a linear list of
 *   length @table_n, where @table_n is a multiple of
 *   16 and is smaller than or equal to 256.
 *   When stored in local memory, @table is a tensor
 *   of shape (1, npu_num, 1, @table_n), that is, the
 *   data of the linear list should be copied across
 *   each NPU's local memory by user. The behavior when
 *   these copies differ is undefined.
 */
typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  const TensorLmem *table;
  uint16_t layer_id;
} LookupTableParam;

typedef struct {
  const TensorLmem *ifmap;
  const TensorLmem *buf;
  const TensorLmem *tbl_answer;
  const TensorLmem *tbl_answer_mantissa;
  const TensorLmem *ofmap;
  uint16_t layer_id;
  /*
   * \brief
   * we support 2 method of lut depends on \is_scientific:
   * scientific: \tbl_answer_mantissa as mantissa part
   * interpolation: \tbl_answer_mantissa as slope part
   * e.g:
   * interpolation we use activation function to achieve high accuracy
   * scientific uses to calucate reciprocal or sqrt
   * \is_scientific 1 means set scientific, otherwise is interpolation
   */
  uint8_t is_scientific;
  uint8_t eu_align;
  /*
   * for achieving high accuracy, we quant activation function
   * with is constrained by a pair ofhorizontal asymptotes that x->infinity
   * from [-infinity, infinity] to [\min, \max]
   */
  float min;
  float max;
} Bf16LookupInterpTableParam;

/*
 * Convolution weight shape:
 *   Calibration output (oc, ic, kh, kw)
 *   bm_build transforms (oc, ic, kh, kw) -> (1, oc, kh*kw, ic)
 *   TDMA load global (1, oc, kh*w, ic) -> local (1, oc, kh*kw, ic)
 *   TIU conv opd1 (ic, oc, kh, kw)
 *
 * Bias (2, oc, 1, 1)
 *   int8: int16, n=0 [7:0], n=1 [15:8]
 *   bf16: fp32, n=0 [31:16], n=1 [15:0]
 */
typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  const TensorLmem *weight;
  const TensorLmem *bias;
  uint8_t ins_h, ins_last_h;
  uint8_t ins_w, ins_last_w;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  uint8_t dilation_h, dilation_w;
  int relu_enable;
  uint8_t rshift_bits;
  uint8_t ps32_mode;
  uint8_t w_is_const;
  uint16_t layer_id;
  uint8_t fp_round_typ;
  uint8_t cmd_pre_exe_typ;  // tiu execute cmd when channel data is ready
                            // wait type:
                            // 0: activation
                            // 1: weight
  uint8_t cmd_pre_exe;      // tiu execute cmd when channel data is ready
                            // 0: disable
                            // 1: load pre exec
                            // 2: store pre exec
                            // 3: load and store pre exec
  int8_t ins_val;           // padding value for int8
  uint16_t ins_fp;          // padding value for bf16
} PtConvolutionParam;

typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  const TensorLmem *weight;
  const TensorLmem *chl_quan_param;
  uint8_t ins_h, ins_last_h;
  uint8_t ins_w, ins_last_w;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  uint8_t dilation_h, dilation_w;
  uint8_t has_bias;
  uint8_t relu_enable;
  uint8_t ps32_mode;
  uint8_t w_is_const;
  uint16_t layer_id;
  uint8_t cmd_pre_exe_typ;  // tiu execute cmd when channel data is ready
                            // wait type:
                            // 0: activation
                            // 1: weight
  uint8_t cmd_pre_exe;      // tiu execute cmd when channel data is ready
                            // 0: disable
                            // 1: load pre exec
                            // 2: store pre exec
                            // 3: load and store pre exec
  int8_t ins_val;           // padding value for int8
  uint16_t ins_fp;          // padding value for bf16
} ConvolutionParam;

typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  uint16_t kh, kw;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  int8_t ins_val;   // padding value for int8
  uint16_t ins_fp;  // padding value for bf16
  uint16_t layer_id;
} MaxPoolingParam;

typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  uint16_t kh, kw;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  uint16_t ins_fp;
  uint16_t layer_id;
} MinPoolingParam;

typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  uint16_t kh, kw;
  uint8_t ins_h, ins_last_h;
  uint8_t ins_w, ins_last_w;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  uint16_t avg_pooling_const;
  uint8_t rshift_bits;
  uint16_t layer_id;
  int8_t ins_val;   // padding value for int8
  uint16_t ins_fp;  // padding value for bf16
} AveragePoolingParam;

typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  const TensorLmem *weight;
  const TensorLmem *bias;
  int weight_is_const;
  struct {
    int16_t val;
    int is_signed;
  } weight_const;
  uint8_t ins_h, ins_last_h;
  uint8_t ins_w, ins_last_w;
  uint8_t dilation_h, dilation_w;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  uint8_t rshift_bits;
  int relu_enable;
  uint16_t layer_id;
  uint8_t cmd_pre_exe_typ;  // tiu execute cmd when channel data is ready
                            // wait type:
                            // 0: activation
                            // 1: weight
  uint8_t cmd_pre_exe;      // tiu execute cmd when channel data is ready
                            // 0: disable
                            // 1: load pre exec
                            // 2: store pre exec
                            // 3: load and store pre exec
  uint8_t ps32_mode;        // output fp32 result if ps32_mode == 2
  int8_t ins_val;           // padding value for int8
  uint16_t ins_fp;          // padding value for bf16
} DepthwisePtConvolutionParam;

typedef struct {
  const TensorLmem *ofmap;
  const TensorLmem *ifmap;
  const TensorLmem *weight;
  const TensorLmem *chl_quan_param;
  int weight_is_const;
  struct {
    int16_t val;
    int is_signed;
  } weight_const;
  uint8_t ins_h, ins_last_h;
  uint8_t ins_w, ins_last_w;
  uint8_t dilation_h, dilation_w;
  uint8_t pad_top, pad_bottom;
  uint8_t pad_left, pad_right;
  uint8_t stride_h, stride_w;
  uint8_t has_bias;
  uint8_t relu_enable;
  uint16_t layer_id;
  uint8_t cmd_pre_exe_typ;  // tiu execute cmd when channel data is ready
                            // wait type:
                            // 0: activation
                            // 1: weight
  uint8_t cmd_pre_exe;      // tiu execute cmd when channel data is ready
                            // 0: disable
                            // 1: load pre exec
                            // 2: store pre exec
                            // 3: load and store pre exec
  int8_t ins_val;           // padding value for int8
  uint16_t ins_fp;          // padding value for bf16
} DepthwiseConvolutionParam;

typedef struct {
  const MatrixLmem *res;
  const MatrixLmem *left;
  const MatrixLmem *right;
  const MatrixLmem *bias;
  uint8_t lshift_bits;
  uint8_t rshift_bits;
  int res_is_int8;
  int relu_enable;
  int add_result;
  uint8_t ps32_mode;
  uint16_t layer_id;
} MatrixMultiplicationParam;

typedef struct {
  const MatrixLmem *res;
  const MatrixLmem *left;
  const MatrixLmem *right;
  const MatrixLmem *bias;
  uint8_t lshift_bits;
  uint8_t rshift_bits;
  int res_is_int8;
  int relu_enable;
  int add_result;
  uint8_t ps32_mode;
  int32_t quan_m;
  uint16_t layer_id;
} MatrixMultiplicationQmParam;

typedef struct {
  CVI_RT_MEM rtmem = NULL;   // 如果未初始化则设置为NULL
  uint64_t paddr = -1;       // 如果未初始化则设置为uint64_t的最大值
  uint8_t *vaddr = nullptr;  // 如果未初始化则设置为nullptr
} Rinfo;

typedef struct {
  int8_t *ptr;
  uint32_t feature_length;
  uint32_t data_num;
} FeatureArray;

typedef struct {
  FeatureArray feature_array;
  float *feature_unit_length = nullptr;
  float *feature_array_buffer = nullptr;
} CPUFeatureArrayInfo;

typedef struct {
  uint32_t feature_length;
  uint32_t data_num;
  Rinfo feature_input;
  Rinfo feature_array;
  Rinfo buffer_array;
  size_t *slice_num = nullptr;
  float *feature_unit_length = nullptr;
  uint32_t *array_buffer_32 = nullptr;
  float *array_buffer_f = nullptr;
} TPUFeatureArrayInfo;

/*
 * Kernel operations
 */
struct Context;

typedef struct cvikernel_operations {
  void (*cleanup)(struct Context *ctx);
  void (*reset)(struct Context *ctx);
  uint8_t *(*acquire_cmdbuf)(struct Context *ctx, uint32_t *size);
  void (*dmabuf_size)(uint8_t *cmdbuf, uint32_t sz, uint32_t *psize,
                      uint32_t *pmu_size);
  void (*dmabuf_convert)(uint8_t *cmdbuf, uint32_t sz, uint8_t *dmabuf);

  // Concurrent TDMA and TIU command execution:
  //   TDMA command runs without waiting previous TIU command:
  //     1. parallel_disable
  //     2. parallel_enable
  //     3. tiu command
  //     4. tdma command (not wait TIU command)
  //     5. tdma command (not wait TIU command)
  void (*parallel_enable)(struct Context *ctx);
  void (*parallel_disable)(struct Context *ctx);

  void (*set_layer_id)(struct Context *ctx, uint16_t layer_id);

  TensorLmem *(*lmem_alloc_tensor)(struct Context *ctx, TensorLmemShape shape,
                                   KernelFmt fmt, int eu_align);

  MatrixLmem *(*lmem_alloc_matrix)(struct Context *ctx, MatrixLmemShape shape,
                                   KernelFmt fmt, int eu_align);

  MatrixLmem *(*lmem_alloc_ps32_matrix)(struct Context *ctx,
                                        MatrixLmemShape shape, KernelFmt fmt,
                                        int eu_align);

  void (*lmem_free_tensor)(struct Context *ctx, const TensorLmem *tl);

  void (*lmem_free_matrix)(struct Context *ctx, const MatrixLmem *ml);

  void (*lmem_init_tensor)(struct Context *ctx, TensorLmem *tl,
                           TensorLmemShape shape, KernelFmt fmt, int eu_align);

  void (*lmem_init_matrix)(struct Context *ctx, MatrixLmem *ml,
                           MatrixLmemShape shape, KernelFmt fmt, int eu_align);

  TensorLmemStride (*tl_default_stride)(struct Context *ctx,
                                        TensorLmemShape shape, KernelFmt fmt,
                                        int eu_align);

  TensorTgmemStride (*tg_default_stride)(struct Context *ctx,
                                         TensorTgmemShape shape, KernelFmt fmt);

  MatrixLmemShape (*ml_default_shape)(struct Context *ctx, uint32_t row,
                                      uint32_t col, KernelFmt fmt);

  MatrixLmemStride (*ml_default_stride)(struct Context *ctx,
                                        MatrixLmemShape shape, KernelFmt fmt,
                                        int eu_align);

  MatrixLmemShape (*ml_shape_t1)(struct Context *ctx, uint32_t len,
                                 KernelFmt fmt);

  uint32_t (*lmem_tensor_to_size)(struct Context *ctx, TensorLmemShape shape,
                                  KernelFmt fmt, int eu_align);

  uint32_t (*lmem_matrix_to_size)(struct Context *ctx, MatrixLmemShape shape,
                                  KernelFmt fmt, int eu_align);

  uint32_t (*lmem_ps32_matrix_to_size)(struct Context *ctx,
                                       MatrixLmemShape shape, KernelFmt fmt,
                                       int eu_align);

  void (*gmem_init_tensor)(struct Context *ctx, TensorGmem *tg,
                           TensorTgmemShape shape, KernelFmt fmt);

  /* Local to Local DMA API */
  void (*tdma_l2l_tensor_copy)(struct Context *ctx,
                               const TdmaL2lTensorCopyParam *param);
  void (*tdma_l2l_bf16_tensor_copy)(struct Context *ctx,
                                    const TdmaL2lTensorCopyParam *param);
  void (*tdma_l2l_tensor_lrn_shift)(struct Context *ctx,
                                    const TdmaL2lTensorLrnShiftParam *param);

  /* Local to Global DMA API */
  void (*tdma_l2g_tensor_copy)(struct Context *ctx,
                               const TdmaL2gTensorCopyParam *param);
  void (*tdma_l2g_bf16_tensor_copy)(struct Context *ctx,
                                    const TdmaL2gTensorCopyParam *param);
  void (*tdma_l2g_tensor_copy_nc_transposed)(
      struct Context *ctx, const TdmaL2gTensorCopyNcTransposedParam *param);
  void (*tdma_l2g_bf16_tensor_copy_nc_transposed)(
      struct Context *ctx, const TdmaL2gTensorCopyNcTransposedParam *param);
  void (*tdma_l2g_tensor_copy_compressed)(
      struct Context *ctx, const TdmaL2gTensorCopyCompressedParam *param);
  void (*tdma_l2g_tensor_fill_constant)(
      struct Context *ctx, const TdmaL2gTensorFillConstantParam *param);
  void (*tdma_l2g_tensor_copy_cw_transposed)(
      struct Context *ctx, const TdmaL2gTensorCopyCwTransposedParam *param);
  void (*tdma_l2g_bf16_tensor_copy_cw_transposed)(
      struct Context *ctx, const TdmaL2gTensorCopyCwTransposedParam *param);
  void (*tdma_l2g_matrix_copy)(struct Context *ctx,
                               const TdmaL2gMatrixCopyParam *param);
  void (*tdma_l2g_bf16_matrix_copy)(struct Context *ctx,
                                    const TdmaL2gMatrixCopyParam *param);
  void (*tdma_l2g_general_copy)(struct Context *ctx,
                                const TdmaL2gGeneralCopyParam *param);
  void (*tdma_l2g_bf16_general_copy)(struct Context *ctx,
                                     const TdmaL2gBf16GeneralCopyParam *param);

  /* Global to Local DMA API */
  void (*tdma_g2l_tensor_copy)(struct Context *ctx,
                               const TdmaG2lTensorCopyParam *param);
  void (*tdma_g2l_bf16_tensor_copy)(struct Context *ctx,
                                    const TdmaG2lTensorCopyParam *param);
  void (*tdma_g2l_tensor_copy_nc_transposed)(
      struct Context *ctx, const TdmaG2lTensorCopyNcTransposedParam *param);
  void (*tdma_g2l_bf16_tensor_copy_nc_transposed)(
      struct Context *ctx, const TdmaG2lTensorCopyNcTransposedParam *param);
  void (*tdma_g2l_tensor_copy_chw_rotated)(
      struct Context *ctx, const TdmaG2lTensorCopyChwRotatedParam *param);
  void (*tdma_g2l_tensor_copy_decompressed)(
      struct Context *ctx, const TdmaG2lTensorCopyDecompressedParam *param);
  void (*tdma_g2l_tensor_fill_constant)(
      struct Context *ctx, const TdmaG2lTensorFillConstantParam *param);
  void (*tdma_g2l_bf16_tensor_fill_constant)(
      struct Context *ctx, const TdmaG2lTensorFillConstantParam *param);
  void (*tdma_g2l_matrix_copy_decompressed)(
      struct Context *ctx, const TdmaG2lMatrixCopyDecompressedParam *param);
  void (*tdma_l2g_matrix_copy_compressed)(
      struct Context *ctx, const TdmaL2gMatrixCopyCompressedParam *param);
  void (*tdma_g2l_matrix_copy)(struct Context *ctx,
                               const TdmaG2lMatrixCopyParam *param);
  void (*tdma_g2l_bf16_matrix_copy)(struct Context *ctx,
                                    const TdmaG2lMatrixCopyParam *param);
  void (*tdma_g2l_matrix_copy_row_col_transposed)(
      struct Context *ctx, const TdmaG2lMatrixCopyRowColTransposedParam *param);
  void (*tdma_g2l_general_copy)(struct Context *ctx,
                                const TdmaG2lGeneralCopyParam *param);
  void (*tdma_g2l_bf16_general_copy)(struct Context *ctx,
                                     const TdmaG2lBf16GeneralCopyParam *param);

  /* Global to Global DMA API */
  void (*tdma_g2g_tensor_copy)(struct Context *ctx,
                               const TdmaG2gTensorCopyParam *param);
  void (*tdma_g2g_general_copy)(struct Context *ctx,
                                const TdmaG2gTensorCopyParam *param);
  void (*tdma_g2g_bf16_general_copy)(struct Context *ctx,
                                     const TdmaG2gTensorCopyParam *param);
  void (*tdma_g2g_bf16_tensor_copy)(struct Context *ctx,
                                    const TdmaG2gTensorCopyParam *param);

  /* TIU API */
  void (*tiu_mul)(struct Context *ctx, const MulParam *param);
  void (*tiu_mul_qm)(struct Context *ctx, const MulQmParam *param);
  void (*tiu_mac)(struct Context *ctx, const MacParam *param);
  void (*tiu_add)(struct Context *ctx, const AddParam *param);
  void (*tiu_sub)(struct Context *ctx, const SubParam *param);
  void (*tiu_max)(struct Context *ctx, const MaxParam *param);
  void (*tiu_min)(struct Context *ctx, const MinParam *param);
  void (*tiu_and_int8)(struct Context *ctx, const AndInt8Param *param);
  void (*tiu_arith_shift)(struct Context *ctx, const ArithShiftParam *param);
  void (*tiu_and_int16)(struct Context *ctx, const AndInt16Param *param);
  void (*tiu_or_int8)(struct Context *ctx, const OrInt8Param *param);
  void (*tiu_or_int16)(struct Context *ctx, const OrInt16Param *param);
  void (*tiu_xor_int8)(struct Context *ctx, const XorInt8Param *param);
  void (*tiu_xor_int16)(struct Context *ctx, const XorInt16Param *param);
  void (*tiu_copy)(struct Context *ctx, const CopyParam *param);
  void (*tiu_lookup_table)(struct Context *ctx, const LookupTableParam *param);
  void (*tiu_bf16_lookup_interp_table)(struct Context *ctx,
                                       const Bf16LookupInterpTableParam *param);
  void (*tiu_pt_convolution)(struct Context *ctx,
                             const PtConvolutionParam *param);
  void (*tiu_convolution)(struct Context *ctx, const ConvolutionParam *param);
  void (*tiu_max_pooling)(struct Context *ctx, const MaxPoolingParam *param);
  void (*tiu_average_pooling)(struct Context *ctx,
                              const AveragePoolingParam *param);
  void (*tiu_pt_depthwise_convolution)(
      struct Context *ctx, const DepthwisePtConvolutionParam *param);
  void (*tiu_depthwise_convolution)(struct Context *ctx,
                                    const DepthwiseConvolutionParam *param);
  void (*tiu_matrix_multiplication)(struct Context *ctx,
                                    const MatrixMultiplicationParam *param);
  void (*tiu_matrix_multiplication_qm)(
      struct Context *ctx, const MatrixMultiplicationQmParam *param);
  void (*tiu_ge)(struct Context *ctx, const GeParam *param);
  void (*tiu_min_pooling)(struct Context *ctx, const MinPoolingParam *param);
} Operations;

/*
 * Miscellaneous helper function
 *   Not directly related to tiu/tdma operation
 *   or not ready to move into official kernel operation yet.
 */
typedef struct {
  uint16_t (*float_to_bfloat16)(struct Context *ctx, float data);
  void (*bf16_table_shape)(struct Context *ctx, TensorLmemShape *shape);
} MiscOperations;

/*
 * Kernel Context
 */
typedef struct Context {
  ChipInfo info;
  Operations *ops;
  MiscOperations *misc_ops;
  void *priv_data;
} KernelContext;

/*
 * Register information
 */
typedef struct cvikernel_register_info {
  char chip_ver_str[16];
  uint32_t cmdbuf_size;
  uint8_t *cmdbuf;
} RegisterInfo;

KernelContext *cvikernel_register(RegisterInfo *req_info);

#ifdef __cplusplus
}
#endif

#endif /* CVIKERNEL_H */
