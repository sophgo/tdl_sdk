#include "blas_npu.hpp"
#include "bmtap2.h"
#include "core/net_loader.hpp"
#include <bmkernel/bm1880/bmkernel_1880.h>
#include <cassert>
#include <libbmruntime/bmruntime.h>

namespace qnn {
namespace math {

using qnn::NetLoader;
using std::vector;

int NBlas::Dot(vector<int8_t> &a, vector<int8_t> &b) {
    assert(a.size() == b.size());

    bmctx_t ctx = NetLoader::Get().GetBmCtxt();
    if (!ctx) {
        assert(false);
        return 0;
    }

    // alloc device memory
    bmshape_t bmshape = BM_TENSOR_INT8(1, 1, 1, int(a.size()));
    bmmem_device_t devmem_a = bmmem_device_alloc(ctx, &bmshape);
    bmmem_device_t devmem_b = bmmem_device_alloc(ctx, &bmshape);
    bmmem_device_t devmem_r_low = bmmem_device_alloc(ctx, &bmshape);
    bmmem_device_t devmem_r_high = bmmem_device_alloc(ctx, &bmshape);

    gaddr_t gaddr_a = bmmem_device_addr(ctx, devmem_a);
    gaddr_t gaddr_b = bmmem_device_addr(ctx, devmem_b);
    gaddr_t gaddr_r_low = bmmem_device_addr(ctx, devmem_r_low);
    gaddr_t gaddr_r_high = bmmem_device_addr(ctx, devmem_r_high);

    // copy to device memory
    bm_memcpy_s2d(ctx, devmem_a, (uint8_t *)a.data());
    bm_memcpy_s2d(ctx, devmem_b, (uint8_t *)b.data());

    // do computation with bmkernel
    bmk1880_context_t *bmk_ctx;
    bmruntime_bmkernel_create(ctx, (void **)&bmk_ctx);
    shape_t shape = shape_t4(1, 1, 1, a.size());
    tensor_lmem *tl_a = bmk1880_tl_alloc(bmk_ctx, shape, FMT_I8, CTRL_AL);
    tensor_lmem *tl_b = bmk1880_tl_alloc(bmk_ctx, shape, FMT_I8, CTRL_AL);
    tensor_lmem *tl_c_low = bmk1880_tl_alloc(bmk_ctx, shape, FMT_I8, CTRL_AL);
    tensor_lmem *tl_c_high = bmk1880_tl_alloc(bmk_ctx, shape, FMT_I8, CTRL_AL);

    bmk1880_gdma_load(bmk_ctx, tl_a, gaddr_a, CTRL_NULL);
    bmk1880_gdma_load(bmk_ctx, tl_b, gaddr_b, CTRL_NULL);
    bmk1880_gdma_load(bmk_ctx, tl_c_low, gaddr_r_low, CTRL_NULL);
    bmk1880_gdma_load(bmk_ctx, tl_c_high, gaddr_r_high, CTRL_NULL);

    bmk1880_mul_param_t p1;
    p1.res_high = NULL;
    p1.res_low = tl_c_low;
    p1.a = tl_a;
    p1.b = tl_b;
    p1.rshift_width = 0;
    bmk1880_tpu_mul(bmk_ctx, &p1);

    bmruntime_bmkernel_submit(ctx);

    vector<int8_t> ret_low(a.size(), 0);
    vector<int8_t> ret_high(a.size(), 0);
    bmk1880_gdma_store(bmk_ctx, tl_c_low, gaddr_r_low, CTRL_NULL);
    bmk1880_gdma_store(bmk_ctx, tl_c_high, gaddr_r_high, CTRL_NULL);
    bmruntime_bmkernel_submit(ctx);
    bm_memcpy_d2s(ctx, (uint8_t *)ret_low.data(), devmem_r_low);
    // bm_memcpy_d2s(ctx, (uint8_t *)ret_high.data(), devmem_r_high);
    bmmem_device_free(ctx, devmem_a);
    bmmem_device_free(ctx, devmem_b);
    bmmem_device_free(ctx, devmem_r_low);
    bmmem_device_free(ctx, devmem_r_high);
    bmruntime_bmkernel_destroy(ctx);

    int sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (ret_low[i] + (ret_high[i] << 8));
    }
    return sum;
}

}  // namespace math
}  // namespace qnn
