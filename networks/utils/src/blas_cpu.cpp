#include "blas_cpu.hpp"
#include <cassert>
#include <cblas.h>

namespace qnn {
namespace math {

using std::vector;

float CBlas::Dot(vector<float> &a, vector<float> &b) {
    assert(a.size() == b.size());

    int M = 1;         // a's rows, c's rows
    int N = 1;         // b's cols, c's cols
    int K = a.size();  // a's cols, b's rows
    int lda = K;       // a's cols
    int ldb = N;       // b's cols
    int ldc = N;       // c's cols

    float ret;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a.data(), lda, b.data(),
                ldb, 0.0f, &ret, ldc);
    return ret;
}

}  // namespace math
}  // namespace qnn
