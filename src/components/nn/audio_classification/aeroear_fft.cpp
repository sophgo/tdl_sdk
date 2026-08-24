/*
 * aeroear_fft.c — Mixed-radix real FFT implementation
 *
 * Implements Cooley-Tukey decomposition + radix-2 DIT FFT for N=3×2^k real
 * sequences. aeroear_fft_init() precomputes trig tables so the merge step never
 * calls sinf/cosf. Python equivalent: numpy.fft.rfft(windowed, axis=1)
 *
 * Only supports n=3×2^k (1536/6144). No dynamic allocation — all
 * stack-allocated. Precomputed tables: 769×4 + 3073×4 = 15,368 floats ≈ 60 KB.
 * Eliminates ~3.8M sinf/cosf calls per window, saving ~1-2s on RISC-V.
 */

#include "aeroear_fft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* Precomputed trig table sizes */
#define FFT_TABLE_SHORT_NF 769 /* 1536/2+1 */
#define FFT_TABLE_LONG_NF 3073 /* 6144/2+1 */

/*
 * Precomputed trig tables (filled in aeroear_fft_init).
 * Merge formula: X[k] = X0[k_mod] + W_N^k · X1[k_mod] + W_N^{2k} · X2[k_mod]
 * w1 = cos(-2πk/N) + j·sin(-2πk/N) = W_N^k
 * w2 = cos(-4πk/N) + j·sin(-4πk/N) = W_N^{2k}
 *
 * w1r/w1i: real/imag of W_N^k, used to rotate X1 term in merge
 * w2r/w2i: real/imag of W_N^{2k}, used to rotate X2 term in merge
 */
static float g_w1r_short[FFT_TABLE_SHORT_NF];
static float g_w1i_short[FFT_TABLE_SHORT_NF];
static float g_w2r_short[FFT_TABLE_SHORT_NF];
static float g_w2i_short[FFT_TABLE_SHORT_NF];

static float g_w1r_long[FFT_TABLE_LONG_NF];
static float g_w1i_long[FFT_TABLE_LONG_NF];
static float g_w2r_long[FFT_TABLE_LONG_NF];
static float g_w2i_long[FFT_TABLE_LONG_NF];

static int g_fft_table_inited = 0;

void aeroear_fft_init(void) {
  if (g_fft_table_inited) return;

  /* Short window (n=1536, nf=769) */
  for (int k = 0; k < FFT_TABLE_SHORT_NF; k++) {
    float angle1 = -2.0f * M_PI * k / 1536.0f;
    float angle2 = 2.0f * angle1;
    g_w1r_short[k] = cosf(angle1);
    g_w1i_short[k] = sinf(angle1);
    g_w2r_short[k] = cosf(angle2);
    g_w2i_short[k] = sinf(angle2);
  }

  /* Long window (n=6144, nf=3073) */
  for (int k = 0; k < FFT_TABLE_LONG_NF; k++) {
    float angle1 = -2.0f * M_PI * k / 6144.0f;
    float angle2 = 2.0f * angle1;
    g_w1r_long[k] = cosf(angle1);
    g_w1i_long[k] = sinf(angle1);
    g_w2r_long[k] = cosf(angle2);
    g_w2i_long[k] = sinf(angle2);
  }

  g_fft_table_inited = 1;
}

/* Check if n = 3 × 2^k */
static int is_3x2k(int n, int *k) {
  if (n % 3 != 0) return 0;
  int m = n / 3;
  int p = 0;
  while (m > 1) {
    if (m % 2 != 0) return 0;
    m /= 2;
    p++;
  }
  *k = p;
  return 1;
}

/*
 * Radix-2 DIT (Decimation In Time) complex FFT.
 * x_real, x_imag: input/output (in-place), length n (must be power of 2).
 */
static void fft_radix2_c2c(float *x_real, float *x_imag, int n) {
  /* Bit-reversal permutation */
  int j = 0;
  for (int i = 0; i < n; i++) {
    if (i < j) {
      float tr = x_real[i], ti = x_imag[i];
      x_real[i] = x_real[j];
      x_imag[i] = x_imag[j];
      x_real[j] = tr;
      x_imag[j] = ti;
    }
    int m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  /* Butterfly */
  for (int s = 1; s < n; s <<= 1) {
    int m = s << 1;
    float wm_real = (float)cos(M_PI / s);
    float wm_imag = (float)-sin(M_PI / s);
    for (int k = 0; k < n; k += m) {
      float w_real = 1.0f, w_imag = 0.0f;
      for (int i = 0; i < s; i++) {
        int a = k + i, b = k + i + s;
        float tr = w_real * x_real[b] - w_imag * x_imag[b];
        float ti = w_real * x_imag[b] + w_imag * x_real[b];
        x_real[b] = x_real[a] - tr;
        x_imag[b] = x_imag[a] - ti;
        x_real[a] = x_real[a] + tr;
        x_imag[a] = x_imag[a] + ti;
        /* Update twiddle factor: w *= wm */
        float wr = w_real * wm_real - w_imag * wm_imag;
        float wi = w_real * wm_imag + w_imag * wm_real;
        w_real = wr;
        w_imag = wi;
      }
    }
  }
}

void aeroear_rfft(const float *x, int n, float *real_out, float *imag_out) {
  int k;
  if (!is_3x2k(n, &k)) {
    /* Unsupported size — zero output */
    int nf = n / 2 + 1;
    memset(real_out, 0, nf * sizeof(float));
    memset(imag_out, 0, nf * sizeof(float));
    return;
  }
  int m = n / 3; /* 2^k */
  int nf = n / 2 + 1;

  /*
   * Stack-allocated temp buffers:
   *   3 complex sequences, each m points = 3 × 2 × m floats = 2n floats
   *   Short window n=1536 -> 3072 floats ≈ 12 KB
   *   Long window n=6144 -> 12288 floats ≈ 48 KB
   *   Safe on stack (Linux default stack 8MB).
   */
  float r0[m], r1[m], r2[m];
  float i0[m], i1[m], i2[m];

  /* Step 1: decimate by mod 3, init imag to 0 */
  for (int t = 0; t < m; t++) {
    r0[t] = x[3 * t];
    r1[t] = x[3 * t + 1];
    r2[t] = x[3 * t + 2];
    i0[t] = 0.0f;
    i1[t] = 0.0f;
    i2[t] = 0.0f;
  }

  /* Step 2: three independent radix-2 FFTs */
  fft_radix2_c2c(r0, i0, m);
  fft_radix2_c2c(r1, i1, m);
  fft_radix2_c2c(r2, i2, m);

  /* Step 3: merge X[k] = X0[k_mod] + W_N^k · X1[k_mod] + W_N^{2k} · X2[k_mod]
   *   where k_mod = k % M (since X0 has period M).
   *   Only need k = 0..n/2.
   *   Uses precomputed trig tables instead of real-time sinf/cosf,
   *   eliminating ~3.8M sinf/cosf calls per window.
   */
  /* Select the precomputed table for this size */
  const float *w1r_tbl, *w1i_tbl, *w2r_tbl, *w2i_tbl;
  if (n == 1536) {
    w1r_tbl = g_w1r_short;
    w1i_tbl = g_w1i_short;
    w2r_tbl = g_w2r_short;
    w2i_tbl = g_w2i_short;
  } else {
    w1r_tbl = g_w1r_long;
    w1i_tbl = g_w1i_long;
    w2r_tbl = g_w2r_long;
    w2i_tbl = g_w2i_long;
  }

  for (int k_idx = 0; k_idx < nf; k_idx++) {
    int km = k_idx % m;

    float w1r = w1r_tbl[k_idx], w1i = w1i_tbl[k_idx];
    float w2r = w2r_tbl[k_idx], w2i = w2i_tbl[k_idx];

    /* X1[k_mod] * W_N^k */
    float t1r = w1r * r1[km] - w1i * i1[km];
    float t1i = w1r * i1[km] + w1i * r1[km];

    /* X2[k_mod] * W_N^{2k} */
    float t2r = w2r * r2[km] - w2i * i2[km];
    float t2i = w2r * i2[km] + w2i * r2[km];

    real_out[k_idx] = r0[km] + t1r + t2r;
    imag_out[k_idx] = i0[km] + t1i + t2i;
  }
}

void aeroear_power_spectrum(const float *real, const float *imag, float *power,
                            int n_freq) {
  for (int i = 0; i < n_freq; i++) {
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }
}
