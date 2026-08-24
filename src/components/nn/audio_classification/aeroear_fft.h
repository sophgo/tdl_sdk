/*
 * aeroear_fft.h — Mixed-radix real FFT (optimized for 1536/6144)
 *
 * Implements Cooley-Tukey decomposition + radix-2 DIT FFT for N=3×2^k real
 * sequences. aeroear_fft_init() precomputes trig tables to avoid sinf/cosf in
 * the hot path. Python equivalent: numpy.fft.rfft(windowed, axis=1)
 *
 * Supports n_fft=1536 (3×512) and n_fft=6144 (3×2048).
 * Input: real array. Output: complex spectrum (n_fft/2+1 bins).
 * Precomputed trig tables eliminate ~3.8M sinf/cosf calls per window.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Precompute FFT merge-step trig tables.
 * Call once at module init; eliminates ~3.8M sinf/cosf calls per window in the
 * hot path. For 1536-point FFT: 769×4 floats. For 6144-point FFT: 3073×4
 * floats.
 */
void aeroear_fft_init(void);

/*
 * Compute mixed-radix FFT of a real sequence.
 *
 * Algorithm: Cooley-Tukey decomposition + radix-2 DIT FFT
 *   N = 3 × M (M is a power of 2)
 *   Step 1: decimate x[0..N-1] by mod 3 into 3 subsequences of length M
 *   Step 2: radix-2 complex FFT on each subsequence
 *   Step 3: merge X[k] = X0[k] + W_N^k · X1[k] + W_N^{2k} · X2[k]
 *
 * Args:
 *   x:        input real array, length n
 *   n:        FFT size, only 3×2^k form (e.g. 1536=3×512, 6144=3×2048)
 *   real_out: output real part, length n/2+1
 *   imag_out: output imaginary part, length n/2+1
 *
 * Internally allocates temp buffers on stack; caller may reuse.
 */
void aeroear_rfft(const float *x, int n, float *real_out, float *imag_out);

/*
 * Compute power spectrum: power[k] = real[k]^2 + imag[k]^2
 *
 * Args:
 *   real:    FFT real output, length n_freq (n_fft/2+1)
 *   imag:    FFT imag output, length n_freq
 *   power:   output power spectrum, length n_freq
 *   n_freq:  number of frequency bins
 */
void aeroear_power_spectrum(const float *real, const float *imag, float *power,
                            int n_freq);

#ifdef __cplusplus
}
#endif
