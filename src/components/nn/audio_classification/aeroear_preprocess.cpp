/*
 * aeroear_preprocess.c — Audio feature extraction implementation
 *
 * C implementation of causal STFT + mel filterbank + log10.
 * Python equivalent: aeroear/features.py np_log_mel() + dsp.py stft_power_mel()
 * Uses aeroear_fft.h for FFT. Hann windows and mel filterbanks are generated at
 * runtime from closed-form formulas (see aeroear_context_init / mel_fb_build),
 * bit-for-bit matching the former auto-generated tables. No external deps (libm
 * only).
 */

#include "aeroear_preprocess.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aeroear_fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Runtime generation of Hann windows and mel filterbanks (replaces the former
 * auto-generated mel_filterbank_*.h / hann_window_*.h tables, ~3.7MB). Values
 * match the old tables to float32 precision.
 *   Hann (periodic): w[n] = 0.5 * (1 - cos(2*pi*n / N))
 *   Mel (HTK scale, triangular, no area normalization):
 *     mel(f) = 2595 * log10(1 + f/700);  hz(m) = 700 * (10^(m/2595) - 1)
 *     fftfreqs   = linspace(0, SR/2, n_freq)
 *     mel points = linspace(mel(FMIN), mel(FMAX), N_MELS+2)
 *     fb[m][k]   = max(0, min((f-lo)/(ce-lo), (hi-f)/(hi-ce)))
 */

/* Mel filterbank matrices, filled once by mel_fb_build (.bss, mirrors the
 * g_*_short/long trig-table style in aeroear_fft.c). */
static float g_mel_fb_short[AEROEAR_N_MELS][AEROEAR_SHORT_NFREQ];
static float g_mel_fb_long[AEROEAR_N_MELS][AEROEAR_LONG_NFREQ];
static int g_mel_fb_inited = 0;

/* HTK mel scale forward / inverse */
static float hz_to_mel_htk(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static float mel_to_hz_htk(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/*
 * Build one mel filterbank matrix.
 *   fb:     output matrix (N_MELS × n_freq), row-major
 *   n_freq: number of frequency bins (nfft/2+1)
 */
static void mel_fb_build(float *fb, int n_freq) {
  const float mel_min = hz_to_mel_htk(AEROEAR_FMIN);
  const float mel_max = hz_to_mel_htk(AEROEAR_FMAX);
  const float nyquist = (float)AEROEAR_SR / 2.0f;

  /* N_MELS+2 band edges, evenly spaced in mel domain, mapped back to Hz */
  float f_pts[AEROEAR_N_MELS + 2];
  for (int i = 0; i < AEROEAR_N_MELS + 2; i++) {
    float mel =
        mel_min + (mel_max - mel_min) * (float)i / (float)(AEROEAR_N_MELS + 1);
    f_pts[i] = mel_to_hz_htk(mel);
  }

  for (int m = 0; m < AEROEAR_N_MELS; m++) {
    float lo = f_pts[m], ce = f_pts[m + 1], hi = f_pts[m + 2];
    float *row = fb + m * n_freq;
    for (int k = 0; k < n_freq; k++) {
      /* fftfreqs[k] = linspace(0, SR/2, n_freq) */
      float freq = nyquist * (float)k / (float)(n_freq - 1);
      float left = (freq - lo) / (ce - lo);
      float right = (hi - freq) / (hi - ce);
      float v = left < right ? left : right;
      row[k] = v > 0.0f ? v : 0.0f;
    }
  }
}

/* ── Ring Buffer implementation ────────────────────────────────────────── */

void ringbuf_init(RingBuffer *rb, int capacity) {
  rb->buf = (float *)calloc(capacity, sizeof(float));
  rb->head = 0;
  rb->capacity = capacity;
  rb->count = 0;
}

void ringbuf_free(RingBuffer *rb) {
  free(rb->buf);
  rb->buf = NULL;
  rb->capacity = 0;
}

void ringbuf_push(RingBuffer *rb, const float *data, int n) {
  for (int i = 0; i < n; i++) {
    rb->buf[rb->head] = data[i];
    rb->head = (rb->head + 1) % rb->capacity;
  }
  rb->count += n;
  if (rb->count > rb->capacity) {
    rb->count = rb->capacity;
  }
}

void ringbuf_tail(const RingBuffer *rb, float *out, int n) {
  if (n > rb->count) n = rb->count;
  int start = (rb->head - n + rb->capacity) % rb->capacity;
  for (int i = 0; i < n; i++) {
    out[i] = rb->buf[(start + i) % rb->capacity];
  }
}

int ringbuf_count(const RingBuffer *rb) { return rb->count; }

/* ── Preprocessing context ─────────────────────────────────────────────── */

void aeroear_context_init(AeroEarContext *ctx) {
  /* Hann (periodic) windows: w[n] = 0.5 * (1 - cos(2*pi*n / N)) */
  for (int n = 0; n < AEROEAR_SHORT_NFFT; n++) {
    ctx->hann_window_short[n] =
        0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)n /
                            (float)AEROEAR_SHORT_NFFT));
  }
  for (int n = 0; n < AEROEAR_LONG_NFFT; n++) {
    ctx->hann_window_long[n] =
        0.5f *
        (1.0f - cosf(2.0f * (float)M_PI * (float)n / (float)AEROEAR_LONG_NFFT));
  }

  /* Build mel filterbanks once (see mel_fb_build) */
  if (!g_mel_fb_inited) {
    mel_fb_build((float *)g_mel_fb_short, AEROEAR_SHORT_NFREQ);
    mel_fb_build((float *)g_mel_fb_long, AEROEAR_LONG_NFREQ);
    g_mel_fb_inited = 1;
  }

  /* Precompute FFT trig tables via aeroear_fft_init to eliminate
   * ~3.8M sinf/cosf calls per window in the inference hot path. */
  aeroear_fft_init();
}

/* ── Mel filterbank application ────────────────────────────────────────── */

void aeroear_mel_filterbank_apply(const float *power, int n_fft,
                                  float *mel_power) {
  /*
   * power:     linear power spectrum (n_freq,) — input to mel filterbank
   * n_fft:    FFT size — selects the correct filterbank matrix
   * mel_power: output mel power (64,) — result of matrix multiply
   *
   * Equivalent to: mel_power = mel_fb @ power  (64×n_freq @ n_freq×1 -> 64×1)
   */
  const float *mel_fb;
  int n_freq;

  if (n_fft == AEROEAR_SHORT_NFFT) {
    mel_fb = (const float *)g_mel_fb_short;
    n_freq = AEROEAR_SHORT_NFREQ;
  } else if (n_fft == AEROEAR_LONG_NFFT) {
    mel_fb = (const float *)g_mel_fb_long;
    n_freq = AEROEAR_LONG_NFREQ;
  } else {
    memset(mel_power, 0, AEROEAR_N_MELS * sizeof(float));
    return;
  }

  /* Matrix-vector multiply: mel_fb (N_MELS × n_freq) @ power (n_freq) */
  for (int m = 0; m < AEROEAR_N_MELS; m++) {
    float acc = 0.0f;
    const float *row = mel_fb + m * n_freq;
    for (int f = 0; f < n_freq; f++) {
      acc += row[f] * power[f];
    }
    mel_power[m] = acc;
  }
}

/* ── Single-frame log-mel computation ──────────────────────────────────── */

void aeroear_log_mel_frame(AeroEarContext *ctx, const float *audio, int n_fft,
                           float *mel_out, float *work_buf) {
  /*
   * Causal STFT one frame:
   *   1. Apply Hann window: windowed[t] = audio[t] * hann[t]
   *   2. FFT: rfft(windowed) -> real[0..nf], imag[0..nf]
   *   3. Power spectrum: power[f] = real[f]^2 + imag[f]^2
   *   4. Mel filterbank: mel_power[m] = sum_f mel_fb[m][f] * power[f]
   *   5. log10: mel_out[m] = log10(mel_power[m] + 1e-10)
   *
   * Temp buffer layout:
   *   work_buf[0..n_fft-1]               = windowed audio
   *   work_buf[n_fft..n_fft+nf-1]        = real (FFT output)
   *   work_buf[n_fft+nf..n_fft+2*nf-1]   = imag (FFT output)
   *   work_buf[n_fft+2*nf..n_fft+3*nf-1] = power (power spectrum)
   *
   * audio:    input PCM samples 48kHz float32 — raw audio frame
   * n_fft:    FFT size — selects window/FFT size
   * mel_out:  output log-mel (64,) — input to TPU CNN model
   * work_buf: temp buffer — avoids malloc in inference hot path
   *
   * window length = n_fft
   * n_freq = n_fft / 2 + 1 (e.g. 1536 -> 769, 6144 -> 3073)
   */
  int n_freq = n_fft / 2 + 1;
  const float *hann = (n_fft == AEROEAR_SHORT_NFFT) ? ctx->hann_window_short
                                                    : ctx->hann_window_long;

  float *windowed = work_buf;          /* n_fft floats */
  float *fft_real = work_buf + n_fft;  /* n_freq floats */
  float *fft_imag = fft_real + n_freq; /* n_freq floats */
  float *power = fft_imag + n_freq;    /* n_freq floats */

  /*
   * Step 1: Apply Hann window
   *   Python: windowed = frames * win (broadcasting)
   *   frames[t] = audio[t*hop : t*hop+n_fft]
   *   (audio is already a single frame, no hop slicing needed)
   *
   * windowed: windowed signal — reduces spectral leakage
   * hann:     Hann window coefficients — precomputed from checkpoint
   */
  for (int i = 0; i < n_fft; i++) {
    windowed[i] = audio[i] * hann[i];
  }

  /*
   * Step 2: Real FFT
   *   Python: spec = np.fft.rfft(windowed)
   *   Output: real + j*imag, n_freq bins (0Hz -> sr/2)
   *
   * fft_real/fft_imag: complex spectrum — intermediate for power computation
   */
  aeroear_rfft(windowed, n_fft, fft_real, fft_imag);

  /*
   * Step 3: Power spectrum
   *   Python: spec.real**2 + spec.imag**2
   *
   * power: linear power spectrum — input to mel filterbank
   */
  aeroear_power_spectrum(fft_real, fft_imag, power, n_freq);

  /*
   * Step 4: Mel filterbank
   *   Python: mel_fb @ power
   *   output: mel_power[0..63], linear power of 64 mel bins
   */
  aeroear_mel_filterbank_apply(power, n_fft, mel_out);

  /*
   * Step 5: log10
   *   Python: np.log10(mel_power + 1e-10)
   *
   * mel_out: initially holds mel_power (linear), converted in-place to log
   * domain via log10f eps=1e-10: prevents log10(0) = -inf
   */
  for (int m = 0; m < AEROEAR_N_MELS; m++) {
    mel_out[m] = log10f(mel_out[m] + 1e-10f);
  }
}
