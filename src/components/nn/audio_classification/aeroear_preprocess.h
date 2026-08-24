/*
 * aeroear_preprocess.h — Audio feature extraction (C equivalent of Python
 * features.py)
 *
 * C-side audio preprocessing API: STFT -> mel filterbank -> log10.
 * Input: raw PCM 48kHz float32 audio. Output: log-mel spectrogram for TPU
 * inference. Streaming dual ring buffer design: short window 8ms hop, long
 * window 32ms hop.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants (strictly aligned with Python features.py) ──────────────── */
#define AEROEAR_SR 48000
#define AEROEAR_N_MELS 64
#define AEROEAR_FMIN 80.0f
#define AEROEAR_FMAX 16000.0f

/* Short window parameters */
#define AEROEAR_SHORT_NFFT 1536  /* 32ms @ 48kHz */
#define AEROEAR_SHORT_HOP 384    /* 8ms @ 48kHz */
#define AEROEAR_SHORT_NFREQ 769  /* nfft/2+1 */
#define AEROEAR_SHORT_FRAMES 622 /* (240000-1536)/384+1 */

/* Long window parameters */
#define AEROEAR_LONG_NFFT 6144  /* 128ms @ 48kHz */
#define AEROEAR_LONG_HOP 1536   /* 32ms @ 48kHz */
#define AEROEAR_LONG_NFREQ 3073 /* nfft/2+1 */
#define AEROEAR_LONG_FRAMES 153 /* (240000-6144)/1536+1 */

/* Alignment parameters */
#define AEROEAR_ALIGN_OFFSET 9
#define AEROEAR_POOL_SIZE 4

/* TPU input dimensions */
/* Keep 621 (was mistakenly changed to 622, reverted).
 * tpu-mlir trims mel_short from 622 to 621 during conversion —
 * align_short_to_long only uses index 0..620 (slice [9:621]),
 * frame 622 (index 621) is redundant. The actual cvimodel input is
 * <1,64,621,1>. Must match the cvimodel tensor time dimension (621) to avoid
 * (T,C)->(C,T) layout misalignment and out-of-bounds writes. */
#define AEROEAR_TPU_T_SHORT 621
#define AEROEAR_TPU_T_LONG 153

/* 5-second clip sample count */
#define AEROEAR_CLIP_SAMPLES 240000 /* 5s @ 48kHz */

/* ── Ring Buffer (for streaming capture) ─────────────────────────────── */

/*
 * Float ring buffer.
 *
 * buf:      data array pointer — holds audio samples circularly
 * head:     write position — tracks where next sample is written
 * capacity: buffer capacity — prevents overflow
 * count:    current data count — indicates if enough samples are available
 */
typedef struct {
  float *buf;
  int head;
  int capacity;
  int count;
} RingBuffer;

/* Init ring buffer, allocate capacity floats */
void ringbuf_init(RingBuffer *rb, int capacity);
/* Free ring buffer memory */
void ringbuf_free(RingBuffer *rb);
/* Push n samples */
void ringbuf_push(RingBuffer *rb, const float *data, int n);
/* Get last n samples into out (out[0] is oldest, out[n-1] is newest) */
void ringbuf_tail(const RingBuffer *rb, float *out, int n);
/* Get current sample count */
int ringbuf_count(const RingBuffer *rb);

/* ── Preprocessing API ────────────────────────────────────────────────── */

/* STFT preprocessing context — stores window functions etc. */
typedef struct {
  float hann_window_short[AEROEAR_SHORT_NFFT];
  float hann_window_long[AEROEAR_LONG_NFFT];
} AeroEarContext;

/* Init preprocessing context (load Hann windows etc.). Call once at startup. */
void aeroear_context_init(AeroEarContext *ctx);

/*
 * Compute one frame of log-mel spectrogram.
 *
 * Core preprocessing function, equivalent to:
 *   Python: np_log_mel(audio, n_fft, hop)
 *   = stft(audio, n_fft, hop, hann) -> mel filterbank -> log10
 *
 * Args:
 *   ctx:      preprocessing context (contains window functions)
 *   audio:    input PCM samples, length n_fft, 48kHz float32 mono
 *   n_fft:    FFT size (1536 or 6144)
 *   mel_out:  output log-mel spectrogram, length AEROEAR_N_MELS (64)
 *   work_buf: temp work buffer, at least (n_fft + 2 * (n_fft/2+1)) * 4 bytes
 *
 * Notes:
 *   - Causal STFT: no center padding, no lookahead
 *   - Frame t covers samples [t*hop, t*hop + n_fft)
 *   - mel_out contains 64 mel bin log10 values for the current frame
 */
void aeroear_log_mel_frame(AeroEarContext *ctx, const float *audio, int n_fft,
                           float *mel_out, float *work_buf);

/*
 * Apply mel filterbank to power spectrum (internal use only).
 * mel_power[k] = sum_j mel_fb[k][j] * power[j]
 */
void aeroear_mel_filterbank_apply(const float *power, int n_fft,
                                  float *mel_power);

#ifdef __cplusplus
}
#endif
