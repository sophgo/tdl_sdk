/*
 * aeroear_gru.c — CPU-side GRU + detection heads implementation
 *
 * Implements CPU-side single-step GRU inference + snr/act detection heads
 * for streaming state carry. Weights from aeroear_gru_weights.h (fp32).
 * Gate order [r, z, n], numerically verified against PyTorch nn.GRU.
 * Only depends on libm (expf/tanhf).
 */

#include "aeroear_gru.h"
#include <math.h>
#include <string.h>
#include "aeroear_gru_weights.h" /* AEROEAR_GRU_* macros + weight arrays */

/* Gate block size = hidden = 192 */
#define GRU_H (AEROEAR_GRU_HIDDEN)
/* 3-gate total rows = 3 * 192 = 576 */
#define GRU_3H (AEROEAR_GRU_GATES * AEROEAR_GRU_HIDDEN)

static inline float sigmoid(float x) {
  /* Numerically stable sigmoid */
  if (x >= 0.0f) {
    float e = expf(-x);
    return 1.0f / (1.0f + e);
  } else {
    float e = expf(x);
    return e / (1.0f + e);
  }
}

void aeroear_gru_init(AeroEarGRU *g) { memset(g->h, 0, GRU_H * sizeof(float)); }

void aeroear_gru_step(AeroEarGRU *g, const float *x, float *snr, float *act) {
  /*
   * g:     read old h, write new h — carries GRU hidden state across frames
   * x:     256-dim proj feature — GRU input
   * snr/act: output current-frame detection head results — used for
   * thresholding
   *
   * Temp arrays gi/gh (3h=576) are stack-allocated to avoid malloc in hot path.
   */
  float gi[GRU_3H];
  float gh[GRU_3H];

  /* 1. Input projection: gi = W_ih @ x + b_ih (576 = 256-dim input) */
  for (int i = 0; i < GRU_3H; i++) {
    float acc = gru_b_ih[i];
    for (int j = 0; j < AEROEAR_GRU_INPUT; j++) {
      acc += gru_w_ih[i][j] * x[j];
    }
    gi[i] = acc;
  }

  /* 2. Recurrent projection: gh = W_hh @ h + b_hh (576 = 192-dim hidden) */
  for (int i = 0; i < GRU_3H; i++) {
    float acc = gru_b_hh[i];
    for (int j = 0; j < GRU_H; j++) {
      acc += gru_w_hh[i][j] * g->h[j];
    }
    gh[i] = acc;
  }

  /*
   * 3. Gate computation: blocks [0,192)=r(reset), [192,384)=z(update),
   * [384,576)=n(new) r = sigmoid(gi_r + gh_r) z = sigmoid(gi_z + gh_z) n =
   * tanh(gi_n + r * gh_n) h' = (1-z)*n + z*h
   */
  for (int j = 0; j < GRU_H; j++) {
    float r = sigmoid(gi[j] + gh[j]);                 /* reset gate */
    float z = sigmoid(gi[GRU_H + j] + gh[GRU_H + j]); /* update gate */
    float n = tanhf(gi[2 * GRU_H + j] + r * gh[2 * GRU_H + j]); /* new */
    float h_prev = g->h[j];
    g->h[j] = (1.0f - z) * n + z * h_prev;
  }

  /* 4. Detection heads: snr = snr_w @ h + snr_b; act = sigmoid(act_w @ h +
   * act_b) */
  float s = snr_b[0];
  float a = act_b[0];
  for (int j = 0; j < GRU_H; j++) {
    s += snr_w[j] * g->h[j];
    a += act_w[j] * g->h[j];
  }
  *snr = s;
  *act = sigmoid(a);
}
