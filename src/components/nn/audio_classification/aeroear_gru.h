/*
 * aeroear_gru.h — CPU-side GRU + detection heads (streaming state carry)
 *
 * TPU only runs the encoder (CNN + proj -> 256-dim); GRU(192) + snr/act head
 * runs on CPU, carrying hidden state across 32ms frames.
 * Weights from aeroear_gru_weights.h (fp32, exported by export_gru_weights.py).
 * Gate order [r, z, n] (PyTorch semantics), numerically verified against
 * nn.GRU. ~258K MAC per frame, 32ms/frame ≈ 8 MMAC/s — well within C906 budget.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Dimension constants (match macros in aeroear_gru_weights.h; #ifndef to avoid
 * redefinition) */
#ifndef AEROEAR_GRU_INPUT
#define AEROEAR_GRU_INPUT 256 /* encoder proj output dim == GRU input_size */
#endif
#ifndef AEROEAR_GRU_HIDDEN
#define AEROEAR_GRU_HIDDEN 192 /* GRU hidden_size */
#endif
#ifndef AEROEAR_GRU_GATES
#define AEROEAR_GRU_GATES 3 /* reset, update, new */
#endif

/*
 * GRU inference state.
 * h: hidden state (192,), used to carry state across 32ms frames.
 */
typedef struct {
  float h[AEROEAR_GRU_HIDDEN];
} AeroEarGRU;

/* Init GRU state: zero h (matches training/inference h0=0). */
void aeroear_gru_init(AeroEarGRU *g);

/*
 * Single GRU step + detection heads.
 *
 * Formulas (PyTorch semantics, gate order [r, z, n]):
 *   gi = W_ih @ x + b_ih;  gh = W_hh @ h + b_hh
 *   r = sigmoid(gi_r + gh_r);  z = sigmoid(gi_z + gh_z)
 *   n = tanh(gi_n + r * gh_n); h' = (1-z)*n + z*h
 *   snr = snr_w @ h' + snr_b;  act = sigmoid(act_w @ h' + act_b)
 *
 * g:   GRU state (read h, write new h), carries hidden state across calls
 * x:   encoder 256-dim proj feature, used as GRU input at current frame
 * snr: output SNR estimate (dB), used for drone detection thresholding
 * act: output activity probability [0,1], used for drone detection thresholding
 */
void aeroear_gru_step(AeroEarGRU *g, const float *x, float *snr, float *act);

#ifdef __cplusplus
}
#endif
