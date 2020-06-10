// Copyright 2018 Bitmain Inc.
// License
// Author: tim.zhang@bitmain.com

# prototype
# float VectorNormNEON(const vector<float> &vec);


.global VectorNormNEON
.text

VectorNormNEON:
  movi    v0.4s, #0x0
  ldp     x4, x5, [x0]
  sub     x6, x5, x4
  asr     x6, x6, #2
  lsr     x6, x6, #2
  cbz     x6, sum

loop:
  ld1     {v1.4s}, [x4], #16
  subs    x6, x6, #1
  fmla    v0.4s, v1.4s, v1.4s
  bne     loop

sum:
  faddp   v0.4s, v0.4s, v0.4s
  faddp   v0.4s, v0.4s, v0.4s

  cmp     x4, x5
  bhs     exit

leftover:
  ldr     s1, [x4], #4
  cmp     x4, x5
  fmadd   s0, s1, s1, s0
  bne     leftover

exit:
  fsqrt   s0, s0
  ret

.end
