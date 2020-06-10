// Copyright 2018 Bitmain Inc.
// License
// Author: tim.zhang@bitmain.com

# prototype
# float VectorDotNEON(const vector<float> &vec1, const vector<float> &vec2);

.global VectorDotNEON
.text

VectorDotNEON:
  movi    v0.4s, #0
  ldp     x4, x5, [x0]
  ldr     x7, [x1]
  sub     x6, x5, x4
  asr     x6, x6, #2
  lsr     x6, x6, #2
  cbz     x6, sum

loop:
  ld1     {v1.4s}, [x4], #16
  ld1     {v2.4s}, [x7], #16
  subs    x6, x6, #1
  fmla    v0.4s, v1.4s, v2.4s
  bne     loop

sum:
  faddp   v0.4s, v0.4s, v0.4s
  faddp   v0.4s, v0.4s, v0.4s

  cmp     x4, x5
  bhs     exit

leftover:
  ldr     s1, [x4], #4
  ldr     s2, [x7], #4
  cmp     x4, x5
  fmadd   s0, s1, s2, s0
  bne     leftover

exit:
  ret

.end
