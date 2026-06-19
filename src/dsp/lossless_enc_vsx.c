// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC64) version of the lossless encoding functions.
//
// Authors: Trung Lê (8@tle.id.au)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector signed char i8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector signed int i32x4;
typedef __vector unsigned long long u64x2;

// Equivalent of _mm_movemask_epi8: bit i = the MSB of byte i.
static WEBP_INLINE uint32_t MoveMask8(u8x16 v) {
  const u8x16 perm = {120, 112, 104, 96, 88, 80, 72, 64,
                      56,  48,  40,  32, 24, 16, 8,  0};
  return (uint32_t)((u64x2)vec_vbpermq(v, perm))[1];
}

// Signed multiply-high of 16-bit lanes: (a * b) >> 16, matching
// _mm_mulhi_epi16.
static WEBP_INLINE i16x8 MulHiS16(i16x8 a, i16x8 b) {
  const u32x4 sh = vec_splats((unsigned int)16);
  const i32x4 e = vec_sra(vec_mule(a, b), sh);
  const i32x4 o = vec_sra(vec_mulo(a, b), sh);
  return (i16x8)vec_pack(vec_mergeh(e, o), vec_mergel(e, o));
}

//------------------------------------------------------------------------------
// Subtract-Green transform.

static void SubtractGreenFromBlueAndRed_VSX(uint32_t* argb_data,
                                            int num_pixels) {
  const u8x16 zero = vec_splats((unsigned char)0);
  // Replicate the green byte (offset 1 of each pixel) into the blue/red slots.
  const u8x16 kSpreadGreen = {1, 16, 1, 16, 5,  16, 5,  16,
                              9, 16, 9, 16, 13, 16, 13, 16};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&argb_data[i]);
    const u8x16 g = vec_perm(in, zero, kSpreadGreen);  // 0 g 0 g per pixel
    vec_xst((u32x4)vec_sub(in, g), 0, &argb_data[i]);
  }
  if (i != num_pixels) {
    VP8LSubtractGreenFromBlueAndRed_C(argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
// Color transform.

static void TransformColor_VSX(const VP8LMultipliers* WEBP_RESTRICT const m,
                               uint32_t* WEBP_RESTRICT argb_data,
                               int num_pixels) {
// sign-extended multiplying constants, pre-shifted by 5 (see lossless_sse2.c).
#define CST(X) (((int16_t)((uint16_t)(m->X) << 8)) >> 5)
  const i16x8 mults_rb =
      (i16x8)vec_splats((int)(((uint32_t)(uint16_t)CST(green_to_red) << 16) |
                              ((uint16_t)CST(green_to_blue))));
  const i16x8 mults_b2 =
      (i16x8)vec_splats((int)((uint32_t)(uint16_t)CST(red_to_blue) << 16));
#undef CST
  const u8x16 zero = vec_splats((unsigned char)0);
  const u32x4 mask_ag = vec_splats((uint32_t)0xff00ff00);  // alpha/green
  const u32x4 mask_rb = vec_splats((uint32_t)0x00ff00ff);  // red/blue
  const u16x8 sh8_16 = vec_splats((unsigned short)8);
  const u32x4 sh16_32 = vec_splats((unsigned int)16);
  // Broadcast the green byte (offset 1) into the high byte of both 16-bit
  // halves of each pixel: yields g << 8 in each lane.
  const u8x16 kGreenHi = {16, 1, 16, 1, 16, 5,  16, 5,
                          16, 9, 16, 9, 16, 13, 16, 13};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&argb_data[i]);
    const u8x16 A = (u8x16)vec_and((u32x4)in, mask_ag);   // a 0 g 0
    const i16x8 C = (i16x8)vec_perm(A, zero, kGreenHi);   // g0g0 (g << 8)
    const u8x16 D = (u8x16)MulHiS16(C, mults_rb);         // x dr x db1
    const u16x8 E = vec_sl((u16x8)in, sh8_16);            // r 0 b 0
    const u8x16 F = (u8x16)MulHiS16((i16x8)E, mults_b2);  // x db2 0 0
    const u8x16 G = (u8x16)vec_sr((u32x4)F, sh16_32);     // 0 0 x db2
    const u8x16 H = vec_add(G, D);                        // x dr x db
    const u8x16 I = (u8x16)vec_and((u32x4)H, mask_rb);    // 0 dr 0 db
    vec_xst((u32x4)vec_sub(in, I), 0, &argb_data[i]);
  }
  if (i != num_pixels) {
    VP8LTransformColor_C(m, argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
// Batch predictor-transform subtraction. The forward predictors read only
// input pixels, so all four pixels of a group are computed in parallel.

// Per-byte floor average (a + b) >> 1, matching the C Average2().
static WEBP_INLINE u8x16 Average2_u8(u8x16 a, u8x16 b) {
  const u8x16 one = vec_splats((unsigned char)1);
  const u8x16 avg1 = vec_avg(a, b);  // (a + b + 1) >> 1
  return vec_sub(avg1, vec_and(vec_xor(a, b), one));
}

static void PredictorSub0_VSX(const uint32_t* in, const uint32_t* upper,
                              int num_pixels, uint32_t* WEBP_RESTRICT out) {
  const u8x16 black = (u8x16)vec_splats((uint32_t)ARGB_BLACK);
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    vec_xst((u32x4)vec_sub(src, black), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

#define GENERATE_PREDICTOR_SUB_1_VSX(X, IN)                                    \
  static void PredictorSub##X##_VSX(const uint32_t* in, const uint32_t* upper, \
                                    int num_pixels,                            \
                                    uint32_t* WEBP_RESTRICT out) {             \
    int i;                                                                     \
    for (i = 0; i + 4 <= num_pixels; i += 4) {                                 \
      const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);                   \
      const u8x16 pred = (u8x16)vec_xl(0, (uint32_t*)&(IN));                   \
      vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);                          \
    }                                                                          \
    if (i != num_pixels) {                                                     \
      VP8LPredictorsSub_C[(X)](in + i, WEBP_OFFSET_PTR(upper, i),              \
                               num_pixels - i, out + i);                       \
    }                                                                          \
  }
GENERATE_PREDICTOR_SUB_1_VSX(1, in[i - 1])     // L
GENERATE_PREDICTOR_SUB_1_VSX(2, upper[i])      // T
GENERATE_PREDICTOR_SUB_1_VSX(3, upper[i + 1])  // TR
GENERATE_PREDICTOR_SUB_1_VSX(4, upper[i - 1])  // TL
#undef GENERATE_PREDICTOR_SUB_1_VSX

// Predictor5: avg2(avg2(L, TR), T).
static void PredictorSub5_VSX(const uint32_t* in, const uint32_t* upper,
                              int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 L = (u8x16)vec_xl(0, (uint32_t*)&in[i - 1]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TR = (u8x16)vec_xl(0, (uint32_t*)&upper[i + 1]);
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u8x16 pred = Average2_u8(Average2_u8(L, TR), T);
    vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[5](in + i, upper + i, num_pixels - i, out + i);
  }
}

#define GENERATE_PREDICTOR_SUB_2_VSX(X, A, B)                                  \
  static void PredictorSub##X##_VSX(const uint32_t* in, const uint32_t* upper, \
                                    int num_pixels,                            \
                                    uint32_t* WEBP_RESTRICT out) {             \
    int i;                                                                     \
    for (i = 0; i + 4 <= num_pixels; i += 4) {                                 \
      const u8x16 tA = (u8x16)vec_xl(0, (uint32_t*)&(A));                      \
      const u8x16 tB = (u8x16)vec_xl(0, (uint32_t*)&(B));                      \
      const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);                   \
      const u8x16 pred = Average2_u8(tA, tB);                                  \
      vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);                          \
    }                                                                          \
    if (i != num_pixels) {                                                     \
      VP8LPredictorsSub_C[(X)](in + i, upper + i, num_pixels - i, out + i);    \
    }                                                                          \
  }
GENERATE_PREDICTOR_SUB_2_VSX(6, in[i - 1], upper[i - 1])  // avg(L, TL)
GENERATE_PREDICTOR_SUB_2_VSX(7, in[i - 1], upper[i])      // avg(L, T)
GENERATE_PREDICTOR_SUB_2_VSX(8, upper[i - 1], upper[i])   // avg(TL, T)
GENERATE_PREDICTOR_SUB_2_VSX(9, upper[i], upper[i + 1])   // avg(T, TR)
#undef GENERATE_PREDICTOR_SUB_2_VSX

// Predictor10: avg(avg(L, TL), avg(T, TR)).
static void PredictorSub10_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 L = (u8x16)vec_xl(0, (uint32_t*)&in[i - 1]);
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TR = (u8x16)vec_xl(0, (uint32_t*)&upper[i + 1]);
    const u8x16 pred = Average2_u8(Average2_u8(L, TL), Average2_u8(T, TR));
    vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[10](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Per-pixel sum of absolute byte differences (4 pixels -> u32x4).
static WEBP_INLINE u32x4 SumAbsDiff32_VSX(u8x16 a, u8x16 b) {
  return vec_sum4s(vec_or(vec_subs(a, b), vec_subs(b, a)),
                   vec_splats((unsigned int)0));
}

// Predictor11: select between T and L by |T-TL| vs |L-TL|.
static void PredictorSub11_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 L = (u8x16)vec_xl(0, (uint32_t*)&in[i - 1]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u32x4 pa = SumAbsDiff32_VSX(T, TL);
    const u32x4 pb = SumAbsDiff32_VSX(L, TL);
    const u8x16 mask = (u8x16)vec_cmpgt(pb, pa);  // pb > pa ? L : T
    const u8x16 pred = vec_sel(T, L, mask);
    vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[11](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor12: ClampedAddSubtractFull (L + T - TL, per channel, saturated).
static void PredictorSub12_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  const u8x16 zero = vec_splats((unsigned char)0);
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u8x16 L = (u8x16)vec_xl(0, (uint32_t*)&in[i - 1]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    const i16x8 pred_lo = vec_add(
        (i16x8)vec_mergeh(L, zero),
        vec_sub((i16x8)vec_mergeh(T, zero), (i16x8)vec_mergeh(TL, zero)));
    const i16x8 pred_hi = vec_add(
        (i16x8)vec_mergel(L, zero),
        vec_sub((i16x8)vec_mergel(T, zero), (i16x8)vec_mergel(TL, zero)));
    const u8x16 pred = vec_packsu(pred_lo, pred_hi);
    vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[12](in + i, upper + i, num_pixels - i, out + i);
  }
}

// ClampedAddSubtractHalf on one 8->16 unpacked half (matches lossless_sse2.c).
static WEBP_INLINE i16x8 ClampedHalf_VSX(i16x8 L, i16x8 T, i16x8 TL) {
  const u16x8 one = vec_splats((unsigned short)1);
  const i16x8 avg = (i16x8)vec_sr((u16x8)vec_add(T, L), one);
  const i16x8 A1 = vec_sub(avg, TL);
  const i16x8 bit_fix = (i16x8)vec_cmpgt(TL, avg);
  const i16x8 A3 = vec_sra(vec_sub(A1, bit_fix), one);
  return vec_add(avg, A3);
}

// Predictor13: ClampedAddSubtractHalf.
static void PredictorSub13_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  const u8x16 zero = vec_splats((unsigned char)0);
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 L = (u8x16)vec_xl(0, (uint32_t*)&in[i - 1]);
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    const i16x8 A4_lo =
        ClampedHalf_VSX((i16x8)vec_mergeh(L, zero), (i16x8)vec_mergeh(T, zero),
                        (i16x8)vec_mergeh(TL, zero));
    const i16x8 A4_hi =
        ClampedHalf_VSX((i16x8)vec_mergel(L, zero), (i16x8)vec_mergel(T, zero),
                        (i16x8)vec_mergel(TL, zero));
    const u8x16 pred = vec_packsu(A4_lo, A4_hi);
    vec_xst((u32x4)vec_sub(src, pred), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[13](in + i, upper + i, num_pixels - i, out + i);
  }
}

//------------------------------------------------------------------------------
// Color-transform histogram collection.

#define SPAN 8
static void CollectColorBlueTransforms_VSX(const uint32_t* WEBP_RESTRICT argb,
                                           int stride, int tile_width,
                                           int tile_height, int green_to_blue,
                                           int red_to_blue, uint32_t histo[]) {
#define CST(X) (((int16_t)((uint16_t)(X) << 8)) >> 5)
  const i16x8 mults_r =
      (i16x8)vec_splats((int)((uint32_t)(uint16_t)CST(red_to_blue) << 16));
  const i16x8 mults_g = (i16x8)vec_splats((int)((uint16_t)CST(green_to_blue)));
#undef CST
  const u32x4 mask_g = vec_splats((uint32_t)0x00ff00);
  const u32x4 mask_b = vec_splats((uint32_t)0x0000ff);
  const u16x8 sh8 = vec_splats((unsigned short)8);
  const u32x4 sh16 = vec_splats((unsigned int)16);
  int y;
  for (y = 0; y < tile_height; ++y) {
    const uint32_t* const src = argb + y * stride;
    int i, x;
    for (x = 0; x + SPAN <= tile_width; x += SPAN) {
      uint16_t values[SPAN];
      const u8x16 in0 = (u8x16)vec_xl(0, (uint32_t*)&src[x + 0]);
      const u8x16 in1 = (u8x16)vec_xl(0, (uint32_t*)&src[x + SPAN / 2]);
      const i16x8 A0 = (i16x8)vec_sl((u16x8)in0, sh8);
      const i16x8 A1 = (i16x8)vec_sl((u16x8)in1, sh8);
      const i16x8 B0 = (i16x8)vec_and((u32x4)in0, mask_g);
      const i16x8 B1 = (i16x8)vec_and((u32x4)in1, mask_g);
      const u8x16 C0 = (u8x16)MulHiS16(A0, mults_r);
      const u8x16 C1 = (u8x16)MulHiS16(A1, mults_r);
      const u8x16 D0 = (u8x16)MulHiS16(B0, mults_g);
      const u8x16 D1 = (u8x16)MulHiS16(B1, mults_g);
      const u8x16 E0 = vec_sub(in0, D0);
      const u8x16 E1 = vec_sub(in1, D1);
      const u8x16 F0 = (u8x16)vec_sr((u32x4)C0, sh16);
      const u8x16 F1 = (u8x16)vec_sr((u32x4)C1, sh16);
      const u32x4 G0 = vec_and((u32x4)vec_sub(E0, F0), mask_b);
      const u32x4 G1 = vec_and((u32x4)vec_sub(E1, F1), mask_b);
      const i16x8 I = vec_packs((i32x4)G0, (i32x4)G1);
      vec_xst((u16x8)I, 0, values);
      for (i = 0; i < SPAN; ++i) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & (SPAN - 1);
    if (left_over > 0) {
      VP8LCollectColorBlueTransforms_C(argb + tile_width - left_over, stride,
                                       left_over, tile_height, green_to_blue,
                                       red_to_blue, histo);
    }
  }
}

static void CollectColorRedTransforms_VSX(const uint32_t* WEBP_RESTRICT argb,
                                          int stride, int tile_width,
                                          int tile_height, int green_to_red,
                                          uint32_t histo[]) {
#define CST(X) (((int16_t)((uint16_t)(X) << 8)) >> 5)
  const i16x8 mults_g = (i16x8)vec_splats((int)((uint16_t)CST(green_to_red)));
#undef CST
  const u32x4 mask_g = vec_splats((uint32_t)0x00ff00);
  const u32x4 mask = vec_splats((uint32_t)0xff);
  const u32x4 sh16 = vec_splats((unsigned int)16);
  int y;
  for (y = 0; y < tile_height; ++y) {
    const uint32_t* const src = argb + y * stride;
    int i, x;
    for (x = 0; x + SPAN <= tile_width; x += SPAN) {
      uint16_t values[SPAN];
      const u8x16 in0 = (u8x16)vec_xl(0, (uint32_t*)&src[x + 0]);
      const u8x16 in1 = (u8x16)vec_xl(0, (uint32_t*)&src[x + SPAN / 2]);
      const i16x8 A0 = (i16x8)vec_and((u32x4)in0, mask_g);
      const i16x8 A1 = (i16x8)vec_and((u32x4)in1, mask_g);
      const u8x16 B0 = (u8x16)vec_sr((u32x4)in0, sh16);
      const u8x16 B1 = (u8x16)vec_sr((u32x4)in1, sh16);
      const u8x16 C0 = (u8x16)MulHiS16(A0, mults_g);
      const u8x16 C1 = (u8x16)MulHiS16(A1, mults_g);
      const u32x4 E0 = vec_and((u32x4)vec_sub(B0, C0), mask);
      const u32x4 E1 = vec_and((u32x4)vec_sub(B1, C1), mask);
      const i16x8 I = vec_packs((i32x4)E0, (i32x4)E1);
      vec_xst((u16x8)I, 0, values);
      for (i = 0; i < SPAN; ++i) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & (SPAN - 1);
    if (left_over > 0) {
      VP8LCollectColorRedTransforms_C(argb + tile_width - left_over, stride,
                                      left_over, tile_height, green_to_red,
                                      histo);
    }
  }
}
#undef SPAN

//------------------------------------------------------------------------------
// Combined Shannon entropy.

#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)
static uint64_t CombinedShannonEntropy_VSX(const uint32_t X[256],
                                           const uint32_t Y[256]) {
  int i;
  uint64_t retval = 0;
  uint32_t sumX = 0, sumXY = 0;
  const i8x16 zero = vec_splats((signed char)0);
  for (i = 0; i < 256; i += 16) {
    const i32x4 x0 = (i32x4)vec_xl(0, (uint32_t*)(X + i + 0));
    const i32x4 x1 = (i32x4)vec_xl(0, (uint32_t*)(X + i + 4));
    const i32x4 x2 = (i32x4)vec_xl(0, (uint32_t*)(X + i + 8));
    const i32x4 x3 = (i32x4)vec_xl(0, (uint32_t*)(X + i + 12));
    const i32x4 y0 = (i32x4)vec_xl(0, (uint32_t*)(Y + i + 0));
    const i32x4 y1 = (i32x4)vec_xl(0, (uint32_t*)(Y + i + 4));
    const i32x4 y2 = (i32x4)vec_xl(0, (uint32_t*)(Y + i + 8));
    const i32x4 y3 = (i32x4)vec_xl(0, (uint32_t*)(Y + i + 12));
    const i8x16 x4 = vec_packs(vec_packs(x0, x1), vec_packs(x2, x3));
    const i8x16 y4 = vec_packs(vec_packs(y0, y1), vec_packs(y2, y3));
    const int32_t mx = (int32_t)MoveMask8((u8x16)vec_cmpgt(x4, zero));
    int32_t my = (int32_t)MoveMask8((u8x16)vec_cmpgt(y4, zero)) | mx;
    while (my) {
      const int32_t j = BitsCtz((uint32_t)my);
      uint32_t xy;
      if ((mx >> j) & 1) {
        const uint32_t x = X[i + j];
        sumXY += x;
        retval += VP8LFastSLog2(x);
      }
      xy = X[i + j] + Y[i + j];
      sumX += xy;
      retval += VP8LFastSLog2(xy);
      my &= my - 1;
    }
  }
  retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
  return retval;
}
#endif  // !WEBP_HAVE_SLOW_CLZ_CTZ

//------------------------------------------------------------------------------
// Histogram vector helpers.

static void AddVector_VSX(const uint32_t* WEBP_RESTRICT a,
                          const uint32_t* WEBP_RESTRICT b,
                          uint32_t* WEBP_RESTRICT out, int size) {
  int i;
  for (i = 0; i + 4 <= size; i += 4) {
    const u32x4 va = vec_xl(0, (uint32_t*)&a[i]);
    const u32x4 vb = vec_xl(0, (uint32_t*)&b[i]);
    vec_xst(vec_add(va, vb), 0, &out[i]);
  }
  for (; i < size; ++i) out[i] = a[i] + b[i];
}

static void AddVectorEq_VSX(const uint32_t* WEBP_RESTRICT a,
                            uint32_t* WEBP_RESTRICT out, int size) {
  int i;
  for (i = 0; i + 4 <= size; i += 4) {
    const u32x4 va = vec_xl(0, (uint32_t*)&a[i]);
    const u32x4 vo = vec_xl(0, (uint32_t*)&out[i]);
    vec_xst(vec_add(va, vo), 0, &out[i]);
  }
  for (; i < size; ++i) out[i] += a[i];
}

static int VectorMismatch_VSX(const uint32_t* const array1,
                              const uint32_t* const array2, int length) {
  int i = 0;
  for (; i + 4 <= length; i += 4) {
    const u32x4 a = vec_xl(0, (uint32_t*)&array1[i]);
    const u32x4 b = vec_xl(0, (uint32_t*)&array2[i]);
    if (!vec_all_eq(a, b)) break;
  }
  for (; i < length && array1[i] == array2[i]; ++i) {
  }
  return i;
}

//------------------------------------------------------------------------------
// Bundle color map.

static void BundleColorMap_VSX(const uint8_t* WEBP_RESTRICT const row,
                               int width, int xbits,
                               uint32_t* WEBP_RESTRICT dst) {
  int x = 0;
  if (xbits == 0) {
    const u8x16 zero = vec_splats((unsigned char)0);
    const u16x8 ff = vec_splats((unsigned short)0xff00);
    for (x = 0; x + 16 <= width; x += 16, dst += 16) {
      const u8x16 in = vec_xl(0, (unsigned char*)&row[x]);
      const u16x8 in_lo = (u16x8)vec_mergeh(zero, in);  // 0 v per byte (v << 8)
      const u16x8 in_hi = (u16x8)vec_mergel(zero, in);
      vec_xst((u32x4)vec_mergeh(in_lo, ff), 0, &dst[0]);
      vec_xst((u32x4)vec_mergel(in_lo, ff), 0, &dst[4]);
      vec_xst((u32x4)vec_mergeh(in_hi, ff), 0, &dst[8]);
      vec_xst((u32x4)vec_mergel(in_hi, ff), 0, &dst[12]);
    }
  } else if (xbits == 1) {
    const u16x8 zero = vec_splats((unsigned short)0);
    const u16x8 ff = vec_splats((unsigned short)0xff00);
    const u16x8 mul = vec_splats((unsigned short)0x110);
    for (x = 0; x + 16 <= width; x += 16, dst += 8) {
      const u8x16 in = vec_xl(0, (unsigned char*)&row[x]);
      const u16x8 tmp = vec_mladd((u16x8)in, mul, zero);  // aba0
      const u16x8 pack = vec_and(tmp, ff);                // ab00
      vec_xst((u32x4)vec_mergeh(pack, ff), 0, &dst[0]);
      vec_xst((u32x4)vec_mergel(pack, ff), 0, &dst[4]);
    }
  } else if (xbits == 2) {
    const u16x8 zero = vec_splats((unsigned short)0);
    const u32x4 mask_or = vec_splats((uint32_t)0xff000000);
    const u16x8 mul_cst = vec_splats((unsigned short)0x0104);
    const u16x8 mask_mul = vec_splats((unsigned short)0x0f00);
    const u32x4 sh12 = vec_splats((unsigned int)12);
    for (x = 0; x + 16 <= width; x += 16, dst += 4) {
      const u8x16 in = vec_xl(0, (unsigned char*)&row[x]);
      const u16x8 mul = vec_mladd((u16x8)in, mul_cst, zero);
      const u16x8 tmp = vec_and(mul, mask_mul);
      const u32x4 shift = vec_sr((u32x4)tmp, sh12);
      const u32x4 pack = vec_or(shift, (u32x4)tmp);
      vec_xst(vec_or(pack, mask_or), 0, dst);
    }
  }
  // xbits == 3 (1-bit palette) uses a byte-mask gather; fall through to C.
  if (x != width) {
    VP8LBundleColorMap_C(row + x, width - x, xbits, dst);
  }
}

//------------------------------------------------------------------------------

extern void VP8LEncDspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitVSX(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_VSX;
  VP8LTransformColor = TransformColor_VSX;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_VSX;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_VSX;
  VP8LAddVector = AddVector_VSX;
  VP8LAddVectorEq = AddVectorEq_VSX;
  VP8LVectorMismatch = VectorMismatch_VSX;
  VP8LBundleColorMap = BundleColorMap_VSX;
#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_VSX;
#endif

  VP8LPredictorsSub[0] = PredictorSub0_VSX;
  VP8LPredictorsSub[1] = PredictorSub1_VSX;
  VP8LPredictorsSub[2] = PredictorSub2_VSX;
  VP8LPredictorsSub[3] = PredictorSub3_VSX;
  VP8LPredictorsSub[4] = PredictorSub4_VSX;
  VP8LPredictorsSub[5] = PredictorSub5_VSX;
  VP8LPredictorsSub[6] = PredictorSub6_VSX;
  VP8LPredictorsSub[7] = PredictorSub7_VSX;
  VP8LPredictorsSub[8] = PredictorSub8_VSX;
  VP8LPredictorsSub[9] = PredictorSub9_VSX;
  VP8LPredictorsSub[10] = PredictorSub10_VSX;
  VP8LPredictorsSub[11] = PredictorSub11_VSX;
  VP8LPredictorsSub[12] = PredictorSub12_VSX;
  VP8LPredictorsSub[13] = PredictorSub13_VSX;
  VP8LPredictorsSub[14] = PredictorSub0_VSX;  // padding sentinels
  VP8LPredictorsSub[15] = PredictorSub0_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8LEncDspInitVSX)

#endif  // WEBP_USE_VSX
