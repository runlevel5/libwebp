// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC64) version of distortion calculation.
//
// Authors: Trung Lê (8@tle.id.au)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <assert.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector signed int i32x4;

#if !defined(WEBP_DISABLE_STATS)

// Horizontal sum of the four 32-bit lanes.
static WEBP_INLINE uint32_t HorizontalSum32_VSX(u32x4 v) {
  const u32x4 s = vec_add(v, vec_sld(v, v, 8));
  return vec_extract(vec_add(s, vec_sld(s, s, 4)), 0);
}

// Sum of squared abs differences of two 16-byte vectors -> 4 partial sums.
static WEBP_INLINE u32x4 SubtractAndSquare_VSX(u8x16 a, u8x16 b) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 abs_a_b = vec_or(vec_subs(a, b), vec_subs(b, a));
  const i16x8 c0 = (i16x8)vec_mergeh(abs_a_b, zero);
  const i16x8 c1 = (i16x8)vec_mergel(abs_a_b, zero);
  const i32x4 sum1 = vec_msum(c0, c0, (i32x4)vec_splats(0));
  const i32x4 sum2 = vec_msum(c1, c1, (i32x4)vec_splats(0));
  return (u32x4)vec_add(sum1, sum2);
}

static uint32_t AccumulateSSE_VSX(const uint8_t* src1, const uint8_t* src2,
                                  int len) {
  int i = 0;
  uint32_t sse2 = 0;
  if (len >= 16) {
    u32x4 sum = vec_splats((uint32_t)0);
    for (; i + 16 <= len; i += 16) {
      const u8x16 a = vec_xl(0, (unsigned char*)&src1[i]);
      const u8x16 b = vec_xl(0, (unsigned char*)&src2[i]);
      sum = vec_add(sum, SubtractAndSquare_VSX(a, b));
    }
    sse2 += HorizontalSum32_VSX(sum);
  }
  for (; i < len; ++i) {
    const int32_t diff = src1[i] - src2[i];
    sse2 += diff * diff;
  }
  return sse2;
}
#endif  // !defined(WEBP_DISABLE_STATS)

#if !defined(WEBP_REDUCE_SIZE)

// Loads 8 bytes from p into the low half (high half undefined).
static WEBP_INLINE u8x16 Load8(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, 8);
  // uint64_t is 'unsigned long' under LP64, which is ambiguous between the
  // 'unsigned long long' and 'unsigned __int128' vec_splats() overloads on
  // clang; cast to an exact-match type.
  return (u8x16)vec_splats((unsigned long long)v);
}

static const uint16_t kWeight[8] = {1, 2, 3, 4, 3, 2, 1, 0};

#define ACCUMULATE_ROW(WEIGHT)                                   \
  do {                                                           \
    const u16x8 Wy = vec_splats((unsigned short)(WEIGHT));       \
    const u16x8 W = vec_mladd(Wx, Wy, z16);                      \
    const u8x16 a0 = Load8(src1);                                \
    const u8x16 b0 = Load8(src2);                                \
    const u16x8 a1 = (u16x8)vec_mergeh(a0, zero);                \
    const u16x8 b1 = (u16x8)vec_mergeh(b0, zero);                \
    const u16x8 wa1 = vec_mladd(a1, W, z16);                     \
    const u16x8 wb1 = vec_mladd(b1, W, z16);                     \
    xm = vec_add(xm, wa1);                                       \
    ym = vec_add(ym, wb1);                                       \
    xxm = vec_add(xxm, vec_msum((i16x8)a1, (i16x8)wa1, zero32)); \
    xym = vec_add(xym, vec_msum((i16x8)a1, (i16x8)wb1, zero32)); \
    yym = vec_add(yym, vec_msum((i16x8)b1, (i16x8)wb1, zero32)); \
    src1 += stride1;                                             \
    src2 += stride2;                                             \
  } while (0)

static double SSIMGet_VSX(const uint8_t* src1, int stride1, const uint8_t* src2,
                          int stride2) {
  VP8DistoStats stats;
  const u8x16 zero = vec_splats((unsigned char)0);
  const u16x8 z16 = vec_splats((unsigned short)0);
  const i32x4 zero32 = (i32x4)vec_splats(0);
  u16x8 xm = z16, ym = z16;
  i32x4 xxm = zero32, yym = zero32, xym = zero32;
  const u16x8 Wx = vec_xl(0, (uint16_t*)kWeight);
  assert(2 * VP8_SSIM_KERNEL + 1 == 7);
  ACCUMULATE_ROW(1);
  ACCUMULATE_ROW(2);
  ACCUMULATE_ROW(3);
  ACCUMULATE_ROW(4);
  ACCUMULATE_ROW(3);
  ACCUMULATE_ROW(2);
  ACCUMULATE_ROW(1);
  {
    // Only the first 7 of the 8 lanes carry data (kWeight[7] == 0).
    uint16_t tx[8], ty[8];
    vec_xst(xm, 0, tx);
    vec_xst(ym, 0, ty);
    stats.xm = tx[0] + tx[1] + tx[2] + tx[3] + tx[4] + tx[5] + tx[6];
    stats.ym = ty[0] + ty[1] + ty[2] + ty[3] + ty[4] + ty[5] + ty[6];
    stats.xxm = HorizontalSum32_VSX((u32x4)xxm);
    stats.xym = HorizontalSum32_VSX((u32x4)xym);
    stats.yym = HorizontalSum32_VSX((u32x4)yym);
  }
  return VP8SSIMFromStats(&stats);
}

#endif  // !defined(WEBP_REDUCE_SIZE)

//------------------------------------------------------------------------------

extern void VP8SSIMDspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8SSIMDspInitVSX(void) {
#if !defined(WEBP_DISABLE_STATS)
  VP8AccumulateSSE = AccumulateSSE_VSX;
#endif
#if !defined(WEBP_REDUCE_SIZE)
  VP8SSIMGet = SSIMGet_VSX;
#endif
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8SSIMDspInitVSX)

#endif  // WEBP_USE_VSX
