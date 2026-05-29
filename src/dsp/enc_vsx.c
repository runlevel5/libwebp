// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC64) version of the encoding functions.
//
// Authors: Trung Lê (8@tle.id.au)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/enc/vp8i_enc.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector signed char i8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector signed int i32x4;

//------------------------------------------------------------------------------
// Sum of squared errors.

// Sum of squared abs differences of two 16-byte vectors -> 4 partial sums.
static WEBP_INLINE i32x4 SubtractAndAccumulate_VSX(u8x16 a, u8x16 b) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 abs_a_b = vec_or(vec_subs(a, b), vec_subs(b, a));
  const i16x8 c0 = (i16x8)vec_mergeh(abs_a_b, zero);
  const i16x8 c1 = (i16x8)vec_mergel(abs_a_b, zero);
  const i32x4 z = (i32x4)vec_splats(0);
  return vec_add(vec_msum(c0, c0, z), vec_msum(c1, c1, z));
}

static WEBP_INLINE int HorizontalSumS32_VSX(i32x4 v) {
  const i32x4 s = vec_add(v, vec_sld(v, v, 8));
  return vec_extract(vec_add(s, vec_sld(s, s, 4)), 0);
}

static WEBP_INLINE int SSE_16xN_VSX(const uint8_t* WEBP_RESTRICT a,
                                    const uint8_t* WEBP_RESTRICT b,
                                    int num_pairs) {
  i32x4 sum = (i32x4)vec_splats(0);
  int i;
  for (i = 0; i < num_pairs; ++i) {
    const u8x16 a0 = vec_xl(0, (unsigned char*)&a[BPS * 0]);
    const u8x16 b0 = vec_xl(0, (unsigned char*)&b[BPS * 0]);
    const u8x16 a1 = vec_xl(0, (unsigned char*)&a[BPS * 1]);
    const u8x16 b1 = vec_xl(0, (unsigned char*)&b[BPS * 1]);
    sum = vec_add(sum, vec_add(SubtractAndAccumulate_VSX(a0, b0),
                               SubtractAndAccumulate_VSX(a1, b1)));
    a += 2 * BPS;
    b += 2 * BPS;
  }
  return HorizontalSumS32_VSX(sum);
}

static int SSE16x16_VSX(const uint8_t* WEBP_RESTRICT a,
                        const uint8_t* WEBP_RESTRICT b) {
  return SSE_16xN_VSX(a, b, 8);
}

static int SSE16x8_VSX(const uint8_t* WEBP_RESTRICT a,
                       const uint8_t* WEBP_RESTRICT b) {
  return SSE_16xN_VSX(a, b, 4);
}

// Loads 8 bytes from p into the low half of a vector (high half undefined).
static WEBP_INLINE u8x16 Load64_VSX(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, 8);
  // uint64_t is 'unsigned long' under LP64, which is ambiguous between the
  // 'unsigned long long' and 'unsigned __int128' vec_splats() overloads on
  // clang; cast to an exact-match type.
  return (u8x16)vec_splats((unsigned long long)v);
}

// Loads 8 bytes from p and zero-extends to eight 16-bit lanes.
static WEBP_INLINE i16x8 Load8x16_VSX(const uint8_t* p) {
  const u8x16 zero = vec_splats((unsigned char)0);
  return (i16x8)vec_mergeh(Load64_VSX(p), zero);
}

static int SSE8x8_VSX(const uint8_t* WEBP_RESTRICT a,
                      const uint8_t* WEBP_RESTRICT b) {
  const i32x4 z = (i32x4)vec_splats(0);
  i32x4 sum = z;
  int num_pairs = 4;
  while (num_pairs-- > 0) {
    const i16x8 a0 = Load8x16_VSX(&a[BPS * 0]);
    const i16x8 a1 = Load8x16_VSX(&a[BPS * 1]);
    const i16x8 b0 = Load8x16_VSX(&b[BPS * 0]);
    const i16x8 b1 = Load8x16_VSX(&b[BPS * 1]);
    const i16x8 c0 = vec_subs(a0, b0);
    const i16x8 c1 = vec_subs(a1, b1);
    sum = vec_add(sum, vec_add(vec_msum(c0, c0, z), vec_msum(c1, c1, z)));
    a += 2 * BPS;
    b += 2 * BPS;
  }
  return HorizontalSumS32_VSX(sum);
}

static int SSE4x4_VSX(const uint8_t* WEBP_RESTRICT a,
                      const uint8_t* WEBP_RESTRICT b) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const i32x4 z = (i32x4)vec_splats(0);
  // Load 8 bytes per row (over-read is safe: buffers are over-allocated).
  const u8x16 a0 = Load64_VSX(&a[BPS * 0]);
  const u8x16 a1 = Load64_VSX(&a[BPS * 1]);
  const u8x16 a2 = Load64_VSX(&a[BPS * 2]);
  const u8x16 a3 = Load64_VSX(&a[BPS * 3]);
  const u8x16 b0 = Load64_VSX(&b[BPS * 0]);
  const u8x16 b1 = Load64_VSX(&b[BPS * 1]);
  const u8x16 b2 = Load64_VSX(&b[BPS * 2]);
  const u8x16 b3 = Load64_VSX(&b[BPS * 3]);
  // Combine pairs of lines (low 32-bit lanes interleaved).
  const u8x16 a01 = (u8x16)vec_mergeh((u32x4)a0, (u32x4)a1);
  const u8x16 a23 = (u8x16)vec_mergeh((u32x4)a2, (u32x4)a3);
  const u8x16 b01 = (u8x16)vec_mergeh((u32x4)b0, (u32x4)b1);
  const u8x16 b23 = (u8x16)vec_mergeh((u32x4)b2, (u32x4)b3);
  const i16x8 a01s = (i16x8)vec_mergeh(a01, zero);
  const i16x8 a23s = (i16x8)vec_mergeh(a23, zero);
  const i16x8 b01s = (i16x8)vec_mergeh(b01, zero);
  const i16x8 b23s = (i16x8)vec_mergeh(b23, zero);
  const i16x8 d0 = vec_subs(a01s, b01s);
  const i16x8 d1 = vec_subs(a23s, b23s);
  return HorizontalSumS32_VSX(
      vec_add(vec_msum(d0, d0, z), vec_msum(d1, d1, z)));
}

//------------------------------------------------------------------------------

static void Mean16x4_VSX(const uint8_t* WEBP_RESTRICT ref, uint32_t dc[4]) {
  const u16x8 mask = vec_splats((unsigned short)0x00ff);
  const u16x8 sh8 = vec_splats((unsigned short)8);
  const u16x8 a0 = (u16x8)vec_xl(0, (unsigned char*)&ref[BPS * 0]);
  const u16x8 a1 = (u16x8)vec_xl(0, (unsigned char*)&ref[BPS * 1]);
  const u16x8 a2 = (u16x8)vec_xl(0, (unsigned char*)&ref[BPS * 2]);
  const u16x8 a3 = (u16x8)vec_xl(0, (unsigned char*)&ref[BPS * 3]);
  const u32x4 d0 = vec_add((u32x4)vec_sr(a0, sh8), (u32x4)vec_and(a0, mask));
  const u32x4 d1 = vec_add((u32x4)vec_sr(a1, sh8), (u32x4)vec_and(a1, mask));
  const u32x4 d2 = vec_add((u32x4)vec_sr(a2, sh8), (u32x4)vec_and(a2, mask));
  const u32x4 d3 = vec_add((u32x4)vec_sr(a3, sh8), (u32x4)vec_and(a3, mask));
  const u32x4 f0 = vec_add(vec_add(d0, d1), vec_add(d2, d3));
  uint16_t tmp[8];
  vec_xst((u16x8)f0, 0, tmp);
  dc[0] = tmp[0] + tmp[1];
  dc[1] = tmp[2] + tmp[3];
  dc[2] = tmp[4] + tmp[5];
  dc[3] = tmp[6] + tmp[7];
}

//------------------------------------------------------------------------------

extern void VP8EncDspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspInitVSX(void) {
  VP8SSE16x16 = SSE16x16_VSX;
  VP8SSE16x8 = SSE16x8_VSX;
  VP8SSE8x8 = SSE8x8_VSX;
  VP8SSE4x4 = SSE4x4_VSX;
  VP8Mean16x4 = Mean16x4_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8EncDspInitVSX)

#endif  // WEBP_USE_VSX
