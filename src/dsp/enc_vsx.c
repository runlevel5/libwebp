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
#include <stdlib.h>
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
typedef __vector signed long long i64x2;

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
// Inverse transform (shares the decode IDCT math).

// Signed 16-bit multiply-high: (a * b) >> 16.
static WEBP_INLINE i16x8 MulHi16_S(i16x8 a, i16x8 b) {
  const u32x4 sh = vec_splats((unsigned int)16);
  const i32x4 e = vec_sra(vec_mule(a, b), sh);
  const i32x4 o = vec_sra(vec_mulo(a, b), sh);
  return (i16x8)vec_pack(vec_mergeh(e, o), vec_mergel(e, o));
}

// Transpose two interleaved 4x4 blocks of 16-bit values.
static WEBP_INLINE void Transpose2_4x4(i16x8 in0, i16x8 in1, i16x8 in2,
                                       i16x8 in3, i16x8* out0, i16x8* out1,
                                       i16x8* out2, i16x8* out3) {
  const i16x8 t0 = (i16x8)vec_mergeh(in0, in1);
  const i16x8 t1 = (i16x8)vec_mergeh(in2, in3);
  const i16x8 t2 = (i16x8)vec_mergel(in0, in1);
  const i16x8 t3 = (i16x8)vec_mergel(in2, in3);
  const i32x4 u0 = vec_mergeh((i32x4)t0, (i32x4)t1);
  const i32x4 u1 = vec_mergeh((i32x4)t2, (i32x4)t3);
  const i32x4 u2 = vec_mergel((i32x4)t0, (i32x4)t1);
  const i32x4 u3 = vec_mergel((i32x4)t2, (i32x4)t3);
  *out0 = (i16x8)vec_mergeh((i64x2)u0, (i64x2)u1);
  *out1 = (i16x8)vec_mergel((i64x2)u0, (i64x2)u1);
  *out2 = (i16x8)vec_mergeh((i64x2)u2, (i64x2)u3);
  *out3 = (i16x8)vec_mergel((i64x2)u2, (i64x2)u3);
}

static WEBP_INLINE i16x8 Load4Coeffs(const int16_t* WEBP_RESTRICT p) {
  int16_t tmp[8] = {0};
  memcpy(tmp, p, 4 * sizeof(int16_t));
  return *(const i16x8*)tmp;
}

static WEBP_INLINE i16x8 LoadRef4(const uint8_t* WEBP_RESTRICT p, int n) {
  unsigned char tmp[16] = {0};
  memcpy(tmp, p, n);
  return (i16x8)vec_mergeh(vec_xl(0, tmp), vec_splats((unsigned char)0));
}

static void ITransform_One_VSX(const uint8_t* WEBP_RESTRICT ref,
                               const int16_t* WEBP_RESTRICT in,
                               uint8_t* WEBP_RESTRICT dst, int do_two) {
  const i16x8 k1 = vec_splats((short)20091);
  const i16x8 k2 = vec_splats((short)-30068);
  const u16x8 three = vec_splats((unsigned short)3);
  i16x8 in0 = Load4Coeffs(in + 0), in1 = Load4Coeffs(in + 4);
  i16x8 in2 = Load4Coeffs(in + 8), in3 = Load4Coeffs(in + 12);
  i16x8 T0, T1, T2, T3;

  if (do_two) {
    in0 = (i16x8)vec_mergeh((i64x2)in0, (i64x2)Load4Coeffs(in + 16));
    in1 = (i16x8)vec_mergeh((i64x2)in1, (i64x2)Load4Coeffs(in + 20));
    in2 = (i16x8)vec_mergeh((i64x2)in2, (i64x2)Load4Coeffs(in + 24));
    in3 = (i16x8)vec_mergeh((i64x2)in3, (i64x2)Load4Coeffs(in + 28));
  }

  {  // Vertical pass + transpose.
    const i16x8 a = vec_add(in0, in2);
    const i16x8 b = vec_sub(in0, in2);
    const i16x8 c = vec_add(vec_sub(in1, in3),
                            vec_sub(MulHi16_S(in1, k2), MulHi16_S(in3, k1)));
    const i16x8 d = vec_add(vec_add(in1, in3),
                            vec_add(MulHi16_S(in1, k1), MulHi16_S(in3, k2)));
    Transpose2_4x4(vec_add(a, d), vec_add(b, c), vec_sub(b, c), vec_sub(a, d),
                   &T0, &T1, &T2, &T3);
  }
  {  // Horizontal pass + transpose.
    const i16x8 dc = vec_add(T0, vec_splats((short)4));
    const i16x8 a = vec_add(dc, T2);
    const i16x8 b = vec_sub(dc, T2);
    const i16x8 c =
        vec_add(vec_sub(T1, T3), vec_sub(MulHi16_S(T1, k2), MulHi16_S(T3, k1)));
    const i16x8 d =
        vec_add(vec_add(T1, T3), vec_add(MulHi16_S(T1, k1), MulHi16_S(T3, k2)));
    const i16x8 s0 = vec_sra(vec_add(a, d), three);
    const i16x8 s1 = vec_sra(vec_add(b, c), three);
    const i16x8 s2 = vec_sra(vec_sub(b, c), three);
    const i16x8 s3 = vec_sra(vec_sub(a, d), three);
    Transpose2_4x4(s0, s1, s2, s3, &T0, &T1, &T2, &T3);
  }
  {  // Add to the reference pixels and store with saturation.
    const int n = do_two ? 8 : 4;
    const i16x8 r0 = LoadRef4(ref + 0 * BPS, n);
    const i16x8 r1 = LoadRef4(ref + 1 * BPS, n);
    const i16x8 r2 = LoadRef4(ref + 2 * BPS, n);
    const i16x8 r3 = LoadRef4(ref + 3 * BPS, n);
    const u8x16 o0 = vec_packsu(vec_add(r0, T0), vec_add(r0, T0));
    const u8x16 o1 = vec_packsu(vec_add(r1, T1), vec_add(r1, T1));
    const u8x16 o2 = vec_packsu(vec_add(r2, T2), vec_add(r2, T2));
    const u8x16 o3 = vec_packsu(vec_add(r3, T3), vec_add(r3, T3));
    unsigned char b0[16], b1[16], b2[16], b3[16];
    memcpy(b0, &o0, 16);
    memcpy(b1, &o1, 16);
    memcpy(b2, &o2, 16);
    memcpy(b3, &o3, 16);
    memcpy(dst + 0 * BPS, b0, n);
    memcpy(dst + 1 * BPS, b1, n);
    memcpy(dst + 2 * BPS, b2, n);
    memcpy(dst + 3 * BPS, b3, n);
  }
}

static void ITransform_VSX(const uint8_t* WEBP_RESTRICT ref,
                           const int16_t* WEBP_RESTRICT in,
                           uint8_t* WEBP_RESTRICT dst, int do_two) {
  ITransform_One_VSX(ref, in, dst, do_two);
}

//------------------------------------------------------------------------------
// Forward transform.

// Transpose four 4-lane 32-bit row vectors into column vectors.
static WEBP_INLINE void Transpose4x4_S32(i32x4 r0, i32x4 r1, i32x4 r2, i32x4 r3,
                                         i32x4* c0, i32x4* c1, i32x4* c2,
                                         i32x4* c3) {
  const i32x4 a = vec_mergeh(r0, r2);
  const i32x4 b = vec_mergeh(r1, r3);
  const i32x4 c = vec_mergel(r0, r2);
  const i32x4 d = vec_mergel(r1, r3);
  *c0 = vec_mergeh(a, b);
  *c1 = vec_mergel(a, b);
  *c2 = vec_mergeh(c, d);
  *c3 = vec_mergel(c, d);
}

// Loads 4 bytes, zero-extended to four 32-bit lanes.
static WEBP_INLINE i32x4 Load4u8_S32(const uint8_t* p) {
  const u8x16 zero = vec_splats((unsigned char)0);
  uint32_t v;
  memcpy(&v, p, 4);
  const u8x16 b = (u8x16)vec_splats(v);
  return (i32x4)vec_mergeh((u16x8)vec_mergeh(b, zero), (u16x8)zero);
}

static void FTransform_One_VSX(const uint8_t* WEBP_RESTRICT src,
                               const uint8_t* WEBP_RESTRICT ref,
                               int16_t* WEBP_RESTRICT out) {
  const i32x4 k2217 = vec_splats(2217);
  const i32x4 k5352 = vec_splats(5352);
  const i32x4 zero = vec_splats(0);
  const i32x4 one = vec_splats(1);
  const u32x4 sh3 = vec_splats((unsigned int)3);
  const u32x4 sh4 = vec_splats((unsigned int)4);
  const u32x4 sh9 = vec_splats((unsigned int)9);
  const u32x4 sh16 = vec_splats((unsigned int)16);
  i32x4 r0 = vec_sub(Load4u8_S32(src + 0 * BPS), Load4u8_S32(ref + 0 * BPS));
  i32x4 r1 = vec_sub(Load4u8_S32(src + 1 * BPS), Load4u8_S32(ref + 1 * BPS));
  i32x4 r2 = vec_sub(Load4u8_S32(src + 2 * BPS), Load4u8_S32(ref + 2 * BPS));
  i32x4 r3 = vec_sub(Load4u8_S32(src + 3 * BPS), Load4u8_S32(ref + 3 * BPS));
  i32x4 d0, d1, d2, d3;  // columns of the pixel diff
  Transpose4x4_S32(r0, r1, r2, r3, &d0, &d1, &d2, &d3);
  {  // pass 1
    const i32x4 a0 = vec_add(d0, d3);
    const i32x4 a1 = vec_add(d1, d2);
    const i32x4 a2 = vec_sub(d1, d2);
    const i32x4 a3 = vec_sub(d0, d3);
    const i32x4 t0 = vec_sl(vec_add(a0, a1), sh3);
    const i32x4 t2 = vec_sl(vec_sub(a0, a1), sh3);
    const i32x4 t1 =
        vec_sra(vec_add(vec_add(vec_mul(a2, k2217), vec_mul(a3, k5352)),
                        vec_splats(1812)),
                sh9);
    const i32x4 t3 =
        vec_sra(vec_add(vec_sub(vec_mul(a3, k2217), vec_mul(a2, k5352)),
                        vec_splats(937)),
                sh9);
    Transpose4x4_S32(t0, t1, t2, t3, &d0, &d1, &d2, &d3);
  }
  {  // pass 2
    const i32x4 a0 = vec_add(d0, d3);
    const i32x4 a1 = vec_add(d1, d2);
    const i32x4 a2 = vec_sub(d1, d2);
    const i32x4 a3 = vec_sub(d0, d3);
    const i32x4 o0 = vec_sra(vec_add(vec_add(a0, a1), vec_splats(7)), sh4);
    const i32x4 o2 = vec_sra(vec_add(vec_sub(a0, a1), vec_splats(7)), sh4);
    // (a3 != 0) correction: 1 where a3 is non-zero.
    const i32x4 a3nz = (i32x4)vec_andc(one, (i32x4)vec_cmpeq(a3, zero));
    const i32x4 o1 =
        vec_add(vec_sra(vec_add(vec_add(vec_mul(a2, k2217), vec_mul(a3, k5352)),
                                vec_splats(12000)),
                        sh16),
                a3nz);
    const i32x4 o3 =
        vec_sra(vec_add(vec_sub(vec_mul(a3, k2217), vec_mul(a2, k5352)),
                        vec_splats(51000)),
                sh16);
    const i16x8 r01 = vec_pack(o0, o1);
    const i16x8 r23 = vec_pack(o2, o3);
    int16_t b01[8], b23[8];
    vec_xst(r01, 0, b01);
    vec_xst(r23, 0, b23);
    memcpy(out + 0, b01, 4 * sizeof(int16_t));
    memcpy(out + 4, b01 + 4, 4 * sizeof(int16_t));
    memcpy(out + 8, b23, 4 * sizeof(int16_t));
    memcpy(out + 12, b23 + 4, 4 * sizeof(int16_t));
  }
}

static void FTransform_VSX(const uint8_t* WEBP_RESTRICT src,
                           const uint8_t* WEBP_RESTRICT ref,
                           int16_t* WEBP_RESTRICT out) {
  FTransform_One_VSX(src, ref, out);
}

static void FTransform2_VSX(const uint8_t* WEBP_RESTRICT src,
                            const uint8_t* WEBP_RESTRICT ref,
                            int16_t* WEBP_RESTRICT out) {
  FTransform_One_VSX(src + 0, ref + 0, out + 0);
  FTransform_One_VSX(src + 4, ref + 4, out + 16);
}

//------------------------------------------------------------------------------
// Walsh-Hadamard transform.

// Gathers in[0], in[stride], in[2*stride], in[3*stride] (in int16 units).
static WEBP_INLINE i32x4 GatherWHT(const int16_t* p, int stride) {
  const i32x4 v = {p[0], p[stride], p[2 * stride], p[3 * stride]};
  return v;
}

static void FTransformWHT_VSX(const int16_t* WEBP_RESTRICT in,
                              int16_t* WEBP_RESTRICT out) {
  const u32x4 one = vec_splats((unsigned int)1);
  // Columns: Dk[i] = in[i * 64 + k * 16].
  i32x4 d0 = GatherWHT(in + 0 * 16, 64);
  i32x4 d1 = GatherWHT(in + 1 * 16, 64);
  i32x4 d2 = GatherWHT(in + 2 * 16, 64);
  i32x4 d3 = GatherWHT(in + 3 * 16, 64);
  i32x4 t0, t1, t2, t3;
  {  // pass 1
    const i32x4 a0 = vec_add(d0, d2);
    const i32x4 a1 = vec_add(d1, d3);
    const i32x4 a2 = vec_sub(d1, d3);
    const i32x4 a3 = vec_sub(d0, d2);
    Transpose4x4_S32(vec_add(a0, a1), vec_add(a3, a2), vec_sub(a3, a2),
                     vec_sub(a0, a1), &t0, &t1, &t2, &t3);
  }
  {  // pass 2
    const i32x4 a0 = vec_add(t0, t2);
    const i32x4 a1 = vec_add(t1, t3);
    const i32x4 a2 = vec_sub(t1, t3);
    const i32x4 a3 = vec_sub(t0, t2);
    const i32x4 o0 = vec_sra(vec_add(a0, a1), one);
    const i32x4 o1 = vec_sra(vec_add(a3, a2), one);
    const i32x4 o2 = vec_sra(vec_sub(a3, a2), one);
    const i32x4 o3 = vec_sra(vec_sub(a0, a1), one);
    const i16x8 r01 = vec_pack(o0, o1);
    const i16x8 r23 = vec_pack(o2, o3);
    int16_t b01[8], b23[8];
    vec_xst(r01, 0, b01);
    vec_xst(r23, 0, b23);
    memcpy(out + 0, b01, 4 * sizeof(int16_t));
    memcpy(out + 4, b01 + 4, 4 * sizeof(int16_t));
    memcpy(out + 8, b23, 4 * sizeof(int16_t));
    memcpy(out + 12, b23 + 4, 4 * sizeof(int16_t));
  }
}

//------------------------------------------------------------------------------
// Texture distortion (Hadamard transform).

// Combines 4 bytes of inA and 4 bytes of inB into [a0..a3 b0..b3] (16-bit).
static WEBP_INLINE i16x8 LoadCombine_VSX(const uint8_t* a, const uint8_t* b) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 ab =
      (u8x16)vec_mergeh((u32x4)Load64_VSX(a), (u32x4)Load64_VSX(b));
  return (i16x8)vec_mergeh(ab, zero);
}

static int TTransform_VSX(const uint8_t* WEBP_RESTRICT inA,
                          const uint8_t* WEBP_RESTRICT inB,
                          const uint16_t* WEBP_RESTRICT const w) {
  const i16x8 zero = vec_splats((short)0);
  const i32x4 z32 = (i32x4)vec_splats(0);
  i16x8 tmp0 = LoadCombine_VSX(&inA[BPS * 0], &inB[BPS * 0]);
  i16x8 tmp1 = LoadCombine_VSX(&inA[BPS * 1], &inB[BPS * 1]);
  i16x8 tmp2 = LoadCombine_VSX(&inA[BPS * 2], &inB[BPS * 2]);
  i16x8 tmp3 = LoadCombine_VSX(&inA[BPS * 3], &inB[BPS * 3]);

  {  // Vertical pass and transpose.
    const i16x8 a0 = vec_add(tmp0, tmp2);
    const i16x8 a1 = vec_add(tmp1, tmp3);
    const i16x8 a2 = vec_sub(tmp1, tmp3);
    const i16x8 a3 = vec_sub(tmp0, tmp2);
    Transpose2_4x4(vec_add(a0, a1), vec_add(a3, a2), vec_sub(a3, a2),
                   vec_sub(a0, a1), &tmp0, &tmp1, &tmp2, &tmp3);
  }
  {  // Horizontal pass and difference of weighted sums.
    const i16x8 w0 = (i16x8)vec_xl(0, (uint16_t*)&w[0]);
    const i16x8 w8 = (i16x8)vec_xl(0, (uint16_t*)&w[8]);
    const i16x8 a0 = vec_add(tmp0, tmp2);
    const i16x8 a1 = vec_add(tmp1, tmp3);
    const i16x8 a2 = vec_sub(tmp1, tmp3);
    const i16x8 a3 = vec_sub(tmp0, tmp2);
    const i16x8 b0 = vec_add(a0, a1);
    const i16x8 b1 = vec_add(a3, a2);
    const i16x8 b2 = vec_sub(a3, a2);
    const i16x8 b3 = vec_sub(a0, a1);
    // Separate the transforms of inA (low 64b) and inB (high 64b).
    i16x8 A_b0 = (i16x8)vec_mergeh((i64x2)b0, (i64x2)b1);
    i16x8 A_b2 = (i16x8)vec_mergeh((i64x2)b2, (i64x2)b3);
    i16x8 B_b0 = (i16x8)vec_mergel((i64x2)b0, (i64x2)b1);
    i16x8 B_b2 = (i16x8)vec_mergel((i64x2)b2, (i64x2)b3);
    A_b0 = vec_max(A_b0, vec_sub(zero, A_b0));  // abs
    A_b2 = vec_max(A_b2, vec_sub(zero, A_b2));
    B_b0 = vec_max(B_b0, vec_sub(zero, B_b0));
    B_b2 = vec_max(B_b2, vec_sub(zero, B_b2));
    {
      const i32x4 sA =
          vec_add(vec_msum(A_b0, w0, z32), vec_msum(A_b2, w8, z32));
      const i32x4 sB =
          vec_add(vec_msum(B_b0, w0, z32), vec_msum(B_b2, w8, z32));
      return HorizontalSumS32_VSX(vec_sub(sA, sB));
    }
  }
}

static int Disto4x4_VSX(const uint8_t* WEBP_RESTRICT const a,
                        const uint8_t* WEBP_RESTRICT const b,
                        const uint16_t* WEBP_RESTRICT const w) {
  const int diff_sum = TTransform_VSX(a, b, w);
  return abs(diff_sum) >> 5;
}

static int Disto16x16_VSX(const uint8_t* WEBP_RESTRICT const a,
                          const uint8_t* WEBP_RESTRICT const b,
                          const uint16_t* WEBP_RESTRICT const w) {
  int D = 0;
  int x, y;
  for (y = 0; y < 16 * BPS; y += 4 * BPS) {
    for (x = 0; x < 16; x += 4) {
      D += Disto4x4_VSX(a + x + y, b + x + y, w);
    }
  }
  return D;
}

//------------------------------------------------------------------------------
// Histogram collection.

static void CollectHistogram_VSX(const uint8_t* WEBP_RESTRICT ref,
                                 const uint8_t* WEBP_RESTRICT pred,
                                 int start_block, int end_block,
                                 VP8Histogram* WEBP_RESTRICT const histo) {
  const i16x8 zero = vec_splats((short)0);
  const i16x8 max_coeff_thresh = vec_splats((short)MAX_COEFF_THRESH);
  const u16x8 three = vec_splats((unsigned short)3);
  int j;
  int distribution[MAX_COEFF_THRESH + 1] = {0};
  for (j = start_block; j < end_block; ++j) {
    int16_t out[16];
    int k;
    FTransform_One_VSX(ref + VP8DspScan[j], pred + VP8DspScan[j], out);
    {
      const i16x8 out0 = (i16x8)vec_xl(0, (int16_t*)&out[0]);
      const i16x8 out1 = (i16x8)vec_xl(0, (int16_t*)&out[8]);
      const i16x8 abs0 = vec_max(out0, vec_sub(zero, out0));
      const i16x8 abs1 = vec_max(out1, vec_sub(zero, out1));
      const i16x8 v0 = vec_sra(abs0, three);
      const i16x8 v1 = vec_sra(abs1, three);
      vec_xst(vec_min(v0, max_coeff_thresh), 0, &out[0]);
      vec_xst(vec_min(v1, max_coeff_thresh), 0, &out[8]);
    }
    for (k = 0; k < 16; ++k) {
      ++distribution[out[k]];
    }
  }
  VP8SetHistogramData(distribution, histo);
}

//------------------------------------------------------------------------------
// Quantization.

// Permutation controls that gather the natural-order levels into zigzag scan
// order: out[n] = level[kZigzag[n]]. kZigzag = {0,1,4,8,5,2,3,6,
//                                               9,12,13,10,7,11,14,15}.
// Each coefficient is 16-bit, so control byte 2n picks byte 2*kZigzag[n] of
// the concatenated (level[0..7] | level[8..15]) operand pair.
static const u8x16 kZigzagLo = {0,  1,  2, 3, 8, 9, 16, 17,
                                10, 11, 4, 5, 6, 7, 12, 13};
static const u8x16 kZigzagHi = {18, 19, 24, 25, 26, 27, 20, 21,
                                14, 15, 22, 23, 28, 29, 30, 31};

static WEBP_INLINE int DoQuantizeBlock_VSX(
    int16_t in[16], int16_t out[16],
    const uint16_t* WEBP_RESTRICT const sharpen,
    const VP8Matrix* WEBP_RESTRICT const mtx) {
  const i16x8 max_coeff_2047 = vec_splats((short)MAX_LEVEL);
  const i16x8 zero = vec_splats((short)0);
  const u32x4 qfix = vec_splats((unsigned int)QFIX);
  i16x8 in0 = (i16x8)vec_xl(0, &in[0]);
  i16x8 in8 = (i16x8)vec_xl(0, &in[8]);
  const u16x8 iq0 = (u16x8)vec_xl(0, (uint16_t*)&mtx->iq[0]);
  const u16x8 iq8 = (u16x8)vec_xl(0, (uint16_t*)&mtx->iq[8]);
  const i16x8 q0 = (i16x8)vec_xl(0, (uint16_t*)&mtx->q[0]);
  const i16x8 q8 = (i16x8)vec_xl(0, (uint16_t*)&mtx->q[8]);
  // sign(in): 0x0000 if positive, 0xffff if negative.
  const i16x8 sign0 = (i16x8)vec_cmpgt(zero, in0);
  const i16x8 sign8 = (i16x8)vec_cmpgt(zero, in8);
  // coeff = abs(in) = (in ^ sign) - sign.
  u16x8 coeff0 = (u16x8)vec_sub(vec_xor(in0, sign0), sign0);
  u16x8 coeff8 = (u16x8)vec_sub(vec_xor(in8, sign8), sign8);
  i16x8 out0, out8;
  if (sharpen != NULL) {
    coeff0 = vec_add(coeff0, (u16x8)vec_xl(0, (uint16_t*)&sharpen[0]));
    coeff8 = vec_add(coeff8, (u16x8)vec_xl(0, (uint16_t*)&sharpen[8]));
  }
  {  // out = QUANTDIV(coeff, iQ, B, QFIX), 32-bit precision.
    const u32x4 p0e = vec_mule(coeff0, iq0);  // even-lane 32b products
    const u32x4 p0o = vec_mulo(coeff0, iq0);  // odd-lane
    const u32x4 p8e = vec_mule(coeff8, iq8);
    const u32x4 p8o = vec_mulo(coeff8, iq8);
    i32x4 out_00 = (i32x4)vec_mergeh(p0e, p0o);  // products 0..3
    i32x4 out_04 = (i32x4)vec_mergel(p0e, p0o);  // products 4..7
    i32x4 out_08 = (i32x4)vec_mergeh(p8e, p8o);
    i32x4 out_12 = (i32x4)vec_mergel(p8e, p8o);
    out_00 = vec_add(out_00, (i32x4)vec_xl(0, (uint32_t*)&mtx->bias[0]));
    out_04 = vec_add(out_04, (i32x4)vec_xl(0, (uint32_t*)&mtx->bias[4]));
    out_08 = vec_add(out_08, (i32x4)vec_xl(0, (uint32_t*)&mtx->bias[8]));
    out_12 = vec_add(out_12, (i32x4)vec_xl(0, (uint32_t*)&mtx->bias[12]));
    out_00 = vec_sra(out_00, qfix);
    out_04 = vec_sra(out_04, qfix);
    out_08 = vec_sra(out_08, qfix);
    out_12 = vec_sra(out_12, qfix);
    out0 = vec_min(vec_packs(out_00, out_04), max_coeff_2047);
    out8 = vec_min(vec_packs(out_08, out_12), max_coeff_2047);
  }
  // restore sign: if (sign) out = -out.
  out0 = vec_sub(vec_xor(out0, sign0), sign0);
  out8 = vec_sub(vec_xor(out8, sign8), sign8);
  // in = out * Q.
  vec_xst(vec_mladd(out0, q0, zero), 0, &in[0]);
  vec_xst(vec_mladd(out8, q8, zero), 0, &in[8]);
  // zigzag-scan the levels into out[].
  {
    const i16x8 z0 = (i16x8)vec_perm((u8x16)out0, (u8x16)out8, kZigzagLo);
    const i16x8 z8 = (i16x8)vec_perm((u8x16)out0, (u8x16)out8, kZigzagHi);
    vec_xst(z0, 0, &out[0]);
    vec_xst(z8, 0, &out[8]);
    return !vec_all_eq(vec_packs(z0, z8), vec_splats((signed char)0));
  }
}

static int QuantizeBlock_VSX(int16_t in[16], int16_t out[16],
                             const VP8Matrix* WEBP_RESTRICT const mtx) {
  return DoQuantizeBlock_VSX(in, out, &mtx->sharpen[0], mtx);
}

static int QuantizeBlockWHT_VSX(int16_t in[16], int16_t out[16],
                                const VP8Matrix* WEBP_RESTRICT const mtx) {
  return DoQuantizeBlock_VSX(in, out, NULL, mtx);
}

static int Quantize2Blocks_VSX(int16_t in[32], int16_t out[32],
                               const VP8Matrix* WEBP_RESTRICT const mtx) {
  int nz;
  const uint16_t* const sharpen = &mtx->sharpen[0];
  nz = DoQuantizeBlock_VSX(in + 0 * 16, out + 0 * 16, sharpen, mtx) << 0;
  nz |= DoQuantizeBlock_VSX(in + 1 * 16, out + 1 * 16, sharpen, mtx) << 1;
  return nz;
}

//------------------------------------------------------------------------------

extern void VP8EncDspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspInitVSX(void) {
  VP8ITransform = ITransform_VSX;
  VP8EncQuantizeBlock = QuantizeBlock_VSX;
  VP8EncQuantize2Blocks = Quantize2Blocks_VSX;
  VP8EncQuantizeBlockWHT = QuantizeBlockWHT_VSX;
  VP8TDisto4x4 = Disto4x4_VSX;
  VP8TDisto16x16 = Disto16x16_VSX;
  VP8CollectHistogram = CollectHistogram_VSX;
  VP8FTransform = FTransform_VSX;
  VP8FTransform2 = FTransform2_VSX;
  VP8FTransformWHT = FTransformWHT_VSX;
  VP8SSE16x16 = SSE16x16_VSX;
  VP8SSE16x8 = SSE16x8_VSX;
  VP8SSE8x8 = SSE8x8_VSX;
  VP8SSE4x4 = SSE4x4_VSX;
  VP8Mean16x4 = Mean16x4_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8EncDspInitVSX)

#endif  // WEBP_USE_VSX
