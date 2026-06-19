// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC64) version of the cost functions.
//
// Authors: Trung Lê (8@tle.id.au)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <assert.h>

#include "src/dsp/cpu.h"
#include "src/enc/cost_enc.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/utils.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector signed char i8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector unsigned long long u64x2;

// Equivalent of _mm_movemask_epi8: bit i = the MSB of byte i.
static WEBP_INLINE uint32_t MoveMask8(u8x16 v) {
  const u8x16 perm = {120, 112, 104, 96, 88, 80, 72, 64,
                      56,  48,  40,  32, 24, 16, 8,  0};
  return (uint32_t)((u64x2)vec_vbpermq(v, perm))[1];
}

static void SetResidualCoeffs_VSX(const int16_t* WEBP_RESTRICT const coeffs,
                                  VP8Residual* WEBP_RESTRICT const res) {
  const i16x8 c0 = (i16x8)vec_xl(0, (int16_t*)(coeffs + 0));
  const i16x8 c1 = (i16x8)vec_xl(0, (int16_t*)(coeffs + 8));
  const i8x16 zero = vec_splats((signed char)0);
  const i8x16 m0 = vec_packs(c0, c1);
  const u8x16 m1 = (u8x16)vec_cmpeq(m0, zero);
  // Negate the mask to get the position of entries that are not zero.
  const uint32_t mask = 0x0000ffffu ^ MoveMask8(m1);
  assert(res->first == 0 || coeffs[0] == 0);
  res->last = mask ? BitsLog2Floor(mask) : -1;
  res->coeffs = coeffs;
}

static int GetResidualCost_VSX(int ctx0, const VP8Residual* const res) {
  uint8_t levels[16], ctxs[16];
  uint16_t abs_levels[16];
  int n = res->first;
  // should be prob[VP8EncBands[n]], but it's equivalent for n=0 or 1
  const int p0 = res->prob[n][ctx0][0];
  CostArrayPtr const costs = res->costs;
  const uint16_t* t = costs[n][ctx0];
  // bit_cost(1, p0) is already incorporated in t[] tables, but only if ctx != 0
  // (as required by the syntax). For ctx0 == 0, we need to add it here or it'll
  // be missing during the loop.
  int cost = (ctx0 == 0) ? VP8BitCost(1, p0) : 0;

  if (res->last < 0) {
    return VP8BitCost(0, p0);
  }

  {  // precompute clamped levels and contexts, packed to 8b.
    const i16x8 zero = vec_splats((signed short)0);
    const u8x16 kCst2 = vec_splats((unsigned char)2);
    const u8x16 kCst67 = vec_splats((unsigned char)MAX_VARIABLE_LEVEL);
    const i16x8 c0 = (i16x8)vec_xl(0, (int16_t*)&res->coeffs[0]);
    const i16x8 c1 = (i16x8)vec_xl(0, (int16_t*)&res->coeffs[8]);
    const i16x8 E0 = vec_max(c0, vec_sub(zero, c0));  // abs(v), 16b
    const i16x8 E1 = vec_max(c1, vec_sub(zero, c1));
    const u8x16 F = (u8x16)vec_packs(E0, E1);
    const u8x16 G = vec_min(F, kCst2);   // context = 0,1,2
    const u8x16 H = vec_min(F, kCst67);  // clamp_level in [0..67]

    vec_xst(G, 0, ctxs);
    vec_xst(H, 0, levels);
    vec_xst((u16x8)E0, 0, &abs_levels[0]);
    vec_xst((u16x8)E1, 0, &abs_levels[8]);
  }
  for (; n < res->last; ++n) {
    const int ctx = ctxs[n];
    const int level = levels[n];
    const int flevel = abs_levels[n];               // full level
    cost += VP8LevelFixedCosts[flevel] + t[level];  // simplified VP8LevelCost()
    t = costs[n + 1][ctx];
  }
  // Last coefficient is always non-zero
  {
    const int level = levels[n];
    const int flevel = abs_levels[n];
    assert(flevel != 0);
    cost += VP8LevelFixedCosts[flevel] + t[level];
    if (n < 15) {
      const int b = VP8EncBands[n + 1];
      const int ctx = ctxs[n];
      const int last_p0 = res->prob[b][ctx][0];
      cost += VP8BitCost(0, last_p0);
    }
  }
  return cost;
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8EncDspCostInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspCostInitVSX(void) {
  VP8SetResidualCoeffs = SetResidualCoeffs_VSX;
  VP8GetResidualCost = GetResidualCost_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8EncDspCostInitVSX)

#endif  // WEBP_USE_VSX
