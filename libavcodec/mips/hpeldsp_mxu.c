/*
 * Copyright (c) 2024 OpenSensor Project
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Ingenic XBurst2 optimised half-pel DSP functions.
 *
 * Optimisations over -Os compiled C:
 *  - Word-sized (32-bit) loads and stores for pixel copy/averaging
 *  - Byte-parallel rounding/truncating average via bit-manipulation
 *    (MXU1 Q8AVG/Q8AVGR are NOT available on XBurst2 A1/T41)
 *  - Minimised loop overhead with counted loops
 *
 * NOTE: The init path calls ff_mxu_ensure_cu2() before these functions are
 * registered, so executing MXU instructions here is safe on XBurst2.
 */

#include <stdint.h>

#include "libavutil/intreadwrite.h"
#include "libavutil/mem_internal.h"
#include "hpeldsp_mips.h"
#include "mxu.h"

#if HAVE_INLINE_ASM
/* Per-byte mask used by rnd_avg32/no_rnd_avg32 bit-hacks: 0xFE in each byte. */
static const uint32_t vpr_mask_fefefefe[16] __attribute__((aligned(64))) = {
    0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU,
    0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU,
    0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU,
    0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU, 0xFEFEFEFEU,
};
#endif

/*
 * rnd_avg32 / no_rnd_avg32 are provided by rnd_avg.h (included via
 * hpeldsp_mips.h → bit_depth_template.c).  They use portable
 * bit-manipulation which is equivalent to MXU1 Q8AVGR / Q8AVG
 * (the MXU1 instructions SIGILL on XBurst2 A1/T41).
 */

static inline int ptr_is_aligned4(const void *p, ptrdiff_t stride)
{
    return ((((uintptr_t)p) | (uintptr_t)stride) & 3) == 0;
}

/* ---- put_pixels: straight copy ---- */

void ff_put_pixels16_mxu(uint8_t *block, const uint8_t *pixels,
                          ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    if (src_aligned) {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            /* Prefetch a couple of rows ahead to hide SDRAM latency on XBurst2. */
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_STORE(block, pref_off);
            }
#endif
            AV_WN32A(block,      AV_RN32A(pixels));
            AV_WN32A(block + 4,  AV_RN32A(pixels + 4));
            AV_WN32A(block + 8,  AV_RN32A(pixels + 8));
            AV_WN32A(block + 12, AV_RN32A(pixels + 12));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            /* Prefetch a couple of rows ahead to hide SDRAM latency on XBurst2. */
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_STORE(block, pref_off);
            }
#endif
            AV_WN32A(block,      AV_RN32(pixels));
            AV_WN32A(block + 4,  AV_RN32(pixels + 4));
            AV_WN32A(block + 8,  AV_RN32(pixels + 8));
            AV_WN32A(block + 12, AV_RN32(pixels + 12));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels8_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block,     AV_RN32A(pixels));
            AV_WN32A(block + 4, AV_RN32A(pixels + 4));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block,     AV_RN32(pixels));
            AV_WN32A(block + 4, AV_RN32(pixels + 4));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels4_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, AV_RN32A(pixels));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, AV_RN32(pixels));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- avg_pixels: rounding average with destination ---- */

void ff_avg_pixels16_mxu(uint8_t *block, const uint8_t *pixels,
                          ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
    /* VPR constants/scratch (aligned for LA0/SA0) */
    LOCAL_ALIGNED_64(uint32_t, va, [16]);
    LOCAL_ALIGNED_64(uint32_t, vb, [16]);
    LOCAL_ALIGNED_64(uint32_t, vr, [16]);
    LA0_VPR_AT(5, vpr_mask_fefefefe);
#endif
    if (src_aligned) {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_LOAD(block,  pref_off);
                PREF_STORE(block, pref_off);
            }

            /* Compute 4x rnd_avg32(dst, src) in VPR (lanes 0..3). */
            va[0] = AV_RN32A(block);
            va[1] = AV_RN32A(block + 4);
            va[2] = AV_RN32A(block + 8);
            va[3] = AV_RN32A(block + 12);
            vb[0] = AV_RN32A(pixels);
            vb[1] = AV_RN32A(pixels + 4);
            vb[2] = AV_RN32A(pixels + 8);
            vb[3] = AV_RN32A(pixels + 12);
            va[4] = va[5] = va[6] = va[7] = va[8] = va[9] = va[10] = va[11] = va[12] = va[13] = va[14] = va[15] = 0;
            vb[4] = vb[5] = vb[6] = vb[7] = vb[8] = vb[9] = vb[10] = vb[11] = vb[12] = vb[13] = vb[14] = vb[15] = 0;

            LA0_VPR_AT(0, va);
            LA0_VPR_AT(1, vb);
            VPR_OR(2, 0, 1);           /* or  */
            VPR_AND(3, 0, 1);          /* and */
            VPR_SUBUW(4, 2, 3);        /* xor = or - and (borrow-free) */
            VPR_AND(4, 4, 5);          /* xor & 0xFEFEFEFE */
            VPR_SRLW_IMM(4, 4, 1);     /* >> 1 */
            VPR_SUBUW(6, 2, 4);        /* rnd_avg32 */
            SA0_VPR_AT(6, vr);

            AV_WN32A(block,      vr[0]);
            AV_WN32A(block + 4,  vr[1]);
            AV_WN32A(block + 8,  vr[2]);
            AV_WN32A(block + 12, vr[3]);
#else
            AV_WN32A(block,      rnd_avg32(AV_RN32A(block),      AV_RN32A(pixels)));
            AV_WN32A(block + 4,  rnd_avg32(AV_RN32A(block + 4),  AV_RN32A(pixels + 4)));
            AV_WN32A(block + 8,  rnd_avg32(AV_RN32A(block + 8),  AV_RN32A(pixels + 8)));
            AV_WN32A(block + 12, rnd_avg32(AV_RN32A(block + 12), AV_RN32A(pixels + 12)));
#endif
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_LOAD(block,  pref_off);
                PREF_STORE(block, pref_off);
            }

            va[0] = AV_RN32A(block);
            va[1] = AV_RN32A(block + 4);
            va[2] = AV_RN32A(block + 8);
            va[3] = AV_RN32A(block + 12);
            vb[0] = AV_RN32(pixels);
            vb[1] = AV_RN32(pixels + 4);
            vb[2] = AV_RN32(pixels + 8);
            vb[3] = AV_RN32(pixels + 12);
            va[4] = va[5] = va[6] = va[7] = va[8] = va[9] = va[10] = va[11] = va[12] = va[13] = va[14] = va[15] = 0;
            vb[4] = vb[5] = vb[6] = vb[7] = vb[8] = vb[9] = vb[10] = vb[11] = vb[12] = vb[13] = vb[14] = vb[15] = 0;

            LA0_VPR_AT(0, va);
            LA0_VPR_AT(1, vb);
            VPR_OR(2, 0, 1);
            VPR_AND(3, 0, 1);
            VPR_SUBUW(4, 2, 3);
            VPR_AND(4, 4, 5);
            VPR_SRLW_IMM(4, 4, 1);
            VPR_SUBUW(6, 2, 4);
            SA0_VPR_AT(6, vr);

            AV_WN32A(block,      vr[0]);
            AV_WN32A(block + 4,  vr[1]);
            AV_WN32A(block + 8,  vr[2]);
            AV_WN32A(block + 12, vr[3]);
#else
            AV_WN32A(block,      rnd_avg32(AV_RN32A(block),      AV_RN32(pixels)));
            AV_WN32A(block + 4,  rnd_avg32(AV_RN32A(block + 4),  AV_RN32(pixels + 4)));
            AV_WN32A(block + 8,  rnd_avg32(AV_RN32A(block + 8),  AV_RN32(pixels + 8)));
            AV_WN32A(block + 12, rnd_avg32(AV_RN32A(block + 12), AV_RN32(pixels + 12)));
#endif
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_avg_pixels8_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     AV_RN32A(pixels)));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), AV_RN32A(pixels + 4)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     AV_RN32(pixels)));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), AV_RN32(pixels + 4)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_avg_pixels4_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), AV_RN32A(pixels)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), AV_RN32(pixels)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- put_pixels_x2: horizontal half-pel (average of [x] and [x+1]) ---- */

void ff_put_pixels16_x2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
    LOCAL_ALIGNED_64(uint32_t, va, [16]);
    LOCAL_ALIGNED_64(uint32_t, vb, [16]);
    LOCAL_ALIGNED_64(uint32_t, vr, [16]);
    LA0_VPR_AT(5, vpr_mask_fefefefe);
#endif
    if (src_aligned) {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_STORE(block, pref_off);
            }
#endif
            uint32_t a0 = AV_RN32A(pixels),      b0 = AV_RN32(pixels + 1);
            uint32_t a1 = AV_RN32A(pixels + 4),  b1 = AV_RN32(pixels + 5);
            uint32_t a2 = AV_RN32A(pixels + 8),  b2 = AV_RN32(pixels + 9);
            uint32_t a3 = AV_RN32A(pixels + 12), b3 = AV_RN32(pixels + 13);
#if HAVE_INLINE_ASM
            va[0] = a0; va[1] = a1; va[2] = a2; va[3] = a3;
            vb[0] = b0; vb[1] = b1; vb[2] = b2; vb[3] = b3;
            va[4] = va[5] = va[6] = va[7] = va[8] = va[9] = va[10] = va[11] = va[12] = va[13] = va[14] = va[15] = 0;
            vb[4] = vb[5] = vb[6] = vb[7] = vb[8] = vb[9] = vb[10] = vb[11] = vb[12] = vb[13] = vb[14] = vb[15] = 0;
            LA0_VPR_AT(0, va);
            LA0_VPR_AT(1, vb);
            VPR_OR(2, 0, 1);
            VPR_AND(3, 0, 1);
            VPR_SUBUW(4, 2, 3);
            VPR_AND(4, 4, 5);
            VPR_SRLW_IMM(4, 4, 1);
            VPR_SUBUW(6, 2, 4);
            SA0_VPR_AT(6, vr);
            AV_WN32A(block,      vr[0]);
            AV_WN32A(block + 4,  vr[1]);
            AV_WN32A(block + 8,  vr[2]);
            AV_WN32A(block + 12, vr[3]);
#else
            AV_WN32A(block,      rnd_avg32(a0, b0));
            AV_WN32A(block + 4,  rnd_avg32(a1, b1));
            AV_WN32A(block + 8,  rnd_avg32(a2, b2));
            AV_WN32A(block + 12, rnd_avg32(a3, b3));
#endif
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_STORE(block, pref_off);
            }
#endif
            uint32_t a0 = AV_RN32(pixels),      b0 = AV_RN32(pixels + 1);
            uint32_t a1 = AV_RN32(pixels + 4),  b1 = AV_RN32(pixels + 5);
            uint32_t a2 = AV_RN32(pixels + 8),  b2 = AV_RN32(pixels + 9);
            uint32_t a3 = AV_RN32(pixels + 12), b3 = AV_RN32(pixels + 13);
#if HAVE_INLINE_ASM
            va[0] = a0; va[1] = a1; va[2] = a2; va[3] = a3;
            vb[0] = b0; vb[1] = b1; vb[2] = b2; vb[3] = b3;
            va[4] = va[5] = va[6] = va[7] = va[8] = va[9] = va[10] = va[11] = va[12] = va[13] = va[14] = va[15] = 0;
            vb[4] = vb[5] = vb[6] = vb[7] = vb[8] = vb[9] = vb[10] = vb[11] = vb[12] = vb[13] = vb[14] = vb[15] = 0;
            LA0_VPR_AT(0, va);
            LA0_VPR_AT(1, vb);
            VPR_OR(2, 0, 1);
            VPR_AND(3, 0, 1);
            VPR_SUBUW(4, 2, 3);
            VPR_AND(4, 4, 5);
            VPR_SRLW_IMM(4, 4, 1);
            VPR_SUBUW(6, 2, 4);
            SA0_VPR_AT(6, vr);
            AV_WN32A(block,      vr[0]);
            AV_WN32A(block + 4,  vr[1]);
            AV_WN32A(block + 8,  vr[2]);
            AV_WN32A(block + 12, vr[3]);
#else
            AV_WN32A(block,      rnd_avg32(a0, b0));
            AV_WN32A(block + 4,  rnd_avg32(a1, b1));
            AV_WN32A(block + 8,  rnd_avg32(a2, b2));
            AV_WN32A(block + 12, rnd_avg32(a3, b3));
#endif
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels8_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block,     rnd_avg32(AV_RN32A(pixels),     AV_RN32(pixels + 1)));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(pixels + 4), AV_RN32(pixels + 5)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block,     rnd_avg32(AV_RN32(pixels),     AV_RN32(pixels + 1)));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32(pixels + 4), AV_RN32(pixels + 5)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels4_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, rnd_avg32(AV_RN32A(pixels), AV_RN32(pixels + 1)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + 1)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- put_pixels_y2: vertical half-pel (average of [y] and [y+stride]) ---- */

void ff_put_pixels16_y2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
    LOCAL_ALIGNED_64(uint32_t, va, [16]);
    LOCAL_ALIGNED_64(uint32_t, vb, [16]);
    LOCAL_ALIGNED_64(uint32_t, vr, [16]);
    LA0_VPR_AT(5, vpr_mask_fefefefe);
#endif
    if (src_aligned) {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_STORE(block, pref_off);
            }
#endif
            const uint8_t *p1 = pixels + line_size;
#if HAVE_INLINE_ASM
            va[0] = AV_RN32A(pixels);
            va[1] = AV_RN32A(pixels + 4);
            va[2] = AV_RN32A(pixels + 8);
            va[3] = AV_RN32A(pixels + 12);
            vb[0] = AV_RN32A(p1);
            vb[1] = AV_RN32A(p1 + 4);
            vb[2] = AV_RN32A(p1 + 8);
            vb[3] = AV_RN32A(p1 + 12);
            va[4] = va[5] = va[6] = va[7] = va[8] = va[9] = va[10] = va[11] = va[12] = va[13] = va[14] = va[15] = 0;
            vb[4] = vb[5] = vb[6] = vb[7] = vb[8] = vb[9] = vb[10] = vb[11] = vb[12] = vb[13] = vb[14] = vb[15] = 0;
            LA0_VPR_AT(0, va);
            LA0_VPR_AT(1, vb);
            VPR_OR(2, 0, 1);
            VPR_AND(3, 0, 1);
            VPR_SUBUW(4, 2, 3);
            VPR_AND(4, 4, 5);
            VPR_SRLW_IMM(4, 4, 1);
            VPR_SUBUW(6, 2, 4);
            SA0_VPR_AT(6, vr);
            AV_WN32A(block,      vr[0]);
            AV_WN32A(block + 4,  vr[1]);
            AV_WN32A(block + 8,  vr[2]);
            AV_WN32A(block + 12, vr[3]);
#else
            AV_WN32A(block,      rnd_avg32(AV_RN32A(pixels),      AV_RN32A(p1)));
            AV_WN32A(block + 4,  rnd_avg32(AV_RN32A(pixels + 4),  AV_RN32A(p1 + 4)));
            AV_WN32A(block + 8,  rnd_avg32(AV_RN32A(pixels + 8),  AV_RN32A(p1 + 8)));
            AV_WN32A(block + 12, rnd_avg32(AV_RN32A(pixels + 12), AV_RN32A(p1 + 12)));
#endif
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
            if (i + 2 < h) {
                PREF_LOAD(pixels, pref_off);
                PREF_STORE(block, pref_off);
            }
#endif
            const uint8_t *p1 = pixels + line_size;
#if HAVE_INLINE_ASM
            va[0] = AV_RN32(pixels);
            va[1] = AV_RN32(pixels + 4);
            va[2] = AV_RN32(pixels + 8);
            va[3] = AV_RN32(pixels + 12);
            vb[0] = AV_RN32(p1);
            vb[1] = AV_RN32(p1 + 4);
            vb[2] = AV_RN32(p1 + 8);
            vb[3] = AV_RN32(p1 + 12);
            va[4] = va[5] = va[6] = va[7] = va[8] = va[9] = va[10] = va[11] = va[12] = va[13] = va[14] = va[15] = 0;
            vb[4] = vb[5] = vb[6] = vb[7] = vb[8] = vb[9] = vb[10] = vb[11] = vb[12] = vb[13] = vb[14] = vb[15] = 0;
            LA0_VPR_AT(0, va);
            LA0_VPR_AT(1, vb);
            VPR_OR(2, 0, 1);
            VPR_AND(3, 0, 1);
            VPR_SUBUW(4, 2, 3);
            VPR_AND(4, 4, 5);
            VPR_SRLW_IMM(4, 4, 1);
            VPR_SUBUW(6, 2, 4);
            SA0_VPR_AT(6, vr);
            AV_WN32A(block,      vr[0]);
            AV_WN32A(block + 4,  vr[1]);
            AV_WN32A(block + 8,  vr[2]);
            AV_WN32A(block + 12, vr[3]);
#else
            AV_WN32A(block,      rnd_avg32(AV_RN32(pixels),      AV_RN32(p1)));
            AV_WN32A(block + 4,  rnd_avg32(AV_RN32(pixels + 4),  AV_RN32(p1 + 4)));
            AV_WN32A(block + 8,  rnd_avg32(AV_RN32(pixels + 8),  AV_RN32(p1 + 8)));
            AV_WN32A(block + 12, rnd_avg32(AV_RN32(pixels + 12), AV_RN32(p1 + 12)));
#endif
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels8_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block,     rnd_avg32(AV_RN32A(pixels),     AV_RN32A(p1)));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(pixels + 4), AV_RN32A(p1 + 4)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block,     rnd_avg32(AV_RN32(pixels),     AV_RN32(p1)));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32(pixels + 4), AV_RN32(p1 + 4)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels4_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, rnd_avg32(AV_RN32A(pixels), AV_RN32A(pixels + line_size)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            AV_WN32A(block, rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + line_size)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- put_pixels_xy2: bilinear (average of 4 neighbours) ---- */

/**
 * Byte-parallel bilinear average of 4 uint32 values:
 *   result[i] = (a[i] + b[i] + c[i] + d[i] + 2) >> 2
 *
 * Uses the decomposition: avg4 = avg(avg(a,b), avg(c,d)) which
 * approximates the correct result with <=1 error. For exact results
 * we use a correction step.
 */
static inline uint32_t avg4_round(uint32_t a, uint32_t b,
                                  uint32_t c, uint32_t d)
{
    /* Two-stage averaging with rounding correction:
     * First average each pair, then average the results.
     * The rounding bias of +2 is inherent in the two rnd_avg32 stages. */
    return rnd_avg32(rnd_avg32(a, b), rnd_avg32(c, d));
}

void ff_put_pixels16_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                              ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        const uint8_t *p1 = pixels + line_size;
        int j;
        if (src_aligned) {
            for (j = 0; j < 16; j += 4) {
                AV_WN32A(block + j, avg4_round(AV_RN32A(pixels + j),
                                               AV_RN32(pixels + j + 1),
                                               AV_RN32A(p1 + j),
                                               AV_RN32(p1 + j + 1)));
            }
        } else {
            for (j = 0; j < 16; j += 4) {
                AV_WN32A(block + j, avg4_round(AV_RN32(pixels + j),
                                               AV_RN32(pixels + j + 1),
                                               AV_RN32(p1 + j),
                                               AV_RN32(p1 + j + 1)));
            }
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels8_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block,     avg4_round(AV_RN32A(pixels),     AV_RN32(pixels + 1),
                                           AV_RN32A(p1),         AV_RN32(p1 + 1)));
            AV_WN32A(block + 4, avg4_round(AV_RN32A(pixels + 4), AV_RN32(pixels + 5),
                                           AV_RN32A(p1 + 4),     AV_RN32(p1 + 5)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block,     avg4_round(AV_RN32(pixels),     AV_RN32(pixels + 1),
                                           AV_RN32(p1),         AV_RN32(p1 + 1)));
            AV_WN32A(block + 4, avg4_round(AV_RN32(pixels + 4), AV_RN32(pixels + 5),
                                           AV_RN32(p1 + 4),     AV_RN32(p1 + 5)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_put_pixels4_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block, avg4_round(AV_RN32A(pixels), AV_RN32(pixels + 1),
                                       AV_RN32A(p1),     AV_RN32(p1 + 1)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block, avg4_round(AV_RN32(pixels), AV_RN32(pixels + 1),
                                       AV_RN32(p1),     AV_RN32(p1 + 1)));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- put_no_rnd_pixels_x2: horizontal half-pel, truncating ---- */

void ff_put_no_rnd_pixels16_x2_mxu(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int32_t h)
{
    int i;
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        uint32_t a0 = AV_RN32(pixels),      b0 = AV_RN32(pixels + 1);
        uint32_t a1 = AV_RN32(pixels + 4),  b1 = AV_RN32(pixels + 5);
        uint32_t a2 = AV_RN32(pixels + 8),  b2 = AV_RN32(pixels + 9);
        uint32_t a3 = AV_RN32(pixels + 12), b3 = AV_RN32(pixels + 13);
        AV_WN32A(block,      no_rnd_avg32(a0, b0));
        AV_WN32A(block + 4,  no_rnd_avg32(a1, b1));
        AV_WN32A(block + 8,  no_rnd_avg32(a2, b2));
        AV_WN32A(block + 12, no_rnd_avg32(a3, b3));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_no_rnd_pixels8_x2_mxu(uint8_t *block, const uint8_t *pixels,
                                   ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,     no_rnd_avg32(AV_RN32(pixels),     AV_RN32(pixels + 1)));
        AV_WN32A(block + 4, no_rnd_avg32(AV_RN32(pixels + 4), AV_RN32(pixels + 5)));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- put_no_rnd_pixels_y2: vertical half-pel, truncating ---- */

void ff_put_no_rnd_pixels16_y2_mxu(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int32_t h)
{
    int i;
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        const uint8_t *p1 = pixels + line_size;
        AV_WN32A(block,      no_rnd_avg32(AV_RN32(pixels),      AV_RN32(p1)));
        AV_WN32A(block + 4,  no_rnd_avg32(AV_RN32(pixels + 4),  AV_RN32(p1 + 4)));
        AV_WN32A(block + 8,  no_rnd_avg32(AV_RN32(pixels + 8),  AV_RN32(p1 + 8)));
        AV_WN32A(block + 12, no_rnd_avg32(AV_RN32(pixels + 12), AV_RN32(p1 + 12)));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_no_rnd_pixels8_y2_mxu(uint8_t *block, const uint8_t *pixels,
                                   ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        AV_WN32A(block,     no_rnd_avg32(AV_RN32(pixels),     AV_RN32(p1)));
        AV_WN32A(block + 4, no_rnd_avg32(AV_RN32(pixels + 4), AV_RN32(p1 + 4)));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- put_no_rnd_pixels_xy2: bilinear, truncating ---- */

static inline uint32_t avg4_no_rnd(uint32_t a, uint32_t b,
                                   uint32_t c, uint32_t d)
{
    return no_rnd_avg32(no_rnd_avg32(a, b), no_rnd_avg32(c, d));
}

void ff_put_no_rnd_pixels16_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        const uint8_t *p1 = pixels + line_size;
        int j;
        if (src_aligned) {
            for (j = 0; j < 16; j += 4) {
                AV_WN32A(block + j, avg4_no_rnd(AV_RN32A(pixels + j),
                                                AV_RN32(pixels + j + 1),
                                                AV_RN32A(p1 + j),
                                                AV_RN32(p1 + j + 1)));
            }
        } else {
            for (j = 0; j < 16; j += 4) {
                AV_WN32A(block + j, avg4_no_rnd(AV_RN32(pixels + j),
                                                AV_RN32(pixels + j + 1),
                                                AV_RN32(p1 + j),
                                                AV_RN32(p1 + j + 1)));
            }
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_no_rnd_pixels8_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block,     avg4_no_rnd(AV_RN32A(pixels),     AV_RN32(pixels + 1),
                                            AV_RN32A(p1),         AV_RN32(p1 + 1)));
            AV_WN32A(block + 4, avg4_no_rnd(AV_RN32A(pixels + 4), AV_RN32(pixels + 5),
                                            AV_RN32A(p1 + 4),     AV_RN32(p1 + 5)));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            AV_WN32A(block,     avg4_no_rnd(AV_RN32(pixels),     AV_RN32(pixels + 1),
                                            AV_RN32(p1),         AV_RN32(p1 + 1)));
            AV_WN32A(block + 4, avg4_no_rnd(AV_RN32(pixels + 4), AV_RN32(pixels + 5),
                                            AV_RN32(p1 + 4),     AV_RN32(p1 + 5)));
            block  += line_size;
            pixels += line_size;
        }
    }
}


/* ---- avg_pixels_x2: horizontal half-pel + avg with dest ---- */

void ff_avg_pixels16_x2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_LOAD(block,  pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        int j;
        if (src_aligned) {
            for (j = 0; j < 16; j += 4) {
                uint32_t src = rnd_avg32(AV_RN32A(pixels + j), AV_RN32(pixels + j + 1));
                AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
            }
        } else {
            for (j = 0; j < 16; j += 4) {
                uint32_t src = rnd_avg32(AV_RN32(pixels + j), AV_RN32(pixels + j + 1));
                AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
            }
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            uint32_t s0 = rnd_avg32(AV_RN32A(pixels),     AV_RN32(pixels + 1));
            uint32_t s1 = rnd_avg32(AV_RN32A(pixels + 4), AV_RN32(pixels + 5));
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            uint32_t s0 = rnd_avg32(AV_RN32(pixels),     AV_RN32(pixels + 1));
            uint32_t s1 = rnd_avg32(AV_RN32(pixels + 4), AV_RN32(pixels + 5));
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_avg_pixels4_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            uint32_t src = rnd_avg32(AV_RN32A(pixels), AV_RN32(pixels + 1));
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            uint32_t src = rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + 1));
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- avg_pixels_y2: vertical half-pel + avg with dest ---- */

void ff_avg_pixels16_y2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_LOAD(block,  pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        const uint8_t *p1 = pixels + line_size;
        int j;
        if (src_aligned) {
            for (j = 0; j < 16; j += 4) {
                uint32_t src = rnd_avg32(AV_RN32A(pixels + j), AV_RN32A(p1 + j));
                AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
            }
        } else {
            for (j = 0; j < 16; j += 4) {
                uint32_t src = rnd_avg32(AV_RN32(pixels + j), AV_RN32(p1 + j));
                AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
            }
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            uint32_t s0 = rnd_avg32(AV_RN32A(pixels),     AV_RN32A(p1));
            uint32_t s1 = rnd_avg32(AV_RN32A(pixels + 4), AV_RN32A(p1 + 4));
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            uint32_t s0 = rnd_avg32(AV_RN32(pixels),     AV_RN32(p1));
            uint32_t s1 = rnd_avg32(AV_RN32(pixels + 4), AV_RN32(p1 + 4));
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_avg_pixels4_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    if (ptr_is_aligned4(pixels, line_size)) {
        for (i = 0; i < h; i++) {
            uint32_t src = rnd_avg32(AV_RN32A(pixels), AV_RN32A(pixels + line_size));
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            uint32_t src = rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + line_size));
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
            block  += line_size;
            pixels += line_size;
        }
    }
}

/* ---- avg_pixels_xy2: bilinear + avg with dest ---- */

void ff_avg_pixels16_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                              ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
#if HAVE_INLINE_ASM
    const ptrdiff_t pref_off = line_size * 2;
#endif
    for (i = 0; i < h; i++) {
#if HAVE_INLINE_ASM
        if (i + 2 < h) {
            PREF_LOAD(pixels, pref_off);
            PREF_LOAD(block,  pref_off);
            PREF_STORE(block, pref_off);
        }
#endif
        const uint8_t *p1 = pixels + line_size;
        int j;
        if (src_aligned) {
            for (j = 0; j < 16; j += 4) {
                uint32_t src = avg4_round(AV_RN32A(pixels + j),
                                          AV_RN32(pixels + j + 1),
                                          AV_RN32A(p1 + j),
                                          AV_RN32(p1 + j + 1));
                AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
            }
        } else {
            for (j = 0; j < 16; j += 4) {
                uint32_t src = avg4_round(AV_RN32(pixels + j),
                                          AV_RN32(pixels + j + 1),
                                          AV_RN32(p1 + j),
                                          AV_RN32(p1 + j + 1));
                AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
            }
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            uint32_t s0 = avg4_round(AV_RN32A(pixels),     AV_RN32(pixels + 1),
                                     AV_RN32A(p1),         AV_RN32(p1 + 1));
            uint32_t s1 = avg4_round(AV_RN32A(pixels + 4), AV_RN32(pixels + 5),
                                     AV_RN32A(p1 + 4),     AV_RN32(p1 + 5));
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            uint32_t s0 = avg4_round(AV_RN32(pixels),     AV_RN32(pixels + 1),
                                     AV_RN32(p1),         AV_RN32(p1 + 1));
            uint32_t s1 = avg4_round(AV_RN32(pixels + 4), AV_RN32(pixels + 5),
                                     AV_RN32(p1 + 4),     AV_RN32(p1 + 5));
            AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
            AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
            block  += line_size;
            pixels += line_size;
        }
    }
}

void ff_avg_pixels4_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    const int src_aligned = ptr_is_aligned4(pixels, line_size);
    if (src_aligned) {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            uint32_t src = avg4_round(AV_RN32A(pixels), AV_RN32(pixels + 1),
                                      AV_RN32A(p1),     AV_RN32(p1 + 1));
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
            block  += line_size;
            pixels += line_size;
        }
    } else {
        for (i = 0; i < h; i++) {
            const uint8_t *p1 = pixels + line_size;
            uint32_t src = avg4_round(AV_RN32(pixels), AV_RN32(pixels + 1),
                                      AV_RN32(p1),     AV_RN32(p1 + 1));
            AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
            block  += line_size;
            pixels += line_size;
        }
    }
}