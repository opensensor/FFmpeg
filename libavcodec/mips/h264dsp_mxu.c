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
 * Ingenic XBurst2 optimised H.264 IDCT and pixel functions.
 *
 * Optimisations over -Os compiled C:
 *  - Word-sized loads/stores for processing 4 pixels at a time
 *  - Hand-unrolled loops to avoid loop overhead
 *  - DC-only shortcuts for the most common IDCT case
 *  - Branchless uint8 clip for saturating arithmetic
 *
 * These functions are pure MIPS32r2 scalar code — no legacy XBurst1 MXU
 * (XR register) instructions are used, so they run safely on both
 * XBurst1 and XBurst2 (T31/T40/T41) platforms.
 */

#include <string.h>
#include <stdint.h>
#include "libavutil/intreadwrite.h"
#include "libavutil/common.h"
#include "libavcodec/h264dec.h"
#include "h264dsp_mips.h"
#include "mxu.h"

/* ---- Branchless uint8 clip ---- */

static inline uint8_t clip_uint8(int v)
{
    if (v & ~0xFF)
        return (-v) >> 31;
    return v;
}

/* ---- DC-only 4x4 IDCT add (most common case) ---- */

/*
 * Helper: add DC value to 4 packed uint8 bytes with clamping.
 * Reads a word, adds dc to each byte, clamps to [0,255], writes word.
 * Reduces 4 byte loads + 4 byte stores to 1 word load + 1 word store.
 */
static inline void dc_add_word4(uint8_t *dst, int dc)
{
    uint32_t w = AV_RN32A(dst);
    AV_WN32A(dst,
        clip_uint8(( w        & 0xFF) + dc)        |
        (clip_uint8(((w >>  8) & 0xFF) + dc) <<  8) |
        (clip_uint8(((w >> 16) & 0xFF) + dc) << 16) |
        (clip_uint8(( w >> 24)         + dc) << 24));
}

/**
 * DC-only 4x4 IDCT add — word-packed loads/stores.
 *
 * One word load + add + clip + word store per row (was 4 byte loads
 * + 4 byte stores).  This is the most frequently called IDCT path
 * in H.264 decoding since most blocks are DC-only.
 */
void ff_h264_idct_dc_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int dc = (block[0] + 32) >> 6;
    block[0] = 0;

    if (dc == 0)
        return;

    dc_add_word4(dst, dc); dst += stride;
    dc_add_word4(dst, dc); dst += stride;
    dc_add_word4(dst, dc); dst += stride;
    dc_add_word4(dst, dc);
}

/* ---- DC-only 8x8 IDCT add ---- */

/**
 * DC-only 8x8 IDCT add — word-packed loads/stores.
 *
 * Two word ops per row (8 pixels = 2 × 4-byte words).
 */
void ff_h264_idct8_dc_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    int dc = (block[0] + 32) >> 6;
    block[0] = 0;

    if (dc == 0)
        return;

    for (i = 0; i < 8; i++) {
        dc_add_word4(dst,     dc);
        dc_add_word4(dst + 4, dc);
        dst += stride;
    }
}

/* ---- Full 4x4 IDCT add ---- */

void ff_h264_idct_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;

    block[0] += 1 << 5;

    /* Column pass */
#if HAVE_INLINE_ASM
    {
        /*
         * Vectorize the 4 independent columns by treating the 4 columns as
         * 4 word-lanes in VPR.  Each lane holds one column's coefficient.
         */
        LOCAL_ALIGNED_64(int32_t, r0, [16]);
        LOCAL_ALIGNED_64(int32_t, r1, [16]);
        LOCAL_ALIGNED_64(int32_t, r2, [16]);
        LOCAL_ALIGNED_64(int32_t, r3, [16]);
        LOCAL_ALIGNED_64(int32_t, o0, [16]);
        LOCAL_ALIGNED_64(int32_t, o1, [16]);
        LOCAL_ALIGNED_64(int32_t, o2, [16]);
        LOCAL_ALIGNED_64(int32_t, o3, [16]);
        LOCAL_ALIGNED_64(int32_t, vzero, [16]) = { 0 };
        int k;

        for (k = 0; k < 16; k++)
            r0[k] = r1[k] = r2[k] = r3[k] = 0;

        /* lanes 0..3 = columns 0..3 */
        for (k = 0; k < 4; k++) {
            r0[k] = block[k + 0 * 4];
            r1[k] = block[k + 1 * 4];
            r2[k] = block[k + 2 * 4];
            r3[k] = block[k + 3 * 4];
        }

        LA0_VPR_AT(0, r0);
        LA0_VPR_AT(1, r1);
        LA0_VPR_AT(2, r2);
        LA0_VPR_AT(3, r3);
        LA0_VPR_AT(15, vzero);

        /* z0 = r0 + r2, z1 = r0 - r2 */
        VPR_ADDUW(4, 0, 2);
        VPR_SUBUW(5, 0, 2);

        /* Arithmetic >> 1 for r1 and r3 (per-lane) using SRLW + sign fill */
        /* t1 = r1 >> 1 */
        VPR_SRLW_IMM(6, 1, 31);     /* sign bit -> {0,1} */
        VPR_SUBUW(6, 15, 6);       /* 0 - sign -> {0,0xFFFFFFFF} */
        VPR_SLLW_IMM(7, 6, 31);    /* fill mask for >>1 */
        VPR_SRLW_IMM(6, 1, 1);
        VPR_OR(6, 6, 7);

        /* t3 = r3 >> 1 */
        VPR_SRLW_IMM(8, 3, 31);
        VPR_SUBUW(8, 15, 8);
        VPR_SLLW_IMM(9, 8, 31);
        VPR_SRLW_IMM(8, 3, 1);
        VPR_OR(8, 8, 9);

        /* z2 = (r1>>1) - r3, z3 = r1 + (r3>>1) */
        VPR_SUBUW(10, 6, 3);
        VPR_ADDUW(11, 1, 8);

        /* outputs */
        VPR_ADDUW(12, 4, 11);  /* row0 */
        VPR_ADDUW(13, 5, 10);  /* row1 */
        VPR_SUBUW(14, 5, 10);  /* row2 */
        VPR_SUBUW(4, 4, 11);   /* row3 (reuse v4) */

        SA0_VPR_AT(12, o0);
        SA0_VPR_AT(13, o1);
        SA0_VPR_AT(14, o2);
        SA0_VPR_AT(4,  o3);

        for (k = 0; k < 4; k++) {
            block[k + 0 * 4] = (int16_t)o0[k];
            block[k + 1 * 4] = (int16_t)o1[k];
            block[k + 2 * 4] = (int16_t)o2[k];
            block[k + 3 * 4] = (int16_t)o3[k];
        }
    }
#else
    for (i = 0; i < 4; i++) {
        const unsigned int z0 =  block[i + 4*0]     +  (unsigned)block[i + 4*2];
        const unsigned int z1 =  block[i + 4*0]     -  (unsigned)block[i + 4*2];
        const unsigned int z2 = (block[i + 4*1]>>1) -  (unsigned)block[i + 4*3];
        const unsigned int z3 =  block[i + 4*1]     + (unsigned)(block[i + 4*3]>>1);

        block[i + 4*0] = z0 + z3;
        block[i + 4*1] = z1 + z2;
        block[i + 4*2] = z1 - z2;
        block[i + 4*3] = z0 - z3;
    }
#endif

    /* Row pass + add to destination + clip (word-packed store) */
    {
#if HAVE_INLINE_ASM
        /* Vectorize 4 independent rows (lanes=rows) and feed scalar clip/store. */
        LOCAL_ALIGNED_64(int32_t, c0, [16]);
        LOCAL_ALIGNED_64(int32_t, c1, [16]);
        LOCAL_ALIGNED_64(int32_t, c2, [16]);
        LOCAL_ALIGNED_64(int32_t, c3, [16]);
        LOCAL_ALIGNED_64(int32_t, y0, [16]);
        LOCAL_ALIGNED_64(int32_t, y1, [16]);
        LOCAL_ALIGNED_64(int32_t, y2, [16]);
        LOCAL_ALIGNED_64(int32_t, y3, [16]);
        LOCAL_ALIGNED_64(int32_t, vzero, [16]) = { 0 };
        int k;

        for (k = 0; k < 16; k++)
            c0[k] = c1[k] = c2[k] = c3[k] = 0;

        for (k = 0; k < 4; k++) {
            c0[k] = block[0 + 4 * k];
            c1[k] = block[1 + 4 * k];
            c2[k] = block[2 + 4 * k];
            c3[k] = block[3 + 4 * k];
        }

        LA0_VPR_AT(0, c0);
        LA0_VPR_AT(1, c1);
        LA0_VPR_AT(2, c2);
        LA0_VPR_AT(3, c3);
        LA0_VPR_AT(15, vzero);

        /* z0 = c0 + c2, z1 = c0 - c2 */
        VPR_ADDUW(4, 0, 2);
        VPR_SUBUW(5, 0, 2);

        /* t1 = c1 >> 1 (arith) */
        VPR_SRLW_IMM(6, 1, 31);
        VPR_SUBUW(6, 15, 6);
        VPR_SLLW_IMM(7, 6, 31);
        VPR_SRLW_IMM(6, 1, 1);
        VPR_OR(6, 6, 7);

        /* t3 = c3 >> 1 (arith) */
        VPR_SRLW_IMM(8, 3, 31);
        VPR_SUBUW(8, 15, 8);
        VPR_SLLW_IMM(9, 8, 31);
        VPR_SRLW_IMM(8, 3, 1);
        VPR_OR(8, 8, 9);

        /* z2 = (c1>>1) - c3, z3 = c1 + (c3>>1) */
        VPR_SUBUW(10, 6, 3);
        VPR_ADDUW(11, 1, 8);

        /* row outputs: y0..y3 = (z0+z3),(z1+z2),(z1-z2),(z0-z3) */
        VPR_ADDUW(12, 4, 11);
        VPR_ADDUW(13, 5, 10);
        VPR_SUBUW(14, 5, 10);
        VPR_SUBUW(4,  4, 11);

        SA0_VPR_AT(12, y0);
        SA0_VPR_AT(13, y1);
        SA0_VPR_AT(14, y2);
        SA0_VPR_AT(4,  y3);

        for (i = 0; i < 4; i++) {
            uint8_t *row = dst + i * stride;
            uint32_t p = AV_RN32A(row);
            AV_WN32A(row,
                clip_uint8(( p        & 0xFF) + ((int)y0[i] >> 6))        |
                (clip_uint8(((p >>  8) & 0xFF) + ((int)y1[i] >> 6)) <<  8) |
                (clip_uint8(((p >> 16) & 0xFF) + ((int)y2[i] >> 6)) << 16) |
                (clip_uint8(( p >> 24)         + ((int)y3[i] >> 6)) << 24));
        }
#else
        for (i = 0; i < 4; i++) {
            const unsigned int z0 =  block[0 + 4*i]     +  (unsigned int)block[2 + 4*i];
            const unsigned int z1 =  block[0 + 4*i]     -  (unsigned int)block[2 + 4*i];
            const unsigned int z2 = (block[1 + 4*i]>>1) -  (unsigned int)block[3 + 4*i];
            const unsigned int z3 =  block[1 + 4*i]     + (unsigned int)(block[3 + 4*i]>>1);
            uint8_t *row = dst + i * stride;
            uint32_t p = AV_RN32A(row);
            AV_WN32A(row,
                clip_uint8(( p        & 0xFF) + ((int)(z0 + z3) >> 6))        |
                (clip_uint8(((p >>  8) & 0xFF) + ((int)(z1 + z2) >> 6)) <<  8) |
                (clip_uint8(((p >> 16) & 0xFF) + ((int)(z1 - z2) >> 6)) << 16) |
                (clip_uint8(( p >> 24)         + ((int)(z0 - z3) >> 6)) << 24));
        }
#endif
    }

    memset(block, 0, 16 * sizeof(int16_t));
}

/* ---- Full 8x8 IDCT add ---- */

void ff_h264_idct8_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;

    block[0] += 32;

    /* Column pass */
#if HAVE_INLINE_ASM
    {
        /* Lanes 0..7 = columns 0..7; process all 8 columns in parallel. */
        LOCAL_ALIGNED_64(int32_t, r0, [16]);
        LOCAL_ALIGNED_64(int32_t, r1, [16]);
        LOCAL_ALIGNED_64(int32_t, r2, [16]);
        LOCAL_ALIGNED_64(int32_t, r3, [16]);
        LOCAL_ALIGNED_64(int32_t, r4, [16]);
        LOCAL_ALIGNED_64(int32_t, r5, [16]);
        LOCAL_ALIGNED_64(int32_t, r6, [16]);
        LOCAL_ALIGNED_64(int32_t, r7, [16]);
        LOCAL_ALIGNED_64(int32_t, o0, [16]);
        LOCAL_ALIGNED_64(int32_t, o1, [16]);
        LOCAL_ALIGNED_64(int32_t, o2, [16]);
        LOCAL_ALIGNED_64(int32_t, o3, [16]);
        LOCAL_ALIGNED_64(int32_t, o4, [16]);
        LOCAL_ALIGNED_64(int32_t, o5, [16]);
        LOCAL_ALIGNED_64(int32_t, o6, [16]);
        LOCAL_ALIGNED_64(int32_t, o7, [16]);
        int k;

        for (k = 0; k < 16; k++)
            r0[k] = r1[k] = r2[k] = r3[k] = r4[k] = r5[k] = r6[k] = r7[k] = 0;

        for (k = 0; k < 8; k++) {
            r0[k] = block[k + 0 * 8];
            r1[k] = block[k + 1 * 8];
            r2[k] = block[k + 2 * 8];
            r3[k] = block[k + 3 * 8];
            r4[k] = block[k + 4 * 8];
            r5[k] = block[k + 5 * 8];
            r6[k] = block[k + 6 * 8];
            r7[k] = block[k + 7 * 8];
        }

        LA0_VPR_AT(0, r0);
        LA0_VPR_AT(1, r1);
        LA0_VPR_AT(2, r2);
        LA0_VPR_AT(3, r3);
        LA0_VPR_AT(4, r4);
        LA0_VPR_AT(5, r5);
        LA0_VPR_AT(6, r6);
        LA0_VPR_AT(7, r7);
        MXUV3_ZERO_VPR(15);

        /* Helpers: arithmetic shifts (per lane) */
        /* t = sra(src, 1) -> dst */
#define VPR_SRAW1(dst, src, tsign, tmask, tfill) do { \
            VPR_SRLW_IMM((tsign), (src), 31);      \
            VPR_SUBUW((tmask), 15, (tsign));       \
            VPR_SLLW_IMM((tfill), (tmask), 31);    \
            VPR_SRLW_IMM((dst), (src), 1);         \
            VPR_OR((dst), (dst), (tfill));         \
        } while (0)
        /* t = sra(src, 2) -> dst */
#define VPR_SRAW2(dst, src, tsign, tmask, tfill) do { \
            VPR_SRLW_IMM((tsign), (src), 31);      \
            VPR_SUBUW((tmask), 15, (tsign));       \
            VPR_SLLW_IMM((tfill), (tmask), 30);    \
            VPR_SRLW_IMM((dst), (src), 2);         \
            VPR_OR((dst), (dst), (tfill));         \
        } while (0)

        /* a0 = r0 + r4, a2 = r0 - r4 */
        VPR_ADDUW(8, 0, 4);
        VPR_SUBUW(9, 0, 4);

        /* a4 = (r2>>1) - r6, a6 = (r6>>1) + r2 */
        VPR_SRAW1(10, 2, 11, 12, 13); /* t_r2 */
        VPR_SUBUW(10, 10, 6);
        VPR_SRAW1(14, 6, 11, 12, 13); /* t_r6 */
        VPR_ADDUW(14, 14, 2);

        /* b0..b6 */
        VPR_ADDUW(16, 8, 14);  /* b0 */
        VPR_ADDUW(17, 9, 10);  /* b2 */
        VPR_SUBUW(18, 9, 10);  /* b4 */
        VPR_SUBUW(19, 8, 14);  /* b6 */

        /* a1 = r5 - r3 - r7 - (r7>>1) */
        VPR_SUBUW(20, 5, 3);
        VPR_SUBUW(20, 20, 7);
        VPR_SRAW1(21, 7, 11, 12, 13);
        VPR_SUBUW(20, 20, 21);

        /* a3 = r1 + r7 - r3 - (r3>>1) */
        VPR_ADDUW(22, 1, 7);
        VPR_SUBUW(22, 22, 3);
        VPR_SRAW1(21, 3, 11, 12, 13);
        VPR_SUBUW(22, 22, 21);

        /* a5 = r7 + r5 + (r5>>1) - r1 */
        VPR_ADDUW(23, 7, 5);
        VPR_SRAW1(21, 5, 11, 12, 13);
        VPR_ADDUW(23, 23, 21);
        VPR_SUBUW(23, 23, 1);

        /* a7 = r3 + r5 + r1 + (r1>>1) */
        VPR_ADDUW(24, 3, 5);
        VPR_ADDUW(24, 24, 1);
        VPR_SRAW1(21, 1, 11, 12, 13);
        VPR_ADDUW(24, 24, 21);

        /* b1 = (a7>>2) + a1 */
        VPR_SRAW2(25, 24, 11, 12, 13);
        VPR_ADDUW(25, 25, 20);

        /* b3 = a3 + (a5>>2) */
        VPR_SRAW2(26, 23, 11, 12, 13);
        VPR_ADDUW(26, 22, 26);

        /* b5 = (a3>>2) - a5 */
        VPR_SRAW2(27, 22, 11, 12, 13);
        VPR_SUBUW(27, 27, 23);

        /* b7 = a7 - (a1>>2) */
        VPR_SRAW2(28, 20, 11, 12, 13);
        VPR_SUBUW(28, 24, 28);

        /* Store back into block as in scalar code */
        VPR_ADDUW(29, 16, 28); /* row0 = b0 + b7 */
        VPR_SUBUW(30, 16, 28); /* row7 = b0 - b7 */
        VPR_ADDUW(31, 17, 27); /* row1 = b2 + b5 */
        VPR_SUBUW(8,  17, 27); /* row6 = b2 - b5 (reuse v8) */
        VPR_ADDUW(9,  18, 26); /* row2 = b4 + b3 */
        VPR_SUBUW(10, 18, 26); /* row5 = b4 - b3 */
        VPR_ADDUW(11, 19, 25); /* row3 = b6 + b1 */
        VPR_SUBUW(12, 19, 25); /* row4 = b6 - b1 */

        SA0_VPR_AT(29, o0);
        SA0_VPR_AT(31, o1);
        SA0_VPR_AT(9,  o2);
        SA0_VPR_AT(11, o3);
        SA0_VPR_AT(12, o4);
        SA0_VPR_AT(10, o5);
        SA0_VPR_AT(8,  o6);
        SA0_VPR_AT(30, o7);

        for (k = 0; k < 8; k++) {
            block[k + 0 * 8] = (int16_t)o0[k];
            block[k + 1 * 8] = (int16_t)o1[k];
            block[k + 2 * 8] = (int16_t)o2[k];
            block[k + 3 * 8] = (int16_t)o3[k];
            block[k + 4 * 8] = (int16_t)o4[k];
            block[k + 5 * 8] = (int16_t)o5[k];
            block[k + 6 * 8] = (int16_t)o6[k];
            block[k + 7 * 8] = (int16_t)o7[k];
        }

#undef VPR_SRAW1
#undef VPR_SRAW2
    }
#else
    for (i = 0; i < 8; i++) {
        const unsigned int a0 =  block[i+0*8] + (unsigned)block[i+4*8];
        const unsigned int a2 =  block[i+0*8] - (unsigned)block[i+4*8];
        const unsigned int a4 = (block[i+2*8]>>1) - (unsigned)block[i+6*8];
        const unsigned int a6 = (block[i+6*8]>>1) + (unsigned)block[i+2*8];

        const unsigned int b0 = a0 + a6;
        const unsigned int b2 = a2 + a4;
        const unsigned int b4 = a2 - a4;
        const unsigned int b6 = a0 - a6;

        const int a1 = -block[i+3*8] + (unsigned)block[i+5*8] - block[i+7*8] - (block[i+7*8]>>1);
        const int a3 =  block[i+1*8] + (unsigned)block[i+7*8] - block[i+3*8] - (block[i+3*8]>>1);
        const int a5 = -block[i+1*8] + (unsigned)block[i+7*8] + block[i+5*8] + (block[i+5*8]>>1);
        const int a7 =  block[i+3*8] + (unsigned)block[i+5*8] + block[i+1*8] + (block[i+1*8]>>1);

        const int b1 = (a7>>2) + (unsigned)a1;
        const int b3 =  (unsigned)a3 + (a5>>2);
        const int b5 = (a3>>2) - (unsigned)a5;
        const int b7 =  (unsigned)a7 - (a1>>2);

        block[i+0*8] = b0 + b7;
        block[i+7*8] = b0 - b7;
        block[i+1*8] = b2 + b5;
        block[i+6*8] = b2 - b5;
        block[i+2*8] = b4 + b3;
        block[i+5*8] = b4 - b3;
        block[i+3*8] = b6 + b1;
        block[i+4*8] = b6 - b1;
    }
#endif

    /* Row pass + add to destination + clip */
    {
#if HAVE_INLINE_ASM
        /* Lanes 0..7 = rows 0..7; process all 8 rows in parallel. */
        LOCAL_ALIGNED_64(int32_t, c0, [16]);
        LOCAL_ALIGNED_64(int32_t, c1, [16]);
        LOCAL_ALIGNED_64(int32_t, c2, [16]);
        LOCAL_ALIGNED_64(int32_t, c3, [16]);
        LOCAL_ALIGNED_64(int32_t, c4, [16]);
        LOCAL_ALIGNED_64(int32_t, c5, [16]);
        LOCAL_ALIGNED_64(int32_t, c6, [16]);
        LOCAL_ALIGNED_64(int32_t, c7, [16]);
        LOCAL_ALIGNED_64(int32_t, y0, [16]);
        LOCAL_ALIGNED_64(int32_t, y1, [16]);
        LOCAL_ALIGNED_64(int32_t, y2, [16]);
        LOCAL_ALIGNED_64(int32_t, y3, [16]);
        LOCAL_ALIGNED_64(int32_t, y4, [16]);
        LOCAL_ALIGNED_64(int32_t, y5, [16]);
        LOCAL_ALIGNED_64(int32_t, y6, [16]);
        LOCAL_ALIGNED_64(int32_t, y7, [16]);
        int k;

        for (k = 0; k < 16; k++)
            c0[k] = c1[k] = c2[k] = c3[k] = c4[k] = c5[k] = c6[k] = c7[k] = 0;

        for (k = 0; k < 8; k++) {
            c0[k] = block[0 + k * 8];
            c1[k] = block[1 + k * 8];
            c2[k] = block[2 + k * 8];
            c3[k] = block[3 + k * 8];
            c4[k] = block[4 + k * 8];
            c5[k] = block[5 + k * 8];
            c6[k] = block[6 + k * 8];
            c7[k] = block[7 + k * 8];
        }

        LA0_VPR_AT(0, c0);
        LA0_VPR_AT(1, c1);
        LA0_VPR_AT(2, c2);
        LA0_VPR_AT(3, c3);
        LA0_VPR_AT(4, c4);
        LA0_VPR_AT(5, c5);
        LA0_VPR_AT(6, c6);
        LA0_VPR_AT(7, c7);
        MXUV3_ZERO_VPR(15);

        /* Arithmetic shifts helpers */
#define VPR_SRAW1(dst, src, tsign, tmask, tfill) do { \
            VPR_SRLW_IMM((tsign), (src), 31);      \
            VPR_SUBUW((tmask), 15, (tsign));       \
            VPR_SLLW_IMM((tfill), (tmask), 31);    \
            VPR_SRLW_IMM((dst), (src), 1);         \
            VPR_OR((dst), (dst), (tfill));         \
        } while (0)
#define VPR_SRAW2(dst, src, tsign, tmask, tfill) do { \
            VPR_SRLW_IMM((tsign), (src), 31);      \
            VPR_SUBUW((tmask), 15, (tsign));       \
            VPR_SLLW_IMM((tfill), (tmask), 30);    \
            VPR_SRLW_IMM((dst), (src), 2);         \
            VPR_OR((dst), (dst), (tfill));         \
        } while (0)

        /* a0 = c0 + c4, a2 = c0 - c4 */
        VPR_ADDUW(8, 0, 4);
        VPR_SUBUW(9, 0, 4);

        /* a4 = (c2>>1) - c6, a6 = (c6>>1) + c2 */
        VPR_SRAW1(10, 2, 11, 12, 13);
        VPR_SUBUW(10, 10, 6);
        VPR_SRAW1(14, 6, 11, 12, 13);
        VPR_ADDUW(14, 14, 2);

        /* b0..b6 */
        VPR_ADDUW(16, 8, 14);
        VPR_ADDUW(17, 9, 10);
        VPR_SUBUW(18, 9, 10);
        VPR_SUBUW(19, 8, 14);

        /* a1 = c5 - c3 - c7 - (c7>>1) */
        VPR_SUBUW(20, 5, 3);
        VPR_SUBUW(20, 20, 7);
        VPR_SRAW1(21, 7, 11, 12, 13);
        VPR_SUBUW(20, 20, 21);

        /* a3 = c1 + c7 - c3 - (c3>>1) */
        VPR_ADDUW(22, 1, 7);
        VPR_SUBUW(22, 22, 3);
        VPR_SRAW1(21, 3, 11, 12, 13);
        VPR_SUBUW(22, 22, 21);

        /* a5 = c7 + c5 + (c5>>1) - c1 */
        VPR_ADDUW(23, 7, 5);
        VPR_SRAW1(21, 5, 11, 12, 13);
        VPR_ADDUW(23, 23, 21);
        VPR_SUBUW(23, 23, 1);

        /* a7 = c3 + c5 + c1 + (c1>>1) */
        VPR_ADDUW(24, 3, 5);
        VPR_ADDUW(24, 24, 1);
        VPR_SRAW1(21, 1, 11, 12, 13);
        VPR_ADDUW(24, 24, 21);

        /* b1..b7 */
        VPR_SRAW2(25, 24, 11, 12, 13);
        VPR_ADDUW(25, 25, 20);

        VPR_SRAW2(26, 23, 11, 12, 13);
        VPR_ADDUW(26, 22, 26);

        VPR_SRAW2(27, 22, 11, 12, 13);
        VPR_SUBUW(27, 27, 23);

        VPR_SRAW2(28, 20, 11, 12, 13);
        VPR_SUBUW(28, 24, 28);

        /* Pixel deltas in output order */
        VPR_ADDUW(29, 16, 28); /* d0 */
        VPR_ADDUW(30, 17, 27); /* d1 */
        VPR_ADDUW(31, 18, 26); /* d2 */
        VPR_ADDUW(8,  19, 25); /* d3 */
        VPR_SUBUW(9,  19, 25); /* d4 */
        VPR_SUBUW(10, 18, 26); /* d5 */
        VPR_SUBUW(11, 17, 27); /* d6 */
        VPR_SUBUW(12, 16, 28); /* d7 */

        SA0_VPR_AT(29, y0);
        SA0_VPR_AT(30, y1);
        SA0_VPR_AT(31, y2);
        SA0_VPR_AT(8,  y3);
        SA0_VPR_AT(9,  y4);
        SA0_VPR_AT(10, y5);
        SA0_VPR_AT(11, y6);
        SA0_VPR_AT(12, y7);

        for (i = 0; i < 8; i++) {
            uint8_t *row = dst + i * stride;
            uint32_t p0 = AV_RN32A(row);
            uint32_t p1 = AV_RN32A(row + 4);
            AV_WN32A(row,
                clip_uint8(( p0        & 0xFF) + ((int)y0[i] >> 6))        |
                (clip_uint8(((p0 >>  8) & 0xFF) + ((int)y1[i] >> 6)) <<  8) |
                (clip_uint8(((p0 >> 16) & 0xFF) + ((int)y2[i] >> 6)) << 16) |
                (clip_uint8(( p0 >> 24)         + ((int)y3[i] >> 6)) << 24));
            AV_WN32A(row + 4,
                clip_uint8(( p1        & 0xFF) + ((int)y4[i] >> 6))        |
                (clip_uint8(((p1 >>  8) & 0xFF) + ((int)y5[i] >> 6)) <<  8) |
                (clip_uint8(((p1 >> 16) & 0xFF) + ((int)y6[i] >> 6)) << 16) |
                (clip_uint8(( p1 >> 24)         + ((int)y7[i] >> 6)) << 24));
        }

#undef VPR_SRAW1
#undef VPR_SRAW2
#else
        for (i = 0; i < 8; i++) {
            const unsigned a0 =  block[0+i*8] + (unsigned)block[4+i*8];
            const unsigned a2 =  block[0+i*8] - (unsigned)block[4+i*8];
            const unsigned a4 = (block[2+i*8]>>1) - (unsigned)block[6+i*8];
            const unsigned a6 = (block[6+i*8]>>1) + (unsigned)block[2+i*8];

            const unsigned b0 = a0 + a6;
            const unsigned b2 = a2 + a4;
            const unsigned b4 = a2 - a4;
            const unsigned b6 = a0 - a6;

            const int a1 = -(unsigned)block[3+i*8] + block[5+i*8] - block[7+i*8] - (block[7+i*8]>>1);
            const int a3 =  (unsigned)block[1+i*8] + block[7+i*8] - block[3+i*8] - (block[3+i*8]>>1);
            const int a5 = -(unsigned)block[1+i*8] + block[7+i*8] + block[5+i*8] + (block[5+i*8]>>1);
            const int a7 =  (unsigned)block[3+i*8] + block[5+i*8] + block[1+i*8] + (block[1+i*8]>>1);

            const unsigned b1 = (a7>>2) + (unsigned)a1;
            const unsigned b3 =  (unsigned)a3 + (a5>>2);
            const unsigned b5 = (a3>>2) - (unsigned)a5;
            const unsigned b7 =  (unsigned)a7 - (a1>>2);

            {
                uint8_t *row = dst + i * stride;
                uint32_t p0 = AV_RN32A(row);
                uint32_t p1 = AV_RN32A(row + 4);
                AV_WN32A(row,
                    clip_uint8(( p0        & 0xFF) + ((int)(b0 + b7) >> 6))        |
                    (clip_uint8(((p0 >>  8) & 0xFF) + ((int)(b2 + b5) >> 6)) <<  8) |
                    (clip_uint8(((p0 >> 16) & 0xFF) + ((int)(b4 + b3) >> 6)) << 16) |
                    (clip_uint8(( p0 >> 24)         + ((int)(b6 + b1) >> 6)) << 24));
                AV_WN32A(row + 4,
                    clip_uint8(( p1        & 0xFF) + ((int)(b6 - b1) >> 6))        |
                    (clip_uint8(((p1 >>  8) & 0xFF) + ((int)(b4 - b3) >> 6)) <<  8) |
                    (clip_uint8(((p1 >> 16) & 0xFF) + ((int)(b2 - b5) >> 6)) << 16) |
                    (clip_uint8(( p1 >> 24)         + ((int)(b0 - b7) >> 6)) << 24));
            }
        }
#endif
    }

    memset(block, 0, 64 * sizeof(int16_t));
}

/* ---- Add pixels 4x4 + clear ---- */

void ff_h264_add_pixels4_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    for (i = 0; i < 4; i++) {
        uint32_t p = AV_RN32A(dst);
        AV_WN32A(dst,
            (uint8_t)(( p        & 0xFF) + (unsigned)block[0])        |
            ((uint8_t)(((p >>  8) & 0xFF) + (unsigned)block[1]) <<  8) |
            ((uint8_t)(((p >> 16) & 0xFF) + (unsigned)block[2]) << 16) |
            ((uint8_t)(( p >> 24)         + (unsigned)block[3]) << 24));
        dst   += stride;
        block += 4;
    }
    memset(block - 16, 0, 16 * sizeof(int16_t));
}

/* ---- Add pixels 8x8 + clear ---- */

void ff_h264_add_pixels8_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    for (i = 0; i < 8; i++) {
        uint32_t p0 = AV_RN32A(dst);
        uint32_t p1 = AV_RN32A(dst + 4);
        AV_WN32A(dst,
            (uint8_t)(( p0        & 0xFF) + (unsigned)block[0])        |
            ((uint8_t)(((p0 >>  8) & 0xFF) + (unsigned)block[1]) <<  8) |
            ((uint8_t)(((p0 >> 16) & 0xFF) + (unsigned)block[2]) << 16) |
            ((uint8_t)(( p0 >> 24)         + (unsigned)block[3]) << 24));
        AV_WN32A(dst + 4,
            (uint8_t)(( p1        & 0xFF) + (unsigned)block[4])        |
            ((uint8_t)(((p1 >>  8) & 0xFF) + (unsigned)block[5]) <<  8) |
            ((uint8_t)(((p1 >> 16) & 0xFF) + (unsigned)block[6]) << 16) |
            ((uint8_t)(( p1 >> 24)         + (unsigned)block[7]) << 24));
        dst   += stride;
        block += 8;
    }
    memset(block - 64, 0, 64 * sizeof(int16_t));
}

/* ---- Multi-block dispatchers ---- */

void ff_h264_idct_add16_8_mxu(uint8_t *dst, const int *block_offset,
                               int16_t *block, int stride,
                               const uint8_t nnzc[5 * 8])
{
    int i;
    for (i = 0; i < 16; i++) {
        int nnz = nnzc[scan8[i]];
        if (nnz) {
            if (nnz == 1 && block[i * 16])
                ff_h264_idct_dc_add_8_mxu(dst + block_offset[i],
                                           block + i * 16, stride);
            else
                ff_h264_idct_add_8_mxu(dst + block_offset[i],
                                        block + i * 16, stride);
        }
    }
}

void ff_h264_idct_add16intra_8_mxu(uint8_t *dst, const int *block_offset,
                                    int16_t *block, int stride,
                                    const uint8_t nnzc[5 * 8])
{
    int i;
    for (i = 0; i < 16; i++) {
        if (nnzc[scan8[i]])
            ff_h264_idct_add_8_mxu(dst + block_offset[i],
                                    block + i * 16, stride);
        else if (block[i * 16])
            ff_h264_idct_dc_add_8_mxu(dst + block_offset[i],
                                       block + i * 16, stride);
    }
}

void ff_h264_idct8_add4_8_mxu(uint8_t *dst, const int *block_offset,
                               int16_t *block, int stride,
                               const uint8_t nnzc[5 * 8])
{
    int i;
    for (i = 0; i < 16; i += 4) {
        int nnz = nnzc[scan8[i]];
        if (nnz) {
            if (nnz == 1 && block[i * 16])
                ff_h264_idct8_dc_add_8_mxu(dst + block_offset[i],
                                            block + i * 16, stride);
            else
                ff_h264_idct8_add_8_mxu(dst + block_offset[i],
                                         block + i * 16, stride);
        }
    }
}

void ff_h264_idct_add8_8_mxu(uint8_t **dest, const int *block_offset,
                              int16_t *block, int stride,
                              const uint8_t nnzc[15 * 8])
{
    int i, j;
    for (j = 1; j < 3; j++) {
        for (i = j * 16; i < j * 16 + 4; i++) {
            if (nnzc[scan8[i]])
                ff_h264_idct_add_8_mxu(dest[j-1] + block_offset[i],
                                        block + i * 16, stride);
            else if (block[i * 16])
                ff_h264_idct_dc_add_8_mxu(dest[j-1] + block_offset[i],
                                           block + i * 16, stride);
        }
    }
}

void ff_h264_idct_add8_422_8_mxu(uint8_t **dest, const int *block_offset,
                                  int16_t *block, int stride,
                                  const uint8_t nnzc[15 * 8])
{
    int i, j;

    for (j = 1; j < 3; j++) {
        for (i = j * 16; i < j * 16 + 4; i++) {
            if (nnzc[scan8[i]])
                ff_h264_idct_add_8_mxu(dest[j-1] + block_offset[i],
                                        block + i * 16, stride);
            else if (block[i * 16])
                ff_h264_idct_dc_add_8_mxu(dest[j-1] + block_offset[i],
                                           block + i * 16, stride);
        }
    }

    for (j = 1; j < 3; j++) {
        for (i = j * 16 + 4; i < j * 16 + 8; i++) {
            if (nnzc[scan8[i+4]])
                ff_h264_idct_add_8_mxu(dest[j-1] + block_offset[i+4],
                                        block + i * 16, stride);
            else if (block[i * 16])
                ff_h264_idct_dc_add_8_mxu(dest[j-1] + block_offset[i+4],
                                           block + i * 16, stride);
        }
    }
}

/* ---- Luma DC dequant + 4x4 Hadamard IDCT ---- */

void ff_h264_luma_dc_dequant_idct_8_mxu(int16_t *output, int16_t *input,
                                          int qmul)
{
#define stride 16
    int i;
    int temp[16];
    static const uint8_t x_offset[4] = {0, 2*stride, 8*stride, 10*stride};

    for (i = 0; i < 4; i++) {
        const int z0 = input[4*i+0] + input[4*i+1];
        const int z1 = input[4*i+0] - input[4*i+1];
        const int z2 = input[4*i+2] - input[4*i+3];
        const int z3 = input[4*i+2] + input[4*i+3];

        temp[4*i+0] = z0 + z3;
        temp[4*i+1] = z0 - z3;
        temp[4*i+2] = z1 - z2;
        temp[4*i+3] = z1 + z2;
    }

    for (i = 0; i < 4; i++) {
        const int offset = x_offset[i];
        const unsigned int z0 = temp[4*0+i] + temp[4*2+i];
        const unsigned int z1 = temp[4*0+i] - temp[4*2+i];
        const unsigned int z2 = temp[4*1+i] - temp[4*3+i];
        const unsigned int z3 = temp[4*1+i] + temp[4*3+i];

        output[stride* 0+offset] = (int)((z0 + z3)*qmul + 128) >> 8;
        output[stride* 1+offset] = (int)((z1 + z2)*qmul + 128) >> 8;
        output[stride* 4+offset] = (int)((z1 - z2)*qmul + 128) >> 8;
        output[stride* 5+offset] = (int)((z0 - z3)*qmul + 128) >> 8;
    }
#undef stride
}



/* ================================================================== */
/* H.264 Deblocking Loop Filter — optimised for MIPS32r2 / XBurst2   */
/* ================================================================== */

/*
 * Fast paths for the common case xstride==1 (contiguous pixels across the edge):
 * use 32-bit loads to reduce byte-load count. Keep endian-safe using AV_RL32/AV_RB32.
 */
#if HAVE_BIGENDIAN
#define H264_LD32(p) AV_RB32(p)
#define H264_B0(w)   ((int)(((w) >> 24) & 0xFF))
#define H264_B1(w)   ((int)(((w) >> 16) & 0xFF))
#define H264_B2(w)   ((int)(((w) >>  8) & 0xFF))
#define H264_B3(w)   ((int)(((w) >>  0) & 0xFF))
#else
#define H264_LD32(p) AV_RL32(p)
#define H264_B0(w)   ((int)(((w) >>  0) & 0xFF))
#define H264_B1(w)   ((int)(((w) >>  8) & 0xFF))
#define H264_B2(w)   ((int)(((w) >> 16) & 0xFF))
#define H264_B3(w)   ((int)(((w) >> 24) & 0xFF))
#endif

/*
 * Branchless av_clip replacement: clamp v to [lo, hi].
 */
static inline int clip3(int v, int lo, int hi)
{
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

/* ---- Luma inter (weak) loop filter ---- */

static av_always_inline void h264_loop_filter_luma_mxu(uint8_t *pix,
        ptrdiff_t xstride, ptrdiff_t ystride,
        int inner_iters, int alpha, int beta, int8_t *tc0)
{
    int i, d;
    for (i = 0; i < 4; i++) {
        const int tc_orig = tc0[i];
        if (tc_orig < 0) {
            pix += inner_iters * ystride;
            continue;
        }
        if (xstride == 1) {
            for (d = 0; d < inner_iters; d++) {
#if HAVE_INLINE_ASM
                /* Prefetch a couple of steps ahead along the iteration dimension. */
                if (d + 2 < inner_iters)
                    PREF_LOAD(pix, ystride * 2);
#endif
                const uint32_t wP = H264_LD32(pix - 3); /* p2 p1 p0 q0 */
                const uint32_t wQ = H264_LD32(pix + 0); /* q0 q1 q2 q3 */
                const int p2 = H264_B0(wP);
                const int p1 = H264_B1(wP);
                const int p0 = H264_B2(wP);
                const int q0 = H264_B3(wP);
                const int q1 = H264_B1(wQ);
                const int q2 = H264_B2(wQ);

            const int abs_p0q0 = FFABS(p0 - q0);
            const int abs_p1p0 = FFABS(p1 - p0);
            const int abs_q1q0 = FFABS(q1 - q0);

            if (abs_p0q0 < alpha &&
                abs_p1p0 < beta &&
                abs_q1q0 < beta) {

                int tc = tc_orig;
                int i_delta;
                const int p0q0_avg = (p0 + q0 + 1) >> 1;

                if (FFABS(p2 - p0) < beta) {
                    if (tc_orig)
                        pix[-2*xstride] = p1 + clip3((((p2 + p0q0_avg) >> 1) - p1), -tc_orig, tc_orig);
                    tc++;
                }
                if (FFABS(q2 - q0) < beta) {
                    if (tc_orig)
                        pix[xstride] = q1 + clip3((((q2 + p0q0_avg) >> 1) - q1), -tc_orig, tc_orig);
                    tc++;
                }

                i_delta = clip3((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3, -tc, tc);
                pix[-xstride] = clip_uint8(p0 + i_delta);
                pix[0]        = clip_uint8(q0 - i_delta);
            }
            pix += ystride;
        }
        } else {
            for (d = 0; d < inner_iters; d++) {
#if HAVE_INLINE_ASM
                /* Prefetch a couple of steps ahead along the iteration dimension. */
                if (d + 2 < inner_iters)
                    PREF_LOAD(pix, ystride * 2);
#endif
                const int p0 = pix[-1*xstride];
                const int p1 = pix[-2*xstride];
                const int p2 = pix[-3*xstride];
                const int q0 = pix[0];
                const int q1 = pix[1*xstride];
                const int q2 = pix[2*xstride];

                const int abs_p0q0 = FFABS(p0 - q0);
                const int abs_p1p0 = FFABS(p1 - p0);
                const int abs_q1q0 = FFABS(q1 - q0);

                if (abs_p0q0 < alpha &&
                    abs_p1p0 < beta &&
                    abs_q1q0 < beta) {

                    int tc = tc_orig;
                    int i_delta;
                    const int p0q0_avg = (p0 + q0 + 1) >> 1;

                    if (FFABS(p2 - p0) < beta) {
                        if (tc_orig)
                            pix[-2*xstride] = p1 + clip3((((p2 + p0q0_avg) >> 1) - p1), -tc_orig, tc_orig);
                        tc++;
                    }
                    if (FFABS(q2 - q0) < beta) {
                        if (tc_orig)
                            pix[xstride] = q1 + clip3((((q2 + p0q0_avg) >> 1) - q1), -tc_orig, tc_orig);
                        tc++;
                    }

                    i_delta = clip3((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3, -tc, tc);
                    pix[-xstride] = clip_uint8(p0 + i_delta);
                    pix[0]        = clip_uint8(q0 - i_delta);
                }
                pix += ystride;
            }
        }
    }
}

/*
 * Horizontal luma inter loop filter — delegates to the scalar
 * h264_loop_filter_luma_mxu with xstride=1.
 */

void ff_h264_v_loop_filter_luma_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                       int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_mxu(pix, stride, 1, 4, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_luma_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                       int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_mxu(pix, 1, stride, 4, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_luma_mbaff_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                              int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_mxu(pix, 1, stride, 2, alpha, beta, tc0);
}

/* ---- Luma intra (strong) loop filter ---- */

static av_always_inline void h264_loop_filter_luma_intra_mxu(uint8_t *pix,
        ptrdiff_t xstride, ptrdiff_t ystride,
        int inner_iters, int alpha, int beta)
{
    int d;
    const int iters = 4 * inner_iters;
    if (xstride == 1) {
        for (d = 0; d < iters; d++) {
#if HAVE_INLINE_ASM
            if (d + 4 < iters)
                PREF_LOAD(pix, ystride * 4);
#endif
            const uint32_t wP = H264_LD32(pix - 4); /* p3 p2 p1 p0 */
            const uint32_t wQ = H264_LD32(pix + 0); /* q0 q1 q2 q3 */
            const int p3 = H264_B0(wP);
            const int p2 = H264_B1(wP);
            const int p1 = H264_B2(wP);
            const int p0 = H264_B3(wP);
            const int q0 = H264_B0(wQ);
            const int q1 = H264_B1(wQ);
            const int q2 = H264_B2(wQ);

        const int abs_p0q0 = FFABS(p0 - q0);
        const int abs_p1p0 = FFABS(p1 - p0);
        const int abs_q1q0 = FFABS(q1 - q0);

        if (abs_p0q0 < alpha &&
            abs_p1p0 < beta &&
            abs_q1q0 < beta) {

            if (abs_p0q0 < ((alpha >> 2) + 2)) {
                if (FFABS(p2 - p0) < beta) {
                    pix[-1*xstride] = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3;
                    pix[-2*xstride] = (p2 + p1 + p0 + q0 + 2) >> 2;
                    pix[-3*xstride] = (2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3;
                } else {
                    pix[-1*xstride] = (2*p1 + p0 + q1 + 2) >> 2;
                }
                if (FFABS(q2 - q0) < beta) {
                    const int q3 = H264_B3(wQ);
                    pix[0*xstride] = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3;
                    pix[1*xstride] = (p0 + q0 + q1 + q2 + 2) >> 2;
                    pix[2*xstride] = (2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3;
                } else {
                    pix[0*xstride] = (2*q1 + q0 + p1 + 2) >> 2;
                }
            } else {
                pix[-1*xstride] = (2*p1 + p0 + q1 + 2) >> 2;
                pix[ 0*xstride] = (2*q1 + q0 + p1 + 2) >> 2;
            }
        }
        pix += ystride;
        }
    } else {
        for (d = 0; d < iters; d++) {
#if HAVE_INLINE_ASM
            if (d + 4 < iters)
                PREF_LOAD(pix, ystride * 4);
#endif
            const int p2 = pix[-3*xstride];
            const int p1 = pix[-2*xstride];
            const int p0 = pix[-1*xstride];
            const int q0 = pix[0*xstride];
            const int q1 = pix[1*xstride];
            const int q2 = pix[2*xstride];

            const int abs_p0q0 = FFABS(p0 - q0);
            const int abs_p1p0 = FFABS(p1 - p0);
            const int abs_q1q0 = FFABS(q1 - q0);

            if (abs_p0q0 < alpha &&
                abs_p1p0 < beta &&
                abs_q1q0 < beta) {

                if (abs_p0q0 < ((alpha >> 2) + 2)) {
                    if (FFABS(p2 - p0) < beta) {
                        const int p3 = pix[-4*xstride];
                        pix[-1*xstride] = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3;
                        pix[-2*xstride] = (p2 + p1 + p0 + q0 + 2) >> 2;
                        pix[-3*xstride] = (2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3;
                    } else {
                        pix[-1*xstride] = (2*p1 + p0 + q1 + 2) >> 2;
                    }
                    if (FFABS(q2 - q0) < beta) {
                        const int q3 = pix[3*xstride];
                        pix[0*xstride] = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3;
                        pix[1*xstride] = (p0 + q0 + q1 + q2 + 2) >> 2;
                        pix[2*xstride] = (2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3;
                    } else {
                        pix[0*xstride] = (2*q1 + q0 + p1 + 2) >> 2;
                    }
                } else {
                    pix[-1*xstride] = (2*p1 + p0 + q1 + 2) >> 2;
                    pix[ 0*xstride] = (2*q1 + q0 + p1 + 2) >> 2;
                }
            }
            pix += ystride;
        }
    }
}

void ff_h264_v_loop_filter_luma_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                              int alpha, int beta)
{
    h264_loop_filter_luma_intra_mxu(pix, stride, 1, 4, alpha, beta);
}

void ff_h264_h_loop_filter_luma_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                              int alpha, int beta)
{
    h264_loop_filter_luma_intra_mxu(pix, 1, stride, 4, alpha, beta);
}

void ff_h264_h_loop_filter_luma_mbaff_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                    int alpha, int beta)
{
    h264_loop_filter_luma_intra_mxu(pix, 1, stride, 2, alpha, beta);
}

/* ---- Chroma inter loop filter ---- */

static av_always_inline void h264_loop_filter_chroma_mxu(uint8_t *pix,
        ptrdiff_t xstride, ptrdiff_t ystride,
        int inner_iters, int alpha, int beta, int8_t *tc0)
{
    int i, d;
    for (i = 0; i < 4; i++) {
        const int tc = (int)((tc0[i] - 1U) + 1);  /* for 8-bit: ((tc0[i] - 1U) << 0) + 1 */
        if (tc <= 0) {
            pix += inner_iters * ystride;
            continue;
        }
        if (xstride == 1) {
            for (d = 0; d < inner_iters; d++) {
#if HAVE_INLINE_ASM
                if (d + 2 < inner_iters)
                    PREF_LOAD(pix, ystride * 2);
#endif
                const uint32_t w = H264_LD32(pix - 2); /* p1 p0 q0 q1 */
                const int p1 = H264_B0(w);
                const int p0 = H264_B1(w);
                const int q0 = H264_B2(w);
                const int q1 = H264_B3(w);

            const int abs_p0q0 = FFABS(p0 - q0);
            const int abs_p1p0 = FFABS(p1 - p0);
            const int abs_q1q0 = FFABS(q1 - q0);

            if (abs_p0q0 < alpha &&
                abs_p1p0 < beta &&
                abs_q1q0 < beta) {

                int delta = clip3(((q0 - p0) * 4 + (p1 - q1) + 4) >> 3, -(int)tc, (int)tc);
                pix[-xstride] = clip_uint8(p0 + delta);
                pix[0]        = clip_uint8(q0 - delta);
            }
            pix += ystride;
        }
        } else {
            for (d = 0; d < inner_iters; d++) {
#if HAVE_INLINE_ASM
                if (d + 2 < inner_iters)
                    PREF_LOAD(pix, ystride * 2);
#endif
                const int p0 = pix[-1*xstride];
                const int p1 = pix[-2*xstride];
                const int q0 = pix[0];
                const int q1 = pix[1*xstride];

                const int abs_p0q0 = FFABS(p0 - q0);
                const int abs_p1p0 = FFABS(p1 - p0);
                const int abs_q1q0 = FFABS(q1 - q0);

                if (abs_p0q0 < alpha &&
                    abs_p1p0 < beta &&
                    abs_q1q0 < beta) {

                    int delta = clip3(((q0 - p0) * 4 + (p1 - q1) + 4) >> 3, -(int)tc, (int)tc);
                    pix[-xstride] = clip_uint8(p0 + delta);
                    pix[0]        = clip_uint8(q0 - delta);
                }
                pix += ystride;
            }
        }
    }
}

void ff_h264_v_loop_filter_chroma_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                          int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_mxu(pix, stride, 1, 2, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_chroma_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                          int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_mxu(pix, 1, stride, 2, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_chroma_mbaff_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_mxu(pix, 1, stride, 1, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_chroma422_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                             int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_mxu(pix, 1, stride, 4, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_chroma422_mbaff_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                   int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_mxu(pix, 1, stride, 2, alpha, beta, tc0);
}

/* ---- Chroma intra loop filter ---- */

static av_always_inline void h264_loop_filter_chroma_intra_mxu(uint8_t *pix,
        ptrdiff_t xstride, ptrdiff_t ystride,
        int inner_iters, int alpha, int beta)
{
    int d;
    const int iters = 4 * inner_iters;
    if (xstride == 1) {
        for (d = 0; d < iters; d++) {
#if HAVE_INLINE_ASM
            if (d + 4 < iters)
                PREF_LOAD(pix, ystride * 4);
#endif
            const uint32_t w = H264_LD32(pix - 2); /* p1 p0 q0 q1 */
            const int p1 = H264_B0(w);
            const int p0 = H264_B1(w);
            const int q0 = H264_B2(w);
            const int q1 = H264_B3(w);

        const int abs_p0q0 = FFABS(p0 - q0);
        const int abs_p1p0 = FFABS(p1 - p0);
        const int abs_q1q0 = FFABS(q1 - q0);

        if (abs_p0q0 < alpha &&
            abs_p1p0 < beta &&
            abs_q1q0 < beta) {

            pix[-xstride] = (2*p1 + p0 + q1 + 2) >> 2;
            pix[0]        = (2*q1 + q0 + p1 + 2) >> 2;
        }
        pix += ystride;
    }
    } else {
        for (d = 0; d < iters; d++) {
#if HAVE_INLINE_ASM
            if (d + 4 < iters)
                PREF_LOAD(pix, ystride * 4);
#endif
            const int p0 = pix[-1*xstride];
            const int p1 = pix[-2*xstride];
            const int q0 = pix[0];
            const int q1 = pix[1*xstride];

            const int abs_p0q0 = FFABS(p0 - q0);
            const int abs_p1p0 = FFABS(p1 - p0);
            const int abs_q1q0 = FFABS(q1 - q0);

            if (abs_p0q0 < alpha &&
                abs_p1p0 < beta &&
                abs_q1q0 < beta) {

                pix[-xstride] = (2*p1 + p0 + q1 + 2) >> 2;
                pix[0]        = (2*q1 + q0 + p1 + 2) >> 2;
            }
            pix += ystride;
        }
    }
}

void ff_h264_v_loop_filter_chroma_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                int alpha, int beta)
{
    h264_loop_filter_chroma_intra_mxu(pix, stride, 1, 2, alpha, beta);
}

void ff_h264_h_loop_filter_chroma_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                int alpha, int beta)
{
    h264_loop_filter_chroma_intra_mxu(pix, 1, stride, 2, alpha, beta);
}

void ff_h264_h_loop_filter_chroma_mbaff_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                      int alpha, int beta)
{
    h264_loop_filter_chroma_intra_mxu(pix, 1, stride, 1, alpha, beta);
}

void ff_h264_h_loop_filter_chroma422_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                   int alpha, int beta)
{
    h264_loop_filter_chroma_intra_mxu(pix, 1, stride, 4, alpha, beta);
}

void ff_h264_h_loop_filter_chroma422_mbaff_intra_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                                         int alpha, int beta)
{
    h264_loop_filter_chroma_intra_mxu(pix, 1, stride, 2, alpha, beta);
}