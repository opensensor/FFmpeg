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
 * Ingenic XBurst2 optimised IDCT and pixel clamping functions.
 *
 * Optimisations over -Os compiled C:
 *  - MXUv3 VPR MAXSH/MINSH bulk clamping (32 int16 values per VPR op)
 *  - VPR ADDUH for int16 bias addition (put_signed_pixels_clamped)
 *  - Word-packed stores (2 per row instead of 8 byte stores)
 *  - Hand-scheduled IDCT butterfly with DC-only shortcut
 *  - Full 8-row unrolling where beneficial
 *  - Branchless uint8 clip for saturating arithmetic
 *
 * The pixel clamping functions use 512-bit VPR registers (MXUv3) to clamp
 * entire IDCT blocks in bulk, then narrow to uint8 with word-packed stores.
 * The IDCT butterfly is pure MIPS32r2 scalar code.
 */

#include <string.h>
#include "libavutil/intreadwrite.h"
#include "libavutil/common.h"
#include "libavutil/mem_internal.h"
#include "idctdsp_mips.h"
#include "mxu.h"

/* ---- Pixel clamping functions ---- */

static inline uint8_t clip_uint8(int v)
{
    if (v & ~0xFF)
        return (-v) >> 31;
    return v;
}

/*
 * Pack 4 clamped-to-[0,255] int16 values into a uint32 (little-endian).
 * Caller must ensure each value is already in [0,255].
 */
static inline uint32_t pack4_u8(int16_t a, int16_t b, int16_t c, int16_t d)
{
    return (uint32_t)(uint8_t)a        | ((uint32_t)(uint8_t)b <<  8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

#if HAVE_INLINE_ASM
/*
 * Upper-bound vector for VPR MINSH clamping: 32 × int16 all set to 255.
 * Must be 64-byte aligned for LA0_VPR_AT.
 */
DECLARE_ASM_CONST(64, int16_t, vpr_clamp_255)[32] = {
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
};

/*
 * Bias vector for signed-to-unsigned conversion: 32 × int16 all set to 128.
 * Must be 64-byte aligned for LA0_VPR_AT.
 */
DECLARE_ASM_CONST(64, int16_t, vpr_bias_128)[32] = {
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
};
#endif /* HAVE_INLINE_ASM */

void ff_put_pixels_clamped_mxu(const int16_t *block,
                                uint8_t *restrict pixels,
                                ptrdiff_t line_size)
{
    int i;
#if HAVE_INLINE_ASM
    /*
     * VPR fast path: clamp 32 int16 values at a time using
     * MAXSH(x, 0) + MINSH(x, 255), then narrow to uint8.
     * Uses a 64-byte-aligned temp buffer for VPR load/store.
     */
    LOCAL_ALIGNED_64(int16_t, clamped, [64]);

    memcpy(clamped, block, 128);

    /* VPR0 = all zeros (lower bound) */
    VPR_ZERO_INIT();
    /* VPR1 = all 255s (upper bound) */
    LA0_VPR_AT(1, vpr_clamp_255);

    /* Clamp rows 0-3 (32 int16 = 64 bytes) */
    LA0_VPR_AT(2, clamped);
    VPR_MAXSH(2, 2, 0);    /* >= 0   */
    VPR_MINSH(2, 2, 1);    /* <= 255 */
    SA0_VPR_AT(2, clamped);

    /* Clamp rows 4-7 (next 64 bytes) */
    LA0_VPR_AT(2, (uint8_t *)clamped + 64);
    VPR_MAXSH(2, 2, 0);
    VPR_MINSH(2, 2, 1);
    SA0_VPR_AT(2, (uint8_t *)clamped + 64);

    /* Narrow clamped int16 → uint8 with word-packed stores */
    for (i = 0; i < 8; i++) {
        const int16_t *row = clamped + i * 8;
        AV_WN32A(pixels,     pack4_u8(row[0], row[1], row[2], row[3]));
        AV_WN32A(pixels + 4, pack4_u8(row[4], row[5], row[6], row[7]));
        pixels += line_size;
    }
#else
    for (i = 0; i < 8; i++) {
        pixels[0] = clip_uint8(block[0]);
        pixels[1] = clip_uint8(block[1]);
        pixels[2] = clip_uint8(block[2]);
        pixels[3] = clip_uint8(block[3]);
        pixels[4] = clip_uint8(block[4]);
        pixels[5] = clip_uint8(block[5]);
        pixels[6] = clip_uint8(block[6]);
        pixels[7] = clip_uint8(block[7]);
        pixels += line_size;
        block  += 8;
    }
#endif
}

void ff_put_signed_pixels_clamped_mxu(const int16_t *block,
                                       uint8_t *restrict pixels,
                                       ptrdiff_t line_size)
{
    int i;
#if HAVE_INLINE_ASM
    /*
     * VPR path: add 128 bias then clamp to [0,255].
     */
    LOCAL_ALIGNED_64(int16_t, clamped, [64]);

    memcpy(clamped, block, 128);

    /* VPR0 = all zeros, VPR1 = all 255s, VPR3 = all 128s */
    VPR_ZERO_INIT();
    LA0_VPR_AT(1, vpr_clamp_255);
    LA0_VPR_AT(3, vpr_bias_128);

    /* Add 128 bias + clamp rows 0-3 */
    LA0_VPR_AT(2, clamped);
    VPR_ADDUH(2, 2, 3);    /* + 128  */
    VPR_MAXSH(2, 2, 0);    /* >= 0   */
    VPR_MINSH(2, 2, 1);    /* <= 255 */
    SA0_VPR_AT(2, clamped);

    /* Add 128 bias + clamp rows 4-7 */
    LA0_VPR_AT(2, (uint8_t *)clamped + 64);
    VPR_ADDUH(2, 2, 3);
    VPR_MAXSH(2, 2, 0);
    VPR_MINSH(2, 2, 1);
    SA0_VPR_AT(2, (uint8_t *)clamped + 64);

    for (i = 0; i < 8; i++) {
        const int16_t *row = clamped + i * 8;
        AV_WN32A(pixels,     pack4_u8(row[0], row[1], row[2], row[3]));
        AV_WN32A(pixels + 4, pack4_u8(row[4], row[5], row[6], row[7]));
        pixels += line_size;
    }
#else
    for (i = 0; i < 8; i++) {
        pixels[0] = clip_uint8(block[0] + 128);
        pixels[1] = clip_uint8(block[1] + 128);
        pixels[2] = clip_uint8(block[2] + 128);
        pixels[3] = clip_uint8(block[3] + 128);
        pixels[4] = clip_uint8(block[4] + 128);
        pixels[5] = clip_uint8(block[5] + 128);
        pixels[6] = clip_uint8(block[6] + 128);
        pixels[7] = clip_uint8(block[7] + 128);
        pixels += line_size;
        block  += 8;
    }
#endif
}

/**
 * Add IDCT residuals to pixels with clamping.
 *
 * Uses word-sized loads/stores with branchless clip for per-pixel
 * saturation.  Processes 4 pixels per word to reduce store pressure
 * on the in-order XBurst2 pipeline.
 */
void ff_add_pixels_clamped_mxu(const int16_t *block,
                                uint8_t *restrict pixels,
                                ptrdiff_t line_size)
{
    int i;
    for (i = 0; i < 8; i++) {
        uint32_t p0 = AV_RN32A(pixels);
        uint32_t p1 = AV_RN32A(pixels + 4);
        AV_WN32A(pixels,
            clip_uint8(( p0        & 0xFF) + block[0])        |
            (clip_uint8(((p0 >>  8) & 0xFF) + block[1]) <<  8) |
            (clip_uint8(((p0 >> 16) & 0xFF) + block[2]) << 16) |
            (clip_uint8(( p0 >> 24)         + block[3]) << 24));
        AV_WN32A(pixels + 4,
            clip_uint8(( p1        & 0xFF) + block[4])        |
            (clip_uint8(((p1 >>  8) & 0xFF) + block[5]) <<  8) |
            (clip_uint8(((p1 >> 16) & 0xFF) + block[6]) << 16) |
            (clip_uint8(( p1 >> 24)         + block[7]) << 24));
        pixels += line_size;
        block  += 8;
    }
}

/* ---- Simple 8x8 IDCT ---- */

#define W1  22725
#define W2  21407
#define W3  19266
#define W4  16383
#define W5  12873
#define W6   8867
#define W7   4520

#define ROW_SHIFT 11
#define COL_SHIFT 20

#define MUL16(a, b) ((int16_t)(a) * (int16_t)(b))
#define MAC16(rt, a, b) rt += (int16_t)(a) * (int16_t)(b)

static inline void idct_row(int16_t *row)
{
    int a0, a1, a2, a3, b0, b1, b2, b3;

    /* DC-only shortcut */
    if (!(AV_RN32A(row + 2) | AV_RN32A(row + 4) |
          AV_RN32A(row + 6) | row[1])) {
        int val = row[0] * (1 << 3); /* DC_SHIFT = 3 */
        int16_t v = val;
        AV_WN32A(row,     v | (v << 16));
        AV_WN32A(row + 2, v | (v << 16));
        AV_WN32A(row + 4, v | (v << 16));
        AV_WN32A(row + 6, v | (v << 16));
        return;
    }

    a0 = W4 * row[0] + (1 << (ROW_SHIFT - 1));
    a1 = a0;
    a2 = a0;
    a3 = a0;

    a0 += W2 * row[2];
    a1 += W6 * row[2];
    a2 -= W6 * row[2];
    a3 -= W2 * row[2];

    b0 = MUL16(W1, row[1]);
    MAC16(b0, W3, row[3]);
    b1 = MUL16(W3, row[1]);
    MAC16(b1, -W7, row[3]);
    b2 = MUL16(W5, row[1]);
    MAC16(b2, -W1, row[3]);
    b3 = MUL16(W7, row[1]);

    if (AV_RN32A(row + 4) | AV_RN32A(row + 6)) {
        a0 +=  W4 * row[4] + W6 * row[6];
        a1 += -W4 * row[4] - W2 * row[6];
        a2 += -W4 * row[4] + W2 * row[6];
        a3 +=  W4 * row[4] - W6 * row[6];

        MAC16(b0,  W5, row[5]);
        MAC16(b0,  W7, row[7]);
        MAC16(b1, -W1, row[5]);
        MAC16(b1, -W5, row[7]);
        MAC16(b2,  W7, row[5]);
        MAC16(b2,  W3, row[7]);
        MAC16(b3,  W3, row[5]);
        MAC16(b3, -W1, row[7]);
    }

    row[0] = (a0 + b0) >> ROW_SHIFT;
    row[7] = (a0 - b0) >> ROW_SHIFT;
    row[1] = (a1 + b1) >> ROW_SHIFT;
    row[6] = (a1 - b1) >> ROW_SHIFT;
    row[2] = (a2 + b2) >> ROW_SHIFT;
    row[5] = (a2 - b2) >> ROW_SHIFT;
    row[3] = (a3 + b3) >> ROW_SHIFT;
    row[4] = (a3 - b3) >> ROW_SHIFT;
}

static inline void idct_col(int16_t *col)
{
    int a0, a1, a2, a3, b0, b1, b2, b3;

    a0 = W4 * col[8*0] + (1 << (COL_SHIFT - 1));
    a1 = a0;
    a2 = a0;
    a3 = a0;

    a0 += W2 * col[8*2];
    a1 += W6 * col[8*2];
    a2 -= W6 * col[8*2];
    a3 -= W2 * col[8*2];

    b0 = MUL16(W1, col[8*1]);
    MAC16(b0, W3, col[8*3]);
    b1 = MUL16(W3, col[8*1]);
    MAC16(b1, -W7, col[8*3]);
    b2 = MUL16(W5, col[8*1]);
    MAC16(b2, -W1, col[8*3]);
    b3 = MUL16(W7, col[8*1]);
    MAC16(b3, -W5, col[8*3]);

    a0 +=  W4 * col[8*4] + W6 * col[8*6];
    a1 += -W4 * col[8*4] - W2 * col[8*6];
    a2 += -W4 * col[8*4] + W2 * col[8*6];
    a3 +=  W4 * col[8*4] - W6 * col[8*6];

    MAC16(b0,  W5, col[8*5]);
    MAC16(b0,  W7, col[8*7]);
    MAC16(b1, -W1, col[8*5]);
    MAC16(b1, -W5, col[8*7]);
    MAC16(b2,  W7, col[8*5]);
    MAC16(b2,  W3, col[8*7]);
    MAC16(b3,  W3, col[8*5]);
    MAC16(b3, -W1, col[8*7]);

    col[8*0] = (a0 + b0) >> COL_SHIFT;
    col[8*7] = (a0 - b0) >> COL_SHIFT;
    col[8*1] = (a1 + b1) >> COL_SHIFT;
    col[8*6] = (a1 - b1) >> COL_SHIFT;
    col[8*2] = (a2 + b2) >> COL_SHIFT;
    col[8*5] = (a2 - b2) >> COL_SHIFT;
    col[8*3] = (a3 + b3) >> COL_SHIFT;
    col[8*4] = (a3 - b3) >> COL_SHIFT;
}

void ff_simple_idct_mxu(int16_t *block)
{
    int i;

    for (i = 0; i < 8; i++)
        idct_row(block + i * 8);

    for (i = 0; i < 8; i++)
        idct_col(block + i);
}

void ff_simple_idct_put_mxu(uint8_t *dest, ptrdiff_t stride, int16_t *block)
{
    ff_simple_idct_mxu(block);
    ff_put_pixels_clamped_mxu(block, dest, stride);
}

void ff_simple_idct_add_mxu(uint8_t *dest, ptrdiff_t stride, int16_t *block)
{
    ff_simple_idct_mxu(block);
    ff_add_pixels_clamped_mxu(block, dest, stride);
}
