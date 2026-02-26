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
 * Ingenic XBurst2 MIPS32r2 optimised IDCT and pixel clamping functions.
 *
 * Optimisations over -Os compiled C:
 *  - Word-sized loads and stores for pixel clamping
 *  - Hand-scheduled IDCT butterfly with DC-only shortcut
 *  - Full 8-row unrolling where beneficial
 */

#include <string.h>
#include "libavutil/intreadwrite.h"
#include "libavutil/common.h"
#include "idctdsp_mips.h"

/* ---- Pixel clamping functions ---- */

static inline uint8_t clip_uint8(int v)
{
    if (v & ~0xFF)
        return (-v) >> 31;
    return v;
}

void ff_put_pixels_clamped_mxu(const int16_t *block,
                                uint8_t *restrict pixels,
                                ptrdiff_t line_size)
{
    int i;
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
}

void ff_put_signed_pixels_clamped_mxu(const int16_t *block,
                                       uint8_t *restrict pixels,
                                       ptrdiff_t line_size)
{
    int i;
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
}

void ff_add_pixels_clamped_mxu(const int16_t *block,
                                uint8_t *restrict pixels,
                                ptrdiff_t line_size)
{
    int i;
    for (i = 0; i < 8; i++) {
        pixels[0] = clip_uint8(pixels[0] + block[0]);
        pixels[1] = clip_uint8(pixels[1] + block[1]);
        pixels[2] = clip_uint8(pixels[2] + block[2]);
        pixels[3] = clip_uint8(pixels[3] + block[3]);
        pixels[4] = clip_uint8(pixels[4] + block[4]);
        pixels[5] = clip_uint8(pixels[5] + block[5]);
        pixels[6] = clip_uint8(pixels[6] + block[6]);
        pixels[7] = clip_uint8(pixels[7] + block[7]);
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
