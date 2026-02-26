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
 * Ingenic XBurst2 MIPS32r2 optimised pixel block DSP functions.
 *
 * These functions are dispatched via the MXU CPU flag path, which
 * identifies XBurst2 hardware where MIPS32r2 hand-tuned code provides
 * significant benefit over -Os compiled generic C.
 *
 * Optimisations:
 *  - Word-sized loads (lw) to read 4 pixels at once
 *  - Packed int16 stores (sw) to write 2 coefficients at once
 *  - Full 8-row unrolling to eliminate loop overhead
 */

#include "libavutil/intreadwrite.h"
#include "pixblockdsp_mips.h"

/**
 * Convert one row of 8 uint8 pixels to 8 int16 coefficients using
 * word-sized loads and packed stores.
 */
static inline void get_pixels_row(int16_t *block, const uint8_t *pixels)
{
    uint32_t w0 = AV_RN32A(pixels);
    uint32_t w1 = AV_RN32A(pixels + 4);

    /* Little-endian: byte 0 is in bits [7:0] of w0 */
    /* Pack two int16 values into one uint32 store: low half | high half << 16 */
    AV_WN32A(block,     ( w0        & 0xFF) | (((w0 >>  8) & 0xFF) << 16));
    AV_WN32A(block + 2, ((w0 >> 16) & 0xFF) | (((w0 >> 24)       ) << 16));
    AV_WN32A(block + 4, ( w1        & 0xFF) | (((w1 >>  8) & 0xFF) << 16));
    AV_WN32A(block + 6, ((w1 >> 16) & 0xFF) | (((w1 >> 24)       ) << 16));
}

void ff_get_pixels_8_mxu(int16_t *restrict block, const uint8_t *pixels,
                          ptrdiff_t stride)
{
    get_pixels_row(block,      pixels);  pixels += stride;
    get_pixels_row(block + 8,  pixels);  pixels += stride;
    get_pixels_row(block + 16, pixels);  pixels += stride;
    get_pixels_row(block + 24, pixels);  pixels += stride;
    get_pixels_row(block + 32, pixels);  pixels += stride;
    get_pixels_row(block + 40, pixels);  pixels += stride;
    get_pixels_row(block + 48, pixels);  pixels += stride;
    get_pixels_row(block + 56, pixels);
}

/**
 * Compute one row of 8 pixel differences (s1[i] - s2[i]) as int16.
 */
static inline void diff_pixels_row(int16_t *block, const uint8_t *s1,
                                   const uint8_t *s2)
{
    uint32_t a0 = AV_RN32A(s1);
    uint32_t a1 = AV_RN32A(s1 + 4);
    uint32_t b0 = AV_RN32A(s2);
    uint32_t b1 = AV_RN32A(s2 + 4);

    int d0 = (int)( a0        & 0xFF) - (int)( b0        & 0xFF);
    int d1 = (int)((a0 >>  8) & 0xFF) - (int)((b0 >>  8) & 0xFF);
    int d2 = (int)((a0 >> 16) & 0xFF) - (int)((b0 >> 16) & 0xFF);
    int d3 = (int)((a0 >> 24)       ) - (int)((b0 >> 24)       );
    int d4 = (int)( a1        & 0xFF) - (int)( b1        & 0xFF);
    int d5 = (int)((a1 >>  8) & 0xFF) - (int)((b1 >>  8) & 0xFF);
    int d6 = (int)((a1 >> 16) & 0xFF) - (int)((b1 >> 16) & 0xFF);
    int d7 = (int)((a1 >> 24)       ) - (int)((b1 >> 24)       );

    /* Store as packed int16 pairs (little-endian) */
    AV_WN32A(block,     (d0 & 0xFFFF) | (d1 << 16));
    AV_WN32A(block + 2, (d2 & 0xFFFF) | (d3 << 16));
    AV_WN32A(block + 4, (d4 & 0xFFFF) | (d5 << 16));
    AV_WN32A(block + 6, (d6 & 0xFFFF) | (d7 << 16));
}

void ff_diff_pixels_mxu(int16_t *restrict block, const uint8_t *s1,
                         const uint8_t *s2, ptrdiff_t stride)
{
    diff_pixels_row(block,      s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 8,  s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 16, s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 24, s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 32, s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 40, s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 48, s1, s2);  s1 += stride;  s2 += stride;
    diff_pixels_row(block + 56, s1, s2);
}

