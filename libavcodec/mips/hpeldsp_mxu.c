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
 * Ingenic XBurst2 MXU SIMD optimised half-pel DSP functions.
 *
 * Optimisations over -Os compiled C:
 *  - Word-sized (32-bit) loads and stores for pixel copy/averaging
 *  - MXU Q8AVGR/Q8AVG hardware SIMD for byte-parallel averaging
 *    (replaces 5-operation bit-manipulation with single instruction)
 *  - Minimised loop overhead with counted loops
 *
 * MXU instruction usage:
 *  - S32I2M: move GPR → XR register
 *  - S32M2I: move XR register → GPR
 *  - Q8AVGR: packed byte rounding average (4 bytes in parallel)
 *    xra[i] = (xrb[i] + xrc[i] + 1) >> 1 for each byte lane
 *  - Q8AVG:  packed byte truncating average (4 bytes in parallel)
 *    xra[i] = (xrb[i] + xrc[i]) >> 1 for each byte lane
 */

#include "libavutil/intreadwrite.h"
#include "hpeldsp_mips.h"

/* XBurst2 XR (MXU) intrinsics for packed 8-bit SIMD */
#include "../../../thingino-accel/include/mxu.h"

/**
 * MXU-accelerated byte-parallel rounding average of 4 packed bytes.
 *
 * Uses the Q8AVGR hardware instruction which computes:
 *   result[i] = (a[i] + b[i] + 1) >> 1
 * for each of the 4 byte lanes simultaneously.
 *
 * This replaces the 5-operation software implementation:
 *   (a | b) - (((a ^ b) & 0xFEFEFEFE) >> 1)
 */
static inline uint32_t rnd_avg32(uint32_t a, uint32_t b)
{
    S32I2M(xr1, a);
    S32I2M(xr2, b);
    Q8AVGR(xr1, xr1, xr2);
    return (uint32_t)S32M2I(xr1);
}

/**
 * MXU-accelerated byte-parallel truncating average of 4 packed bytes.
 *
 * Uses the Q8AVG hardware instruction which computes:
 *   result[i] = (a[i] + b[i]) >> 1
 * for each of the 4 byte lanes simultaneously.
 *
 * This replaces the 5-operation software implementation:
 *   (a & b) + (((a ^ b) & 0xFEFEFEFE) >> 1)
 */
static inline uint32_t no_rnd_avg32(uint32_t a, uint32_t b)
{
    S32I2M(xr1, a);
    S32I2M(xr2, b);
    Q8AVG(xr1, xr1, xr2);
    return (uint32_t)S32M2I(xr1);
}

/* ---- put_pixels: straight copy ---- */

void ff_put_pixels16_mxu(uint8_t *block, const uint8_t *pixels,
                          ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,      AV_RN32(pixels));
        AV_WN32A(block + 4,  AV_RN32(pixels + 4));
        AV_WN32A(block + 8,  AV_RN32(pixels + 8));
        AV_WN32A(block + 12, AV_RN32(pixels + 12));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels8_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,     AV_RN32(pixels));
        AV_WN32A(block + 4, AV_RN32(pixels + 4));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels4_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block, AV_RN32(pixels));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- avg_pixels: rounding average with destination ---- */

void ff_avg_pixels16_mxu(uint8_t *block, const uint8_t *pixels,
                          ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,      rnd_avg32(AV_RN32A(block),      AV_RN32(pixels)));
        AV_WN32A(block + 4,  rnd_avg32(AV_RN32A(block + 4),  AV_RN32(pixels + 4)));
        AV_WN32A(block + 8,  rnd_avg32(AV_RN32A(block + 8),  AV_RN32(pixels + 8)));
        AV_WN32A(block + 12, rnd_avg32(AV_RN32A(block + 12), AV_RN32(pixels + 12)));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     AV_RN32(pixels)));
        AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), AV_RN32(pixels + 4)));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels4_mxu(uint8_t *block, const uint8_t *pixels,
                         ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block, rnd_avg32(AV_RN32A(block), AV_RN32(pixels)));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- put_pixels_x2: horizontal half-pel (average of [x] and [x+1]) ---- */

void ff_put_pixels16_x2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a0 = AV_RN32(pixels),      b0 = AV_RN32(pixels + 1);
        uint32_t a1 = AV_RN32(pixels + 4),  b1 = AV_RN32(pixels + 5);
        uint32_t a2 = AV_RN32(pixels + 8),  b2 = AV_RN32(pixels + 9);
        uint32_t a3 = AV_RN32(pixels + 12), b3 = AV_RN32(pixels + 13);
        AV_WN32A(block,      rnd_avg32(a0, b0));
        AV_WN32A(block + 4,  rnd_avg32(a1, b1));
        AV_WN32A(block + 8,  rnd_avg32(a2, b2));
        AV_WN32A(block + 12, rnd_avg32(a3, b3));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels8_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,     rnd_avg32(AV_RN32(pixels),     AV_RN32(pixels + 1)));
        AV_WN32A(block + 4, rnd_avg32(AV_RN32(pixels + 4), AV_RN32(pixels + 5)));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels4_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block, rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + 1)));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- put_pixels_y2: vertical half-pel (average of [y] and [y+stride]) ---- */

void ff_put_pixels16_y2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        AV_WN32A(block,      rnd_avg32(AV_RN32(pixels),      AV_RN32(p1)));
        AV_WN32A(block + 4,  rnd_avg32(AV_RN32(pixels + 4),  AV_RN32(p1 + 4)));
        AV_WN32A(block + 8,  rnd_avg32(AV_RN32(pixels + 8),  AV_RN32(p1 + 8)));
        AV_WN32A(block + 12, rnd_avg32(AV_RN32(pixels + 12), AV_RN32(p1 + 12)));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels8_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        AV_WN32A(block,     rnd_avg32(AV_RN32(pixels),     AV_RN32(p1)));
        AV_WN32A(block + 4, rnd_avg32(AV_RN32(pixels + 4), AV_RN32(p1 + 4)));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels4_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block, rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + line_size)));
        block  += line_size;
        pixels += line_size;
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
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        int j;
        for (j = 0; j < 16; j += 4) {
            AV_WN32A(block + j, avg4_round(AV_RN32(pixels + j),
                                           AV_RN32(pixels + j + 1),
                                           AV_RN32(p1 + j),
                                           AV_RN32(p1 + j + 1)));
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_pixels8_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
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

void ff_put_pixels4_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        AV_WN32A(block, avg4_round(AV_RN32(pixels), AV_RN32(pixels + 1),
                                   AV_RN32(p1),     AV_RN32(p1 + 1)));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- put_no_rnd_pixels_x2: horizontal half-pel, truncating ---- */

void ff_put_no_rnd_pixels16_x2_mxu(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
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
    for (i = 0; i < h; i++) {
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
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        int j;
        for (j = 0; j < 16; j += 4) {
            AV_WN32A(block + j, avg4_no_rnd(AV_RN32(pixels + j),
                                            AV_RN32(pixels + j + 1),
                                            AV_RN32(p1 + j),
                                            AV_RN32(p1 + j + 1)));
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_put_no_rnd_pixels8_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int32_t h)
{
    int i;
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


/* ---- avg_pixels_x2: horizontal half-pel + avg with dest ---- */

void ff_avg_pixels16_x2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        int j;
        for (j = 0; j < 16; j += 4) {
            uint32_t src = rnd_avg32(AV_RN32(pixels + j), AV_RN32(pixels + j + 1));
            AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t s0 = rnd_avg32(AV_RN32(pixels),     AV_RN32(pixels + 1));
        uint32_t s1 = rnd_avg32(AV_RN32(pixels + 4), AV_RN32(pixels + 5));
        AV_WN32A(block,     rnd_avg32(AV_RN32A(block),     s0));
        AV_WN32A(block + 4, rnd_avg32(AV_RN32A(block + 4), s1));
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels4_x2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t src = rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + 1));
        AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- avg_pixels_y2: vertical half-pel + avg with dest ---- */

void ff_avg_pixels16_y2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        int j;
        for (j = 0; j < 16; j += 4) {
            uint32_t src = rnd_avg32(AV_RN32(pixels + j), AV_RN32(p1 + j));
            AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
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

void ff_avg_pixels4_y2_mxu(uint8_t *block, const uint8_t *pixels,
                            ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t src = rnd_avg32(AV_RN32(pixels), AV_RN32(pixels + line_size));
        AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
        block  += line_size;
        pixels += line_size;
    }
}

/* ---- avg_pixels_xy2: bilinear + avg with dest ---- */

void ff_avg_pixels16_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                              ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        int j;
        for (j = 0; j < 16; j += 4) {
            uint32_t src = avg4_round(AV_RN32(pixels + j),
                                      AV_RN32(pixels + j + 1),
                                      AV_RN32(p1 + j),
                                      AV_RN32(p1 + j + 1));
            AV_WN32A(block + j, rnd_avg32(AV_RN32A(block + j), src));
        }
        block  += line_size;
        pixels += line_size;
    }
}

void ff_avg_pixels8_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
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

void ff_avg_pixels4_xy2_mxu(uint8_t *block, const uint8_t *pixels,
                             ptrdiff_t line_size, int32_t h)
{
    int i;
    for (i = 0; i < h; i++) {
        const uint8_t *p1 = pixels + line_size;
        uint32_t src = avg4_round(AV_RN32(pixels), AV_RN32(pixels + 1),
                                  AV_RN32(p1),     AV_RN32(p1 + 1));
        AV_WN32A(block, rnd_avg32(AV_RN32A(block), src));
        block  += line_size;
        pixels += line_size;
    }
}