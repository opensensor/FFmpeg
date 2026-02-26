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
 * Ingenic XBurst2 MXU SIMD optimised H.264 IDCT and pixel functions.
 *
 * Optimisations over -Os compiled C:
 *  - MXU Q8ADD_AA/Q8ADD_SS for saturating pixel add/sub (4 bytes at once)
 *  - Word-sized loads/stores for processing 4 pixels at a time
 *  - Hand-unrolled loops to avoid loop overhead
 *  - DC-only shortcuts for the most common IDCT case
 *
 * MXU instruction usage:
 *  - S32I2M: move GPR → XR register
 *  - S32M2I: move XR register → GPR
 *  - Q8ADD_AA: saturating unsigned byte add (4 lanes, clamps to 255)
 *  - Q8ADD_SS: saturating unsigned byte sub (4 lanes, clamps to 0)
 */

#include <string.h>
#include <stdint.h>
#include "libavutil/intreadwrite.h"
#include "libavutil/common.h"
#include "libavcodec/h264dec.h"
#include "h264dsp_mips.h"

/* XBurst2 XR (MXU) intrinsics for packed 8-bit/16-bit math */
#include "../../../thingino-accel/include/mxu.h"

/* ---- Branchless uint8 clip ---- */

static inline uint8_t clip_uint8(int v)
{
    if (v & ~0xFF)
        return (-v) >> 31;
    return v;
}

/* ---- MXU-accelerated DC-only 4x4 IDCT add (most common case) ---- */

/**
 * DC-only 4x4 IDCT add using MXU packed byte saturating arithmetic.
 *
 * Instead of 16 individual clip_uint8(dst[i] + dc) calls (each with a
 * branch), we process 4 pixels per iteration using Q8ADD_AA (saturating
 * add) or Q8ADD_SS (saturating subtract):
 *
 *   Q8ADD_AA: xra[i] = min(xrb[i] + xrc[i], 255) for 4 byte lanes
 *   Q8ADD_SS: xra[i] = max(xrb[i] - xrc[i], 0)   for 4 byte lanes
 *
 * This handles uint8 clamping automatically in hardware.
 */
void ff_h264_idct_dc_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    int dc = (block[0] + 32) >> 6;
    block[0] = 0;

    if (dc == 0)
        return;

    if (dc > 0) {
        /* Clamp dc to byte range for Q8ADD_AA */
        uint32_t dcb = (dc > 255) ? 255 : (uint32_t)dc;
        /* Splat dc byte into all 4 lanes */
        dcb |= dcb << 8;
        dcb |= dcb << 16;
        S32I2M(xr2, dcb);

        for (i = 0; i < 4; i++) {
            S32I2M(xr1, AV_RN32(dst));
            Q8ADD_AA(xr1, xr1, xr2);
            AV_WN32(dst, (uint32_t)S32M2I(xr1));
            dst += stride;
        }
    } else {
        /* dc < 0: subtract |dc| with saturation to 0 */
        uint32_t dcb = (-dc > 255) ? 255 : (uint32_t)(-dc);
        dcb |= dcb << 8;
        dcb |= dcb << 16;
        S32I2M(xr2, dcb);

        for (i = 0; i < 4; i++) {
            S32I2M(xr1, AV_RN32(dst));
            Q8ADD_SS(xr1, xr1, xr2);
            AV_WN32(dst, (uint32_t)S32M2I(xr1));
            dst += stride;
        }
    }
}

/* ---- MXU-accelerated DC-only 8x8 IDCT add ---- */

void ff_h264_idct8_dc_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    int dc = (block[0] + 32) >> 6;
    block[0] = 0;

    if (dc == 0)
        return;

    if (dc > 0) {
        uint32_t dcb = (dc > 255) ? 255 : (uint32_t)dc;
        dcb |= dcb << 8;
        dcb |= dcb << 16;
        S32I2M(xr2, dcb);

        for (i = 0; i < 8; i++) {
            /* Process 8 pixels per row: two 4-byte groups */
            S32I2M(xr1, AV_RN32(dst));
            Q8ADD_AA(xr1, xr1, xr2);
            AV_WN32(dst, (uint32_t)S32M2I(xr1));

            S32I2M(xr3, AV_RN32(dst + 4));
            Q8ADD_AA(xr3, xr3, xr2);
            AV_WN32(dst + 4, (uint32_t)S32M2I(xr3));

            dst += stride;
        }
    } else {
        uint32_t dcb = (-dc > 255) ? 255 : (uint32_t)(-dc);
        dcb |= dcb << 8;
        dcb |= dcb << 16;
        S32I2M(xr2, dcb);

        for (i = 0; i < 8; i++) {
            S32I2M(xr1, AV_RN32(dst));
            Q8ADD_SS(xr1, xr1, xr2);
            AV_WN32(dst, (uint32_t)S32M2I(xr1));

            S32I2M(xr3, AV_RN32(dst + 4));
            Q8ADD_SS(xr3, xr3, xr2);
            AV_WN32(dst + 4, (uint32_t)S32M2I(xr3));

            dst += stride;
        }
    }
}

/* ---- Full 4x4 IDCT add ---- */

void ff_h264_idct_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;

    block[0] += 1 << 5;

    /* Column pass */
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

    /* Row pass + add to destination + clip */
    for (i = 0; i < 4; i++) {
        const unsigned int z0 =  block[0 + 4*i]     +  (unsigned int)block[2 + 4*i];
        const unsigned int z1 =  block[0 + 4*i]     -  (unsigned int)block[2 + 4*i];
        const unsigned int z2 = (block[1 + 4*i]>>1) -  (unsigned int)block[3 + 4*i];
        const unsigned int z3 =  block[1 + 4*i]     + (unsigned int)(block[3 + 4*i]>>1);

        dst[i*stride + 0] = clip_uint8(dst[i*stride + 0] + ((int)(z0 + z3) >> 6));
        dst[i*stride + 1] = clip_uint8(dst[i*stride + 1] + ((int)(z1 + z2) >> 6));
        dst[i*stride + 2] = clip_uint8(dst[i*stride + 2] + ((int)(z1 - z2) >> 6));
        dst[i*stride + 3] = clip_uint8(dst[i*stride + 3] + ((int)(z0 - z3) >> 6));
    }

    memset(block, 0, 16 * sizeof(int16_t));
}

/* ---- Full 8x8 IDCT add ---- */

void ff_h264_idct8_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;

    block[0] += 32;

    /* Column pass */
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

    /* Row pass + add to destination + clip */
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

        dst[i*stride + 0] = clip_uint8(dst[i*stride + 0] + ((int)(b0 + b7) >> 6));
        dst[i*stride + 1] = clip_uint8(dst[i*stride + 1] + ((int)(b2 + b5) >> 6));
        dst[i*stride + 2] = clip_uint8(dst[i*stride + 2] + ((int)(b4 + b3) >> 6));
        dst[i*stride + 3] = clip_uint8(dst[i*stride + 3] + ((int)(b6 + b1) >> 6));
        dst[i*stride + 4] = clip_uint8(dst[i*stride + 4] + ((int)(b6 - b1) >> 6));
        dst[i*stride + 5] = clip_uint8(dst[i*stride + 5] + ((int)(b4 - b3) >> 6));
        dst[i*stride + 6] = clip_uint8(dst[i*stride + 6] + ((int)(b2 - b5) >> 6));
        dst[i*stride + 7] = clip_uint8(dst[i*stride + 7] + ((int)(b0 - b7) >> 6));
    }

    memset(block, 0, 64 * sizeof(int16_t));
}

/* ---- Add pixels 4x4 + clear ---- */

void ff_h264_add_pixels4_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    for (i = 0; i < 4; i++) {
        dst[0] += (unsigned)block[0];
        dst[1] += (unsigned)block[1];
        dst[2] += (unsigned)block[2];
        dst[3] += (unsigned)block[3];
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
        dst[0] += (unsigned)block[0];
        dst[1] += (unsigned)block[1];
        dst[2] += (unsigned)block[2];
        dst[3] += (unsigned)block[3];
        dst[4] += (unsigned)block[4];
        dst[5] += (unsigned)block[5];
        dst[6] += (unsigned)block[6];
        dst[7] += (unsigned)block[7];
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
        for (d = 0; d < inner_iters; d++) {
            const int p0 = pix[-1*xstride];
            const int p1 = pix[-2*xstride];
            const int p2 = pix[-3*xstride];
            const int q0 = pix[0];
            const int q1 = pix[1*xstride];
            const int q2 = pix[2*xstride];

            if (FFABS(p0 - q0) < alpha &&
                FFABS(p1 - p0) < beta &&
                FFABS(q1 - q0) < beta) {

                int tc = tc_orig;
                int i_delta;

                if (FFABS(p2 - p0) < beta) {
                    if (tc_orig)
                        pix[-2*xstride] = p1 + clip3(((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1, -tc_orig, tc_orig);
                    tc++;
                }
                if (FFABS(q2 - q0) < beta) {
                    if (tc_orig)
                        pix[xstride] = q1 + clip3(((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1, -tc_orig, tc_orig);
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

/*
 * Horizontal luma inter loop filter (xstride == 1) accelerated with XR MXU.
 *
 * For each edge we process "inner_iters" rows. On each row the relevant
 * pixels are laid out as:
 *   p2 p1 p0 | q0 q1 q2
 * and are byte-contiguous in memory, which maps naturally onto the 4-byte
 * XR registers:
 *   S32LDD(xr0, row-3) → [p2, p1, p0, q0]
 *   S32LDD(xr1, row-2) → [p1, p0, q0, q1]
 *   S32LDD(xr2, row-1) → [p0, q0, q1, q2]
 *
 * Two Q8ABD operations then give us the absolute differences needed for the
 * H.264 conditions:
 *   diffA = |xr0 - xr1| → [|p2-p1|, |p1-p0|, |p0-q0|, |q0-q1|]
 *   diffB = |xr0 - xr2| → [|p2-p0|,  ... ,  ... , |q0-q2|]
 *
 * We still perform the scalar arithmetic for tc/i_delta exactly as in the
 * scalar helper but use S32LDD/Q8ABD/S32M2I/S32I2M/S32STD to load,
 * compute abs-diffs and store 4 pixels at a time.
 */
static av_always_inline void h264_loop_filter_luma_h_xr_mxu(uint8_t *pix,
        ptrdiff_t stride, int inner_iters,
        int alpha, int beta, int8_t *tc0)
{
    int i, d;

    for (i = 0; i < 4; i++) {
        const int tc_orig = tc0[i];

        if (tc_orig < 0) {
            pix += inner_iters * stride;
            continue;
        }

        for (d = 0; d < inner_iters; d++) {
            uint8_t *row = pix;
            intptr_t base_m3 = (intptr_t)(row - 3); /* p2 */
            intptr_t base_m2 = (intptr_t)(row - 2); /* p1 */
            intptr_t base_m1 = (intptr_t)(row - 1); /* p0 */

            /* Load p2..q2 into XR registers (4 bytes each) */
            S32LDD(xr0, base_m3, 0); /* [p2, p1, p0, q0] */
            S32LDD(xr1, base_m2, 0); /* [p1, p0, q0, q1] */
            S32LDD(xr2, base_m1, 0); /* [p0, q0, q1, q2] */

            /* Preserve packed pixel words */
            uint32_t pack0 = (uint32_t) S32M2I(xr0);
            uint32_t pack1 = (uint32_t) S32M2I(xr1);
            uint32_t pack2 = (uint32_t) S32M2I(xr2);

            /* Vector absolute differences for filter conditions */
            Q8ABD(xr3, xr0, xr1);
            Q8ABD(xr4, xr0, xr2);

            uint32_t diffA = (uint32_t) S32M2I(xr3);
            uint32_t diffB = (uint32_t) S32M2I(xr4);

            /* Byte lanes are in increasing address order in the 32-bit word. */
            int p2 =  pack0        & 0xFF;
            int p1 = (pack0 >>  8) & 0xFF;
            int p0 = (pack0 >> 16) & 0xFF;
            int q0 = (pack0 >> 24) & 0xFF;
            int q1 = (pack1 >> 24) & 0xFF;
            int q2 = (pack2 >> 24) & 0xFF;

            int abs_p1_p0 = (diffA >>  8) & 0xFF;
            int abs_p0_q0 = (diffA >> 16) & 0xFF;
            int abs_q1_q0 = (diffA >> 24) & 0xFF;
            int abs_p2_p0 =  diffB        & 0xFF;
            int abs_q2_q0 = (diffB >> 24) & 0xFF;

            if (abs_p0_q0 < alpha &&
                abs_p1_p0 < beta &&
                abs_q1_q0 < beta) {

                int tc = tc_orig;
                int p1_old = p1;
                int q1_old = q1;
                int p1_new = p1;
                int q1_new = q1;

                if (abs_p2_p0 < beta) {
                    if (tc_orig) {
                        int avg_p0q0 = (p0 + q0 + 1) >> 1;
                        int tmp      = (p2 + avg_p0q0) >> 1;
                        int delta1   = clip3(tmp - p1_old, -tc_orig, tc_orig);
                        p1_new       = p1_old + delta1;
                    }
                    tc++;
                }

                if (abs_q2_q0 < beta) {
                    if (tc_orig) {
                        int avg_p0q0 = (p0 + q0 + 1) >> 1;
                        int tmp      = (q2 + avg_p0q0) >> 1;
                        int delta2   = clip3(tmp - q1_old, -tc_orig, tc_orig);
                        q1_new       = q1_old + delta2;
                    }
                    tc++;
                }

                /* Main delta and p0/q0 update as in scalar implementation */
                {
                    int i_delta = clip3((((q0 - p0) * 4) + (p1_old - q1_old) + 4) >> 3,
                                        -tc, tc);
                    int p0_new = clip_uint8(p0 + i_delta);
                    int q0_new = clip_uint8(q0 - i_delta);

                    /* Re-pack updated p2..q0 and p1..q1 and store via XR */
                    uint32_t pack0_new = (pack0 & 0x000000FFu) |
                                         ((uint32_t)p1_new << 8) |
                                         ((uint32_t)p0_new << 16) |
                                         ((uint32_t)q0_new << 24);
                    uint32_t pack1_new = ((uint32_t)p1_new << 0) |
                                         ((uint32_t)p0_new << 8) |
                                         ((uint32_t)q0_new << 16) |
                                         ((uint32_t)q1_new << 24);

                    S32I2M(xr5, pack0_new);
                    S32I2M(xr6, pack1_new);
                    S32STD(xr5, base_m3, 0);
                    S32STD(xr6, base_m2, 0);
                }
            }

            pix += stride;
        }
    }
}

void ff_h264_v_loop_filter_luma_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                       int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_mxu(pix, stride, 1, 4, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_luma_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                       int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_h_xr_mxu(pix, stride, 4, alpha, beta, tc0);
}

void ff_h264_h_loop_filter_luma_mbaff_8_mxu(uint8_t *pix, ptrdiff_t stride,
                                              int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_h_xr_mxu(pix, stride, 2, alpha, beta, tc0);
}

/* ---- Luma intra (strong) loop filter ---- */

static av_always_inline void h264_loop_filter_luma_intra_mxu(uint8_t *pix,
        ptrdiff_t xstride, ptrdiff_t ystride,
        int inner_iters, int alpha, int beta)
{
    int d;
    for (d = 0; d < 4 * inner_iters; d++) {
        const int p2 = pix[-3*xstride];
        const int p1 = pix[-2*xstride];
        const int p0 = pix[-1*xstride];
        const int q0 = pix[0*xstride];
        const int q1 = pix[1*xstride];
        const int q2 = pix[2*xstride];

        if (FFABS(p0 - q0) < alpha &&
            FFABS(p1 - p0) < beta &&
            FFABS(q1 - q0) < beta) {

            if (FFABS(p0 - q0) < ((alpha >> 2) + 2)) {
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
        for (d = 0; d < inner_iters; d++) {
            const int p0 = pix[-1*xstride];
            const int p1 = pix[-2*xstride];
            const int q0 = pix[0];
            const int q1 = pix[1*xstride];

            if (FFABS(p0 - q0) < alpha &&
                FFABS(p1 - p0) < beta &&
                FFABS(q1 - q0) < beta) {

                int delta = clip3(((q0 - p0) * 4 + (p1 - q1) + 4) >> 3, -(int)tc, (int)tc);
                pix[-xstride] = clip_uint8(p0 + delta);
                pix[0]        = clip_uint8(q0 - delta);
            }
            pix += ystride;
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
    for (d = 0; d < 4 * inner_iters; d++) {
        const int p0 = pix[-1*xstride];
        const int p1 = pix[-2*xstride];
        const int q0 = pix[0];
        const int q1 = pix[1*xstride];

        if (FFABS(p0 - q0) < alpha &&
            FFABS(p1 - p0) < beta &&
            FFABS(q1 - q0) < beta) {

            pix[-xstride] = (2*p1 + p0 + q1 + 2) >> 2;
            pix[0]        = (2*q1 + q0 + p1 + 2) >> 2;
        }
        pix += ystride;
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