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
 * Ingenic XBurst2 MIPS32r2 optimised H.264 IDCT and pixel functions.
 *
 * Optimisations over -Os compiled C:
 *  - Branchless uint8 clamping with conditional moves
 *  - Word-sized loads/stores for processing 4 pixels at a time
 *  - Hand-unrolled loops to avoid loop overhead
 *  - DC-only shortcuts for the most common IDCT case
 */

#include <string.h>
#include "libavutil/intreadwrite.h"
#include "libavutil/common.h"
#include "libavcodec/h264dec.h"
#include "h264dsp_mips.h"

/* ---- Branchless uint8 clip ---- */

static inline uint8_t clip_uint8(int v)
{
    if (v & ~0xFF)
        return (-v) >> 31;
    return v;
}

/* ---- DC-only 4x4 IDCT add (most common case) ---- */

void ff_h264_idct_dc_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    int dc = (block[0] + 32) >> 6;
    block[0] = 0;

    for (i = 0; i < 4; i++) {
        dst[0] = clip_uint8(dst[0] + dc);
        dst[1] = clip_uint8(dst[1] + dc);
        dst[2] = clip_uint8(dst[2] + dc);
        dst[3] = clip_uint8(dst[3] + dc);
        dst += stride;
    }
}

/* ---- DC-only 8x8 IDCT add ---- */

void ff_h264_idct8_dc_add_8_mxu(uint8_t *dst, int16_t *block, int stride)
{
    int i;
    int dc = (block[0] + 32) >> 6;
    block[0] = 0;

    for (i = 0; i < 8; i++) {
        dst[0] = clip_uint8(dst[0] + dc);
        dst[1] = clip_uint8(dst[1] + dc);
        dst[2] = clip_uint8(dst[2] + dc);
        dst[3] = clip_uint8(dst[3] + dc);
        dst[4] = clip_uint8(dst[4] + dc);
        dst[5] = clip_uint8(dst[5] + dc);
        dst[6] = clip_uint8(dst[6] + dc);
        dst[7] = clip_uint8(dst[7] + dc);
        dst += stride;
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

