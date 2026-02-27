/*
 * Ingenic XBurst2 (MXU) H.264 QPEL helpers.
 *
 * Keep this file dependency-free w.r.t. CONFIG_HPELDSP.
 *
 * Some build configurations enable H264QPEL without enabling HPELDSP.
 * In those cases, referencing ff_put/avg_pixels*_mxu() (from hpeldsp_mxu.c)
 * causes undefined references at link time.
 *
 * For mc00 (integer-pel copy / avg), use simple word-at-a-time scalar
 * routines. This keeps the MXU QPEL dispatch plumbing valid without
 * introducing cross-module link dependencies.
 */

#include <stddef.h>
#include <stdint.h>

#include "h264dsp_mips.h"

#include "libavutil/intreadwrite.h"

static inline uint32_t rnd_avg32(uint32_t a, uint32_t b)
{
    const uint32_t xor = a ^ b;
    /* Rounded per-byte average: (a + b + 1) >> 1, without cross-byte carries. */
    return (a & b) + ((xor & 0xFEFEFEFEU) >> 1) + (xor & 0x01010101U);
}

static inline void put_block_w4(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride, int h)
{
    for (int y = 0; y < h; y++) {
        AV_WN32(dst, AV_RN32(src));
        dst += stride;
        src += stride;
    }
}

static inline void put_block_w8(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride, int h)
{
    for (int y = 0; y < h; y++) {
        AV_WN32(dst + 0, AV_RN32(src + 0));
        AV_WN32(dst + 4, AV_RN32(src + 4));
        dst += stride;
        src += stride;
    }
}

static inline void put_block_w16(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t stride, int h)
{
    for (int y = 0; y < h; y++) {
        AV_WN32(dst +  0, AV_RN32(src +  0));
        AV_WN32(dst +  4, AV_RN32(src +  4));
        AV_WN32(dst +  8, AV_RN32(src +  8));
        AV_WN32(dst + 12, AV_RN32(src + 12));
        dst += stride;
        src += stride;
    }
}

static inline void avg_block_w4(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride, int h)
{
    for (int y = 0; y < h; y++) {
        const uint32_t s0 = AV_RN32(src);
        const uint32_t d0 = AV_RN32(dst);
        AV_WN32(dst, rnd_avg32(d0, s0));
        dst += stride;
        src += stride;
    }
}

static inline void avg_block_w8(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t stride, int h)
{
    for (int y = 0; y < h; y++) {
        const uint32_t s0 = AV_RN32(src + 0);
        const uint32_t s1 = AV_RN32(src + 4);
        const uint32_t d0 = AV_RN32(dst + 0);
        const uint32_t d1 = AV_RN32(dst + 4);
        AV_WN32(dst + 0, rnd_avg32(d0, s0));
        AV_WN32(dst + 4, rnd_avg32(d1, s1));
        dst += stride;
        src += stride;
    }
}

static inline void avg_block_w16(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t stride, int h)
{
    for (int y = 0; y < h; y++) {
        const uint32_t s0 = AV_RN32(src +  0);
        const uint32_t s1 = AV_RN32(src +  4);
        const uint32_t s2 = AV_RN32(src +  8);
        const uint32_t s3 = AV_RN32(src + 12);
        const uint32_t d0 = AV_RN32(dst +  0);
        const uint32_t d1 = AV_RN32(dst +  4);
        const uint32_t d2 = AV_RN32(dst +  8);
        const uint32_t d3 = AV_RN32(dst + 12);
        AV_WN32(dst +  0, rnd_avg32(d0, s0));
        AV_WN32(dst +  4, rnd_avg32(d1, s1));
        AV_WN32(dst +  8, rnd_avg32(d2, s2));
        AV_WN32(dst + 12, rnd_avg32(d3, s3));
        dst += stride;
        src += stride;
    }
}

void ff_put_h264_qpel16_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    put_block_w16(dst, src, dst_stride, 16);
}

void ff_put_h264_qpel8_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    put_block_w8(dst, src, dst_stride, 8);
}

void ff_put_h264_qpel4_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    put_block_w4(dst, src, dst_stride, 4);
}

void ff_avg_h264_qpel16_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    avg_block_w16(dst, src, dst_stride, 16);
}

void ff_avg_h264_qpel8_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    avg_block_w8(dst, src, dst_stride, 8);
}

void ff_avg_h264_qpel4_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    avg_block_w4(dst, src, dst_stride, 4);
}
