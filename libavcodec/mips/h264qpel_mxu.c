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

#include "libavutil/common.h"
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

static inline int qpel_6tap_u8(const uint8_t *p)
{
    /* 6-tap H.264 luma filter: [ 1, -5, 20, 20, -5, 1 ] */
    return p[0] + p[5] - 5 * (p[1] + p[4]) + 20 * (p[2] + p[3]);
}

static inline uint8_t qpel_clip_shift5(int v)
{
    return av_clip_uint8((v + 16) >> 5);
}

static inline uint8_t qpel_clip_shift10(int v)
{
    return av_clip_uint8((v + 512) >> 10);
}

static inline void qpel_put_or_avg_row(uint8_t *dst, const uint8_t *pred,
                                       ptrdiff_t stride, int w, int h,
                                       int do_avg)
{
    for (int y = 0; y < h; y++) {
        if (!do_avg) {
            for (int x = 0; x < w; x++)
                dst[x] = pred[x];
        } else {
            for (int x = 0; x < w; x++)
                dst[x] = (dst[x] + pred[x] + 1) >> 1;
        }
        dst  += stride;
        pred += w;
    }
}

static inline void qpel_h_lowpass(uint8_t *dst, const uint8_t *src,
                                  ptrdiff_t stride, int w, int h,
                                  int do_avg)
{
    uint8_t pred[16 * 16];
    uint8_t *p = pred;

    for (int y = 0; y < h; y++) {
        const uint8_t *s = src + y * stride - 2;
        for (int x = 0; x < w; x++) {
            const int v = qpel_6tap_u8(s + x);
            p[x] = qpel_clip_shift5(v);
        }
        p += w;
    }

    qpel_put_or_avg_row(dst, pred, stride, w, h, do_avg);
}

static inline void qpel_v_lowpass(uint8_t *dst, const uint8_t *src,
                                  ptrdiff_t stride, int w, int h,
                                  int do_avg)
{
    uint8_t pred[16 * 16];
    uint8_t *p = pred;

    for (int y = 0; y < h; y++) {
        const uint8_t *s0 = src + (y - 2) * stride;
        const uint8_t *s1 = s0 + stride;
        const uint8_t *s2 = s1 + stride;
        const uint8_t *s3 = s2 + stride;
        const uint8_t *s4 = s3 + stride;
        const uint8_t *s5 = s4 + stride;

        for (int x = 0; x < w; x++) {
            const int v = s0[x] + s5[x] - 5 * (s1[x] + s4[x]) + 20 * (s2[x] + s3[x]);
            p[x] = qpel_clip_shift5(v);
        }
        p += w;
    }

    qpel_put_or_avg_row(dst, pred, stride, w, h, do_avg);
}

static inline void qpel_hv_lowpass(uint8_t *dst, const uint8_t *src,
                                   ptrdiff_t stride, int w, int h,
                                   int do_avg)
{
    /* Ring buffer of 6 horizontally-filtered rows (int16 intermediate). */
    int16_t hbuf[6][16];
    uint8_t pred[16 * 16];
    uint8_t *p = pred;

    for (int r = 0; r < 6; r++) {
        const uint8_t *s = src + (r - 2) * stride - 2;
        for (int x = 0; x < w; x++)
            hbuf[r][x] = qpel_6tap_u8(s + x);
    }

    for (int y = 0; y < h; y++) {
        const int i0 = (y + 0) % 6;
        const int i1 = (y + 1) % 6;
        const int i2 = (y + 2) % 6;
        const int i3 = (y + 3) % 6;
        const int i4 = (y + 4) % 6;
        const int i5 = (y + 5) % 6;

        for (int x = 0; x < w; x++) {
            const int v = hbuf[i0][x] + hbuf[i5][x]
                        - 5 * (hbuf[i1][x] + hbuf[i4][x])
                        + 20 * (hbuf[i2][x] + hbuf[i3][x]);
            p[x] = qpel_clip_shift10(v);
        }
        p += w;

        /* Compute next horizontal intermediate row (source row y+4) into slot y%6. */
        const uint8_t *sn = src + (y + 4) * stride - 2;
        for (int x = 0; x < w; x++)
            hbuf[y % 6][x] = qpel_6tap_u8(sn + x);
    }

    qpel_put_or_avg_row(dst, pred, stride, w, h, do_avg);
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

/* ---- mc20: horizontal half-pel (6-tap) ---- */

void ff_put_h264_qpel16_mc20_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    qpel_h_lowpass(dst, src, dst_stride, 16, 16, 0);
}

void ff_put_h264_qpel8_mc20_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_h_lowpass(dst, src, dst_stride, 8, 8, 0);
}

void ff_put_h264_qpel4_mc20_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_h_lowpass(dst, src, dst_stride, 4, 4, 0);
}

void ff_avg_h264_qpel16_mc20_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    qpel_h_lowpass(dst, src, dst_stride, 16, 16, 1);
}

void ff_avg_h264_qpel8_mc20_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_h_lowpass(dst, src, dst_stride, 8, 8, 1);
}

void ff_avg_h264_qpel4_mc20_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_h_lowpass(dst, src, dst_stride, 4, 4, 1);
}

/* ---- mc02: vertical half-pel (6-tap) ---- */

void ff_put_h264_qpel16_mc02_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    qpel_v_lowpass(dst, src, dst_stride, 16, 16, 0);
}

void ff_put_h264_qpel8_mc02_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_v_lowpass(dst, src, dst_stride, 8, 8, 0);
}

void ff_put_h264_qpel4_mc02_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_v_lowpass(dst, src, dst_stride, 4, 4, 0);
}

void ff_avg_h264_qpel16_mc02_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    qpel_v_lowpass(dst, src, dst_stride, 16, 16, 1);
}

void ff_avg_h264_qpel8_mc02_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_v_lowpass(dst, src, dst_stride, 8, 8, 1);
}

void ff_avg_h264_qpel4_mc02_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_v_lowpass(dst, src, dst_stride, 4, 4, 1);
}

/* ---- mc22: horizontal+vertical half-pel (6-tap + 6-tap) ---- */

void ff_put_h264_qpel16_mc22_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    qpel_hv_lowpass(dst, src, dst_stride, 16, 16, 0);
}

void ff_put_h264_qpel8_mc22_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_hv_lowpass(dst, src, dst_stride, 8, 8, 0);
}

void ff_put_h264_qpel4_mc22_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_hv_lowpass(dst, src, dst_stride, 4, 4, 0);
}

void ff_avg_h264_qpel16_mc22_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    qpel_hv_lowpass(dst, src, dst_stride, 16, 16, 1);
}

void ff_avg_h264_qpel8_mc22_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_hv_lowpass(dst, src, dst_stride, 8, 8, 1);
}

void ff_avg_h264_qpel4_mc22_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    qpel_hv_lowpass(dst, src, dst_stride, 4, 4, 1);
}
