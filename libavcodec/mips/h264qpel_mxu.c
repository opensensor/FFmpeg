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

#include "mxu.h"

static inline uint8_t clip_uint8(int v)
{
    /* Branchless clamp to [0,255]. Works for v in a reasonable int range. */
    if (v & ~0xFF)
        return (-v) >> 31;
    return v;
}

static inline uint32_t rnd_avg32(uint32_t a, uint32_t b)
{
    /*
     * Rounded per-byte average: (a + b + 1) >> 1, without cross-byte carries.
     * Portable bit-manipulation — MXU1 Q8AVGR SIGILLs on XBurst2 (A1/T41).
     */
    const uint32_t xor_val = a ^ b;
    return (a & b) + ((xor_val & 0xFEFEFEFEU) >> 1) + (xor_val & 0x01010101U);
}

static inline uint32_t pack_u8x4(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3)
{
#if HAVE_BIGENDIAN
    return ((uint32_t)b0 << 24) | ((uint32_t)b1 << 16) | ((uint32_t)b2 << 8) | (uint32_t)b3;
#else
    return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
#endif
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

static inline int qpel_6tap_vals(int p0, int p1, int p2, int p3, int p4, int p5)
{
    /*
     * v = p0 + p5 - 5*(p1+p4) + 20*(p2+p3)
     * Use shifts/adds so -Os doesn't choose slower integer mul on XBurst2.
     */
    const int s05 = p0 + p5;
    const int s14 = p1 + p4;
    const int s23 = p2 + p3;
    return s05 - ((s14 << 2) + s14) + ((s23 << 4) + (s23 << 2));
}

static inline uint8_t qpel_clip_shift5(int v)
{
    return clip_uint8((v + 16) >> 5);
}

static inline uint8_t qpel_clip_shift10(int v)
{
    return clip_uint8((v + 512) >> 10);
}

static inline void qpel_h_lowpass(uint8_t *dst, const uint8_t *src,
                                  ptrdiff_t stride, int w, int h,
                                  int do_avg)
{
    /* Sliding-window horizontal 6-tap: reduces loads from 6/pixel to ~1/pixel. */
    for (int y = 0; y < h; y++) {
        const uint8_t *s = src + y * stride - 2;
        int p0 = s[0], p1 = s[1], p2 = s[2], p3 = s[3], p4 = s[4], p5 = s[5];

        for (int x = 0; x < w; x += 4) {
            uint8_t o0, o1, o2, o3;

            o0 = qpel_clip_shift5(qpel_6tap_vals(p0, p1, p2, p3, p4, p5));
            p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = p5; p5 = s[x + 6];

            o1 = qpel_clip_shift5(qpel_6tap_vals(p0, p1, p2, p3, p4, p5));
            p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = p5; p5 = s[x + 7];

            o2 = qpel_clip_shift5(qpel_6tap_vals(p0, p1, p2, p3, p4, p5));
            p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = p5; p5 = s[x + 8];

            o3 = qpel_clip_shift5(qpel_6tap_vals(p0, p1, p2, p3, p4, p5));
            p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = p5;
            if (x + 4 < w)
                p5 = s[x + 9];

            const uint32_t pred = pack_u8x4(o0, o1, o2, o3);
            if (!do_avg) {
                AV_WN32(dst + x, pred);
            } else {
                const uint32_t d = AV_RN32(dst + x);
                AV_WN32(dst + x, rnd_avg32(d, pred));
            }
        }
        dst += stride;
    }
}

static inline void qpel_v_lowpass(uint8_t *dst, const uint8_t *src,
                                  ptrdiff_t stride, int w, int h,
                                  int do_avg)
{
    /* Keep a 6-row sliding window of source bytes, rotated by pointer swap. */
    uint8_t rows_mem[6][16];
    uint8_t *r0 = rows_mem[0];
    uint8_t *r1 = rows_mem[1];
    uint8_t *r2 = rows_mem[2];
    uint8_t *r3 = rows_mem[3];
    uint8_t *r4 = rows_mem[4];
    uint8_t *r5 = rows_mem[5];

    for (int r = 0; r < 6; r++) {
        const uint8_t *s = src + (r - 2) * stride;
        for (int x = 0; x < w; x += 4)
            AV_WN32(rows_mem[r] + x, AV_RN32(s + x));
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x += 4) {
            const uint8_t o0 = qpel_clip_shift5(qpel_6tap_vals(r0[x + 0], r1[x + 0], r2[x + 0], r3[x + 0], r4[x + 0], r5[x + 0]));
            const uint8_t o1 = qpel_clip_shift5(qpel_6tap_vals(r0[x + 1], r1[x + 1], r2[x + 1], r3[x + 1], r4[x + 1], r5[x + 1]));
            const uint8_t o2 = qpel_clip_shift5(qpel_6tap_vals(r0[x + 2], r1[x + 2], r2[x + 2], r3[x + 2], r4[x + 2], r5[x + 2]));
            const uint8_t o3 = qpel_clip_shift5(qpel_6tap_vals(r0[x + 3], r1[x + 3], r2[x + 3], r3[x + 3], r4[x + 3], r5[x + 3]));
            const uint32_t pred = pack_u8x4(o0, o1, o2, o3);

            if (!do_avg) {
                AV_WN32(dst + x, pred);
            } else {
                const uint32_t d = AV_RN32(dst + x);
                AV_WN32(dst + x, rnd_avg32(d, pred));
            }
        }
        dst += stride;

        /* Slide window down by one row and refill the newest (y+4). */
        if (y + 1 < h) {
            uint8_t *tmp = r0;
            r0 = r1; r1 = r2; r2 = r3; r3 = r4; r4 = r5; r5 = tmp;

            const uint8_t *sn = src + (y + 4) * stride;
            for (int x = 0; x < w; x += 4)
                AV_WN32(r5 + x, AV_RN32(sn + x));
        }
    }
}

static inline void qpel_hv_lowpass(uint8_t *dst, const uint8_t *src,
                                   ptrdiff_t stride, int w, int h,
                                   int do_avg)
{
    /* Ring buffer of 6 horizontally-filtered rows (int16 intermediate). */
    int16_t hmem[6][16];
    int16_t *r0 = hmem[0];
    int16_t *r1 = hmem[1];
    int16_t *r2 = hmem[2];
    int16_t *r3 = hmem[3];
    int16_t *r4 = hmem[4];
    int16_t *r5 = hmem[5];

    /* Horizontal stage (sliding window) into the 6-row buffer. */
    for (int r = 0; r < 6; r++) {
        const uint8_t *s = src + (r - 2) * stride - 2;
        int p0 = s[0], p1 = s[1], p2 = s[2], p3 = s[3], p4 = s[4], p5 = s[5];
        for (int x = 0; x < w; x++) {
            hmem[r][x] = (int16_t)qpel_6tap_vals(p0, p1, p2, p3, p4, p5);
            p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = p5;
            if (x + 1 < w)
                p5 = s[x + 6];
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x += 4) {
            const uint8_t o0 = qpel_clip_shift10(qpel_6tap_vals(r0[x + 0], r1[x + 0], r2[x + 0], r3[x + 0], r4[x + 0], r5[x + 0]));
            const uint8_t o1 = qpel_clip_shift10(qpel_6tap_vals(r0[x + 1], r1[x + 1], r2[x + 1], r3[x + 1], r4[x + 1], r5[x + 1]));
            const uint8_t o2 = qpel_clip_shift10(qpel_6tap_vals(r0[x + 2], r1[x + 2], r2[x + 2], r3[x + 2], r4[x + 2], r5[x + 2]));
            const uint8_t o3 = qpel_clip_shift10(qpel_6tap_vals(r0[x + 3], r1[x + 3], r2[x + 3], r3[x + 3], r4[x + 3], r5[x + 3]));
            const uint32_t pred = pack_u8x4(o0, o1, o2, o3);

            if (!do_avg) {
                AV_WN32(dst + x, pred);
            } else {
                const uint32_t d = AV_RN32(dst + x);
                AV_WN32(dst + x, rnd_avg32(d, pred));
            }
        }
        dst += stride;

        /* Slide vertical window and compute the next horizontal row (y+4). */
        if (y + 1 < h) {
            int16_t *tmp = r0;
            r0 = r1; r1 = r2; r2 = r3; r3 = r4; r4 = r5; r5 = tmp;

            const uint8_t *sn = src + (y + 4) * stride - 2;
            int p0 = sn[0], p1 = sn[1], p2 = sn[2], p3 = sn[3], p4 = sn[4], p5 = sn[5];
            for (int x = 0; x < w; x++) {
                r5[x] = (int16_t)qpel_6tap_vals(p0, p1, p2, p3, p4, p5);
                p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = p5;
                if (x + 1 < w)
                    p5 = sn[x + 6];
            }
        }
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
