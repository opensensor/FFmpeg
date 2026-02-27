/*
 * Ingenic XBurst2 (MXU) H.264 QPEL helpers.
 *
 * This is intentionally a minimal, low-risk hook: implement only mc00
 * (integer-pel copy / avg) by delegating to the already-optimised MXU
 * put/avg pixels primitives from hpeldsp_mxu.c.
 */

#include "h264dsp_mips.h"
#include "hpeldsp_mips.h"

void ff_put_h264_qpel16_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    ff_put_pixels16_mxu(dst, src, dst_stride, 16);
}

void ff_put_h264_qpel8_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    ff_put_pixels8_mxu(dst, src, dst_stride, 8);
}

void ff_put_h264_qpel4_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    ff_put_pixels4_mxu(dst, src, dst_stride, 4);
}

void ff_avg_h264_qpel16_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                 ptrdiff_t dst_stride)
{
    ff_avg_pixels16_mxu(dst, src, dst_stride, 16);
}

void ff_avg_h264_qpel8_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    ff_avg_pixels8_mxu(dst, src, dst_stride, 8);
}

void ff_avg_h264_qpel4_mc00_mxu(uint8_t *dst, const uint8_t *src,
                                ptrdiff_t dst_stride)
{
    ff_avg_pixels4_mxu(dst, src, dst_stride, 4);
}
