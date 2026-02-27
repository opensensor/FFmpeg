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
 * Ingenic XBurst2 MXUv3 optimised block DSP functions.
 *
 * MXUv3 background
 * ----------------
 * VPR registers are 512 bits (64 bytes) wide.  The SA0 store instruction
 * transfers one 256-bit half of a VPR to/from a 64-byte aligned memory
 * address, so two SA0 operations clear the full 64 bytes.
 *
 * Verified instruction encodings (hardware-tested on XBurst2 T31/T40):
 *   VPR_ZERO(0) = 0x4a80000b  — zeros VPR0 (COP2: VPR0 = VPR0 - VPR0)
 *   SA0 low  half (bytes  0-31 of VPR0): 0x710000d5  ($t0 = base)
 *   SA0 high half (bytes 32-63 of VPR0): 0x710102d5  ($t0 = base)
 *
 * Important: only offset values 0 (low half) and 1 (high half) work
 * reliably.  Larger offsets in the SA0 encoding do NOT produce correct
 * results on hardware — the base pointer ($t0) must be updated for each
 * 64-byte chunk.
 *
 * Alignment requirement
 * ---------------------
 * SA0 requires 64-byte aligned addresses.  When HAVE_SIMD_ALIGN_64=1
 * (enabled by --enable-mxuv3) FFmpeg allocates 64-byte aligned frame
 * buffers.  In cases where the block pointer is only 32-byte aligned we
 * fall back to memset so correctness is never sacrificed.
 */

#include <string.h>
#include "libavutil/intreadwrite.h"
#include "blockdsp_mips.h"
#include "mxu.h"

/**
 * Zero a single 64-element int16_t DCT coefficient block (128 bytes).
 *
 * Uses VPR_ZERO(0) once to zero VPR0, then two SA0 store pairs to write
 * the 128 bytes.  Falls back to memset when the block is not 64-byte
 * aligned so that the SA0 alignment requirement is always satisfied.
 */
void ff_clear_block_mxu(int16_t *block)
{
    if (__builtin_expect(((uintptr_t)block & 63) != 0, 0)) {
        memset(block, 0, 128);
        return;
    }

    VPR_ZERO_INIT();

    SA0_VPR0_AT(block);              /* bytes   0-63  */
    SA0_VPR0_AT((int8_t *)block + 64); /* bytes  64-127 */
}

/**
 * Zero six consecutive int16_t DCT coefficient blocks (6 × 128 = 768 bytes).
 *
 * VPR_ZERO(0) is executed once, then 12 SA0 store pairs write the full
 * 768 bytes.  Called once per macroblock in the H.264/MPEG-2/4 decode path.
 * Falls back to memset when the block pointer is not 64-byte aligned.
 */
void ff_clear_blocks_mxu(int16_t *block)
{
    if (__builtin_expect(((uintptr_t)block & 63) != 0, 0)) {
        memset(block, 0, 6 * 128);
        return;
    }

    VPR_ZERO_INIT();

    int8_t *b = (int8_t *)block;

    /* 12 × 64-byte SA0 store pairs → 768 bytes total */
    SA0_VPR0_AT(b +   0);  SA0_VPR0_AT(b +  64);
    SA0_VPR0_AT(b + 128);  SA0_VPR0_AT(b + 192);
    SA0_VPR0_AT(b + 256);  SA0_VPR0_AT(b + 320);
    SA0_VPR0_AT(b + 384);  SA0_VPR0_AT(b + 448);
    SA0_VPR0_AT(b + 512);  SA0_VPR0_AT(b + 576);
    SA0_VPR0_AT(b + 640);  SA0_VPR0_AT(b + 704);
}

