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
 * VPR registers are 512 bits (64 bytes) wide.  The LA0/SA0 load/store
 * instructions transfer a full VPR to/from a 64-byte aligned memory address.
 * Two SA0 operations are therefore required to zero the 128-byte DCT
 * coefficient block used by MPEG-2/4 and H.264.
 *
 * Alignment note
 * --------------
 * DCT coefficient arrays may be declared DECLARE_ALIGNED(32, …) and are
 * therefore only guaranteed to be 32-byte aligned at runtime.  The MXUv3
 * SA0 instruction requires 64-byte alignment.
 *
 * When frame buffers are 64-byte aligned (HAVE_SIMD_ALIGN_64=1, enabled by
 * --enable-mxuv3 in configure) and coefficient arrays are embedded inside
 * those buffers, the 64-byte alignment condition is satisfied.  In cases
 * where the block pointer is only 32-byte aligned we fall back to memset so
 * that correctness is never sacrificed.
 *
 * TODO: MXUv3 instruction encodings
 * -----------------------------------
 * The inline-assembly paths below are prepared but intentionally left as
 * memset stubs until the VPR_ZERO / SA0_VPR instruction encodings have been
 * verified against actual Ingenic XBurst2 hardware.  The RE-derived
 * encodings in thingino-accel/include/mxuv3.h should be cross-checked
 * against an Ingenic SDK disassembly before enabling them here.
 *
 * Once verified, replace the memset calls with:
 *
 *   // Zero VPR0
 *   __asm__ volatile (".word 0x71000000" ::: "memory");  // VPR_ZERO(0)
 *   // Store VPR0 to block[0..63]
 *   __asm__ volatile (".word 0x78000000" :: "r"(block) : "memory"); // SA0_VPR(0, block)
 *   // Store VPR0 to block[64..127]  (block+32 int16 = +64 bytes)
 *   __asm__ volatile (".word 0x78000000" :: "r"((block) + 32) : "memory");
 *
 * Encodings above are PLACEHOLDERS — do not enable without verification.
 */

#include <string.h>
#include "libavutil/intreadwrite.h"
#include "blockdsp_mips.h"

/**
 * Zero a single 64-element int16_t DCT coefficient block (128 bytes).
 *
 * Falls back to memset when the block is not 64-byte aligned so that
 * the MXUv3 SA0 alignment requirement is always satisfied.
 */
void ff_clear_block_mxu(int16_t *block)
{
    /*
     * TODO: replace memset with MXUv3 VPR_ZERO + SA0 once instruction
     * encodings are confirmed on hardware (see file-level comment above).
     */
    memset(block, 0, 128);
}

/**
 * Zero six consecutive int16_t DCT coefficient blocks (6 × 128 = 768 bytes).
 *
 * In the MPEG-2/4 / H.264 macroblock path this is called once per MB to
 * clear all six luma+chroma coefficient arrays.
 */
void ff_clear_blocks_mxu(int16_t *block)
{
    /*
     * TODO: replace memset with a loop of 12 × (VPR_ZERO + SA0) pairs once
     * instruction encodings are confirmed on hardware (see file-level comment).
     */
    memset(block, 0, 6 * 128);
}

