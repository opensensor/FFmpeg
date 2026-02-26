/*
 * MXU Raw Opcode Intrinsics for FFmpeg
 *
 * Ingenic XBurst MXU (Media Extension Unit) instructions using raw .word
 * encodings.  Works with any MIPS assembler — no MXU mnemonic support needed.
 *
 * Instruction encodings verified against:
 *   - QEMU  target/mips/tcg/mxu_translate.c
 *   - Binutils  opcodes/mxu-opc.c
 *   - Ingenic kernel  arch/mips/xburst2/core/include/mxu_media.h
 *
 * Copyright (c) 2024-2026 OpenSensor Project
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef AVCODEC_MIPS_MXU_H
#define AVCODEC_MIPS_MXU_H

#include <stdint.h>
#include "libavutil/intreadwrite.h"

/* ---- XR register names (4-bit encoding, XR0-XR15) ---- */
#define xr0   0
#define xr1   1
#define xr2   2
#define xr3   3
#define xr4   4
#define xr5   5
#define xr6   6
#define xr7   7
#define xr8   8
#define xr9   9
#define xr10  10
#define xr11  11
#define xr12  12
#define xr13  13
#define xr14  14
#define xr15  15

/*
 * Map XR register number to the 5-bit 'sa' field used by S32I2M / S32M2I.
 * Hardware quirk: XR0 is encoded as 16 in the 5-bit field.
 */
#define _MXU_XR5(n) (((n) == 0) ? 16 : (n))

/* MIPS GPR number for the pinned temporary register ($t0 = $8). */
#define _MXU_GPR_T0 8

/* ------------------------------------------------------------------ */
/*  GPR ↔ XR transfers                                                */
/* ------------------------------------------------------------------ */

/*
 * S32I2M — Move 32-bit value from GPR to XR register.
 * Encoding: SPECIAL2 | (rt << 16) | (xr_hw << 6) | 0x2F
 */
#define S32I2M(xr, val) do {                                               \
    register uint32_t __mxu_v __asm__("t0") = (uint32_t)(val);            \
    __asm__ __volatile__(".word %1"                                        \
        :: "r"(__mxu_v),                                                   \
           "i"(0x7000002F | (_MXU_GPR_T0 << 16) | (_MXU_XR5(xr) << 6))); \
} while (0)

/*
 * S32M2I — Move 32-bit value from XR register to GPR.  Returns uint32_t.
 * Encoding: SPECIAL2 | (rt << 16) | (xr_hw << 6) | 0x2E
 */
#define S32M2I(xr) ({                                                      \
    register uint32_t __mxu_v __asm__("t0");                               \
    __asm__ __volatile__(".word %1"                                        \
        : "=r"(__mxu_v)                                                    \
        : "i"(0x7000002E | (_MXU_GPR_T0 << 16) | (_MXU_XR5(xr) << 6))); \
    __mxu_v; })

/* ------------------------------------------------------------------ */
/*  Load / Store (emulated via S32I2M/S32M2I + C memory access)       */
/* ------------------------------------------------------------------ */

/*
 * S32LDD — Load 32-bit word from (base + off) into XR register.
 * S32STD — Store 32-bit word from XR register to (base + off).
 *
 * Implemented as C memory access + GPR↔XR transfer so that no native
 * S32LDD/S32STD opcode encoding is required.  AV_RN32 / AV_WN32 handle
 * potentially-unaligned pointers safely.
 */
#define S32LDD(xr, base, off) \
    S32I2M(xr, AV_RN32((const uint8_t *)(base) + (off)))

#define S32STD(xr, base, off) \
    AV_WN32((uint8_t *)(base) + (off), (uint32_t)S32M2I(xr))

/* ------------------------------------------------------------------ */
/*  Q8ADD — Quad 8-bit add/subtract with saturation                   */
/* ------------------------------------------------------------------ */
/*
 * Encoding:  SPECIAL2 | (eptn2 << 24) | (7 << 18) |
 *            (XRc << 14) | (XRb << 10) | (XRa << 6) | 0x06
 *
 * EPTN2:  0 = AA  (add-add, clamp 0..255)
 *         1 = AS  (add-sub)
 *         2 = SA  (sub-add)
 *         3 = SS  (sub-sub, clamp 0..255)
 */
#define Q8ADD_AA(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x701C0006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

#define Q8ADD_AS(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x711C0006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

#define Q8ADD_SA(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x721C0006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

#define Q8ADD_SS(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x731C0006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* ------------------------------------------------------------------ */
/*  Q8AVG / Q8AVGR — Quad 8-bit unsigned average                     */
/* ------------------------------------------------------------------ */

/*
 * Q8AVG  — truncating:  byte_n = (XRb[n] + XRc[n]) >> 1
 * Q8AVGR — rounded:     byte_n = (XRb[n] + XRc[n] + 1) >> 1
 *
 * Encoding (POOL01, func6 = 0x06):
 *   Q8AVG:   bits 20:18 = 100  →  base 0x70100006
 *   Q8AVGR:  bits 20:18 = 101  →  base 0x70140006
 */
#define Q8AVG(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x70100006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

#define Q8AVGR(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x70140006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* ------------------------------------------------------------------ */
/*  Q8ABD — Quad 8-bit absolute difference                            */
/* ------------------------------------------------------------------ */
/*
 * byte_n = |XRb[n] - XRc[n]|
 *
 * Encoding (POOL02, func6 = 0x07):
 *   bits 20:18 = 100  →  base 0x70100007
 */
#define Q8ABD(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x70100007 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

#endif /* AVCODEC_MIPS_MXU_H */
