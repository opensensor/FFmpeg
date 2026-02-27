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

/* ------------------------------------------------------------------ */
/*  Software prefetch (MIPS PREF instruction)                          */
/* ------------------------------------------------------------------ */

/*
 * PREF hint=0 → prefetch for load
 * PREF hint=1 → prefetch for store (prepare cache line for writing)
 *
 * On the in-order XBurst2 core, issuing PREF one or two loop iterations
 * ahead hides the memory-access latency that would otherwise stall the
 * pipeline.  The cost of a PREF to an already-cached line is negligible.
 */
#define PREF_LOAD(base, off) \
    __asm__ __volatile__("pref 0, %0" :: "m"(*((const uint8_t *)(base) + (off))))

#define PREF_STORE(base, off) \
    __asm__ __volatile__("pref 1, %0" :: "m"(*((uint8_t *)(base) + (off))))

/* ------------------------------------------------------------------ */
/*  SA0 VPR0 zero store (512-bit / 64 bytes per pair)                  */
/* ------------------------------------------------------------------ */

/*
 * SA0_VPR0_AT(ptr) — store all 64 bytes of VPR0 (which must already
 * be zeroed) to the given 64-byte-aligned address.  Two SA0 ops write
 * the low and high 256-bit halves.
 *
 * Shared between blockdsp_mxu.c and h264dsp_mxu.c so that the
 * zero-store pattern is consistent everywhere.
 */
#define SA0_VPR0_AT(ptr) do {                                   \
    register void *_base __asm__("t0") = (void *)(ptr);         \
    __asm__ __volatile__(                                        \
        ".set push\n\t"                                          \
        ".set noreorder\n\t"                                     \
        ".word 0x710000d5\n\t"  /* SA0 VPR0 low  -> t0+0  */   \
        ".word 0x710102d5\n\t"  /* SA0 VPR0 high -> t0+32 */   \
        ".set pop\n\t"                                           \
        :: "r"(_base) : "memory"                                 \
    );                                                           \
} while (0)

/*
 * VPR_ZERO_BLOCK_32(ptr) — zero 32 bytes (4x4 int16_t block) using VPR0.
 * Falls back to memset if not 64-byte aligned.  For 32-byte blocks that
 * are 64-byte aligned, only the low half of SA0 is needed.
 */

/*
 * VPR_ZERO_INIT() — zero VPR0 once; call before SA0 stores.
 */
#define VPR_ZERO_INIT() \
    __asm__ __volatile__(".word 0x4a80000b\n\tsync\n" ::: "memory")

/* ------------------------------------------------------------------ */
/*  CU2 (Coprocessor 2) lazy enablement for XBurst2 MXUv3             */
/* ------------------------------------------------------------------ */

/*
 * ff_mxu_ensure_cu2() — trigger the kernel's lazy CU2 enablement.
 *
 * On Ingenic XBurst2 (T31/T40/T41) the kernel enables CU2 lazily on
 * the first coprocessor exception.  However the do_ri handler (which
 * catches SPECIAL2-encoded MXU instructions like S32I2M, Q8AVG, LA0,
 * SA0) has its CU2 enablement code disabled (#if 0 in traps.c).
 * Only the do_cpu handler (exc_code=11, CpU) works, and it is only
 * reached by COP2-encoded instructions.
 *
 * VPR_ZERO (0x4a80000b) is COP2-encoded (bits 31:26 = 010010).
 * Executing it as the very first MXU instruction triggers exc_code=11
 * with CE=2, the kernel enables CU2, and all subsequent SPECIAL2
 * instructions (XR ops, LA0, SA0) work without faulting.
 *
 * Call this once from each *_init_mips() function before registering
 * any MXU function pointers.  After the first call CU2 stays enabled
 * for the lifetime of the process, so the cost is one trapped
 * instruction total.
 */
static inline void ff_mxu_ensure_cu2(void)
{
    /* VPR0 = VPR0 - VPR0  (COP2 opcode, zeroes VPR0 as side-effect) */
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");
}

#endif /* AVCODEC_MIPS_MXU_H */
