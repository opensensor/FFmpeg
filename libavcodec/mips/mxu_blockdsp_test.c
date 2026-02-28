/*
 * MXUv3 block-DSP hardware verification test.
 *
 * Cross-compile with:
 *   mipsel-linux-gcc -O2 -march=mips32r2 -mips32r2 -o mxu_blockdsp_test \
 *       FFmpeg/libavcodec/mips/mxu_blockdsp_test.c
 * Run on device:
 *   scp mxu_blockdsp_test root@<device>:/tmp/
 *   ssh root@<device> /tmp/mxu_blockdsp_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * Instruction encodings (confirmed from thingino-accel/include/mxuv3.h RE):
 *
 * VPR_ZERO(0) = MXUV3_COP2_INST(20,0,0,0,11) = 0x4a80000b
 *   VPR0 = VPR0 - VPR0 = 0, followed by sync for ordering.
 *
 * SA0_VPR encoding: 0x710000d5 | (offset<<16) | (vprn<<11) | (n<<9)
 *   offset = byte_offset / 32
 *   n = 0 → low 256-bit half (bytes 0-31 of VPR), n = 1 → high (bytes 32-63)
 *   Base address always from $t0.
 */

static int tests_run   = 0;
static int tests_passed = 0;

static void check(const char *name, int ok)
{
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("PASS  %s\n", name);
    } else {
        printf("FAIL  %s\n", name);
    }
}

/* --- test 1: clear 64 bytes with explicit base-pointer per store pair --- */
static void test_clear_64_base_update(void)
{
    void *buf;
    if (posix_memalign(&buf, 64, 128)) { printf("SKIP  alloc failed\n"); return; }
    memset(buf, 0xAB, 128);

    register void *_base __asm__("t0");

    __asm__ __volatile__(".word 0x4a80000b\n\tsync\n" ::: "memory");

    _base = buf;
    __asm__ __volatile__(
        ".word 0x710000d5\n\t"   /* SA0 VPR0 low  → base+0  */
        ".word 0x710102d5\n\t"   /* SA0 VPR0 high → base+32 */
        :: "r"(_base) : "memory"
    );

    int ok = (memcmp(buf, "\0\0\0\0", 4) == 0) &&
             (((uint8_t *)buf)[63] == 0) &&
             (((uint8_t *)buf)[64] == 0xAB);  /* only first 64 bytes cleared */
    check("clear_64_base_update (low 64 bytes are zero)", ok);
    free(buf);
}

/* --- test 2: clear 128 bytes using offset encoding, single base pointer --- */
static void test_clear_128_offset_encoding(void)
{
    void *buf;
    if (posix_memalign(&buf, 64, 128)) { printf("SKIP  alloc failed\n"); return; }
    memset(buf, 0xAB, 128);

    register void *_base __asm__("t0") = buf;

    __asm__ __volatile__(".word 0x4a80000b\n\tsync\n" ::: "memory");

    /* Store VPR0 to base+0..63 (offset=0 low, offset=1 high) */
    /* Store VPR0 to base+64..127 (offset=2 low, offset=3 high) */
    __asm__ __volatile__(
        ".word 0x710000d5\n\t"   /* off=0 n=0: base+0  */
        ".word 0x710102d5\n\t"   /* off=1 n=1: base+32 */
        ".word 0x710200d5\n\t"   /* off=2 n=0: base+64 */
        ".word 0x710302d5\n\t"   /* off=3 n=1: base+96 */
        :: "r"(_base) : "memory"
    );

    int all_zero = 1;
    for (int i = 0; i < 128; i++) {
        if (((uint8_t *)buf)[i] != 0) { all_zero = 0; break; }
    }
    check("clear_128_offset_encoding (all 128 bytes zero)", all_zero);
    free(buf);
}

/* --- test 3: clear 768 bytes (6 DCT blocks × 128 B) using offset encoding --- */
static void test_clear_768_offset_encoding(void)
{
    void *buf;
    if (posix_memalign(&buf, 64, 768)) { printf("SKIP  alloc failed\n"); return; }
    memset(buf, 0xCD, 768);

    register void *_base __asm__("t0") = buf;

    __asm__ __volatile__(".word 0x4a80000b\n\tsync\n" ::: "memory");

    /* 12 × 64-byte stores (offsets 0..23) */
    __asm__ __volatile__(
        ".word 0x710000d5\n\t" ".word 0x710102d5\n\t"   /* block 0: base+0   */
        ".word 0x710200d5\n\t" ".word 0x710302d5\n\t"   /* block 1: base+64  */
        ".word 0x710400d5\n\t" ".word 0x710502d5\n\t"   /* block 2: base+128 */
        ".word 0x710600d5\n\t" ".word 0x710702d5\n\t"   /* block 3: base+192 */
        ".word 0x710800d5\n\t" ".word 0x710902d5\n\t"   /* block 4: base+256 */
        ".word 0x710a00d5\n\t" ".word 0x710b02d5\n\t"   /* block 5: base+320 */
        ".word 0x710c00d5\n\t" ".word 0x710d02d5\n\t"   /* block 6: base+384 */
        ".word 0x710e00d5\n\t" ".word 0x710f02d5\n\t"   /* block 7: base+448 */
        ".word 0x711000d5\n\t" ".word 0x711102d5\n\t"   /* block 8: base+512 */
        ".word 0x711200d5\n\t" ".word 0x711302d5\n\t"   /* block 9: base+576 */
        ".word 0x711400d5\n\t" ".word 0x711502d5\n\t"   /* block10: base+640 */
        ".word 0x711600d5\n\t" ".word 0x711702d5\n\t"   /* block11: base+704 */
        :: "r"(_base) : "memory"
    );

    int all_zero = 1;
    for (int i = 0; i < 768; i++) {
        if (((uint8_t *)buf)[i] != 0) { all_zero = 0; break; }
    }
    check("clear_768_offset_encoding (all 768 bytes zero)", all_zero);
    free(buf);
}

int main(void)
{
    printf("=== MXUv3 block-DSP verification ===\n");
    test_clear_64_base_update();
    test_clear_128_offset_encoding();
    test_clear_768_offset_encoding();
    printf("=== %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

