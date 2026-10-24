//go:build cuda && amd64

#include "textflag.h"

// The scale-broadcast indices: a vector of eight words at index 8k within a 32-word block takes the f16 groups 2k and 2k+1,
// four words each.
DATA moeIdx<>+0(SB)/4, $0
DATA moeIdx<>+4(SB)/4, $0
DATA moeIdx<>+8(SB)/4, $0
DATA moeIdx<>+12(SB)/4, $0
DATA moeIdx<>+16(SB)/4, $1
DATA moeIdx<>+20(SB)/4, $1
DATA moeIdx<>+24(SB)/4, $1
DATA moeIdx<>+28(SB)/4, $1
DATA moeIdx<>+32(SB)/4, $2
DATA moeIdx<>+36(SB)/4, $2
DATA moeIdx<>+40(SB)/4, $2
DATA moeIdx<>+44(SB)/4, $2
DATA moeIdx<>+48(SB)/4, $3
DATA moeIdx<>+52(SB)/4, $3
DATA moeIdx<>+56(SB)/4, $3
DATA moeIdx<>+60(SB)/4, $3
DATA moeIdx<>+64(SB)/4, $4
DATA moeIdx<>+68(SB)/4, $4
DATA moeIdx<>+72(SB)/4, $4
DATA moeIdx<>+76(SB)/4, $4
DATA moeIdx<>+80(SB)/4, $5
DATA moeIdx<>+84(SB)/4, $5
DATA moeIdx<>+88(SB)/4, $5
DATA moeIdx<>+92(SB)/4, $5
DATA moeIdx<>+96(SB)/4, $6
DATA moeIdx<>+100(SB)/4, $6
DATA moeIdx<>+104(SB)/4, $6
DATA moeIdx<>+108(SB)/4, $6
DATA moeIdx<>+112(SB)/4, $7
DATA moeIdx<>+116(SB)/4, $7
DATA moeIdx<>+120(SB)/4, $7
DATA moeIdx<>+124(SB)/4, $7
GLOBL moeIdx<>(SB), RODATA|NOPTR, $128

// P computes the integer products of eight packed words at byte offset off (from row base SI and the activation tables R8,
// R9, R10), converted to float, into Y6. Clobbers Y9.
#define P(off) \
	VMOVDQU off(SI)(BX*1), Y9; \
	VPAND Y15, Y9, Y6; \
	VPSRLD $4, Y9, Y9; \
	VPAND Y15, Y9, Y9; \
	VPMADDUBSW off(R8)(BX*1), Y6, Y6; \
	VPMADDUBSW off(R9)(BX*1), Y9, Y9; \
	VPADDW Y9, Y6, Y6; \
	VPMADDWD Y14, Y6, Y6; \
	VPSUBD off(R10)(BX*1), Y6, Y6; \
	VCVTDQ2PS Y6, Y6

// PAIR is one step of the full loop for the lanes in accumulator ACC, whose words sit at byte offsets OFF0 (p0) and OFF1
// (p1, 32 words on): T = round(S1*P1); T = fma(S0, P0, T); ACC = ACC + T.
#define PAIR(OFF1, OFF0, IDX, ACC) \
	P(OFF1); \
	VPERMPS Y5, IDX, Y7; \
	VMULPS Y6, Y7, Y7; \
	P(OFF0); \
	VPERMPS Y4, IDX, Y8; \
	VFMADD231PS Y8, Y6, Y7; \
	VADDPS Y7, ACC, ACC

// func w4a8RowsAVX2(w *uint32, s *uint16, aLo, aHi, sum8 *int32, kwords, rows int, out *float32)
//
// For each of rows rows (kwords words and kwords/4 f16 scales each, contiguous): gemv_w4a8_moe's per-lane sums and shuffle
// tree, before the activation scale, into out. aLo[w] and aHi[w] are the activation's int8 quads for word w's low and high
// nibbles, sum8[w] is 8 times the sum of those eight int8. kwords must be a multiple of 8.
TEXT ·w4a8RowsAVX2(SB), NOSPLIT, $0-64
	MOVQ w+0(FP), SI
	MOVQ s+8(FP), DI
	MOVQ aLo+16(FP), R8
	MOVQ aHi+24(FP), R9
	MOVQ sum8+32(FP), R10
	MOVQ kwords+40(FP), R11
	MOVQ rows+48(FP), R12
	MOVQ out+56(FP), R13

	MOVL $0x0F0F0F0F, AX
	VMOVD AX, X15
	VPBROADCASTD X15, Y15
	MOVL $0x00010001, AX
	VMOVD AX, X14
	VPBROADCASTD X14, Y14
	LEAQ moeIdx<>(SB), AX
	VMOVDQU 0(AX), Y10
	VMOVDQU 32(AX), Y11
	VMOVDQU 64(AX), Y12
	VMOVDQU 96(AX), Y13

	TESTQ R12, R12
	JEQ done

row:
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	XORQ BX, BX // byte offset of the current base word (4 bytes per word)
	XORQ CX, CX // the base word

full:
	LEAQ 64(CX), DX
	CMPQ DX, R11
	JGT tail
	// the f16 scales of the two 32-word blocks: groups base/4 .. +7 and +8 .. +15
	MOVQ CX, AX
	SHRQ $1, AX
	VCVTPH2PS 0(DI)(AX*1), Y4
	VCVTPH2PS 16(DI)(AX*1), Y5
	PAIR(128, 0, Y10, Y0)
	PAIR(160, 32, Y11, Y1)
	PAIR(192, 64, Y12, Y2)
	PAIR(224, 96, Y13, Y3)
	ADDQ $256, BX
	ADDQ $64, CX
	JMP full

tail:
	CMPQ CX, R11
	JGE reduce
	// up to four vectors of eight words, lanes 8k..8k+7 into accumulator k: ACC = fma(S, P, ACC)
	MOVQ CX, AX
	SHRQ $1, AX
	MOVL 0(DI)(AX*1), DX
	VMOVD DX, X4
	VCVTPH2PS X4, Y4
	VPERMPS Y4, Y10, Y8
	P(0)
	VFMADD231PS Y8, Y6, Y0
	LEAQ 16(CX), DX
	CMPQ DX, R11
	JGT tailnext
	MOVL 4(DI)(AX*1), DX
	VMOVD DX, X4
	VCVTPH2PS X4, Y4
	VPERMPS Y4, Y10, Y8
	P(32)
	VFMADD231PS Y8, Y6, Y1
	LEAQ 24(CX), DX
	CMPQ DX, R11
	JGT tailnext
	MOVL 8(DI)(AX*1), DX
	VMOVD DX, X4
	VCVTPH2PS X4, Y4
	VPERMPS Y4, Y10, Y8
	P(64)
	VFMADD231PS Y8, Y6, Y2
	LEAQ 32(CX), DX
	CMPQ DX, R11
	JGT tailnext
	MOVL 12(DI)(AX*1), DX
	VMOVD DX, X4
	VCVTPH2PS X4, Y4
	VPERMPS Y4, Y10, Y8
	P(96)
	VFMADD231PS Y8, Y6, Y3

tailnext:
	ADDQ $128, BX
	ADDQ $32, CX
	JMP tail

reduce:
	// the warp's shuffle tree: offsets 16, 8, 4, 2, 1, each lane l < off adding lane l+off
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VMOVHLPS X0, X0, X1
	VADDPS X1, X0, X0
	VMOVSHDUP X0, X1
	VADDSS X1, X0, X0
	VMOVSS X0, (R13)

	LEAQ (SI)(R11*4), SI // the next row's words
	MOVQ R11, AX
	SHRQ $1, AX
	ADDQ AX, DI // the next row's scales: kwords/4 halves
	ADDQ $4, R13
	DECQ R12
	JNE row

done:
	VZEROUPPER
	RET

// func cpuidex(leaf, sub uint32) (a, b, c, d uint32)
TEXT ·cpuidex(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL sub+4(FP), CX
	CPUID
	MOVL AX, a+8(FP)
	MOVL BX, b+12(FP)
	MOVL CX, c+16(FP)
	MOVL DX, d+20(FP)
	RET

// func xgetbv0() (lo uint32)
TEXT ·xgetbv0(SB), NOSPLIT, $0-4
	XORL CX, CX
	XGETBV
	MOVL AX, lo+0(FP)
	RET
