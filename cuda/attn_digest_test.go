//go:build cuda

package cuda

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// The drivers of G1-K (see attn_digest_rig_test.go). Each case is one kernel at one shape, on fixed inputs, with the launch geometry the
// resident uses; its output buffers are hashed. A kernel that serves only some head dimensions is driven at those only.

var (
	adGlue     = [2]adMod{{"glue", gluePTX}, {"glue_kv16", glueKV16PTX}}
	adPrefill  = [2]adMod{{"prefill_batched", prefillBatchedPTX}, {"prefill_batched_kv16", prefillBatchedKV16PTX}}
	adSplit    = [2]adMod{{"decode_splitkv", decodeSplitKVPTX}, {"decode_splitkv_kv16", decodeSplitKVKV16PTX}}
	adSplitRow = [2]adMod{{"prefill_splitkv", prefillSplitKVPTX}, {"prefill_splitkv_kv16", prefillSplitKVKV16PTX}}
	adFA       = [2]adMod{{"decode_fa", decodeFAPTX}, {"decode_fa_kv16", decodeFAKV16PTX}}
	adFused    = [2]adMod{{"attn_fused", attnFusedPTX}, {"attn_fused_kv16", attnFusedKV16PTX}}
	adFusedBM  = [2]adMod{{"attn_fused_bm", attnFusedBMPTX}, {"attn_fused_bm_kv16", attnFusedBMKV16PTX}}
	adBlock    = adMod{"attn_block", attnBlockPTX}
	adImg      = adMod{"attn_img_prefill", attnImgPrefillPTX}
	adMLA      = adMod{"mla", mlaPTX}
	adKVNames  = [2]string{"f32", "f16"}
	adHeadDims = []struct{ hd, nH, nKV int }{{64, 4, 2}, {128, 8, 2}, {256, 8, 4}, {512, 4, 1}}
)

func adCfg(gx, gy, gz, bx int, shared int) gc.LaunchConfig {
	return gc.LaunchConfig{GridX: uint32(gx), GridY: uint32(gy), GridZ: uint32(gz), BlockX: uint32(bx), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(shared)}
}

func adI(v int) gc.KernelArg { return gc.ArgValue(int32(v)) }

// adSink is the sink argument: nothing, or one learned value per head.
func adSink(r *adRig, nH int, on bool) gc.KernelArg {
	if !on {
		return gc.ArgDevicePtr(0)
	}
	return gc.Arg(r.f32(adRand(nH, 77, 2)))
}

func adWindowText(w int) string {
	if w == 0 {
		return "dense"
	}
	return fmt.Sprintf("win%d", w)
}

func adSinkText(on bool) string {
	if on {
		return "sink"
	}
	return "nosink"
}

// attention (glue.cu): decode, one query, nKeys keys either side of the 128-thread tile.
func adCasesAttention() []adCase {
	var cs []adCase
	for kvi, m := range adGlue {
		for _, g := range adHeadDims {
			for _, nKeys := range []int{1, 2, 127, 128, 129, 300} {
				for _, window := range []int{0, 64} {
					f16, nH, nKV, hd := kvi == 1, g.nH, g.nKV, g.hd
					cs = append(cs, adCase{"attention", fmt.Sprintf("attention/%s/hd%d/k%d/%s", adKVNames[kvi], hd, nKeys, adWindowText(window)),
						func(r *adRig) [][]float32 {
							kvDim := nKV * hd
							q := r.f32(adRand(nH*hd, 11, 1))
							kc, vc := r.kv(adRand(nKeys*kvDim, 12, 1), f16), r.kv(adRand(nKeys*kvDim, 13, 1), f16)
							ctx := r.zeros(nH * hd)
							nWin := nKeys
							if window > 0 && nKeys > window {
								nWin = window
							}
							r.launch(m, "attention", adCfg(nH, 1, 1, 128, (nWin+128)*4), gc.Arg(q), kc, vc, r.kvt(nKeys),
								adI(nH), adI(nKV), adI(hd), adI(nKeys), gc.ArgValue(adScale(hd)), adI(window), gc.Arg(ctx))
							return [][]float32{r.read(ctx, nH*hd)}
						}})
				}
			}
		}
	}
	return cs
}

// adRows is the (startPos, M) set the batched-prefill kernels are driven at: one row, a tile, a tile and a bit, both sides of 64 and 128 keys.
var adRows = [][2]int{{0, 1}, {0, 17}, {63, 2}, {64, 2}, {127, 2}, {130, 70}, {300, 5}, {200, 200}}

// attn_batched (prefill_batched.cu): the exact prefill kernel, which decode also takes at M = 1.
func adCasesBatched() []adCase {
	var cs []adCase
	for kvi, m := range adPrefill {
		for _, g := range adHeadDims {
			for _, sm := range adRows {
				for _, window := range []int{0, 50} {
					for _, sink := range []bool{false, true} {
						f16, nH, nKV, hd, startPos, M := kvi == 1, g.nH, g.nKV, g.hd, sm[0], sm[1]
						cs = append(cs, adCase{"attn_batched", fmt.Sprintf("attn_batched/%s/hd%d/s%d_m%d/%s/%s", adKVNames[kvi], hd, startPos, M, adWindowText(window), adSinkText(sink)),
							func(r *adRig) [][]float32 {
								nTot, kvDim := startPos+M, nKV*hd
								q := r.f32(adRand(M*nH*hd, 21, 1))
								kc, vc := r.kv(adRand(nTot*kvDim, 22, 1), f16), r.kv(adRand(nTot*kvDim, 23, 1), f16)
								ctx := r.zeros(M * nH * hd)
								maxWin := nTot
								if window > 0 && window < maxWin {
									maxWin = window
								}
								r.launch(m, "attn_batched", adCfg(nH, M, 1, 128, (maxWin+128)*4), gc.Arg(q), kc, vc, r.kvt(nTot),
									adI(nH), adI(nKV), adI(hd), adI(startPos), gc.ArgValue(adScale(hd)), adI(window), adI(M), gc.Arg(ctx), adSink(r, nH, sink))
								return [][]float32{r.read(ctx, M*nH*hd)}
							}})
					}
				}
			}
		}
	}
	return cs
}

// The split-KV chain (decode_splitkv.cu): scores, softmax in place, the V-sum, and the opt-in split V-sum with its combine. A case
// hashes the stored (scaled) scores, the inverse denominators and both contexts.
func adCasesSplitKV() []adCase {
	var cs []adCase
	for kvi, m := range adSplit {
		for _, g := range adHeadDims {
			for _, nKeys := range []int{1, 127, 128, 129, 257, 500} {
				for _, window := range []int{0, 100} {
					for _, sink := range []bool{false, true} {
						f16, nH, nKV, hd := kvi == 1, g.nH, g.nKV, g.hd
						id := fmt.Sprintf("%s/hd%d/k%d/%s/%s", adKVNames[kvi], hd, nKeys, adWindowText(window), adSinkText(sink))
						run := func(r *adRig) [][]float32 {
							kvDim := nKV * hd
							winStart := 0
							if window > 0 && nKeys > window {
								winStart = nKeys - window
							}
							nWin := nKeys - winStart
							q := r.f32(adRand(nH*hd, 31, 1))
							kc, vc := r.kv(adRand(nKeys*kvDim, 32, 1), f16), r.kv(adRand(nKeys*kvDim, 33, 1), f16)
							sc, inv, ctx := r.zeros(nH*nWin), r.zeros(nH), r.zeros(nH*hd)
							r.launch(m, "splitkv_scores", adCfg(nH, (nWin+127)/128, 1, 128, hd*4), gc.Arg(q), kc, r.kvt(nKeys), adI(nH), adI(nKV), adI(hd),
								adI(winStart), adI(nKeys), gc.ArgValue(adScale(hd)), gc.Arg(sc), adI(nWin))
							raw := r.read(sc, nH*nWin)
							r.launch(m, "splitkv_softmax", adCfg(nH, 1, 1, 128, 128*4), gc.Arg(sc), adI(nH), adI(nWin), gc.Arg(inv), adSink(r, nH, sink))
							r.launch(m, "splitkv_vsum", adCfg(nH, (hd+31)/32, 1, 32, 0), gc.Arg(sc), vc, r.kvt(nKeys), gc.Arg(inv), adI(nH), adI(nKV), adI(hd),
								adI(winStart), adI(nKeys), adI(nWin), gc.Arg(ctx))
							const nSplit = 4
							part, ctx2 := r.zeros(nH*hd*nSplit), r.zeros(nH*hd)
							r.launch(m, "splitkv_vsum_partial", adCfg(nH, (hd+31)/32, nSplit, 32, 0), gc.Arg(sc), vc, r.kvt(nKeys), adI(nH), adI(nKV), adI(hd),
								adI(winStart), adI(nKeys), adI(nWin), adI(nSplit), gc.Arg(part))
							r.launch(m, "splitkv_vsum_combine", adCfg(nH, (hd+31)/32, 1, 32, 0), gc.Arg(part), gc.Arg(inv), adI(nH), adI(hd), adI(nSplit), gc.Arg(ctx2))
							return [][]float32{raw, r.read(sc, nH*nWin), r.read(inv, nH), r.read(ctx, nH*hd), r.read(ctx2, nH*hd)}
						}
						cs = append(cs, adCase{"splitkv_scores", "splitkv_scores/" + id, run})
					}
				}
			}
		}
	}
	return cs
}

// The row-batched exact attention (prefill_splitkv.cu), the batched prefill's attention past attn_batched's shared-memory bound: scores for every row of a
// launch group, the softmax in place, the V-sum, with the image-block table of attn_img_batched's contract (none, or one block). A case hashes the raw
// scores, the stored softmaxed scores, the inverse denominators and the context.
func adCasesSplitKVRows() []adCase {
	var cs []adCase
	for kvi, m := range adSplitRow {
		for _, g := range adHeadDims {
			for _, sm := range [][2]int{{0, 40}, {100, 60}} {
				for _, window := range []int{0, 50} {
					for _, nBlocks := range []int{0, 1} {
						for _, sink := range []bool{false, true} {
							f16, nH, nKV, hd, startPos, M := kvi == 1, g.nH, g.nKV, g.hd, sm[0], sm[1]
							id := fmt.Sprintf("%s/hd%d/s%d_m%d/%s/blocks%d/%s", adKVNames[kvi], hd, startPos, M, adWindowText(window), nBlocks, adSinkText(sink))
							cs = append(cs, adCase{"splitkv_scores_rows", "splitkv_scores_rows/" + id, func(r *adRig) [][]float32 {
								nTot, kvDim := startPos+M, nKV*hd
								blocks := [][2]int{}
								blkArg, nb := gc.KernelArg(gc.ArgDevicePtr(0)), 0
								if nBlocks == 1 {
									blocks = [][2]int{{startPos + 5, startPos + 25}}
									blkArg, nb = gc.Arg(r.i32([]int32{int32(startPos + 5), int32(startPos + 25)})), 1
								}
								stride := nTot
								if window > 0 && window < stride {
									stride = window
								}
								stride = imgBlocksMaxNWin(stride, window, blocks)
								q := r.f32(adRand(M*nH*hd, 71, 1))
								kc, vc := r.kv(adRand(nTot*kvDim, 72, 1), f16), r.kv(adRand(nTot*kvDim, 73, 1), f16)
								sc, inv, ctx := r.zeros(M*nH*stride), r.zeros(M*nH), r.zeros(M*nH*hd)
								r.launch(m, "splitkv_scores_rows", adCfg(nH, (stride+127)/128, M, 128, hd*4), gc.Arg(q), kc, r.kvt(nTot), adI(nH), adI(nKV), adI(hd),
									adI(startPos), adI(window), adI(0), gc.ArgValue(adScale(hd)), gc.Arg(sc), adI(stride), blkArg, adI(nb))
								raw := r.read(sc, M*nH*stride)
								r.launch(m, "splitkv_softmax_rows", adCfg(nH, M, 1, 128, 128*4), gc.Arg(sc), adI(nH), adI(startPos), adI(window), adI(0),
									adI(stride), gc.Arg(inv), adSink(r, nH, sink), blkArg, adI(nb))
								r.launch(m, "splitkv_vsum_rows", adCfg(nH, (hd+31)/32, M, 32, 0), gc.Arg(sc), vc, r.kvt(nTot), gc.Arg(inv), adI(nH), adI(nKV), adI(hd),
									adI(startPos), adI(window), adI(0), adI(stride), gc.Arg(ctx), blkArg, adI(nb))
								return [][]float32{raw, r.read(sc, M*nH*stride), r.read(inv, M*nH), r.read(ctx, M*nH*hd)}
							}})
						}
					}
				}
			}
		}
	}
	return cs
}

// attn_block_full (attn_block.cu): the drafter's non-causal block attention. attn_img_batched (attn_img_prefill.cu): Gemma 3's image blocks.
// attn_span_batched (the same file): Gemma 4's spans, whose window reaches back to the span start; window 4 is shorter than every span.
// All three read an f32 cache only.
func adCasesBlockAndImage() []adCase {
	var cs []adCase
	for _, g := range adHeadDims {
		for _, sm := range [][2]int{{0, 16}, {100, 28}, {300, 12}} {
			for _, window := range []int{0, 64} {
				nH, nKV, hd, startPos, M := g.nH, g.nKV, g.hd, sm[0], sm[1]
				cs = append(cs, adCase{"attn_block_full", fmt.Sprintf("attn_block_full/f32/hd%d/s%d_m%d/%s", hd, startPos, M, adWindowText(window)),
					func(r *adRig) [][]float32 {
						nTot, kvDim := startPos+M, nKV*hd
						q := r.f32(adRand(M*nH*hd, 41, 1))
						kc, vc := r.f32(adRand(nTot*kvDim, 42, 1)), r.f32(adRand(nTot*kvDim, 43, 1))
						ctx := r.zeros(M * nH * hd)
						r.launch(adBlock, "attn_block_full", adCfg(nH, M, 1, 128, (nTot+128)*4), gc.Arg(q), gc.Arg(kc), gc.Arg(vc),
							adI(nH), adI(nKV), adI(hd), adI(startPos), gc.ArgValue(adScale(hd)), adI(window), adI(M), gc.Arg(ctx))
						return [][]float32{r.read(ctx, M*nH*hd)}
					}})
			}
		}
		for _, sm := range [][2]int{{0, 16}, {100, 40}} {
			for _, window := range []int{0, 64} {
				for _, nBlocks := range []int{0, 1, 2} {
					nH, nKV, hd, startPos, M := g.nH, g.nKV, g.hd, sm[0], sm[1]
					cs = append(cs, adCase{"attn_img_batched", fmt.Sprintf("attn_img_batched/f32/hd%d/s%d_m%d/%s/blocks%d", hd, startPos, M, adWindowText(window), nBlocks),
						func(r *adRig) [][]float32 {
							nTot, kvDim := startPos+M, nKV*hd
							q := r.f32(adRand(M*nH*hd, 51, 1))
							kc, vc := r.f32(adRand(nTot*kvDim, 52, 1)), r.f32(adRand(nTot*kvDim, 53, 1))
							ctx := r.zeros(M * nH * hd)
							// block b covers [startPos+3+6b, startPos+3+6b+5): rows inside it see the whole block
							flat := []int32{int32(startPos + 3), int32(startPos + 8), int32(startPos + 9), int32(startPos + 14)}
							r.launch(adImg, "attn_img_batched", adCfg(nH, M, 1, 128, (nTot+128)*4), gc.Arg(q), gc.Arg(kc), gc.Arg(vc), r.kvt(nTot),
								adI(nH), adI(nKV), adI(hd), adI(startPos), gc.ArgValue(adScale(hd)), adI(window), adI(M), gc.Arg(ctx),
								gc.ArgDevicePtr(0), gc.Arg(r.i32(flat)), adI(nBlocks))
							return [][]float32{r.read(ctx, M*nH*hd)}
						}})
				}
			}
			for _, window := range []int{0, 4, 64} {
				for _, nBlocks := range []int{0, 1, 2} {
					nH, nKV, hd, startPos, M := g.nH, g.nKV, g.hd, sm[0], sm[1]
					cs = append(cs, adCase{"attn_span_batched", fmt.Sprintf("attn_span_batched/f32/hd%d/s%d_m%d/%s/blocks%d", hd, startPos, M, adWindowText(window), nBlocks),
						func(r *adRig) [][]float32 {
							nTot, kvDim := startPos+M, nKV*hd
							q := r.f32(adRand(M*nH*hd, 54, 1))
							kc, vc := r.f32(adRand(nTot*kvDim, 55, 1)), r.f32(adRand(nTot*kvDim, 56, 1))
							ctx := r.zeros(M * nH * hd)
							// span b covers [startPos+2+7b, startPos+8+7b): six rows, longer than window 4, inside the shortest prompt (16 rows)
							flat := []int32{int32(startPos + 2), int32(startPos + 8), int32(startPos + 9), int32(startPos + 15)}
							r.launch(adImg, "attn_span_batched", adCfg(nH, M, 1, 128, (nTot+128)*4), gc.Arg(q), gc.Arg(kc), gc.Arg(vc), r.kvt(nTot),
								adI(nH), adI(nKV), adI(hd), adI(startPos), gc.ArgValue(adScale(hd)), adI(window), adI(M), gc.Arg(ctx),
								gc.ArgDevicePtr(0), gc.Arg(r.i32(flat)), adI(nBlocks))
							return [][]float32{r.read(ctx, M*nH*hd)}
						}})
				}
			}
		}
	}
	return cs
}

// fa_partial_{64,128,256} and fa_combine (decode_fa.cu), the flash-decode lane, and its multi-row variants for speculative verify.
func adCasesFlash() []adCase {
	var cs []adCase
	for kvi, m := range adFA {
		for _, hd := range []int{64, 128, 256} {
			for _, G := range []int{1, 2, 3, 4, 5, 6, 7, 8} {
				for _, nSplit := range []int{4, 16} {
					for _, kw := range [][2]int{{70, 0}, {1000, 0}, {2100, 500}} {
						f16, nKV, nKeys, winStart := kvi == 1, 2, kw[0], kw[1]
						nH := nKV * G
						cs = append(cs, adCase{"fa_partial_" + itoa(hd), fmt.Sprintf("fa_partial_%d/%s/g%d/s%d/k%d_w%d", hd, adKVNames[kvi], G, nSplit, nKeys, winStart),
							func(r *adRig) [][]float32 {
								kvDim := nKV * hd
								q := r.f32(adRand(nH*hd, 61, 1))
								kc, vc := r.kv(adRand(nKeys*kvDim, 62, 1), f16), r.kv(adRand(nKeys*kvDim, 63, 1), f16)
								U := nSplit * faWarps
								part, ctx := r.zeros(nH*U*(hd+4)), r.zeros(nH*hd)
								r.launch(m, "fa_partial_"+itoa(hd), adCfg(nKV, nSplit, 1, faWarps*32, (G*hd+faWarps*64)*4), gc.Arg(q), kc, vc, r.kvt(nKeys),
									adI(nH), adI(nKV), adI(winStart), adI(nKeys), gc.ArgValue(adScale(hd)), adI(nSplit), gc.Arg(part))
								p := r.read(part, nH*U*(hd+4))
								r.launch(m, "fa_combine", adCfg(nH, 1, 1, hd, 3*U*4), gc.Arg(part), adI(hd), adI(U), gc.Arg(ctx))
								return [][]float32{p, r.read(ctx, nH*hd)}
							}})
						if nSplit != 16 {
							continue
						}
						// the multi-row variant: a verify batch of nRows rows at consecutive positions, in runs that share a chunk partition
						for _, nRows := range []int{1, 5, 16} {
							window := 0
							if winStart > 0 {
								window = 1500
							}
							n0 := nKeys
							cs = append(cs, adCase{fmt.Sprintf("fa_partial_rows_%d_g%d", hd, G), fmt.Sprintf("fa_partial_rows_%d_g%d/%s/r%d/k%d_w%d", hd, G, adKVNames[kvi], nRows, n0, window),
								func(r *adRig) [][]float32 {
									kvDim := nKV * hd
									nTot := n0 + nRows
									q := r.f32(adRand(nRows*nH*hd, 71, 1))
									kc, vc := r.kv(adRand(nTot*kvDim, 72, 1), f16), r.kv(adRand(nTot*kvDim, 73, 1), f16)
									U := nSplit * faWarps
									part, ctx := r.zeros(faMaxRows*nH*U*(hd+4)), r.zeros(nRows*nH*hd)
									R := faRowsPerCTA(hd, G)
									for _, run := range faRowRuns(n0, nRows, window, nSplit) {
										r.launch(m, fmt.Sprintf("fa_partial_rows_%d_g%d", hd, G), adCfg(nKV, nSplit, (run.n+R-1)/R, faWarps*32, (R*G*hd+faWarps*R*64)*4),
											gc.Arg(q), kc, vc, r.kvt(nTot), adI(nH), adI(nKV), adI(run.winStart), adI(run.nKeys0), adI(run.per), gc.ArgValue(adScale(hd)),
											adI(nSplit), adI(run.n), adI(run.row0), gc.Arg(part))
										r.launch(m, "fa_combine_rows", adCfg(nH, run.n, 1, hd, 3*U*4), gc.Arg(part), adI(nH), adI(hd), adI(U), adI(run.row0), gc.Arg(ctx))
									}
									return [][]float32{r.read(ctx, nRows*nH*hd)}
								}})
						}
					}
				}
			}
		}
	}
	return cs
}

// attn_fused_hd{64,128} and the six attn_fused_bm* tile variants: the tensor-core prefill kernels, head dimensions 64 and 128 only.
func adCasesFused() []adCase {
	type k struct {
		name   string
		mods   [2]adMod
		hd     int
		bm, bn int
	}
	var ks []k
	for _, hd := range []int{64, 128} {
		ks = append(ks, k{fmt.Sprintf("attn_fused_hd%d", hd), adFused, hd, attnFusedBM, attnFusedBM},
			k{fmt.Sprintf("attn_fused_bm32_hd%d", hd), adFusedBM, hd, 32, 64},
			k{fmt.Sprintf("attn_fused_bm32n32_hd%d", hd), adFusedBM, hd, 32, 32},
			k{fmt.Sprintf("attn_fused_bm128_hd%d", hd), adFusedBM, hd, 128, 64})
	}
	var cs []adCase
	for _, kk := range ks {
		for kvi, m := range kk.mods {
			for _, sm := range adRows {
				for _, window := range []int{0, 100} {
					for _, sink := range []bool{false, true} {
						f16, startPos, M, hd := kvi == 1, sm[0], sm[1], kk.hd
						nH, nKV := 4, 2
						if hd == 128 {
							nH = 8
						}
						name, bm, bn := kk.name, kk.bm, kk.bn
						cs = append(cs, adCase{name, fmt.Sprintf("%s/%s/s%d_m%d/%s/%s", name, adKVNames[kvi], startPos, M, adWindowText(window), adSinkText(sink)),
							func(r *adRig) [][]float32 {
								nTot, kvDim := startPos+M, nKV*hd
								q := r.f32(adRand(M*nH*hd, 81, 1))
								kc, vc := r.kv(adRand(nTot*kvDim, 82, 1), f16), r.kv(adRand(nTot*kvDim, 83, 1), f16)
								ctx := r.zeros(M * nH * hd)
								shared := 2 * (bn*(hd+attnFusedKPAD) + hd*(bn+attnFusedKPAD))
								r.launch(m, name, adCfg(nH, (M+bm-1)/bm, 1, bm/16*32, shared), gc.Arg(q), kc, vc, r.kvt(nTot),
									adI(nH), adI(nKV), adI(hd), adI(startPos), gc.ArgValue(adScale(hd)), adI(window), adI(M), gc.Arg(ctx), adSink(r, nH, sink))
								return [][]float32{r.read(ctx, M*nH*hd)}
							}})
					}
				}
			}
		}
	}
	return cs
}

// mla_attn (mla.cu): attention over the latent cache, absorbed form.
func adCasesMLA() []adCase {
	var cs []adCase
	for _, sh := range [][2]int{{16, 8}, {512, 64}} { // rank, qkRope
		for _, sm := range [][2]int{{0, 1}, {36, 1}, {127, 3}, {130, 4}} {
			for _, window := range []int{0, 64} {
				rank, rope, startPos, M, nH := sh[0], sh[1], sm[0], sm[1], 4
				latDim := rank + rope
				cs = append(cs, adCase{"mla_attn", fmt.Sprintf("mla_attn/f32/rank%d/s%d_m%d/%s", rank, startPos, M, adWindowText(window)),
					func(r *adRig) [][]float32 {
						nTot := startPos + M
						qAbs := r.f32(adRand(M*nH*latDim, 91, 1))
						lat := r.f32(adRand(nTot*latDim, 92, 1))
						wsum := r.zeros(M * nH * rank)
						maxWin := nTot
						if window > 0 && window < maxWin {
							maxWin = window
						}
						r.launch(adMLA, "mla_attn", adCfg(nH, M, 1, 128, (maxWin+128)*4), gc.Arg(qAbs), gc.Arg(lat), r.kvt(nTot), gc.Arg(wsum),
							adI(nH), adI(latDim), adI(rank), adI(startPos), gc.ArgValue(adScale(latDim)), adI(window), adI(M))
						return [][]float32{r.read(wsum, M*nH*rank)}
					}})
			}
		}
	}
	return cs
}

// adAllCases is every driven kernel at every shape, in a fixed order.
func adAllCases() []adCase {
	var cs []adCase
	for _, f := range []func() []adCase{adCasesAttention, adCasesBatched, adCasesSplitKV, adCasesSplitKVRows, adCasesBlockAndImage, adCasesFlash, adCasesFused, adCasesMLA} {
		cs = append(cs, f()...)
	}
	return cs
}

// adDrivenKernels is the set of entry names the cases drive. The split-KV chain's other entries and the flash-decode combines are driven
// inside their cases (the case key is the kernel that forms the score).
func adDrivenKernels(cs []adCase) map[string]bool {
	out := map[string]bool{
		// driven inside a chain case, after the score kernel that names it
		"splitkv_softmax": true, "splitkv_vsum": true, "splitkv_vsum_partial": true, "splitkv_vsum_combine": true,
		"splitkv_softmax_rows": true, "splitkv_vsum_rows": true,
		"fa_combine": true, "fa_combine_rows": true,
	}
	for _, c := range cs {
		out[c.kernel] = true
	}
	return out
}

// TestAttnKernelDigests_tableCoversTheSource is the half of G1-K that needs no device: an attention kernel the sources declare and the
// table does not drive fails it, and so does a table entry the sources no longer declare.
func TestAttnKernelDigests_tableCoversTheSource(t *testing.T) {
	driven := adDrivenKernels(adAllCases())
	src := adSourceKernels(t)
	for _, k := range src {
		if !driven[k] {
			t.Errorf("%s is an attention kernel in the .cu sources that G1-K does not drive: add its cases to attn_digest_test.go (or, for a kernel that is not a text-path attention kernel, to adSourceKernels' exemptions with the reason)", k)
		}
	}
	for k := range driven {
		if !slices.Contains(src, k) {
			t.Errorf("%s is driven by G1-K but no .cu source declares it: the table is stale", k)
		}
	}
	if len(src) < 40 {
		t.Errorf("only %d attention kernels found in the sources: the scan is broken, and a broken scan empties this check", len(src))
	}
}

// TestAttnKernelDigests drives every case twice (a case that does not repeat is VOID: nondeterministic) and compares the digests with the
// committed file, which must name exactly the same cases.
func TestAttnKernelDigests(t *testing.T) {
	r, anchor := newAdRig(t)
	cases := adAllCases()
	got := make(map[string]string, len(cases))
	for _, c := range cases {
		if _, dup := got[c.id]; dup {
			t.Fatalf("duplicate case id %s", c.id)
		}
		a := adDigest(c.run(r))
		r.release()
		if b := adDigest(c.run(r)); a != b {
			t.Errorf("%s: two runs in one process give different bytes: the case is nondeterministic, so no digest of it can be compared", c.id)
		}
		r.release()
		got[c.id] = a
	}
	if *updateAttnDigests {
		b, err := json.MarshalIndent(adFile{Anchor: anchor, Digests: got}, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(attnDigestFile, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("recorded %d digests on %+v", len(got), anchor)
		return
	}
	b, err := os.ReadFile(attnDigestFile)
	if err != nil {
		t.Fatalf("%v: record the digests at base with -update-attn-digests", err)
	}
	var want adFile
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	if want.Anchor != anchor {
		t.Fatalf("VOID: the digests were recorded on %+v and this run is on %+v. A driver or device change can move a bit with no source change; re-anchor deliberately (CLAUDE.md) and record why", want.Anchor, anchor)
	}
	var diff, missing, extra []string
	for id, d := range got {
		w, ok := want.Digests[id]
		switch {
		case !ok:
			extra = append(extra, id)
		case w != d:
			diff = append(diff, id)
		}
	}
	for id := range want.Digests {
		if _, ok := got[id]; !ok {
			missing = append(missing, id)
		}
	}
	sort.Strings(diff)
	sort.Strings(missing)
	sort.Strings(extra)
	show := func(what string, ids []string) {
		if len(ids) == 0 {
			return
		}
		n := min(len(ids), 8)
		t.Errorf("%d case(s) %s, first %d: %v", len(ids), what, n, ids[:n])
	}
	byKernel := map[string]int{}
	for _, id := range diff {
		byKernel[id[:strings.IndexByte(id, '/')]]++
	}
	if len(byKernel) > 0 {
		t.Errorf("cases with a different digest, by kernel: %v", byKernel)
	}
	show("have a different digest from base's", diff)
	show("are in the committed file and no longer driven", missing)
	show("are driven and not in the committed file", extra)
	if len(diff)+len(missing)+len(extra) == 0 {
		t.Logf("%d digests equal base's on %+v", len(got), anchor)
	}
}
