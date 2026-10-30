package decoder

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// KS-F2's G0 (docs/measurements/ks-f2-2026-10/README.md): attendSinksToken against the code it replaced and attendQueryP against its own, kept below
// verbatim as the references. Every ctx word and every KV row is compared by its bits, over shapes that cover the eight-key chains'
// edges (key counts not a multiple of 8), GQA groups, windows, a wrapped ring and softcap, with the cache's scratch present (heads fan
// out) and absent (one worker), at GOMAXPROCS 1 and 8.

func ksf2Floats(r *rand.Rand, n int, amp float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((r.Float64()*2 - 1) * amp)
	}
	return out
}

func ksf2SameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d words, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: word %d is %v (%08x), the reference %v (%08x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

type ksf2Shape struct{ nH, nKV, hd int }

var ksf2Shapes = []ksf2Shape{{8, 8, 64}, {8, 4, 80}, {16, 2, 128}, {64, 8, 64}}
var ksf2Keys = []int{1, 7, 8, 9, 15, 64, 129, 300}

func TestKSF2_sinksBitIdentical(t *testing.T) {
	cells, fanned := 0, 0
	for _, procs := range []int{1, 8} {
		old := runtime.GOMAXPROCS(procs)
		for _, sh := range ksf2Shapes {
			for _, nKeys := range ksf2Keys {
				for _, window := range []int{0, 40} {
					for _, withScr := range []bool{true, false} {
						name := fmt.Sprintf("procs %d, %+v, %d keys, window %d, scratch %v", procs, sh, nKeys, window, withScr)
						r := rand.New(rand.NewSource(int64(nKeys*131 + sh.hd*7 + window + procs)))
						arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
						kvDim := sh.nKV * sh.hd
						lw := &LayerWeights{AttnSinks: ksf2Floats(r, sh.nH, 3)}
						p := attentionParams{Window: window, Scale: 1 / math.Sqrt(float64(sh.hd))}
						a, b := NewKVCache(1, sh.nKV, sh.hd, window, nKeys, nil), NewKVCache(1, sh.nKV, sh.hd, window, nKeys, nil)
						if withScr {
							a.scr = newDecodeScratch(arch)
						}
						for range nKeys - 1 {
							k, v := ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
							a.Append(0, k, v)
							b.Append(0, k, v)
						}
						q, k, v := ksf2Floats(r, sh.nH*sh.hd, 2), ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
						got, want := make([]float32, sh.nH*sh.hd), make([]float32, sh.nH*sh.hd)
						attendSinksToken(p, q, k, v, got, a, arch, lw, 0, nKeys-1)
						attendSinksTokenRef(p, q, k, v, want, b, arch, lw, 0, nKeys-1)
						ksf2SameBits(t, name+": ctx", got, want)
						ksf2SameBits(t, name+": keys", a.Keys(0), b.Keys(0))
						ksf2SameBits(t, name+": vals", a.Vals(0), b.Vals(0))
						cells++
						if withScr && scalarHeadWorkers(sh.nH, nKeys-p.Start(nKeys-1), sh.hd) > 1 {
							fanned++
						}
					}
				}
			}
		}
		runtime.GOMAXPROCS(old)
	}
	t.Logf("%d runs (each cell at GOMAXPROCS 1 and 8), %d of them with the heads fanned out (%d distinct cells), every word equal", cells, fanned, fanned/2)
	if fanned == 0 {
		t.Fatal("no cell fanned the heads out: the parallel path went untested")
	}
}

// KS-F2b: attendQueryP with the heads fanned out and eight keys scored at once is bit-identical to its reference in every cell. In
// every run with a scratch it fans out exactly where scalarHeadWorkers says to (its cache's head worker pool grows there and nowhere
// else).
func TestKSF2_queryPFannedBitIdentical(t *testing.T) {
	cells, fanned := 0, 0
	for _, procs := range []int{1, 8} {
		old := runtime.GOMAXPROCS(procs)
		for _, sh := range ksf2Shapes {
			for _, nKeys := range ksf2Keys {
				for _, mode := range []string{"global", "window", "ring"} {
					for _, softcap := range []float64{0, 30} {
						for _, withScr := range []bool{true, false} {
							name := fmt.Sprintf("procs %d, %+v, %d keys, %s, softcap %v, scratch %v", procs, sh, nKeys, mode, softcap, withScr)
							r := rand.New(rand.NewSource(int64(nKeys*977 + sh.nH*13 + len(mode) + procs)))
							arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
							kvDim := sh.nKV * sh.hd
							p := attentionParams{Scale: 1 / math.Sqrt(float64(sh.hd)), Softcap: softcap}
							const W = 48
							if mode != "global" {
								p.Window = W
							}
							a, b := NewKVCache(1, sh.nKV, sh.hd, p.Window, nKeys, nil), NewKVCache(1, sh.nKV, sh.hd, p.Window, nKeys, nil)
							if mode == "ring" {
								a.enableRings(W, func(int) bool { return false })
								b.enableRings(W, func(int) bool { return false })
							}
							if withScr {
								a.scr = newDecodeScratch(arch)
							}
							for i := range nKeys {
								k, v := ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
								if mode == "ring" {
									a.rings[0].write(i, k, v)
									b.rings[0].write(i, k, v)
								} else {
									a.Append(0, k, v)
									b.Append(0, k, v)
								}
							}
							if mode == "ring" {
								a.rings[0].count, b.rings[0].count = nKeys, nKeys
							}
							q := ksf2Floats(r, sh.nH*sh.hd, 2)
							got, want := make([]float32, sh.nH*sh.hd), make([]float32, sh.nH*sh.hd)
							attendQueryP(p, q, got, make([]float32, nKeys), a, 0, nKeys-1, arch)
							attendQueryPRef(p, q, want, make([]float32, nKeys), b, 0, nKeys-1, arch)
							ksf2SameBits(t, name+": ctx", got, want)
							cells++
							if withScr {
								eligible := scalarHeadWorkers(sh.nH, nKeys-p.Start(nKeys-1), sh.hd) > 1
								switch {
								case eligible && len(a.scr.headPool) == 0:
									t.Errorf("%s: the fan-out rule takes this cell, but attendQueryP grew no head worker slot", name)
								case !eligible && len(a.scr.headPool) > 0:
									t.Errorf("%s: under the fan-out threshold, but attendQueryP grew %d head worker slots", name, len(a.scr.headPool))
								}
								if eligible {
									fanned++
								}
							}
						}
					}
				}
			}
		}
		runtime.GOMAXPROCS(old)
	}
	t.Logf("%d runs, every word equal; %d in cells the fan-out rule takes, each of them fanned out", cells, fanned)
	if fanned == 0 {
		t.Fatal("no cell the fan-out rule takes: the fanned schedule was not exercised")
	}
}

// attendSinksTokenRef and attendQueryPRef are the two functions as they were before KS-F2, verbatim: the references G0 compares against.
// attendQueryP keeps exactly that code, so its reference also pins it against an unreviewed change.

func attendSinksTokenRef(p attentionParams, q, k, v, ctx []float32, cache *KVCache, arch *Architecture, lw *LayerWeights, l, pos int) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	cache.Append(l, k, v)
	keys, vals := cache.Keys(l), cache.Vals(l)
	nKeys := len(keys) / kvDim
	start := p.Start(pos)
	scale := p.Scale
	group := nH / nKV

	clear(ctx)
	scores := make([]float32, nKeys)
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			base := s*kvDim + kvh*hd
			kHead := keys[base : base+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := dot * scale
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}
		sinkLogit := float64(lw.AttnSinks[qh])
		if sinkLogit > maxS {
			maxS = sinkLogit
		}
		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		sum += math.Exp(sinkLogit - maxS)
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			base := s*kvDim + kvh*hd
			vHead := vals[base : base+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}
}

func attendQueryPRef(p attentionParams, q, ctx, scores []float32, cache *KVCache, layer, pos int, arch *Architecture) {
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim
	// Local (sliding-window) layers store only the last W positions in a ring: absolute key s lives at row s%W, nKeys becomes the
	// logical count, and the window guarantees every read s in [start, nKeys) is resident. wrap=0 means global append-forever,
	// row = s. scores stays indexed by absolute s (its scratch is already context-sized).
	wrap := 0
	if r := cache.rings[layer]; r != nil {
		keys, vals, nKeys, wrap = r.k, r.v, r.count, r.w
	}
	rowBase := func(s int) int {
		if wrap > 0 {
			return (s % wrap) * kvDim
		}
		return s * kvDim
	}
	start := p.Start(pos)
	scale := p.Scale  // query_pre_attn_scalar^-0.5 (Gemma) or 1/sqrt(headDim)
	group := nH / nKV // GQA: query heads per KV head

	clear(ctx) // accumulated into below
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := softcapScore(dot*scale, p.Softcap)
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}

		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			vHead := vals[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
			addScaled(oHead, vHead, w)
		}
	}
}

// Many decode steps sharing one scratch: the fanned path reuses decodeScratch.headScoresPool's buffers across steps, grows them, and
// meets shorter steps after longer ones. Each window (narrower than the span, and wider than it) runs on its own cache, growing a row a
// step to 300 keys; then a second, shorter cache on the same scratch runs 1 to 60 keys, so a slot sized and written for 300 keys serves
// steps that read far fewer. Every step's ctx must equal the reference's: a stale score carried into a shorter step, or a slot grown
// short, shows here and not in the one-call cells. attendQueryP shares the scratch, as a cache's layers do: it scores into the scratch's
// scoresBuf, the buffer attendSinksToken also uses when one worker serves, and must stay on its path.
func TestKSF2_scratchReuseAcrossSteps(t *testing.T) {
	for _, sh := range []ksf2Shape{{64, 8, 64}, {16, 2, 128}} {
		r := rand.New(rand.NewSource(int64(sh.nH)))
		arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
		kvDim := sh.nKV * sh.hd
		lw := &LayerWeights{AttnSinks: ksf2Floats(r, sh.nH, 3)}
		scr := newDecodeScratch(arch)
		fanned, steps := 0, 0
		run := func(name string, window, n int) {
			p := attentionParams{Window: window, Scale: 1 / math.Sqrt(float64(sh.hd)), Softcap: 30}
			a, b := NewKVCache(1, sh.nKV, sh.hd, window, n, nil), NewKVCache(1, sh.nKV, sh.hd, window, n, nil)
			a.scr = scr
			qa, qb := NewKVCache(1, sh.nKV, sh.hd, 0, n, nil), NewKVCache(1, sh.nKV, sh.hd, 0, n, nil)
			qa.scr = scr
			for pos := range n {
				q, k, v := ksf2Floats(r, sh.nH*sh.hd, 2), ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
				got, want := make([]float32, sh.nH*sh.hd), make([]float32, sh.nH*sh.hd)
				attendSinksToken(p, q, k, v, got, a, arch, lw, 0, pos)
				attendSinksTokenRef(p, q, k, v, want, b, arch, lw, 0, pos)
				ksf2SameBits(t, fmt.Sprintf("%+v %s sinks step %d: ctx", sh, name, pos), got, want)
				qa.Append(0, k, v)
				qb.Append(0, k, v)
				pq := attentionParams{Window: window, Scale: p.Scale, Softcap: p.Softcap}
				attendQueryP(pq, q, got, scr.scoresBuf(pos+1), qa, 0, pos, arch) // the cache scratch's buffer, as the walker passes it
				attendQueryPRef(pq, q, want, make([]float32, pos+1), qb, 0, pos, arch)
				ksf2SameBits(t, fmt.Sprintf("%+v %s queryP step %d: ctx", sh, name, pos), got, want)
				if scalarHeadWorkers(sh.nH, pos+1-p.Start(pos), sh.hd) > 1 { // the span the step reads, as attendSinksToken counts it
					fanned++
				}
				steps++
			}
		}
		run("window 40 (narrower than the span)", 40, 300)
		run("window 400 (wider than the span)", 400, 300)
		run("shorter, after the long ones", 400, 60)
		if fanned == 0 {
			t.Fatalf("%+v: no step fanned the heads out", sh)
		}
		t.Logf("%+v: %d steps on one scratch (growing, then shorter), gpt-oss's attention fanned out in %d, every word equal", sh, steps, fanned)
	}
}

// TestKSF2_familiesTakeTheFannedSchedule: each attendQueryP family (Qwen3.5/3.6, Granite-4.0-H, LFM2, Nemotron-H, Llama 4) takes
// the fanned schedule through its own forward. Its tiny fixture, on the CPU, decodes past a prompt long enough that a layer's
// heads × span × head size crosses scalarHeadWorkers' threshold, and attendQueryPFanned must rise. A fixture that is absent here
// (granite-tiny is gitignored) skips. Bit-identity is TestKSF2_queryPFannedBitIdentical's, and each family's full logits are
// gate identity's. It reads the package-wide attendQueryPFanned counter, so it and its subtests run serially: a test calling
// attendQueryP in parallel with it could raise the counter and pass a family that never fanned out.
func TestKSF2_familiesTakeTheFannedSchedule(t *testing.T) {
	for _, fx := range []string{"qwen35-tiny", "granite-tiny", "lfm2-tiny", "nemotron-tiny", "llama4-tiny"} {
		t.Run(fx, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", fx)
			if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			m, err := Load(dir, Options{Backend: "cpu"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			_, _, nH, _, hd, _, vocab := m.Dims()
			n := (1<<15)/(nH*hd) + 64 // keys enough for one layer to cross the threshold
			prompt := make([]int, n)
			for i := range prompt {
				prompt[i] = 3 + (i*131)%(vocab-3)
			}
			before := attendQueryPFanned.Load()
			ch, g := m.Generate(context.Background(), prompt, 2, SamplingParams{})
			for range ch {
			}
			if err := g.Err(); err != nil {
				t.Fatalf("generate: %v", err)
			}
			fanned := attendQueryPFanned.Load() - before
			if fanned == 0 {
				t.Fatalf("%d-token prompt (heads %d × head size %d): attendQueryP never fanned out", n, nH, hd)
			}
			t.Logf("%d-token prompt: attendQueryP fanned out %d times", n, fanned)
		})
	}
}
