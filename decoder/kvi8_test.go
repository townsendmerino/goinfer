package decoder

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// kv-quant Increment 1 gate (docs/task-cpu-kv-quant.md): int8 KV decode quality.
// int8 is lossy, so the bar is cosine vs the f32 cache (not bit-exact). Covers
// BOTH int8 layer kinds — a global (append-forever) layer and a ring
// (sliding-window) layer that wraps — since attendQueryI8 reads them differently.
// Model-free; the real-checkpoint greedy-decode gate is TestKVI8_genParity below.

func cosineV(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestKVI8_decodeQuality(t *testing.T) {
	const nLayers, nH, nKV, hd, W, N = 2, 4, 2, 8, 4, 40
	kvDim, qDim := nKV*hd, nH*hd
	arch := ringTestArch(nH, nKV, hd, W, func(l int) bool { return l == 1 }) // L0 local ring, L1 global

	f32 := NewKVCache(nLayers, nKV, hd, W, N+1)
	f32.enableRings(W, arch.isGlobalLayer)
	f32.scr = newDecodeScratch(arch)

	i8 := NewKVCache(nLayers, nKV, hd, W, N+1)
	i8.setQuant(kvI8)
	i8.enableRings(W, arch.isGlobalLayer)
	i8.scr = newDecodeScratch(arch)

	rng := rand.New(rand.NewSource(7))
	worst := map[int]float64{0: 1, 1: 1}
	for pos := 0; pos < N; pos++ {
		for l := range nLayers {
			global := arch.isGlobalLayer(l)
			k, v, q := randVec(rng, kvDim), randVec(rng, kvDim), randVec(rng, qDim)

			f32.Append(l, append([]float32(nil), k...), append([]float32(nil), v...))
			fCtx := make([]float32, qDim)
			attendQuery(q, fCtx, f32.scr.scoresBuf(f32.storedRows(l, kvDim)), f32, l, pos, global, arch)

			i8.Append(l, append([]float32(nil), k...), append([]float32(nil), v...))
			iCtx := make([]float32, qDim)
			attendQuery(append([]float32(nil), q...), iCtx, i8.scr.scoresBuf(i8.storedRows(l, kvDim)), i8, l, pos, global, arch)

			if c := cosineV(iCtx, fCtx); c < worst[l] {
				worst[l] = c
			}
		}
	}
	t.Logf("int8 vs f32 ctx cosine: ring layer 0 = %.6f, global layer 1 = %.6f (N=%d, W=%d wraps)", worst[0], worst[1], N, W)
	for l, c := range worst {
		if c < 0.999 {
			t.Errorf("layer %d int8 decode cosine %.6f < 0.999", l, c)
		}
	}
	// Storage: int8 ring holds W rows of int8 (¼ the f32 ring bytes).
	if r := i8.rings[0]; len(r.kq) != W*kvDim || len(r.k) != 0 {
		t.Errorf("int8 ring: kq=%d (want %d), k=%d (want 0)", len(r.kq), W*kvDim, len(r.k))
	}
	if len(i8.keysQ[1]) != N*kvDim || len(i8.keys[1]) != 0 {
		t.Errorf("int8 global: keysQ=%d (want %d), keys=%d (want 0)", len(i8.keysQ[1]), N*kvDim, len(i8.keys[1]))
	}
}

// TestKVI8_genParity: real-checkpoint int8 KV vs f32 KV on a COHERENT prompt,
// teacher-forced (both see the same tokens). Gate = per-step argmax agreement +
// avg per-step logit cosine — NOT a 0.999 bar (int8 KV lands ~0.993/~93% over a
// long context, in line with the shipped full-int8-weights precedent of 92.5%;
// the lossy KV cache is opt-in). Skips without a checkpoint dir that carries a
// tokenizer.json (GINFER_TEST_MODEL or ~/models/gemma-3-4b-it). NOTE: the prompt
// MUST be tokenized by THIS model's tokenizer — feeding foreign token ids is
// garbage input where argmax is a coin-flip (the bug that faked an earlier
// "int8 fails" result).
func TestKVI8_genParity(t *testing.T) {
	dir := os.Getenv("GINFER_TEST_MODEL")
	if dir == "" {
		dir = os.Getenv("HOME") + "/models/gemma-3-4b-it"
	}
	tkPath := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(tkPath); err != nil {
		t.Skipf("no checkpoint with tokenizer.json at %s — set GINFER_TEST_MODEL", dir)
	}
	tk, err := tokenizer.Load(tkPath)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	m, err := Load(dir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	if m.w.arch.MoE != nil || m.w.arch.gemma4 != nil || m.w.arch.qwen35 != nil {
		t.Skip("int8 KV excluded for this family")
	}
	ids, err := tk.Encode(kvi8GateText, true)
	if err != nil || len(ids) < 16 {
		t.Skipf("encode: %v (n=%d)", err, len(ids))
	}

	// Teacher-forced per-step argmax + logit cosine, both fed the same ids.
	args := func(kvI8 bool) ([]int, [][]float32) {
		mm := *m
		mm.kvI8 = kvI8
		cache := mm.NewCache(len(ids) + 1)
		am := make([]int, len(ids))
		lg := make([][]float32, len(ids))
		for i, id := range ids {
			logits, _ := mm.forward(id, cache)
			am[i] = argmax(logits)
			lg[i] = append([]float32(nil), logits...)
		}
		return am, lg
	}
	refA, refL := args(false)
	gotA, gotL := args(true)
	match := 0
	var cosSum float64
	for i := range ids {
		if gotA[i] == refA[i] {
			match++
		}
		cosSum += logitCosine(refL[i], gotL[i])
	}
	agree := float64(match) / float64(len(ids))
	cos := cosSum / float64(len(ids))
	t.Logf("int8 KV vs f32 (%d tokens): per-step argmax agree %.1f%%, avg cosine %.5f", len(ids), 100*agree, cos)
	if agree < 0.80 {
		t.Errorf("per-step argmax agreement %.1f%% < 80%% (int8 KV opt-in, lossy)", 100*agree)
	}
	if cos < 0.99 {
		t.Errorf("avg per-step logit cosine %.5f < 0.99", cos)
	}
}

const kvi8GateText = "The history of computing began in the early nineteenth century with the work of Charles Babbage, who designed the Analytical Engine, a general-purpose mechanical computer. Ada Lovelace wrote the first algorithm intended for such a machine. Modern electronic computers emerged in the 1940s with ENIAC and the stored-program architecture described by John von Neumann. The transistor and the integrated circuit made computers smaller and cheaper, leading to the microprocessor and the personal computer."
