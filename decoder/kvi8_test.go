package decoder

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// kv-quant Increment 1 gate (docs/completed/task-cpu-kv-quant.md): int8 KV decode quality.
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
	i8.setQuant(kvI8, N+1)
	i8.enableRings(W, arch.isGlobalLayer)
	i8.scr = newDecodeScratch(arch)

	rng := rand.New(rand.NewSource(7))
	worst := map[int]float64{0: 1, 1: 1}
	for pos := range N {
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

// TestKVI8_truncateReappend: the spec-decode rollback mechanic under int8 —
// append, TruncateTo (drop rejected drafts), re-append (accepted) must leave the
// int8 cache byte-identical to one that only ever saw the final positions. Covers
// global int8 + an int8 ring (unwrapped, as short-context spec-decode is). The
// risk the doc flags (TruncateTo × quant interplay), model-free.
func TestKVI8_truncateReappend(t *testing.T) {
	const nLayers, nH, nKV, hd, W = 2, 4, 2, 8, 16
	kvDim := nKV * hd
	arch := ringTestArch(nH, nKV, hd, W, func(l int) bool { return l == 1 }) // L0 ring, L1 global
	mk := func() *KVCache {
		c := NewKVCache(nLayers, nKV, hd, W, 16)
		c.setQuant(kvI8, 16)
		c.enableRings(W, arch.isGlobalLayer)
		c.scr = newDecodeScratch(arch)
		return c
	}
	rng := rand.New(rand.NewSource(5))
	vec := func() []float32 { return randVec(rng, kvDim) }
	// final[p] = the K/V that position p ends up with (draft positions 6..9 rejected,
	// re-appended with different values).
	final := make([][2][]float32, 10)
	for p := range final {
		final[p] = [2][]float32{vec(), vec()}
	}

	// (1) append 0..5, draft 6..9 (garbage), truncate to 6, re-append 6..9 (final).
	got := mk()
	for p := range 6 {
		for l := range nLayers {
			got.Append(l, append([]float32(nil), final[p][0]...), append([]float32(nil), final[p][1]...))
		}
	}
	for p := 6; p < 10; p++ { // rejected drafts
		for l := range nLayers {
			got.Append(l, vec(), vec())
		}
	}
	got.TruncateTo(6)
	for p := 6; p < 10; p++ { // re-append the accepted tokens
		for l := range nLayers {
			got.Append(l, append([]float32(nil), final[p][0]...), append([]float32(nil), final[p][1]...))
		}
	}
	// (2) reference: only ever the final 10.
	ref := mk()
	for p := range 10 {
		for l := range nLayers {
			ref.Append(l, append([]float32(nil), final[p][0]...), append([]float32(nil), final[p][1]...))
		}
	}
	if got.pos != ref.pos {
		t.Fatalf("pos %d != %d", got.pos, ref.pos)
	}
	// global int8 layer 1: int8 rows + scales must match byte-for-byte.
	if !slices.Equal(got.keysQ[1], ref.keysQ[1]) || !slices.Equal(got.keyScale[1], ref.keyScale[1]) {
		t.Errorf("global int8: truncate+reappend keysQ/scale != fresh")
	}
	// ring int8 layer 0: count + live int8 slots must match.
	if got.rings[0].count != ref.rings[0].count || !slices.Equal(got.rings[0].kq, ref.rings[0].kq) {
		t.Errorf("ring int8: truncate+reappend kq/count != fresh")
	}
}

// TestKVI8_snapshotRoundtrip: snapshot v2 must reconstruct an int8 + ring cache
// exactly — the next forward after snapshot→restore is BIT-IDENTICAL to one that
// never snapshotted. gemma-3-4b-it int8 covers both layer kinds (int8 rings on the
// sliding-window layers + int8 global layers). Skips without the checkpoint.
func TestKVI8_snapshotRoundtrip(t *testing.T) {
	requireHeavyModel(t)
	dir := os.Getenv("GINFER_TEST_MODEL")
	if dir == "" {
		dir = os.Getenv("HOME") + "/models/gemma-3-4b-it"
	}
	tkPath := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(tkPath); err != nil {
		t.Skipf("no checkpoint at %s", dir)
	}
	tk, err := tokenizer.Load(tkPath)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	m, err := Load(dir, Options{Quant: "int8int8", KVQuant: "i8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !m.kvI8 {
		t.Fatal("expected int8 KV")
	}
	ids, _ := tk.Encode(kvi8GateText, true)
	if len(ids) < 8 {
		t.Skip("short encode")
	}
	prefix, next := ids[:len(ids)-1], ids[len(ids)-1]

	prefill := func() *KVCache {
		c := m.NewCache(len(ids) + 1)
		for _, id := range prefix {
			m.runLayers(id, c)
		}
		return c
	}
	// (A) no snapshot: forward the next token.
	logitsA, _ := m.forward(next, prefill())
	// (B) snapshot the prefilled cache, restore, forward the same next token.
	sess := &Session{m: m, cache: prefill(), tokens: prefix}
	blob := sess.Snapshot("rt")
	if blob == nil {
		t.Fatal("Snapshot returned nil (int8+ring should serialize)")
	}
	sess2, err := m.LoadSession(blob, "rt")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	logitsB, _ := m.forward(next, sess2.cache)

	for i := range logitsA {
		if logitsA[i] != logitsB[i] {
			t.Fatalf("snapshot→restore NOT bit-identical: logit[%d] %v != %v", i, logitsA[i], logitsB[i])
		}
	}
	t.Logf("snapshot v2 (int8 + ring) round-trip: next-token logits bit-identical (%d-dim, %d-byte blob)", len(logitsA), len(blob))
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
	requireHeavyModel(t)
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

// TestKVI8_batchedPrefill: int8 KV through the BATCHED prefill path (forwardN,
// Increment 2's dequant-into-scratch) vs batched f32 — teacher-forced per-step
// argmax + cosine. Catches bugs in the int8 dequant/round-trip and confirms int8
// no longer forces sequential prefill. Skips without a checkpoint + tokenizer.
func TestKVI8_batchedPrefill(t *testing.T) {
	dir := os.Getenv("GINFER_TEST_MODEL")
	if dir == "" {
		dir = os.Getenv("HOME") + "/models/gemma-3-4b-it"
	}
	tkPath := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(tkPath); err != nil {
		t.Skipf("no checkpoint with tokenizer.json at %s", dir)
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
	if err != nil || len(ids) < 16 || !m.canBatchN(len(ids)) {
		t.Skipf("encode/canBatchN: %v (n=%d)", err, len(ids))
	}

	prefill := func(kvI8 bool) [][]float32 {
		mm := *m
		mm.kvI8 = kvI8
		lg, _ := mm.forwardN(ids, mm.NewCache(len(ids)+1)) // batched (canBatchN true)
		return lg
	}
	refL := prefill(false)
	gotL := prefill(true)
	match := 0
	var cosSum float64
	for i := range ids {
		if argmax(gotL[i]) == argmax(refL[i]) {
			match++
		}
		cosSum += logitCosine(refL[i], gotL[i])
	}
	agree := 100 * float64(match) / float64(len(ids))
	cos := cosSum / float64(len(ids))
	t.Logf("batched int8 prefill vs batched f32 (%d tokens): argmax %.1f%%, avg cosine %.5f", len(ids), agree, cos)
	if agree < 80 {
		t.Errorf("batched int8 prefill argmax agreement %.1f%% < 80%%", agree)
	}
	if cos < 0.99 {
		t.Errorf("batched int8 prefill avg cosine %.5f < 0.99", cos)
	}
}
