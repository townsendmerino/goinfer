//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMC5_prefillChunkInvariance asks the question chunked prefill's design turns on (docs/tasks/task-concurrency-
// 2026-09.md, chunked prefill): does Metal's fast prefill give the SAME bits when a prompt is ingested in chunks —
// PrefillLast(chunk, startPos) one after another — as when it is ingested whole? The same 1000-token prompt is
// prefilled whole into one resident KV slot and in chunks of C tokens into another; every K and V element of every
// layer, and the last token's logits, are compared bit for bit.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC5_prefillChunkInvariance$' -v ./metal/
func TestMC5_prefillChunkInvariance(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_MC3_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 2048, ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || len(a.r.kvSlotBufs) < 2 || !a.r.kvContig || a.r.kvF32 {
		t.Skip("needs a Metal resident with 2 contiguous f16 KV slots")
	}
	r := a.r
	const N = 1000
	seed := uint32(13579)
	embs := make([][]float32, N)
	for i := range embs {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		embs[i] = mc3Emb(r, int(seed%20000))
	}
	snapshot := func(slot int) [][]uint16 {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			kvDim := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+N*kvDim]...), append([]uint16(nil), r.vc[l].U16s()[o:o+N*kvDim]...))
		}
		return out
	}
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	whole, err := a.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("whole prefill: %v", err)
	}
	whole = append([]float32(nil), whole...)
	wantKV := snapshot(0)
	for _, C := range []int{64, 128, 256, 384} {
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		var lg []float32
		for c := 0; c < N; c += C {
			e := min(c+C, N)
			if lg, err = a.PrefillLast(context.Background(), embs[c:e], c); err != nil {
				t.Fatalf("chunk %d..%d: %v", c, e, err)
			}
		}
		lg = append([]float32(nil), lg...)
		gotKV := snapshot(1)
		kvDiff, kvTotal := 0, 0
		firstBad := -1
		for i := range wantKV {
			for j := range wantKV[i] {
				kvTotal++
				if wantKV[i][j] != gotKV[i][j] {
					kvDiff++
					if firstBad < 0 {
						firstBad = j / r.layers[0].geom.kvDim // position
					}
				}
			}
		}
		lgDiff, worst := 0, 0.0
		for i := range whole {
			if math.Float32bits(whole[i]) != math.Float32bits(lg[i]) {
				lgDiff++
				worst = math.Max(worst, math.Abs(float64(whole[i]-lg[i])))
			}
		}
		fmt.Fprintf(os.Stderr, "[mc5-chunk] C=%d: KV %d of %d elements differ (first at position %d); last logits %d of %d differ (max |diff| %.3g)\n",
			C, kvDiff, kvTotal, firstBad, lgDiff, len(whole), worst)
		if kvDiff != 0 || lgDiff != 0 {
			t.Errorf("C=%d: chunked prefill is not bit-identical to whole prefill (%d KV elements, %d logits differ)", C, kvDiff, lgDiff)
		}
	}
	// Control: the comparison must be able to see a difference. The same chunked prefill with ONE token changed (at
	// position 700) must differ, and only from that position on.
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	pert := append([][]float32(nil), embs...)
	pert[700] = mc3Emb(r, 777)
	for c := 0; c < N; c += 256 {
		if _, err := a.PrefillLast(context.Background(), pert[c:min(c+256, N)], c); err != nil {
			t.Fatal(err)
		}
	}
	gotKV := snapshot(1)
	kvDim := r.layers[0].geom.kvDim
	before, after := 0, 0
	for i := range wantKV {
		for j := range wantKV[i] {
			if wantKV[i][j] != gotKV[i][j] {
				if j/kvDim < 700 {
					before++
				} else {
					after++
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[mc5-chunk] control (token 700 changed): %d KV elements differ before position 700, %d from it on\n", before, after)
	if before != 0 || after == 0 {
		t.Errorf("control: %d differences before the changed token, %d after — the comparison is not seeing what it should", before, after)
	}
}

// TestMC5_chunkCost: EXPLORATORY. What chunking itself costs a long prefill, with no decoding in between: one
// PrefillLast over 3000 tokens against the same prompt in chunks of C (cut as mc3Prefill cuts them — only while two
// chunks' worth remains, so the last pass is C..2C-1), wall time per pass, 3 reps each, interleaved.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC5_chunkCost$' -v ./metal/
func TestMC5_chunkCost(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	m, err := decoder.Load(filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
		decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	const N = 3000
	seed := uint32(97531)
	embs := make([][]float32, N)
	for i := range embs {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		embs[i] = mc3Emb(r, int(seed%20000))
	}
	run := func(C int) (total, maxPass float64, passes int) {
		t0 := time.Now()
		from := 0
		for C > 0 && N-from >= 2*C {
			p0 := time.Now()
			if _, err := a.PrefillLast(context.Background(), embs[from:from+C], from); err != nil {
				t.Fatal(err)
			}
			maxPass = max(maxPass, time.Since(p0).Seconds())
			passes++
			from += C
		}
		p0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), embs[from:], from); err != nil {
			t.Fatal(err)
		}
		maxPass = max(maxPass, time.Since(p0).Seconds())
		return time.Since(t0).Seconds(), maxPass, passes + 1
	}
	arms := []int{0, 128, 256, 512}
	res := map[int][]float64{}
	mx := map[int]float64{}
	np := map[int]int{}
	for rep := 0; rep < 3; rep++ {
		for k := range arms {
			C := arms[(k+rep)%len(arms)]
			tot, mp, n := run(C)
			res[C] = append(res[C], tot)
			mx[C], np[C] = max(mx[C], mp), n
		}
	}
	for _, C := range arms {
		v := res[C]
		sort.Float64s(v)
		fmt.Fprintf(os.Stderr, "[mc5-cost] C=%d: %d passes, total %.3f s (median of 3), longest pass %.3f s\n", C, np[C], v[1], mx[C])
	}
}

// TestMC5_passCost: EXPLORATORY. One PrefillLast pass's time against its length at two depths, to separate a pass's
// fixed cost (the intercept) from its per-token cost: C = 16..256 tokens at startPos 64 and 2048, median of 5.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC5_passCost$' -v ./metal/
func TestMC5_passCost(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	m, err := decoder.Load(filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
		decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	embs := make([][]float32, 2400)
	for i := range embs {
		embs[i] = mc3Emb(r, 1000+(i*37)%15000)
	}
	if _, err := a.PrefillLast(context.Background(), embs[:2048], 0); err != nil {
		t.Fatal(err)
	}
	for _, at := range []int{64, 2048} {
		line := fmt.Sprintf("startPos %d:", at)
		for _, C := range []int{16, 64, 128, 256} {
			var ts []float64
			for rep := 0; rep < 5; rep++ {
				t0 := time.Now()
				if _, err := a.PrefillLast(context.Background(), embs[at:at+C], at); err != nil {
					t.Fatal(err)
				}
				ts = append(ts, time.Since(t0).Seconds()*1e3)
			}
			sort.Float64s(ts)
			line += fmt.Sprintf("  C=%d %.1f ms (GPU %.1f)", C, ts[2], (r.gpuEnd-r.gpuStart)*1e3)
		}
		fmt.Fprintf(os.Stderr, "[mc5-pass] %s\n", line)
	}
}
