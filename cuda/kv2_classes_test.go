//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G4-C, KV1's carry-overs on the device (docs/measurements/kv2-cuda-2026-10-15.md).
//
//	go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKV2_(recordedClass|hitCap|inPlaceCap)' -v ./cuda/
//
// GOINFER_HEAVY_TESTS=1 adds the 1.5B.

// kv2Generate runs one greedy turn of prompt through Generate and returns its reply and diagnostics.
func kv2Generate(t *testing.T, m *decoder.Model, prompt []int, maxTok int) ([]int, *decoder.Generation) {
	t.Helper()
	ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids, g
}

// kv2PredictedClasses is the class KV0's validated class map gives each row of a prefill of rows [start, start+m) of a
// prompt of n tokens, cut into passes of chunk rows, and how many of its passes are fast.
func kv2PredictedClasses(floor, start, m, chunk int) (rows []string, fastPasses int) {
	for _, p := range kv0Passes(start, m, chunk) {
		c := "exact"
		if kv0FastPass(floor, p.promptLen, p.start, p.m) {
			c, fastPasses = "fast", fastPasses+1
		}
		for range p.m {
			rows = append(rows, c)
		}
	}
	return rows, fastPasses
}

// kv2RecordedClass is G4-C(a): for prompts of 985, 1,030 and 4,000 tokens, and a 1,040-token prompt that copies 960 rows
// and prefills the rest, the class the KV manager records for each row is the class that ran it. What ran is read from
// the fused-attention launch count, against the passes the chunker makes; the recorded class is
// KVBoundClassesForTest's.
func kv2RecordedClass(t *testing.T, g *kv2Rig) {
	floor, chunk := g.r.FastPrefillFloor(), prefillDefaultChunk // no GOINFER_PREFILL_CHUNK knob is set here
	layers := len(g.r.layers)
	check := func(name string, rec []string, start, m int) {
		t.Helper()
		want, _ := kv2PredictedClasses(floor, start, m, chunk)
		got := rec[start : start+m]
		if !slices.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Errorf("%s: row %d is recorded %s; the pass that wrote it ran %s", name, start+i, got[i], want[i])
					break
				}
			}
		}
	}
	for _, n := range []int{985, 1030, 4000} {
		prompt := g.toks(n, uint32(n))
		a0 := g.r.fastAttnLaunches
		kv2Generate(t, g.m, prompt, 1)
		launched := g.r.fastAttnLaunches - a0
		_, rec := g.m.KVBoundClassesForTest()
		_, fast := kv2PredictedClasses(floor, 0, n, chunk)
		if launched != fast*layers {
			t.Errorf("%d tokens: %d fused-attention launches, the predicted passes give %d: the prediction does not describe what ran", n, launched, fast*layers)
		}
		if len(rec) < n {
			t.Fatalf("%d tokens: %d rows recorded", n, len(rec))
		}
		check(fmt.Sprintf("a %d-token prompt", n), rec, 0, n)
		t.Logf("%d tokens: %d fast passes launched; recorded classes %s", n, launched/layers, kv2Runs(rec[:n]))
	}
	// a hit followed by a suffix, on two slots: A holds the preamble and more than a block of its own (so B copies rather
	// than taking A's slot in place), B copies 960 rows and prefills 80
	pre := g.toks(1000, 7)
	a := append(slices.Clone(pre), g.toks(100, 8)...)
	b := append(slices.Clone(pre), g.toks(40, 9)...)
	kv2Generate(t, g.m, a, 1)
	a0 := g.r.fastAttnLaunches
	_, gb := kv2Generate(t, g.m, b, 1)
	launched := g.r.fastAttnLaunches - a0
	if gb.PrefillShared != 960 {
		t.Fatalf("B copied %d rows (reused %d, declined %q), want 960", gb.PrefillShared, gb.PrefillReused, gb.KVCopyDeclined)
	}
	_, rec := g.m.KVBoundClassesForTest()
	_, fast := kv2PredictedClasses(floor, 960, len(b)-960, chunk)
	if launched != fast*layers {
		t.Errorf("hit then suffix: %d fused-attention launches, predicted %d", launched, fast*layers)
	}
	check("hit then suffix (copied rows)", rec, 0, 960) // A's passes [0, 512) and [512, 1024) wrote them
	check("hit then suffix (suffix)", rec, 960, len(b)-960)
	t.Logf("hit then suffix: copied %d; recorded classes %s", gb.PrefillShared, kv2Runs(rec[:len(b)]))
}

// kv2Runs renders a row-class list as runs, "fast[0,1024) exact[1024,1030)".
func kv2Runs(rows []string) string {
	out := ""
	for i := 0; i < len(rows); {
		j := i
		for j < len(rows) && rows[j] == rows[i] {
			j++
		}
		out += fmt.Sprintf("%s[%d,%d) ", rows[i], i, j)
		i = j
	}
	return out
}

func TestKV2_recordedClass_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	g := kv2Load(t, kv2WideFixture(t, dir, 8192), decoder.Options{ResidentKVSlots: 2, ResidentContext: 4160})
	kv2RecordedClass(t, g)
}

func TestKV2_recordedClass_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	kv2RecordedClass(t, kv2Load(t, path, decoder.Options{ResidentKVSlots: 2, ResidentContext: 4160, KVPrecision: "f16"}))
}

// kv2HitCap is G4-C(b): a 975-token prompt whose first 970 tokens another slot holds. The block-aligned hit, 960, would
// leave 15 rows, which the fast kernels do not run; the cap shrinks it to 896, leaving 79. The reply and every row equal
// the prompt run cold.
func kv2HitCap(t *testing.T, path string, opts decoder.Options) {
	g := kv2Load(t, path, opts)
	pre := g.toks(970, 31)
	a := append(slices.Clone(pre), g.toks(100, 32)...) // more than a block of A's own, so B copies
	b := append(slices.Clone(pre), g.toks(5, 33)...)
	kv2Generate(t, g.m, a, 4)
	ids, gb := kv2Generate(t, g.m, b, 8)
	slot, _ := g.m.KVBoundClassesForTest()
	got := g.rows(slot, len(b)+len(ids)-1)
	_ = g.m.Close()
	if gb.PrefillShared != 896 {
		t.Errorf("B copied %d rows (reused %d), want the hit capped to 896", gb.PrefillShared, gb.PrefillReused)
	}
	o := opts
	o.ResidentKVSlots = 1
	c := kv2Load(t, path, o)
	want, _ := kv2Generate(t, c.m, b, 8)
	if !slices.Equal(ids, want) {
		t.Errorf("B after the capped copy: %v, cold %v", ids, want)
	}
	if !kv2SameRaw(got, c.rows(0, len(want)+len(b)-1)) {
		t.Errorf("B's K/V rows after the capped copy differ from the cold run's")
	}
}

func TestKV2_hitCap_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			kv2HitCap(t, dir, decoder.Options{ResidentKVSlots: 2, ResidentContext: 2048, KVPrecision: kv})
		})
	}
}

func TestKV2_hitCap_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	kv2HitCap(t, path, decoder.Options{ResidentKVSlots: 2, ResidentContext: 2048, KVPrecision: "f16"})
}

// kv2InPlaceCap is the in-place twin of kv2HitCap: A holds the 970-token preamble and only 4 tokens of its own, so B takes
// A's idle slot in place rather than copying. The shared 970 rows would leave B's 5-token tail to the exact kernels; the cap
// backs the reuse off to 896, leaving 79 rows for the fast ones. The reply and every row equal the prompt run cold.
func kv2InPlaceCap(t *testing.T, path string, opts decoder.Options) {
	g := kv2Load(t, path, opts)
	pre := g.toks(970, 41)
	a := append(slices.Clone(pre), g.toks(4, 42)...)
	b := append(slices.Clone(pre), g.toks(5, 43)...)
	kv2Generate(t, g.m, a, 4)
	ids, gb := kv2Generate(t, g.m, b, 8)
	slot, _ := g.m.KVBoundClassesForTest()
	got := g.rows(slot, len(b)+len(ids)-1)
	_ = g.m.Close()
	if gb.PrefillShared != 0 || gb.PrefillReused != 896 {
		t.Errorf("B reused %d rows (copied %d), want 896 taken in place", gb.PrefillReused, gb.PrefillShared)
	}
	o := opts
	o.ResidentKVSlots = 1
	c := kv2Load(t, path, o)
	want, _ := kv2Generate(t, c.m, b, 8)
	if !slices.Equal(ids, want) {
		t.Errorf("B after the in-place reuse: %v, cold %v", ids, want)
	}
	if !kv2SameRaw(got, c.rows(0, len(want)+len(b)-1)) {
		t.Errorf("B's K/V rows after the in-place reuse differ from the cold run's")
	}
}

func TestKV2_inPlaceCap_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			kv2InPlaceCap(t, dir, decoder.Options{ResidentKVSlots: 2, ResidentContext: 2048, KVPrecision: kv})
		})
	}
}

// kv2FastWindowFixture is llama-attnfa-tiny (head dimension 128, so the fused prefill kernels run) declared as a Mistral
// with a 256-position sliding window: the same weights, every layer windowed. KV1's G2b instrument set the window on the
// resident's layers behind the decoder; declaring it in the config is what makes the decoder's class map windowed too.
func kv2FastWindowFixture(t *testing.T) string {
	t.Helper()
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	out := kv2WideFixture(t, dir, 4096)
	p := filepath.Join(out, "config.json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["model_type"], cfg["architectures"], cfg["sliding_window"] = "mistral", []string{"MistralForCausalLM"}, 256
	if b, err = json.Marshal(cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestKV2_windowedUnalignedNotShared is G4-C(e): on a windowed model on the fast kernels, a pass that starts off a block
// boundary writes rows no other request may take. A's first prompt is 700 tokens; A2 continues it two tokens into A's own
// reply and diverges there, taking A's slot in place, so its pass starts at 702 (a reuse that holds decoded rows is not
// capped to a block edge). B shares A2's first 1,250 tokens and arrives while A2 still decodes, so it copies from A2's busy
// slot: only the 640 rows of the blocks A's aligned first pass wrote whole. B equals its cold run.
func TestKV2_windowedUnalignedNotShared(t *testing.T) {
	path := kv2FastWindowFixture(t)
	g := kv2Load(t, path, decoder.Options{ResidentKVSlots: 2, ResidentContext: 2048})
	if l := g.r.layers[0]; l.window != 256 {
		t.Fatalf("layer 0's window is %d, want 256: the fixture is not windowed", l.window)
	}
	if got := g.m.EnableResidentConcurrency(2); got != 2 {
		t.Fatalf("EnableResidentConcurrency(2) = %d", got)
	}
	a1 := g.toks(700, 41)
	reply, _ := kv2Generate(t, g.m, a1, 4)
	a2 := append(append(slices.Clone(a1), reply[:2]...), g.toks(1100, 42)...)
	a0 := g.r.fastAttnLaunches
	chA, ga2 := g.m.Generate(context.Background(), a2, 32, decoder.SamplingParams{})
	<-chA // A2 has prefilled and holds its slot while it decodes
	if ga2.PrefillReused != 702 || g.r.fastAttnLaunches == a0 {
		t.Fatalf("A2 reused %d (want 702, its slot in place) with %d fused-attention launches: the case needs an unaligned fast pass", ga2.PrefillReused, g.r.fastAttnLaunches-a0)
	}
	done := make(chan struct{})
	go func() {
		for range chA {
		}
		close(done)
	}()
	b := append(slices.Clone(a2[:1250]), g.toks(30, 43)...)
	bids, gb := kv2Generate(t, g.m, b, 8)
	<-done
	if err := ga2.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("B reused %d, copied %d, declined %q", gb.PrefillReused, gb.PrefillShared, gb.KVCopyDeclined)
	n := len(b) + len(bids) - 1
	got0, got1 := g.rows(0, n), g.rows(1, n)
	_ = g.m.Close()
	if gb.PrefillShared != 640 {
		t.Errorf("B copied %d rows, want 640: the blocks of A's aligned first pass and none of the unaligned one", gb.PrefillShared)
	}
	c := kv2Load(t, path, decoder.Options{ResidentKVSlots: 1, ResidentContext: 2048})
	want, _ := kv2Generate(t, c.m, b, 8)
	cold := c.rows(0, len(b)+len(want)-1)
	if !slices.Equal(bids, want) || !(kv2SameRaw(got0, cold) || kv2SameRaw(got1, cold)) {
		t.Errorf("B: %v, cold %v (K/V equal in slot 0 %v, slot 1 %v)", bids, want, kv2SameRaw(got0, cold), kv2SameRaw(got1, cold))
	}
}

// kv2ExactIsDecode is a G0-C arm the Mac registered for its backends and KV0 did not run on CUDA: the KV manager records
// decode-written rows and exact-class prefill rows as one class (kvmgr.ClassExact), which is sound only if a prompt under
// the fast floor, prefilled in one call, writes the bytes the same prompt decoded token by token writes.
func kv2ExactIsDecode(t *testing.T, g *kv2Rig) {
	const n = 300
	if n >= g.r.FastPrefillFloor() {
		t.Fatalf("a %d-token prompt is at or past the fast floor %d", n, g.r.FastPrefillFloor())
	}
	p := g.toks(n, 51)
	a0 := g.r.fastAttnLaunches
	batched := g.fill(0, p, 0, true)
	if g.r.fastAttnLaunches != a0 {
		t.Fatal("the one-call prefill launched the fused kernels: it is not the exact class")
	}
	decoded := g.fill(1, p, 0, false)
	ka, kb := g.rows(0, n), g.rows(1, n)
	for i, b := range g.bufs() {
		if !bytes.Equal(ka[i], kb[i]) {
			first := 0
			for first < n && bytes.Equal(ka[i][first*b.row:(first+1)*b.row], kb[i][first*b.row:(first+1)*b.row]) {
				first++
			}
			t.Errorf("layer %d %s: the one-call prefill and decode differ, first at row %d", b.layer, kv2KV(b.v), first)
		}
	}
	if j, ok := sameBits(batched[0], decoded[len(decoded)-1]); !ok {
		t.Errorf("the last logits differ (first at %d)", j)
	}
}

func TestKV2_exactPrefillIsDecode_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			kv2ExactIsDecode(t, kv2Load(t, dir, decoder.Options{ResidentKVSlots: 2, ResidentContext: 1024, KVPrecision: kv}))
		})
	}
}

func TestKV2_exactPrefillIsDecode_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			kv2ExactIsDecode(t, kv2Load(t, path, decoder.Options{ResidentKVSlots: 2, ResidentContext: 1024, KVPrecision: kv}))
		})
	}
}
