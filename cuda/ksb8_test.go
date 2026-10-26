//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSB8_widthChooser is B8-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-B8 build"): prefillWidthFor on a table of rooms, then
// the knob and the learned cap over the chosen width through prefillBaseRows. The planted chooser ignores the margin and must pick
// a wider pass than the rule at the row whose room exists only inside the margin.
func TestKSB8_widthChooser(t *testing.T) {
	const MiB = int64(1 << 20)
	pass := func(rows int) int64 { return int64(rows) * 135 * MiB / 1000 } // ~1.5B: 138 / 276 MiB at 1,024 / 2,048
	margin := int64(ctxCapMarginBytes)
	for _, c := range []struct {
		name         string
		free, attach int64
		want         int
	}{
		{"room for 2,048", margin + pass(2048) + MiB, 0, 2048},
		{"exactly 2,048's room", margin + pass(2048), 0, 2048},
		{"room for 1,024 only", margin + pass(1024) + MiB, 0, 1024},
		{"room for 512 only", margin + pass(512) + MiB, 0, 512},
		{"less than 512's room", margin + pass(512)/2, 0, 512},
		{"nothing beyond the margin", margin, 0, 512},
		{"an attachment takes 2,048's room", margin + pass(2048) + MiB, pass(2048) - pass(1024), 1024},
	} {
		if got := prefillWidthFor(c.free, margin, c.attach, pass); got != c.want {
			t.Errorf("%s: width %d, want %d", c.name, got, c.want)
		}
	}
	planted := func(free, attach int64) int { return prefillWidthFor(free+margin, margin, attach, pass) } // the margin left out
	if got, rule := planted(margin+pass(1024)+MiB, 0), prefillWidthFor(margin+pass(1024)+MiB, margin, 0, pass); got == rule {
		t.Errorf("the planted chooser (no margin) picks %d, the same as the rule: the table cannot see the margin", got)
	}
	t.Setenv("GOINFER_PREFILL_CHUNK", "") // a bare resident reads its knobs from the environment
	r := &cudaResident{prefillWidth: 2048}
	if got := r.prefillBaseRows(); got != 2048 {
		t.Errorf("no knob, no learned cap: %d rows, want the chosen 2,048", got)
	}
	t.Setenv("GOINFER_PREFILL_CHUNK", "256")
	if got := r.prefillBaseRows(); got != 256 {
		t.Errorf("the knob at 256 over a chosen 2,048: %d rows, want 256", got)
	}
	t.Setenv("GOINFER_PREFILL_CHUNK", "not-a-number")
	if got := r.prefillBaseRows(); got != 2048 {
		t.Errorf("an unusable knob: %d rows, want the chosen 2,048", got)
	}
	t.Setenv("GOINFER_PREFILL_CHUNK", "")
	r.prefillChunkCap.Store(1024)
	if got := r.prefillBaseRows(); got != 1024 {
		t.Errorf("a learned cap of 1,024: %d rows, want 1,024", got)
	}
	r.prefillChunkCap.Store(0)
	r.prefillWidth = 0
	if got := r.prefillBaseRows(); got != prefillDefaultChunk {
		t.Errorf("before the choice: %d rows, want %d", got, prefillDefaultChunk)
	}
}

// ksb8Model is the 1.5B these gates load, skipping when it or a device is absent (the run script counts a skip as a failure).
func ksb8Model(t *testing.T) string {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	return path
}

func ksb8Embeddings(M, hidden int) [][]float32 {
	embs := make([][]float32, M)
	var s uint32 = 20261025
	for i := range embs {
		row := make([]float32, hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}
	return embs
}

// TestKSB8_bitsAcrossWidths is B8-G2: the 1.5B on CUDA prefills a 4,100-row prompt in passes of 512, 1,024, 2,048 and the build's
// default width, and in one pass; each run's seed logits and the 8 greedy decode ids after it must equal the single pass's.
func TestKSB8_bitsAcrossWidths(t *testing.T) {
	path := ksb8Model(t)
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 8192})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no CUDA resident: %s", m.ResidentDecline())
	}
	if batched, why := r.PrefillPath(); !batched {
		t.Fatalf("batched prefill declined: %s", why)
	}
	if r.prefillWidth != 2048 {
		t.Fatalf("the build chose %d rows, not 2,048: the default run would not be the registered arm", r.prefillWidth)
	}
	const M = 4100
	embs := ksb8Embeddings(M, r.hidden)
	run := func(chunk string) ([]float32, []int) {
		t.Helper()
		decoder.SetKnobEnvForTest(t, m, "GOINFER_PREFILL_CHUNK", chunk)
		r.Reset()
		t0 := time.Now()
		lg, err := r.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatalf("chunk %q: %v", chunk, err)
		}
		lg = slices.Clone(lg)
		fmt.Fprintf(os.Stderr, "[B8-G2] passes of %s rows (%d per pass): prefill %s\n", chunk, r.prefillPassRows(), time.Since(t0).Round(time.Millisecond))
		// The run must have used the width it names: no learned narrowing (an out-of-memory pass halves and remembers it, and
		// Reset does not clear that), and the pass rows the knob or the default asks for.
		if c := r.prefillChunkCap.Load(); c != 0 {
			t.Fatalf("chunk %q: a pass narrowed to %d rows after running out of memory, so this run is not the width it names", chunk, c)
		}
		wantRows := r.prefillWidth
		if n, err := strconv.Atoi(chunk); err == nil {
			wantRows = n
		}
		if got := r.prefillPassRows(); got != wantRows {
			t.Fatalf("chunk %q: passes of %d rows, want %d", chunk, got, wantRows)
		}
		ids, cur := make([]int, 0, 8), lg
		for k := range 8 {
			best := 0
			for i, v := range cur {
				if v > cur[best] {
					best = i
				}
			}
			ids = append(ids, best)
			if cur, err = r.Forward(m.EmbedResidentForTest(best), M+k); err != nil {
				t.Fatal(err)
			}
		}
		return lg, ids
	}
	refL, refI := run(strconv.Itoa(M + 100))
	for _, c := range []string{"512", "1024", "2048", ""} {
		l, ids := run(c)
		diff := 0
		for i := range refL {
			if math.Float32bits(l[i]) != math.Float32bits(refL[i]) {
				diff++
			}
		}
		name := c
		if c == "" {
			name = fmt.Sprintf("the default (%d)", r.prefillWidth)
		}
		fmt.Fprintf(os.Stderr, "[B8-G2] %s: %d of %d seed logits differ from the single pass; decode ids %v (single pass %v)\n", name, diff, len(refL), ids, refI)
		if diff != 0 || !slices.Equal(ids, refI) {
			t.Errorf("passes of %s: %d logits differ, ids %v against %v", name, diff, ids, refI)
		}
	}
}

// TestKSB8_buildPricesThePass is B8-G3: the 1.5B loaded with nothing attached, then with Options.ExtraResidentBytes leaving room for
// 1,024 rows but not 2,048, then for less than 1,024; the context is pinned so the KV plan, and the free VRAM it leaves, do not move.
// Each load's chosen width must be 2,048, 1,024 and 512, PrefillChunkRows must report it, and PrefillScratchBytes at that width must
// equal what a pass of that width allocates (prefillScratchSeenForTest). A prompt of width+4 rows, a remainder under
// attnFusedMinRows, must allocate no buffer larger than the width's pass does: prefillChunked takes the short tail's rows from the
// pass before it (prefillTailCut), so no pass is wider than the chosen width.
func TestKSB8_buildPricesThePass(t *testing.T) {
	path := ksb8Model(t)
	type built struct {
		width, reported int
		free            int64
		pass            func(int) int64
		seen            int64
	}
	load := func(extra int64) built {
		t.Helper()
		var free int64
		saved := cudaFreeVRAM
		cudaFreeVRAM = func(r *cudaResident) (uint64, error) { // the figure the chooser read, recorded as it read it
			f, err := saved(r)
			free = int64(f)
			return f, err
		}
		defer func() { cudaFreeVRAM = saved }()
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096, ExtraResidentBytes: extra})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("no CUDA resident (extra %d): %s", extra, m.ResidentDecline())
		}
		b := built{width: r.prefillWidth, reported: r.PrefillChunkRows(), free: free}
		b.pass = func(rows int) int64 { return r.PrefillScratchBytes(rows) + r.PrefillSpanScratchBytes(rows, r.ctxCap) }
		var widest int64
		prefillScratchSeenForTest = func(n int64) { b.seen += n + allocRoundSlack(n); widest = max(widest, n) }
		_, err = r.PrefillLast(context.Background(), ksb8Embeddings(b.width, r.hidden), 0)
		prefillScratchSeenForTest = nil
		if err != nil {
			t.Fatalf("a pass of %d rows: %v", b.width, err)
		}
		if want := r.PrefillScratchBytes(b.width); b.seen != want {
			t.Errorf("extra %d: PrefillScratchBytes(%d) is %d B, the pass allocated %d B", extra, b.width, want, b.seen)
		}
		var widestTail int64
		prefillScratchSeenForTest = func(n int64) { widestTail = max(widestTail, n) }
		r.Reset()
		_, err = r.PrefillLast(context.Background(), ksb8Embeddings(b.width+4, r.hidden), 0)
		prefillScratchSeenForTest = nil
		if err != nil {
			t.Fatalf("a prompt of %d rows: %v", b.width+4, err)
		}
		if widestTail > widest || r.prefillChunkCap.Load() != 0 {
			t.Errorf("extra %d: a %d-row prompt allocated a %d B buffer (learned cap %d), wider than the %d-row pass's largest, %d B",
				extra, b.width+4, widestTail, r.prefillChunkCap.Load(), b.width, widest)
		}
		fmt.Fprintf(os.Stderr, "[B8-G3] extra %d MiB: width %d (reported %d); %.0f MiB free at the choice; pass scratch at 512 / 1,024 / 2,048: %.0f / %.0f / %.0f MiB\n",
			extra>>20, b.width, b.reported, float64(free)/(1<<20), float64(b.pass(512))/(1<<20), float64(b.pass(1024))/(1<<20), float64(b.pass(2048))/(1<<20))
		return b
	}
	b0 := load(0)
	room := b0.free - ctxCapMarginBytes
	cases := []struct {
		extra int64
		want  int
	}{
		{0, 2048},
		{room - (b0.pass(1024)+b0.pass(2048))/2, 1024},
		{room - b0.pass(1024)/2, 512},
	}
	for i, c := range cases {
		b := b0
		if i > 0 {
			b = load(c.extra)
		}
		if b.width != c.want || b.reported != c.want {
			t.Errorf("extra %d MiB: width %d, reported %d; want %d", c.extra>>20, b.width, b.reported, c.want)
		}
	}
}

// TestKSB8_speed is B8-G4: one 1.5B resident, a 3,900-row prompt's PrefillLast at passes of 512 (the knob) against the build's
// default, A B B A over 10 pairs after 2 warm runs each. A pair whose start sees the 1-minute load above 8.0 is void and re-run.
// SHIP iff the median 512/default wall ratio is >= 1.05 with at least 9 of 10 pairs faster; PARK under 1.05 with 6-8 faster; KILL
// at a median <= 1.00. A timed run: hold the timing lock (GOINFER_KSB8_TIMING=1).
func TestKSB8_speed(t *testing.T) {
	if os.Getenv("GOINFER_KSB8_TIMING") == "" {
		t.Skip("set GOINFER_KSB8_TIMING=1 (a timed run under the timing lock)")
	}
	path := ksb8Model(t)
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no CUDA resident: %s", m.ResidentDecline())
	}
	if r.prefillWidth != 2048 {
		t.Fatalf("the build chose %d rows, not 2,048: the A/B would not compare the registered arms", r.prefillWidth)
	}
	embs := ksb8Embeddings(3900, r.hidden)
	run := func(narrow bool) float64 {
		chunk := ""
		if narrow {
			chunk = "512"
		}
		decoder.SetKnobEnvForTest(t, m, "GOINFER_PREFILL_CHUNK", chunk)
		r.Reset()
		t0 := time.Now()
		if _, err := r.PrefillLast(context.Background(), embs, 0); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds()
	}
	for range 2 {
		run(true)
		run(false)
	}
	var ratios []float64
	faster, voids := 0, 0
	for len(ratios) < 10 {
		if voids > 5 {
			t.Fatalf("%d void pairs: the box is busy", voids)
		}
		if la := kso1cLoad1(); la > 8.0 {
			voids++
			fmt.Fprintf(os.Stderr, "[B8-G4] pair %d VOID: load %.2f above 8.0; re-run\n", len(ratios), la)
			time.Sleep(30 * time.Second)
			continue
		}
		la, mem := kso1cLoad1(), kso1cMemAvailGB()
		var narrow, wide float64
		if len(ratios)%2 == 0 {
			narrow, wide = run(true), run(false)
		} else {
			wide, narrow = run(false), run(true)
		}
		ratios = append(ratios, narrow/wide)
		if wide < narrow {
			faster++
		}
		fmt.Fprintf(os.Stderr, "[B8-G4] pair %d (load %.2f, %.1f GB available): 512 rows %.1f ms, default %.1f ms, ratio %.4f\n",
			len(ratios)-1, la, mem, narrow*1e3, wide*1e3, narrow/wide)
	}
	s := slices.Clone(ratios)
	slices.Sort(s)
	med := (s[4] + s[5]) / 2
	verdict := "PARK"
	switch {
	case med <= 1.00:
		verdict = "KILL"
	case med >= 1.05 && faster >= 9:
		verdict = "SHIP"
	}
	fmt.Fprintf(os.Stderr, "[B8-G4] median 512/default %.4f, %d of 10 pairs faster at the default, %d void: %s\n", med, faster, voids, verdict)
	if verdict != "SHIP" {
		t.Errorf("B8-G4 %s: median %.4f, %d of 10 faster", verdict, med, faster)
	}
}

// TestKSB8_admissionPricesTheChosenPass is B8-G3's admission half, through the decoder: on the 1.5B at context 16,384, the
// per-request price (decoder residentPrefillScratch, through ResidentPrefillScratchForTest) of a prompt is one pass at the build's
// chosen width, the span bound at the prompt's length (PrefillSpanScratchBytes), and the prompt's host embeddings. 2,052 and 4,100
// leave a remainder under attnFusedMinRows past the width: the price is still one pass of the width, since prefillChunked takes
// the short tail's rows from the pass before it (TestKSB8_buildPricesThePass holds the execution side).
//
// The 1.5B's fused attention serves past attn_batched's bound, so its span scratch is zero at default settings. With the fused
// attention off, prompts of 13,000 and 16,380 tokens run every pass past the bound on the row kernels, at 2,048 rows and shorter
// spans before the last: each is prefilled, and the largest pass's allocations (prefillScratchSeenForTest, per pass by
// prefillPassDoneForTest) must not exceed the device part of admission's price, whose span term must be non-zero.
func TestKSB8_admissionPricesTheChosenPass(t *testing.T) {
	path := ksb8Model(t)
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 16384})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no CUDA resident: %s", m.ResidentDecline())
	}
	w := r.PrefillChunkRows()
	if w != r.prefillWidth {
		t.Fatalf("PrefillChunkRows %d, the chosen width %d", w, r.prefillWidth)
	}
	price := func(p int) (dev int64) {
		t.Helper()
		rows := min(p, w)
		dev = r.PrefillScratchBytes(rows) + r.PrefillSpanScratchBytes(rows, p)
		want := dev + int64(p)*4*int64(m.ResidentEmbedLen())
		if got, ok := m.ResidentPrefillScratchForTest(p); !ok || got != want {
			t.Errorf("a %d-token prompt: admission prices %d B (ok %v), want %d (a %d-row pass, %d B of span bound, the embeddings)", p, got, ok, want, rows, dev-r.PrefillScratchBytes(rows))
		}
		fmt.Fprintf(os.Stderr, "[B8-G3 admission] %d tokens, fused attention %v: %.1f MiB of device scratch priced (a %d-row pass, %.1f MiB span bound)\n",
			p, r.fastAttn, float64(dev)/(1<<20), rows, float64(r.PrefillSpanScratchBytes(rows, p))/(1<<20))
		return dev
	}
	for _, p := range []int{100, 2052, 3900, 4100, 12200, 13000, 14000, 15000, 16000} {
		price(p)
	}
	fast := r.fastAttn
	r.fastAttn = false
	defer func() { r.fastAttn = fast; prefillScratchSeenForTest, prefillPassDoneForTest = nil, nil }()
	for _, p := range []int{13000, 16380} {
		dev := price(p)
		if span := r.PrefillSpanScratchBytes(w, p); span == 0 {
			t.Errorf("a %d-token prompt with the fused attention off prices no span scratch at %d rows: the term is not exercised", p, w)
		}
		var cur, most int64
		passes := 0
		prefillScratchSeenForTest = func(n int64) { cur += n + allocRoundSlack(n) }
		prefillPassDoneForTest = func() { most, cur = max(most, cur), 0; passes++ }
		r.Reset()
		if _, err := r.PrefillLast(context.Background(), ksb8Embeddings(p, r.hidden), 0); err != nil {
			t.Fatalf("a %d-token prompt, fused attention off: %v", p, err)
		}
		prefillScratchSeenForTest, prefillPassDoneForTest = nil, nil
		fmt.Fprintf(os.Stderr, "[B8-G3 admission] %d tokens, fused attention off: %d passes, the largest allocated %.1f MiB against %.1f MiB priced\n",
			p, passes, float64(most)/(1<<20), float64(dev)/(1<<20))
		if passes < 2 || most == 0 || most > dev {
			t.Errorf("a %d-token prompt: %d passes, the largest allocated %d B, admission's device price %d B", p, passes, most, dev)
		}
	}
}
