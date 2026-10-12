//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

const prefillInt4Steps = 8

// w4MultiAuto leaves the pass's choice of int4 kernel to the row count, as production does.
const w4MultiAuto = w4MultiKind(-1)

// prefillInt4Run loads dir on WebGPU at int4 and returns two readings of the same n rows and the same
// prefillInt4Steps decode tokens after them: the row-by-row walk (the reference, whose own argmax chooses the tokens),
// and the batched pass followed by those tokens through the runner. Each reading is the last row's logits and every
// decode step's. kind pins the multi-row int4 kernel the pass runs; w4MultiAuto pins none.
func prefillInt4Run(t *testing.T, dir string, opts decoder.Options, n int, kind w4MultiKind) (want, got [][]float32) {
	t.Helper()
	return prefillRowsRun(t, dir, opts, n, kind, nil)
}

// prefillRowsRun is prefillInt4Run with the rows chosen by mkRows (nil: the embeddings of random tokens).
func prefillRowsRun(t *testing.T, dir string, opts decoder.Options, n int, kind w4MultiKind, mkRows func(m *decoder.Model, rng *rand.Rand, n int) [][]float32) (want, got [][]float32) {
	t.Helper()
	opts.Backend = "webgpu"
	if opts.Quant == "" {
		opts.Quant = "int4"
	}
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("webgpu declined %s", dir)
	}
	rd, ok := rf.(*residentDecoder)
	if !ok {
		t.Fatalf("%s: the resident is a %T", dir, rf)
	}
	rd.c.w4MultiPin = int(kind) + 1
	if b := rd.c.Backend(); !w4MultiVerified[b] {
		// This run is the backend's check: open the pass to it for the test's duration.
		t.Logf("backend %q is not in w4MultiVerified; a pass here is the evidence for adding it", b)
		w4MultiVerified[b] = true
		defer delete(w4MultiVerified, b)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	rng := rand.New(rand.NewSource(int64(n)*31 + 5))
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = m.EmbedResidentForTest(rng.Intn(vocab - 1))
	}
	if mkRows != nil {
		rows = mkRows(m, rng, n)
	}
	read := func(prefill func() ([]float32, error), toks []int) ([][]float32, []int) {
		rf.Reset()
		last, err := prefill()
		if err != nil {
			t.Fatalf("%s, %d rows: %v", dir, n, err)
		}
		out := [][]float32{append([]float32(nil), last...)}
		chosen := make([]int, 0, prefillInt4Steps)
		tok := argmax(last)
		for i := range prefillInt4Steps {
			if toks != nil {
				tok = toks[i]
			}
			chosen = append(chosen, tok)
			l, err := rf.Forward(m.EmbedResidentForTest(tok), n+i)
			if err != nil {
				t.Fatalf("%s, decode step %d: %v", dir, i, err)
			}
			out = append(out, append([]float32(nil), l...))
			tok = argmax(l)
		}
		return out, chosen
	}
	want, toks := read(func() ([]float32, error) {
		var last []float32
		for i, row := range rows {
			l, err := rf.Forward(row, i)
			if err != nil {
				return nil, err
			}
			last = l
		}
		return last, nil
	}, nil)
	// The pass itself, not PrefillLast: PrefillLast answers a decline on an E-model with the walk, which would compare
	// the walk with itself.
	got, _ = read(func() ([]float32, error) {
		l, err := rd.prefillBlocks(context.Background(), rows, 0, nil)
		applySoftcap(l, rd.finalSoftcap)
		return l, err
	}, toks)
	return want, got
}

// prefillInt4Grade grades got against the walk: every logit of every reading equal bit for bit. It returns "" on a
// pass.
func prefillInt4Grade(want, got [][]float32) string {
	for s := range want {
		w, g := want[s], got[s]
		for j := range w {
			if math.Float32bits(w[j]) != math.Float32bits(g[j]) {
				return fmt.Sprintf("reading %d: logit %d is %g, the walk's is %g", s, j, g[j], w[j])
			}
		}
	}
	return ""
}

var prefillInt4Fixtures = []struct {
	name, dir string
	opts      decoder.Options
	rows      []int
}{
	{"plain", "llama-tiny", decoder.Options{}, []int{7, 40, 70}},
	{"qk-norm", "qwen3asr-tiny", decoder.Options{}, []int{7, 40, 70}},
	{"qkv-bias", "qwen25vl-tiny", decoder.Options{}, []int{7, 40, 70}},
	{"gated-gelu", "gemma1-tiny", decoder.Options{}, []int{7, 40}}, // its context is 64 positions
	{"kv-f16", "llama-tiny", decoder.Options{KVPrecision: "f16"}, []int{7, 40, 70}},
}

// G-S12.4b-2a (docs/tasks/task-multimodal-support-2026-10.md): the batched pass at int4 against the row-by-row walk,
// the last row and the next 8 decode steps from the prefilled cache, at row counts on both sides of the row-shaped
// kernel's 16-row block and the tiled kernel's 64-row tile.
func TestPrefillInt4_matchesWalk(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	for _, f := range prefillInt4Fixtures {
		for _, n := range f.rows {
			for _, kind := range []w4MultiKind{w4MultiRows, w4MultiTiled} {
				t.Run(fmt.Sprintf("%s/%d-rows/%s", f.name, n, kind), func(t *testing.T) {
					opts := f.opts
					opts.ResidentContext = 96
					want, got := prefillInt4Run(t, filepath.Join("..", "testdata", f.dir), opts, n, kind)
					if why := prefillInt4Grade(want, got); why != "" {
						t.Error(why)
					}
				})
			}
		}
	}
}

// A prompt longer than one block of the pass goes through in several, and must still give the walk's bits.
func TestPrefillInt4_blocks(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	defer func(old int) { prefillBlockRows = old }(prefillBlockRows)
	prefillBlockRows = 24
	for _, n := range []int{24, 25, 70} {
		want, got := prefillInt4Run(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{ResidentContext: 96}, n, w4MultiAuto)
		if why := prefillInt4Grade(want, got); why != "" {
			t.Errorf("%d rows in blocks of %d: %s", n, prefillBlockRows, why)
		}
	}
}

// The pass through its production caller: Generate on an int4 model must give the same tokens with the batched
// prefill and with the row-by-row loop (GOINFER_BATCHED_PREFILL=0), and the batched arm must have run the pass.
func TestPrefillInt4_generate(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	const steps = 12
	run := func(batched string) (toks []int, passes int64) {
		t.Setenv("GOINFER_BATCHED_PREFILL", batched)
		m, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 96})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok {
			t.Fatalf("the resident is a %T", m.ResidentForwardForTest())
		}
		if b := rd.c.Backend(); !w4MultiVerified[b] {
			w4MultiVerified[b] = true
			defer delete(w4MultiVerified, b)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		rng := rand.New(rand.NewSource(11))
		prompt := make([]int, 40)
		for i := range prompt {
			prompt[i] = rng.Intn(vocab - 1)
		}
		before := prefillSharedQuants.Load()
		out, gen := m.Generate(context.Background(), prompt, steps, decoder.SamplingParams{Temperature: 0})
		for tok := range out {
			toks = append(toks, tok)
		}
		if err := gen.Err(); err != nil {
			t.Fatalf("generate (GOINFER_BATCHED_PREFILL=%q): %v", batched, err)
		}
		return toks, prefillSharedQuants.Load() - before
	}
	walk, walkPasses := run("0")
	pass, passPasses := run("")
	if walkPasses != 0 {
		t.Errorf("the row-by-row arm ran the batched pass (%d shared quantizations)", walkPasses)
	}
	if passPasses == 0 {
		t.Fatal("the batched arm never ran the pass: the two arms are the same path")
	}
	if len(walk) == 0 || fmt.Sprint(walk) != fmt.Sprint(pass) {
		t.Errorf("tokens differ:\n  row by row %v\n  batched    %v", walk, pass)
	}
}

// The planted defects of G-S12.4b-2a: each must turn the gate red under both kernels.
func TestPrefillInt4_plantedDefects(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	defer func() { prefillDefectForTest = 0 }()
	names := map[int]string{1: "rows one position late", 2: "each row's K/V written one slot late", 3: "the q projection's scales from the k projection"}
	for d := 1; d <= 3; d++ {
		for _, kind := range []w4MultiKind{w4MultiRows, w4MultiTiled} {
			prefillDefectForTest = d
			want, got := prefillInt4Run(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{ResidentContext: 96}, 40, kind)
			prefillDefectForTest = 0
			why := prefillInt4Grade(want, got)
			if why == "" {
				t.Errorf("%s, %s: the gate stayed green", kind, names[d])
				continue
			}
			t.Logf("%s, %s: red (%s)", kind, names[d], why)
		}
	}
}

// prefillInt4RealPath is the real plain dense int4 checkpoint the heavy readings run on.
func prefillInt4RealPath(t *testing.T) string {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	path := os.Getenv("GOINFER_RESIDENT_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3-vl-2b-instruct.int4.e4h.webgpu.v16.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	return path
}

// prefillRowCounts is def, or GOINFER_PREFILL_ROWS's comma-separated counts when it is set.
func prefillRowCounts(t *testing.T, def ...int) []int {
	t.Helper()
	v := os.Getenv("GOINFER_PREFILL_ROWS")
	if v == "" {
		return def
	}
	var counts []int
	for _, f := range strings.Split(v, ",") {
		n, err := strconv.Atoi(f)
		if err != nil {
			t.Fatalf("GOINFER_PREFILL_ROWS: %v", err)
		}
		counts = append(counts, n)
	}
	return counts
}

// G-S12.4b-2b and G-S12.4b-3c: the tiny gates' reading on a real int4 checkpoint, at 64 and 300 rows (or
// GOINFER_PREFILL_ROWS, to cross a real model's sliding window).
func TestPrefillInt4_real(t *testing.T) {
	path := prefillInt4RealPath(t)
	for _, n := range prefillRowCounts(t, 64, 300) {
		for _, kind := range []w4MultiKind{w4MultiRows, w4MultiTiled} {
			want, got := prefillInt4Run(t, path, decoder.Options{}, n, kind)
			why := prefillInt4Grade(want, got)
			fmt.Fprintf(os.Stderr, "REAL %s, %d rows, %s: %s\n", filepath.Base(path), n, kind, map[bool]string{true: "pass", false: why}[why == ""])
			if why != "" {
				t.Errorf("%d rows, %s: %s", n, kind, why)
			}
		}
	}
}

// G-S12.4b-1b and G-S12.4b-2c, in process: the batched pass under each multi-row int4 kernel, the row-by-row walk (the
// production loop: no logits read until the last row), and the route an image turn takes today, the CPU's batched
// prefill of the same tokens uploaded to the resident. At 256 and 1,024 rows: one discarded pass of each arm, then
// five rounds in alternating order (GOINFER_PREFILL_ROWS overrides the row counts; GOINFER_PREFILL_NO_CPU=1 drops the
// CPU arm, which takes minutes a round on a Gemma 4 E-model). It reports the per-round ratios and their medians and
// grades nothing.
func TestPrefillInt4_speed(t *testing.T) {
	path := prefillInt4RealPath(t)
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("the resident is a %T", m.ResidentForwardForTest())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	rng := rand.New(rand.NewSource(3))
	if os.Getenv("GOINFER_PREFILL_ATTN_PER_ROW") == "1" { // a probe of what the once-a-row attention dispatches cost
		prefillAttnPerRow = true
		defer func() { prefillAttnPerRow = false }()
	}
	var ids []int // the tokens rows embed, for the CPU arm
	pass := func(kind w4MultiKind) func([][]float32) error {
		return func(rows [][]float32) error {
			rd.c.w4MultiPin = int(kind) + 1
			_, err := rd.prefillBlocks(context.Background(), rows, 0, nil) // the pass itself: PrefillLast walks an E-model's rows on a decline
			return err
		}
	}
	arms := []struct {
		name string
		run  func([][]float32) error
	}{
		{"rows", pass(w4MultiRows)},
		{"tiled", pass(w4MultiTiled)},
		{"walk", func(rows [][]float32) error {
			for i, row := range rows[:len(rows)-1] {
				if err := rd.ForwardNoLogits(row, i); err != nil {
					return err
				}
			}
			_, err := rd.Forward(rows[len(rows)-1], len(rows)-1)
			return err
		}},
		{"cpu+upload", func([][]float32) error {
			cache := m.NewCache(len(ids) + 1)
			if _, err := m.PrefillLogitsForTest(context.Background(), ids, cache); err != nil {
				return err
			}
			return m.ResidentUploadPrefillForTest(cache)
		}},
	}
	const rounds = 5
	began := time.Now()
	for _, n := range prefillRowCounts(t, 256, 1024) {
		rows := make([][]float32, n)
		ids = make([]int, n)
		for i := range rows {
			ids[i] = rng.Intn(vocab - 1)
			rows[i] = m.EmbedResidentForTest(ids[i])
		}
		secs := make([][]float64, len(arms))
		timeArm := func(a int) (float64, error) {
			rd.Reset()
			t0 := time.Now()
			err := arms[a].run(rows)
			return time.Since(t0).Seconds(), err
		}
		on := make([]bool, len(arms)) // an arm the model has no route for is dropped at its discarded pass
		for a := range arms {
			if arms[a].name == "cpu+upload" && os.Getenv("GOINFER_PREFILL_NO_CPU") == "1" {
				continue
			}
			sec, err := timeArm(a)
			if err != nil {
				if arms[a].name != "cpu+upload" {
					t.Fatalf("%s, %d rows: %v", arms[a].name, n, err)
				}
				fmt.Fprintf(os.Stderr, "[%4.0fs] %d rows, %s: no such route for this model (%v)\n", time.Since(began).Seconds(), n, arms[a].name, err)
				continue
			}
			on[a] = true
			fmt.Fprintf(os.Stderr, "[%4.0fs] %d rows, discarded pass, %s: %.2f s\n", time.Since(began).Seconds(), n, arms[a].name, sec)
		}
		for r := range rounds {
			order := []int{0, 1, 2, 3}
			if r%2 == 1 {
				order = []int{3, 2, 1, 0}
			}
			line := fmt.Sprintf("[%4.0fs] %d rows, round %d of %d:", time.Since(began).Seconds(), n, r+1, rounds)
			for _, a := range order {
				if !on[a] {
					continue
				}
				sec, err := timeArm(a)
				if err != nil {
					t.Fatalf("%s, %d rows: %v", arms[a].name, n, err)
				}
				secs[a] = append(secs[a], sec)
			}
			for a := range arms {
				if on[a] {
					line += fmt.Sprintf(" %s %.2f s", arms[a].name, secs[a][r])
				}
			}
			fmt.Fprintln(os.Stderr, line)
		}
		ratio := func(a, b int) (rs []float64, med float64) {
			for r := range rounds {
				rs = append(rs, secs[a][r]/secs[b][r])
			}
			s := append([]float64(nil), rs...)
			sort.Float64s(s)
			return rs, s[rounds/2]
		}
		for _, pr := range [][2]int{{1, 0}, {0, 2}, {1, 2}, {1, 3}} {
			if !on[pr[0]] || !on[pr[1]] {
				continue
			}
			rs, med := ratio(pr[0], pr[1])
			fmt.Fprintf(os.Stderr, "RESULT %d rows, %s over %s: ratios %.3f, median %.3f\n", n, arms[pr[0]].name, arms[pr[1]].name, rs, med)
		}
	}
}

// The int8 pass on a real checkpoint: the same reading at int8int8, 64 and 300 rows, bit for bit. Its quantization
// runs through the same rows forms of decode's fused kernels as the int4 pass's.
func TestPrefillInt8_real(t *testing.T) {
	prefillInt4RealPath(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no checkpoint at %s: %v", dir, err)
	}
	for _, n := range []int{64, 300} {
		want, got := prefillInt4Run(t, dir, decoder.Options{Quant: "int8int8"}, n, w4MultiRows)
		why := prefillInt4Grade(want, got)
		fmt.Fprintf(os.Stderr, "int8int8 %s, %d rows: %s\n", filepath.Base(dir), n, map[bool]string{true: "pass", false: why}[why == ""])
		if why != "" {
			t.Errorf("%d rows: %s", n, why)
		}
	}
}

// TestPrefillInt4_profile splits the batched pass's time by category (projections, attention, norms and rotary, K/V
// write) on a real checkpoint, at GOINFER_PREFILL_ROWS row counts (default 256 and 1,024), beside the pass's own wall
// time with profiling off. Profiling syncs at every category boundary, so the categories sum to more than that wall
// time; their shares are what it is for. One discarded pass first: the first one compiles pipelines.
func TestPrefillInt4_profile(t *testing.T) {
	path := prefillInt4RealPath(t)
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("the resident is a %T", m.ResidentForwardForTest())
	}
	_, nLayers, nH, nKV, hd, _, vocab := m.Dims()
	rng := rand.New(rand.NewSource(3))
	ctx := context.Background()
	fmt.Fprintf(os.Stderr, "PROFILE %s on %s: %d layers, %d heads of %d (%d KV heads), sliding window %d; key-split attention off: %v\n", filepath.Base(path), rd.c.Backend(), nLayers, nH, hd, nKV, rd.rm.slidingWindow, attnKeysDisabled)
	for _, n := range prefillRowCounts(t, 256, 1024) {
		rows := make([][]float32, n)
		for i := range rows {
			rows[i] = m.EmbedResidentForTest(rng.Intn(vocab - 1))
		}
		pass := func() float64 {
			rd.Reset()
			t0 := time.Now()
			if _, err := rd.prefillBlocks(ctx, rows, 0, nil); err != nil {
				t.Fatalf("%d rows: %v", n, err)
			}
			return time.Since(t0).Seconds()
		}
		pass()
		wall := pass()
		// The row-by-row walk, once, for the same rows: what the pass replaced, and what a decode step at this depth costs.
		rd.Reset()
		t0 := time.Now()
		for i, row := range rows[:n-1] {
			if err := rd.ForwardNoLogits(row, i); err != nil {
				t.Fatalf("walk, row %d: %v", i, err)
			}
		}
		t1 := time.Now()
		if _, err := rd.Forward(rows[n-1], n-1); err != nil {
			t.Fatal(err)
		}
		walk, lastStep := time.Since(t0).Seconds(), time.Since(t1).Seconds()
		rd.c.prefillProf = &prefillProf{}
		profiled := pass()
		p := *rd.c.prefillProf
		rd.c.prefillProf = nil
		sum := (p.gemm + p.attn + p.normsRope + p.kvWrite).Seconds()
		fmt.Fprintf(os.Stderr, "PROFILE %d rows: the walk %.2f s, its last step (a decode step at this depth) %.1f ms\n", n, walk, 1000*lastStep)
		fmt.Fprintf(os.Stderr, "PROFILE %d rows: wall %.2f s unprofiled, %.2f s profiled; projections %.2f s (%.0f%%), attention %.2f s (%.0f%%), norms and rotary %.2f s (%.0f%%), K/V write %.2f s (%.0f%%)\n",
			n, wall, profiled, p.gemm.Seconds(), 100*p.gemm.Seconds()/sum, p.attn.Seconds(), 100*p.attn.Seconds()/sum,
			p.normsRope.Seconds(), 100*p.normsRope.Seconds()/sum, p.kvWrite.Seconds(), 100*p.kvWrite.Seconds()/sum)
	}
}
