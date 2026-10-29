//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// i8Passes is the cut the decoder makes of an n-token prompt with a block of B rows: the rows of each pass, in order.
// It restates residentPrefillSeed's rule so a test can say what the pass count and the scratch should have been.
func i8Passes(n, B int) []int {
	var rows []int
	from := 0
	for B > 0 && n-from >= B+8 {
		rows = append(rows, B)
		from += B
	}
	return append(rows, n-from)
}

// i8Block sets the forced block for one test.
func i8Block(t *testing.T, rows int) {
	t.Helper()
	prev := prefillBlockRowsForTest
	prefillBlockRowsForTest = rows
	t.Cleanup(func() { prefillBlockRowsForTest = prev })
}

// i8KV is the resident's state after a prompt of n rows: each attention layer's first n key and value rows, and each
// Gated-DeltaNet layer's conv window and recurrent state.
func i8KV(r *resident, n int) (out [][]uint32) {
	for l := range r.layers {
		L := &r.layers[l]
		if L.delta != nil {
			out = append(out, slices.Clone(L.delta.win.U32s()), slices.Clone(L.delta.state.U32s()))
			continue
		}
		o, w := r.kvHostOff(l, 4), n*L.geom.kvDim/2 // halves, read as words
		out = append(out, slices.Clone(r.kc[l].U32s()[o:o+w]), slices.Clone(r.vc[l].U32s()[o:o+w]))
	}
	return out
}

type i8Result struct {
	toks    []int
	lp      []float64
	kv      [][]uint32
	scratch int64
}

// i8Generate loads the model fresh, generates from prompt and returns what a block could change: the tokens, their
// log-probabilities, the resident's state, and the scratch the prefill allocated. check sees the resident before the run.
func i8Generate(t *testing.T, path string, opts decoder.Options, prompt []int, check func(a *metalResident)) i8Result {
	t.Helper()
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("%s is not Metal-resident: %s", path, m.ResidentDecline())
	}
	check(a)
	before := prefillScratchSeen.Load()
	toks, lp := mgGen(t, m, prompt, 4)
	return i8Result{toks, lp, i8KV(a.r, len(prompt)), prefillScratchSeen.Load() - before}
}

// i8Check is G-I8-1 and G-I8-3 for one model: the prompt generated with each forced block equals the one-pass
// generation in its tokens, log-probabilities and resident state, bit for bit; and each run's prefill allocated
// exactly what PrefillScratchBytes reports for its passes.
func i8Check(t *testing.T, path string, opts decoder.Options, prompt []int, blocks []int, want func(a *metalResident)) {
	t.Helper()
	var price func(rows []int) int64
	var floor int
	see := func(a *metalResident) {
		want(a)
		if a.PrefillBlockRows() <= 0 {
			t.Fatalf("this resident does not split its prefill; the test would compare one pass with one pass")
		}
		floor = a.PrefillKernelFloor()
		r := a.r
		price = func(rows []int) (n int64) {
			for _, m := range rows {
				n += r.prefillScratchBytes(m)
			}
			return n
		}
	}
	i8Block(t, 1<<30) // one pass, whatever the model's own block
	ref := i8Generate(t, path, opts, prompt, see)
	if ref.scratch != price([]int{len(prompt)}) || ref.scratch == 0 {
		t.Fatalf("the one-pass run allocated %d bytes of scratch; PrefillScratchBytes(%d) is %d", ref.scratch, len(prompt), price([]int{len(prompt)}))
	}
	for _, B := range blocks {
		prefillBlockRowsForTest = B
		got := i8Generate(t, path, opts, prompt, see)
		passes := i8Passes(len(prompt), max(B, floor))
		if len(passes) < 2 {
			t.Fatalf("block %d: one pass over %d tokens, so nothing was split", B, len(prompt))
		}
		if got.scratch != price(passes) {
			t.Errorf("block %d: the prefill allocated %d bytes of scratch, want %d for passes of %v rows", B, got.scratch, price(passes), passes)
		}
		if !slices.Equal(got.toks, ref.toks) {
			t.Errorf("block %d: tokens %v, one pass %v", B, got.toks, ref.toks)
		}
		if !slices.EqualFunc(got.lp, ref.lp, func(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) }) {
			t.Errorf("block %d: the log-probabilities differ from one pass (max |diff| %g)", B, mgMaxDiff(got.lp, ref.lp))
		}
		for i := range ref.kv {
			if !slices.Equal(got.kv[i], ref.kv[i]) {
				t.Errorf("block %d: resident state buffer %d differs from one pass", B, i)
				break
			}
		}
	}
}

func i8IDs(n, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 3) % vocab
	}
	return ids
}

// TestI8_blocksMatchOnePass_dense (G-I8-1, G-I8-3): the dense fixture, a 1,000-token prompt, blocks of 64, 100 and 256.
func TestI8_blocksMatchOnePass_dense(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	opts := decoder.Options{Backend: "metal", Quant: mc3TestQuant, ResidentContext: 2048}
	i8Check(t, mc3FixtureWriter(t, 4096), opts, i8IDs(1000, mfVocab), []int{64, 100, 256}, func(a *metalResident) {
		if a.r.moe != nil || a.r.dnet != nil {
			t.Fatal("the dense fixture built an MoE or a DeltaNet resident")
		}
	})
}

// TestI8_blocksMatchOnePass_MoE (G-I8-1, G-I8-3): the tiny MoE fixtures that take the batched pass, the last of them a
// Gated-DeltaNet hybrid. Their windows hold 64 to 100 positions, so the block is 16 rows, which the resident's
// kernel floor may raise.
func TestI8_blocksMatchOnePass_MoE(t *testing.T) {
	for _, fx := range []string{"../testdata/mixtral-tiny", "../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3", "../decoder/testdata/qwen3_5_moe-tiny"} {
		t.Run(filepath.Base(fx), func(t *testing.T) {
			if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			opts := decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024}
			probe, err := decoder.Load(fx, opts)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			_, _, _, _, _, _, vocab := probe.Dims()
			n := 100
			if a, ok := probe.ResidentForwardForTest().(*metalResident); ok {
				n = min(n, a.r.ctxCap) - 8
			}
			probe.Close()
			i8Check(t, fx, opts, i8IDs(n, vocab), []int{16}, func(a *metalResident) {
				if a.r.moe == nil || a.r.moe.paged {
					t.Fatalf("not a resident generic MoE on Metal")
				}
				if hybrid := filepath.Base(fx) == "qwen3_5_moe-tiny"; hybrid != (a.r.dnet != nil) {
					t.Fatalf("Gated-DeltaNet layers present: %v, want %v", a.r.dnet != nil, hybrid)
				}
			})
		})
	}
}

// TestI8_exactKernelModelIsSplit (KS-H1): Qwen3.5 0.8B (head dim 256) reports a block, and its prefill in blocks of 64 and
// 256 rows gives one pass's tokens, log-probabilities and resident state, as the fused-kernel models do, on both attention
// kernels head dim 256 can take: attention_prefill_steel_split, the default (exactAttnLayer), and the exact kernel, which
// prefillSteelAttnOff restores. At head dim ≤ 256 the exact and tiled kernels a block may choose are bit-identical.
func TestI8_exactKernelModelIsSplit(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen3.5-0.8b")
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	tk, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	ids := decoder.PrefillGateProseIDsForTest(t, tk, files[1], 600)[:600]
	opts := decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024}
	t.Run("split", func(t *testing.T) {
		i8Check(t, path, opts, ids, []int{64, 256}, func(a *metalResident) {
			if a.r.prefillExactAttn() {
				t.Fatalf("exact attention at head dim %d by default: the test would not exercise attention_prefill_steel_split", a.r.exactAttnHeadDim())
			}
		})
	})
	t.Run("exact", func(t *testing.T) {
		prev := prefillSteelAttnOff
		prefillSteelAttnOff = true
		defer func() { prefillSteelAttnOff = prev }()
		i8Check(t, path, opts, ids, []int{64, 256}, func(a *metalResident) {
			if !a.r.prefillExactAttn() || a.r.exactAttnHeadDim() > 256 {
				t.Fatalf("exact attention %v at head dim %d: the test would not exercise a blocked exact-kernel model", a.r.prefillExactAttn(), a.r.exactAttnHeadDim())
			}
		})
	})
}

// TestI8_blockRows: the block is the rows that fit prefillBlockScratch, a multiple of 64 and at least
// prefillBlockRowsMin, and 0 for a resident whose prompts do not all take the plain batched pass.
func TestI8_blockRows(t *testing.T) {
	_, r := mc3FixtureResident(t, 2, 2048)
	a := &metalResident{r: r, hidden: r.H}
	B := a.PrefillBlockRows()
	if B < prefillBlockRowsMin || B%64 != 0 {
		t.Fatalf("PrefillBlockRows = %d, want a multiple of 64 no less than %d", B, prefillBlockRowsMin)
	}
	if got := a.PrefillScratchBytes(B); got > prefillBlockScratch || a.PrefillScratchBytes(B+64) <= prefillBlockScratch {
		t.Errorf("a block of %d rows prices at %d bytes; want the most rows under %d", B, got, int64(prefillBlockScratch))
	}
	exact := &metalResident{r: r, hidden: r.H, exact: true}
	if got := exact.PrefillBlockRows(); got != 0 {
		t.Errorf("PrefillBlockRows = %d with --exact-prefill, want 0", got)
	}
	// An int8 cache takes the pass (P3b-2): it gets a block, and a row costs what it does at f16 plus the fixed dequantize
	// scratch (kvI8ScratchRows positions of K and V at f16).
	f16Price := a.PrefillScratchBytes(256)
	r.kvI8 = true
	if got := a.PrefillBlockRows(); got <= 0 {
		t.Errorf("PrefillBlockRows = %d with an int8 KV cache, want a block", got)
	}
	if got, want := a.PrefillScratchBytes(256)-f16Price, 4*int64(r.kvI8ScratchRows())*int64(r.prefillGeom().kvDim); got != want {
		t.Errorf("an int8 cache adds %d bytes of prefill scratch, want %d (the dequantize scratch)", got, want)
	}
	r.kvI8 = false
}

// TestI8_real is the real-size half, on Qwen2.5-Coder-1.5B from ~/models, chosen by GOINFER_I8_REAL:
//
//	identity (G-I8-2, G-I8-3): a 6,500-token prose prompt in blocks of 2,048 rows gives the one-pass logits bit for
//	bit, and each run allocates what PrefillScratchBytes reports.
//	speed (G-I8-5): a 12,288-token prompt, one pass against blocks of 2,048 rows, five alternating pairs after a
//	warm-up pair; it prints each pair and the median of blocked / whole, and reads the model's own block beside them.
func TestI8_real(t *testing.T) {
	mode := os.Getenv("GOINFER_I8_REAL")
	if mode != "identity" && mode != "speed" {
		t.Skip("set GOINFER_I8_REAL=identity or speed")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("no checkpoint at %s", path)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 16384})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	own := a.PrefillBlockRows()
	if own <= 0 {
		t.Fatal("this resident does not split its prefill")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	run := func(ids []int, block int) ([]float32, int64, time.Duration) {
		prefillBlockRowsForTest = block
		before, t0 := prefillScratchSeen.Load(), time.Now()
		lg, err := m.ResidentPrefillSeedForTest(ctx, ids)
		if err != nil {
			t.Fatalf("prefill with block %d: %v", block, err)
		}
		return slices.Clone(lg), prefillScratchSeen.Load() - before, time.Since(t0)
	}
	defer func(prev int) { prefillBlockRowsForTest = prev }(prefillBlockRowsForTest)
	price := func(n, B int) (sum int64) {
		for _, rows := range i8Passes(n, B) {
			sum += a.PrefillScratchBytes(rows)
		}
		return sum
	}
	const whole, forced = 1 << 30, 2048
	fmt.Fprintf(os.Stderr, "i8: %s, the model's own block is %d rows (%.1f KB a row)\n", filepath.Base(path), own, float64(a.PrefillScratchBytes(64))/64/1024)

	if mode == "identity" {
		const n = 6500
		ids := decoder.PrefillGateProseIDsForTest(t, tk, files[1], n)[:n]
		want, sw, dw := run(ids, whole)
		got, sb, db := run(ids, forced)
		fmt.Fprintf(os.Stderr, "i8 identity: one pass %.1fs, %d MB of scratch at once; blocks of %d (%v rows) %.1fs, %d MB at once\n",
			dw.Seconds(), a.PrefillScratchBytes(n)>>20, forced, i8Passes(n, forced), db.Seconds(), a.PrefillScratchBytes(forced)>>20)
		if sw != price(n, whole) || sb != price(n, forced) {
			t.Errorf("scratch allocated: one pass %d, blocks %d; PrefillScratchBytes says %d and %d", sw, sb, price(n, whole), price(n, forced))
		}
		diff := 0
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				diff++
			}
		}
		if diff != 0 || len(got) != len(want) {
			t.Errorf("%d of %d logits differ between blocks of %d and one pass", diff, len(want), forced)
		}
		return
	}

	const n, pairs = 12288, 5
	ids := decoder.PrefillGateProseIDsForTest(t, tk, files[1], n)[:n]
	var ratios, ownRatios []float64
	start := time.Now()
	for p := 0; p <= pairs; p++ { // pair 0 is the warm-up
		var dw, db time.Duration
		if p%2 == 0 {
			_, _, dw = run(ids, whole)
			_, _, db = run(ids, forced)
		} else {
			_, _, db = run(ids, forced)
			_, _, dw = run(ids, whole)
		}
		_, _, do := run(ids, 0)
		label := fmt.Sprintf("pair %d of %d", p, pairs)
		if p == 0 {
			label = "warm-up"
		} else {
			ratios = append(ratios, db.Seconds()/dw.Seconds())
			ownRatios = append(ownRatios, do.Seconds()/dw.Seconds())
		}
		fmt.Fprintf(os.Stderr, "i8 speed %s: one pass %.2fs, blocks of %d %.2fs (ratio %.4f), own block %d %.2fs (ratio %.4f); elapsed %.0fs\n",
			label, dw.Seconds(), forced, db.Seconds(), db.Seconds()/dw.Seconds(), own, do.Seconds(), do.Seconds()/dw.Seconds(), time.Since(start).Seconds())
	}
	sort.Float64s(ratios)
	sort.Float64s(ownRatios)
	fmt.Fprintf(os.Stderr, "i8 speed: %d tokens, median blocked / whole %.4f (pairs %v); own block, not graded, %.4f (%v)\n",
		n, ratios[len(ratios)/2], ratios, ownRatios[len(ownRatios)/2], ownRatios)
}
