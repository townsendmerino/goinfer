//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math"
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// kse2Run compiles code and dispatches it over wg workgroups with the bindings in order: ins (read storage, as words), then
// outs (read-write storage of the given word counts), then the uniform. It returns each out's words.
func kse2Run(t *testing.T, c *Context, label, code string, ins [][]uint32, outs []int, uni []uint32, wg uint32) [][]uint32 {
	t.Helper()
	_, pl, lay, err := kernels.ContextMkPipeline(c, label, code)
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	var entries []wgpu.BindGroupEntry
	var bufs []*wgpu.Buffer
	add := func(b *wgpu.Buffer) {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(len(entries)), Buffer: b, Size: b.GetSize()})
		bufs = append(bufs, b)
	}
	for _, in := range ins {
		b, err := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "kse2-in", Contents: wgpu.ToBytes(in), Usage: wgpu.BufferUsageStorage})
		if err != nil {
			t.Fatal(err)
		}
		add(b)
	}
	var outB []*wgpu.Buffer
	for _, n := range outs {
		b := zbuf(c, max(n, 1))
		outB = append(outB, b)
		add(b)
	}
	add(ubuf(c, uni))
	defer func() {
		for _, b := range bufs {
			b.Release()
		}
	}()
	bg, err := (*kernels.ContextDevice(c)).TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: lay, Entries: entries})
	if err != nil {
		t.Fatalf("%s bind group: %v", label, err)
	}
	defer bg.Release()
	enc, _ := (*kernels.ContextDevice(c)).TryCreateCommandEncoder(nil)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(wg, 1, 1)
	pass.TryEnd()
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
	(*kernels.ContextQueue(c)).Submit(cmd)
	cmd.Release()
	enc.Release()
	var res [][]uint32
	for i, b := range outB {
		res = append(res, readWords(c, b, max(outs[i], 1)))
	}
	return res
}

// TestKSE2_widenedMatchesNarrow is E2-G1 (a) (docs/tasks/task-kernel-survey-2026-10.md, "KS-E2 gates"): each of the six
// single-workgroup quantizers widened to 256 lanes (widenQuant) against its 64-lane source, and the three prefill row forms
// built by a wide and a narrow Context, on random inputs at n = 1, 3, 255, 896, 4864, 8960 (rows M = 1, 3, 17). The packed int8
// words and the scales byte-equal. GOINFER_KSE2_PLANT=1 widens rmsnorm's sum of squares to all 256 lanes and expects red.
func TestKSE2_widenedMatchesNarrow(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	widen := kernels.WidenQuant
	if os.Getenv("GOINFER_KSE2_PLANT") != "" {
		widen = func(label, src string) (string, error) {
			s, err := kernels.WidenQuant(label, src)
			if err == nil && label == "rmsnormQuant" {
				s = replaceOnceT(t, s, "if (t < 64u) { for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; ss = ss + v*v; } }",
					"for (var i: u32 = t; i < p.h; i = i + 256u) { let v = src[i]; ss = ss + v*v; }")
				s = replaceOnceT(t, s, "    var stride: u32 = 32u;", "    var stride: u32 = 128u;")
			}
			return s, err
		}
	}
	rng := rand.New(rand.NewSource(83))
	f32w := func(n int, sd float64) []uint32 {
		v := make([]uint32, max(n, 1))
		for i := range n {
			v[i] = math.Float32bits(float32(rng.NormFloat64() * sd))
		}
		return v
	}
	total, cells := 0, 0
	compare := func(name string, a, b [][]uint32) {
		d := 0
		for i := range a {
			for j := range a[i] {
				if a[i][j] != b[i][j] {
					d++
				}
			}
		}
		cells++
		total += d
		if d != 0 {
			fmt.Fprintf(os.Stderr, "[E2-G1] %s: %d words differ\n", name, d)
			t.Errorf("%s: %d words differ between the widened and the 64-lane kernel", name, d)
		}
	}
	type shader struct {
		label, src string
		ins        func(n int) [][]uint32
		uni        func(n, np int) []uint32
	}
	eps := math.Float32bits(1e-6)
	shaders := []shader{
		{"rmsnormQuant", kernels.RmsnormQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 3), f32w(n, 1)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), eps, 1, uint32(np)} }},
		{"swigluQuant", kernels.SwigluQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2), f32w(n, 2)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), uint32(np), 0, 0} }},
		{"gegluQuant", kernels.GegluQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2), f32w(n, 2)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), uint32(np), 0, 0} }},
		{"relu2Quant", kernels.Relu2QuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), uint32(np), 0, 0} }},
		{"gptossGluQuant", kernels.GptossGluQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2), f32w(n, 2), f32w(2*n, 0.5), {0}} },
			func(n, np int) []uint32 {
				return []uint32{uint32(n), uint32(np), 0, 1, math.Float32bits(1.702), math.Float32bits(7), 0, 0}
			}},
		{"quantizeRowsInt8", kernels.QuantizeShaderWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 3)} },
			func(n, np int) []uint32 { return []uint32{1, uint32(n), uint32(np), 0} }},
	}
	for _, sh := range shaders {
		wide, err := widen(sh.label, sh.src)
		if err != nil {
			t.Fatalf("%s: %v", sh.label, err)
		}
		for _, n := range []int{1, 3, 255, 896, 4864, 8960} {
			np := (n + 3) / 4 * 4
			ins, uni := sh.ins(n), sh.uni(n, np)
			outs := []int{np / 4, 1}
			a := kse2Run(t, c, sh.label, sh.src, ins, outs, uni, 1)
			b := kse2Run(t, c, sh.label, wide, ins, outs, uni, 1)
			compare(fmt.Sprintf("%s n=%d", sh.label, n), a, b)
		}
	}
	// The prefill row forms, from a narrow and a wide Context's fusedRowsSource.
	for kind, name := range []string{"rmsnormQuantRows", "swigluQuantRows", "gegluQuantRows"} {
		(*kernels.ContextQuantNarrow(c)) = true
		narrowSrc, _, err := kernels.ContextFusedRowsSource(c, kernels.FusedRowsKind(kind))
		(*kernels.ContextQuantNarrow(c)) = false
		wideSrc, _, err2 := kernels.ContextFusedRowsSource(c, kernels.FusedRowsKind(kind))
		if err != nil || err2 != nil {
			t.Fatalf("%s: %v %v", name, err, err2)
		}
		if os.Getenv("GOINFER_KSE2_PLANT") != "" && kind == 0 {
			wideSrc = replaceOnceT(t, wideSrc, "if (t < 64u) { for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[rb + i]; ss = ss + v*v; } }",
				"for (var i: u32 = t; i < p.h; i = i + 256u) { let v = src[rb + i]; ss = ss + v*v; }")
			wideSrc = replaceOnceT(t, wideSrc, "    var stride: u32 = 32u;", "    var stride: u32 = 128u;")
		}
		for _, n := range []int{1, 3, 255, 896, 4864, 8960} {
			for _, M := range []int{1, 3, 17} {
				np := (n + 3) / 4 * 4
				var ins [][]uint32
				var uni []uint32
				if kind == 0 {
					ins, uni = [][]uint32{f32w(M*n, 3), f32w(n, 1)}, []uint32{uint32(n), eps, 1, uint32(np)}
				} else {
					ins, uni = [][]uint32{f32w(M*n, 2), f32w(M*n, 2)}, []uint32{uint32(n), uint32(np), 0, 0}
				}
				outs := []int{M * np / 4, M}
				a := kse2Run(t, c, name, narrowSrc, ins, outs, uni, uint32(M))
				b := kse2Run(t, c, name, wideSrc, ins, outs, uni, uint32(M))
				compare(fmt.Sprintf("%s n=%d M=%d", name, n, M), a, b)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[E2-G1] %d cells: %d words differ in all\n", cells, total)
}

func replaceOnceT(t *testing.T, s, old, new string) string {
	t.Helper()
	if n := strings.Count(s, old); n != 1 {
		t.Fatalf("plant site found %d times: %q", n, old)
	}
	return strings.Replace(s, old, new, 1)
}

// kse2Resident builds m's WebGPU resident on its own backend, its Context narrow (64-lane quantizers) or wide. The returned
// close releases both.
func kse2Resident(t *testing.T, m *decoder.Model, narrow bool) (*residentDecoder, func()) {
	t.Helper()
	prev := kernels.QuantNarrowForTest.Load()
	kernels.QuantNarrowForTest.Store(narrow)
	defer kernels.QuantNarrowForTest.Store(prev)
	b, err := newWebGPUBackend("kse2")
	if err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	rf, ok, err := b.BuildResident(m)
	if err != nil || !ok {
		b.Close()
		t.Fatalf("BuildResident: ok %v, %v", ok, err)
	}
	rd, isRD := rf.(*residentDecoder)
	if !isRD || (*kernels.ContextQuantNarrow(rd.c)) != narrow {
		rf.Close()
		b.Close()
		t.Fatalf("resident %T, narrow %v: the switch did not reach its Context", rf, narrow)
	}
	return rd, func() { rf.Close(); b.Close() }
}

// kse2Decode runs n greedy decode steps from token 13 at positions start.. and returns every step's logits.
func kse2Decode(t *testing.T, m *decoder.Model, rd *residentDecoder, start, n int) [][]float32 {
	t.Helper()
	var out [][]float32
	tok := 13
	for pos := start; pos < start+n; pos++ {
		lg, err := rd.Forward(m.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
		tok = 0
		for i, v := range lg {
			if v > lg[tok] {
				tok = i
			}
		}
	}
	return out
}

// TestKSE2_decodeIdentity is E2-G1 (b): llama-tiny (and, with GOINFER_HEAVY_TESTS=1, the 1.5B's .gguf) at int8int8 on WebGPU,
// a narrow and a wide resident: a 64-row PrefillLast (the widened row forms; llama-tiny's widths reach only one stride of
// them, the multi-stride loops are E2-G1 (a)'s) and 32 greedy decode steps attending over it, every logit byte-equal.
func TestKSE2_decodeIdentity(t *testing.T) {
	newOrSkipHW(t).Close()
	paths := []string{"../testdata/llama-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		paths = append(paths, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	}
	for _, p := range paths {
		m, err := decoder.Load(p, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load %s: %v", p, err)
		}
		// A 64-row prefill first (the widened row forms), then 32 decode steps after it.
		_, _, _, _, _, _, V := m.Dims()
		embs := make([][]float32, 64)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*7919 + 13) % min(V, 1000))
		}
		arm := func(narrow bool) (pre []float32, dec [][]float32) {
			rd, closeRD := kse2Resident(t, m, narrow)
			defer closeRD()
			lg, err := rd.PrefillLast(context.Background(), embs, 0)
			if err != nil && strings.Contains(err.Error(), "declines") {
				// The backend declines this family's batched prefill (the 1.5B's q/k/v bias on Metal): only decode is compared.
				fmt.Fprintf(os.Stderr, "[E2-G1] %s: prefill declined (%v); the row forms are not exercised on this model\n", p, err)
				return nil, kse2Decode(t, m, rd, 0, 32)
			}
			if err != nil {
				t.Fatalf("prefill: %v", err)
			}
			return slices.Clone(lg), kse2Decode(t, m, rd, len(embs), 32) // decode attends over the prefill's KV
		}
		preN, narrow := arm(true)
		preW, wide := arm(false)
		m.Close()
		if len(preN) != len(preW) || (strings.HasSuffix(p, "llama-tiny") && len(preN) == 0) {
			t.Errorf("%s: prefill logits %d and %d long (llama-tiny's prefill must be admitted)", p, len(preN), len(preW))
		}
		dp := 0
		for i := range preN {
			if math.Float32bits(preN[i]) != math.Float32bits(preW[i]) {
				dp++
			}
		}
		fmt.Fprintf(os.Stderr, "[E2-G1] %s: 64-row prefill, %d of %d last logits differ, wide against narrow\n", p, dp, len(preN))
		if dp != 0 {
			t.Errorf("%s: %d prefill logits differ", p, dp)
		}
		d := 0
		for s := range narrow {
			for i := range narrow[s] {
				if math.Float32bits(narrow[s][i]) != math.Float32bits(wide[s][i]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[E2-G1] %s: %d of %d logits differ over 32 steps, wide against narrow\n", p, d, 32*len(narrow[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ", p, d)
		}
	}
}

// TestKSE2_contextWiresEveryQuantizer: every Context entry point that builds a single-workgroup quantizer takes its source through
// quantSrc, so a wide Context compiles all nine (the six decode quantizers and the three prefill row forms) widened and a narrow
// one none. It calls each builder (ensureFuse, ensureQuantize, ensureRelu2, ensureGptOssGluQuant, ensureFusedRows) on a fresh
// Context and reads which labels quantSrc served (quantSrcHookForTest). TestKSE2_widenedMatchesNarrow holds the widened
// sources to the narrow ones' bits; this holds the builders to the widened sources.
func TestKSE2_contextWiresEveryQuantizer(t *testing.T) {
	want := []string{"rmsnormQuant", "swigluQuant", "gegluQuant", "quantizeRowsInt8", "relu2Quant", "gptossGluQuant",
		"rmsnormQuantRows", "swigluQuantRows", "gegluQuantRows"}
	for _, narrow := range []bool{false, true} {
		prevN := kernels.QuantNarrowForTest.Load()
		kernels.QuantNarrowForTest.Store(narrow)
		c := newOrSkipHW(t)
		kernels.QuantNarrowForTest.Store(prevN)
		seen := map[string]bool{}
		kernels.QuantSrcHookForTest = func(label string, wide bool) { seen[label] = wide }
		for name, f := range map[string]func() error{"ensureFuse": kernels.ContextEnsureFuseFunc(c), "ensureQuantize": kernels.ContextEnsureQuantizeFunc(c),
			"ensureRelu2": kernels.ContextEnsureRelu2Func(c), "ensureGptOssGluQuant": kernels.ContextEnsureGptOssGluQuantFunc(c), "ensureFusedRows": kernels.ContextEnsureFusedRowsFunc(c)} {
			if err := f(); err != nil {
				t.Fatalf("narrow %v: %s: %v", narrow, name, err)
			}
		}
		kernels.QuantSrcHookForTest = nil
		c.Close()
		for _, l := range want {
			wide, ok := seen[l]
			if !ok {
				t.Errorf("narrow %v: %s was built without quantSrc", narrow, l)
			} else if wide == narrow {
				t.Errorf("narrow %v: %s compiled wide = %v", narrow, l, wide)
			}
		}
		if len(seen) != len(want) {
			t.Errorf("narrow %v: quantSrc served %d labels %v, want the %d named", narrow, len(seen), seen, len(want))
		}
	}
}

// TestKSE2_decodeSpeed is E2-G2's re-run (docs/tasks/task-kernel-survey-2026-10.md, "E2-G2 re-run" and its amendments): the
// 1.5B's .gguf at int8int8 on WebGPU, one resident alive at a time (each arm builds its resident, warms it twice and times 128
// greedy decode tokens after a 32-token prompt), A B B A over 10 valid pairs: the A/A control (narrow against narrow) first, then
// the A/B (narrow against wide), in one process. It refuses to run outside the timing lock and under 6 GB reclaimable before its
// loads. voidWindow brackets each timed decode; a pair with either arm over the void rule's line is void and re-run, up to 10
// times per phase. It fails when the control's median is outside 1 ± 0.02 (and then does not run the A/B), or on the A/B's bar:
// median wide/narrow over 0.98 or fewer than 9 of 10 faster.
func TestKSE2_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSE2_TIMING") == "" {
		t.Skip("set GOINFER_KSE2_TIMING=1 (a timed run under the timing lock)")
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	// The void rule's memory term is read before the cell's loads: about 4 GB (the host model and a resident) plus 2 GB.
	if r, ok := probe.Read(); !ok || r.Reclaimable < 6<<30 {
		t.Fatalf("reclaimable %.2f GiB before the loads, under 6 GiB: the cell would be void", float64(r.Reclaimable)/(1<<30))
	}
	newOrSkipHW(t).Close()
	m, err := decoder.Load(os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const prompt, gen = 32, 128
	// once runs the 32-token prompt, then times the 128 decode tokens; with guard set, voidWindow brackets exactly the timed loop.
	once := func(rd *residentDecoder, guard bool) (sec float64, valid bool, note string) {
		for pos := range prompt {
			if _, err := rd.Forward(m.EmbedResidentForTest((pos*7919+13)%1000), pos); err != nil {
				t.Fatal(err)
			}
		}
		var end func() (bool, string)
		if guard {
			end = voidWindow(t)
		}
		tok := 13
		t0 := time.Now()
		for pos := prompt; pos < prompt+gen; pos++ {
			lg, err := rd.Forward(m.EmbedResidentForTest(tok), pos)
			if err != nil {
				t.Fatal(err)
			}
			tok = 0
			for i, v := range lg {
				if v > lg[tok] {
					tok = i
				}
			}
		}
		sec = time.Since(t0).Seconds() / gen
		if guard {
			valid, note = end()
		}
		return sec, valid, note
	}
	// runArm builds its arm's resident (one alive at a time: two together leave under the 2 GB margin on a 16 GB Mac), warms it
	// twice and times one guarded run.
	runArm := func(narrow bool) (float64, bool, string) {
		rd, closeRD := kse2Resident(t, m, narrow)
		defer closeRD()
		once(rd, false)
		once(rd, false)
		return once(rd, true)
	}
	// phase runs 10 valid A B B A pairs of arm a (narrow) against arm b and returns the median b/a and how many pairs b was faster.
	phase := func(label, nameA, nameB string, bNarrow bool) (float64, int) {
		var ratios []float64
		faster, voids := 0, 0
		for len(ratios) < 10 {
			var a, b float64
			var va, vb bool
			var na, nb, order string
			if len(ratios)%2 == 0 {
				a, va, na = runArm(true)
				b, vb, nb = runArm(bNarrow)
				order = nameA + " first"
			} else {
				b, vb, nb = runArm(bNarrow)
				a, va, na = runArm(true)
				order = nameB + " first"
			}
			if !va || !vb {
				voids++
				fmt.Fprintf(os.Stderr, "[E2-G2] VOID %s pair, %s (%s: %s; %s: %s)\n", label, order, nameA, na, nameB, nb)
				if voids > 10 {
					t.Fatalf("%s: over 10 void pairs", label)
				}
				time.Sleep(30 * time.Second)
				continue
			}
			ratios = append(ratios, b/a)
			if b < a {
				faster++
			}
			fmt.Fprintf(os.Stderr, "[E2-G2] %s pair %d, %s (%s: %s; %s: %s): %s %.3f ms/token, %s %.3f ms/token, %s/%s %.4f\n",
				label, len(ratios)-1, order, nameA, na, nameB, nb, nameA, a*1e3, nameB, b*1e3, nameB, nameA, b/a)
		}
		slices.Sort(ratios)
		med := (ratios[4] + ratios[5]) / 2
		fmt.Fprintf(os.Stderr, "[E2-G2] %s: median %s/%s %.4f, %d of 10 pairs below 1, %d void pairs\n", label, nameB, nameA, med, faster, voids)
		return med, faster
	}
	if med, _ := phase("A/A", "narrow A", "narrow B", true); med < 0.98 || med > 1.02 {
		t.Fatalf("the A/A control reads %.4f, outside 1 ± 0.02: the setup biases the reading, so the A/B is not run", med)
	}
	if med, faster := phase("A/B", "narrow", "wide", false); med > 0.98 || faster < 9 {
		t.Errorf("E2-G2 below its bar: median %.4f, %d of 10 faster", med, faster)
	}
}
