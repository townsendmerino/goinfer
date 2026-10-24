//go:build gpu

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// kse2Run compiles code and dispatches it over wg workgroups with the bindings in order: ins (read storage, as words), then
// outs (read-write storage of the given word counts), then the uniform. It returns each out's words.
func kse2Run(t *testing.T, c *Context, label, code string, ins [][]uint32, outs []int, uni []uint32, wg uint32) [][]uint32 {
	t.Helper()
	_, pl, lay, err := c.mkPipeline(label, code)
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
		b, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "kse2-in", Contents: wgpu.ToBytes(in), Usage: wgpu.BufferUsageStorage})
		if err != nil {
			t.Fatal(err)
		}
		add(b)
	}
	var outB []*wgpu.Buffer
	for _, n := range outs {
		b := c.zbuf(max(n, 1))
		outB = append(outB, b)
		add(b)
	}
	add(c.ubuf(uni))
	defer func() {
		for _, b := range bufs {
			b.Release()
		}
	}()
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: lay, Entries: entries})
	if err != nil {
		t.Fatalf("%s bind group: %v", label, err)
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(wg, 1, 1)
	pass.TryEnd()
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
	c.queue.Submit(cmd)
	cmd.Release()
	enc.Release()
	var res [][]uint32
	for i, b := range outB {
		res = append(res, c.readWords(b, max(outs[i], 1)))
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
	widen := widenQuant
	if os.Getenv("GOINFER_KSE2_PLANT") != "" {
		widen = func(label, src string) (string, error) {
			s, err := widenQuant(label, src)
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
		{"rmsnormQuant", rmsnormQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 3), f32w(n, 1)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), eps, 1, uint32(np)} }},
		{"swigluQuant", swigluQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2), f32w(n, 2)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), uint32(np), 0, 0} }},
		{"gegluQuant", gegluQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2), f32w(n, 2)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), uint32(np), 0, 0} }},
		{"relu2Quant", relu2QuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2)} },
			func(n, np int) []uint32 { return []uint32{uint32(n), uint32(np), 0, 0} }},
		{"gptossGluQuant", gptossGluQuantWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 2), f32w(n, 2), f32w(2*n, 0.5), {0}} },
			func(n, np int) []uint32 {
				return []uint32{uint32(n), uint32(np), 0, 1, math.Float32bits(1.702), math.Float32bits(7), 0, 0}
			}},
		{"quantizeRowsInt8", quantizeShaderWGSL, func(n int) [][]uint32 { return [][]uint32{f32w(n, 3)} },
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
		c.quantNarrow = true
		narrowSrc, _, err := c.fusedRowsSource(fusedRowsKind(kind))
		c.quantNarrow = false
		wideSrc, _, err2 := c.fusedRowsSource(fusedRowsKind(kind))
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
	prev := quantNarrowForTest.Load()
	quantNarrowForTest.Store(narrow)
	defer quantNarrowForTest.Store(prev)
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
	if !isRD || rd.c.quantNarrow != narrow {
		rf.Close()
		b.Close()
		t.Fatalf("resident %T, narrow %v: the switch did not reach its Context", rf, narrow)
	}
	return rd, func() { rf.Close(); b.Close() }
}

// kse2Decode runs n greedy decode steps from token 13 and returns every step's logits.
func kse2Decode(t *testing.T, m *decoder.Model, rd *residentDecoder, n int) [][]float32 {
	t.Helper()
	var out [][]float32
	tok := 13
	for pos := range n {
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
// a narrow and a wide resident: a 64-row PrefillLast (the widened row forms) and 32 greedy decode steps, every logit byte-equal.
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
				return nil, kse2Decode(t, m, rd, 32)
			}
			if err != nil {
				t.Fatalf("prefill: %v", err)
			}
			return slices.Clone(lg), kse2Decode(t, m, rd, 32)
		}
		preN, narrow := arm(true)
		preW, wide := arm(false)
		m.Close()
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

// TestKSE2_decodeSpeed is E2-G2 and its re-run's A/A control (docs/tasks/task-kernel-survey-2026-10.md, "E2-G2 re-run"): the
// 1.5B's .gguf at int8int8 on WebGPU, two residents in one process, 128 greedy decode tokens after a 32-token prompt, A B B A
// over 10 valid pairs after 2 warm runs each. GOINFER_KSE2_AA=1 makes both residents narrow (the control). It refuses to run
// outside the timing lock, records each pair's start load and reclaimable memory, voids and re-runs (up to 10 times) a pair
// started above half the cores' load or under 6 GB reclaimable, and fails on the bar: the A/B's median wide/narrow over 0.98
// or fewer than 9 of 10 faster; the control's median outside 1 ± 0.02.
func TestKSE2_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSE2_TIMING") == "" {
		t.Skip("set GOINFER_KSE2_TIMING=1 (a timed run under the timing lock)")
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	aa := os.Getenv("GOINFER_KSE2_AA") != ""
	newOrSkipHW(t).Close()
	m, err := decoder.Load(os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	aRD, aClose := kse2Resident(t, m, true)
	defer aClose()
	bRD, bClose := kse2Resident(t, m, aa)
	defer bClose()
	const prompt, gen = 32, 128
	run := func(rd *residentDecoder) float64 {
		for pos := range prompt {
			if _, err := rd.Forward(m.EmbedResidentForTest((pos*7919+13)%1000), pos); err != nil {
				t.Fatal(err)
			}
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
		return time.Since(t0).Seconds() / gen
	}
	for range 2 {
		run(aRD)
		run(bRD)
	}
	label := "A/B narrow, wide"
	if aa {
		label = "A/A narrow, narrow"
	}
	maxLoad := float64(runtime.NumCPU()) / 2
	var ratios []float64
	faster, voids := 0, 0
	for i := 0; len(ratios) < 10; i++ {
		var load1 [3]float64
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		fmt.Sscanf(strings.Trim(strings.TrimSpace(string(la)), "{}"), "%g %g %g", &load1[0], &load1[1], &load1[2])
		recl := int64(0)
		if r, ok := hostmem.Read(); ok {
			recl = r.Reclaimable
		}
		if load1[0] > maxLoad || recl < 6<<30 {
			voids++
			fmt.Fprintf(os.Stderr, "[E2-G2] VOID pair start %d (%s): load %.2f, reclaimable %.2f GiB\n", i, label, load1[0], float64(recl)/(1<<30))
			if voids > 10 {
				t.Fatalf("over 10 void pair starts")
			}
			time.Sleep(30 * time.Second)
			continue
		}
		var a, b float64
		if len(ratios)%2 == 0 {
			a, b = run(aRD), run(bRD)
		} else {
			b, a = run(bRD), run(aRD)
		}
		ratios = append(ratios, b/a)
		if b < a {
			faster++
		}
		fmt.Fprintf(os.Stderr, "[E2-G2] %s pair %d (load %.2f, reclaimable %.2f GiB): first %.3f ms/token, second %.3f ms/token, second/first %.4f\n",
			label, len(ratios)-1, load1[0], float64(recl)/(1<<30), a*1e3, b*1e3, b/a)
	}
	slices.Sort(ratios)
	med := (ratios[4] + ratios[5]) / 2
	fmt.Fprintf(os.Stderr, "[E2-G2] %s: median second/first %.4f, %d of 10 pairs below 1, %d void starts\n", label, med, faster, voids)
	if aa && (med < 0.98 || med > 1.02) {
		t.Errorf("the A/A control reads %.4f: the two-resident setup biases the reading", med)
	}
	if !aa && (med > 0.98 || faster < 9) {
		t.Errorf("E2-G2 below its bar: median %.4f, %d of 10 faster", med, faster)
	}
}
