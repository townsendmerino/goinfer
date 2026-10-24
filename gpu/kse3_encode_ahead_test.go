//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// kse3Resident builds m's WebGPU resident on its own backend.
func kse3Resident(t *testing.T, m *decoder.Model) (*residentDecoder, func()) {
	t.Helper()
	b, err := newWebGPUBackend("kse3")
	if err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	rf, ok, err := b.BuildResident(m)
	if err != nil || !ok {
		b.Close()
		t.Fatalf("BuildResident: ok %v, %v", ok, err)
	}
	rd, isRD := rf.(*residentDecoder)
	if !isRD {
		t.Fatalf("resident %T", rf)
	}
	return rd, func() { rf.Close(); b.Close() }
}

// kse3Plant replaces the runner's pending buffer with one recorded without the logits copy (the plant).
func kse3Plant(t *testing.T, r *DecodeRunner) {
	if r.pending == nil {
		return
	}
	r.dropPending()
	enc, _ := r.c.device.TryCreateCommandEncoder(nil)
	pass := enc.BeginComputePass(nil)
	r.record(pass)
	pass.TryEnd()
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	enc.Release()
	if err != nil {
		t.Fatal(err)
	}
	r.pending = cmd
}

// kse3Sequence runs a fixed mix on rd: 8 Forward steps, a 4-row ForwardN, 2 ForwardNoLogits, 4 Forward; then, with a second
// KV slot, 4 Forward on slot 1 and 4 more on slot 0. It returns every logit row in order.
func kse3Sequence(t *testing.T, m *decoder.Model, rd *residentDecoder, off, plant bool) [][]float32 {
	t.Helper()
	prev := encodeAheadOffForTest
	encodeAheadOffForTest = off
	defer func() { encodeAheadOffForTest = prev }()
	_, _, _, _, _, _, V := m.Dims()
	emb := func(k int) []float32 { return m.EmbedResidentForTest((k*7919 + 13) % min(V, 1000)) }
	var out [][]float32
	fwd := func(pos int) {
		lg, err := rd.Forward(emb(pos), pos)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
		if plant {
			kse3Plant(t, rd.runner)
		}
	}
	for pos := range 8 {
		fwd(pos)
	}
	rows, err := rd.ForwardN([][]float32{emb(8), emb(9), emb(10), emb(11)}, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		out = append(out, slices.Clone(r))
	}
	for pos := 12; pos < 14; pos++ {
		if err := rd.ForwardNoLogits(emb(pos), pos); err != nil {
			t.Fatal(err)
		}
	}
	for pos := 14; pos < 18; pos++ {
		fwd(pos)
	}
	if rd.KVSlots() > 1 {
		if err := rd.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		for pos := range 4 {
			fwd(pos)
		}
		if err := rd.UseKVSlot(0); err != nil {
			t.Fatal(err)
		}
		for pos := 18; pos < 22; pos++ {
			fwd(pos)
		}
	}
	return out
}

// TestKSE3_encodeAheadIdentity is E3-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-E3 gates"): llama-tiny (and, with
// GOINFER_HEAVY_TESTS=1, the 1.5B's .gguf) at int8int8 on WebGPU with two KV slots, kse3Sequence with encode-ahead on against
// off on one resident; every logit byte-equal. GOINFER_KSE3_PLANT=1 swaps in a pending buffer without the logits copy and
// expects red.
func TestKSE3_encodeAheadIdentity(t *testing.T) {
	newOrSkipHW(t).Close()
	paths := []string{"../testdata/llama-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		paths = append(paths, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	}
	plant := os.Getenv("GOINFER_KSE3_PLANT") != ""
	for _, p := range paths {
		m, err := decoder.Load(p, decoder.Options{Quant: "int8int8", ResidentKVSlots: 2})
		if err != nil {
			t.Fatalf("load %s: %v", p, err)
		}
		rd, closeRD := kse3Resident(t, m)
		off := kse3Sequence(t, m, rd, true, false)
		on := kse3Sequence(t, m, rd, false, plant)
		slots := rd.KVSlots()
		closeRD()
		m.Close()
		d := 0
		for s := range off {
			for i := range off[s] {
				if math.Float32bits(off[s][i]) != math.Float32bits(on[s][i]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[E3-G1] %s (%d KV slots, %d logit rows, plant %v): %d logits differ, encode-ahead on against off\n", p, slots, len(off), plant, d)
		if d != 0 {
			t.Errorf("%s: %d logits differ", p, d)
		}
	}
}

// TestKSE3_decodeSpeed is E3-G2: the 1.5B's .gguf at int8int8 on WebGPU, encode-ahead on against off on one resident, 128
// greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each, each pair's start load printed.
// A timed run: hold the timing lock (GOINFER_KSE3_TIMING=1).
func TestKSE3_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSE3_TIMING") == "" {
		t.Skip("set GOINFER_KSE3_TIMING=1 (a timed run under the timing lock)")
	}
	newOrSkipHW(t).Close()
	m, err := decoder.Load(os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rd, closeRD := kse3Resident(t, m)
	defer closeRD()
	const prompt, gen = 32, 128
	run := func(off bool) float64 {
		encodeAheadOffForTest = off
		defer func() { encodeAheadOffForTest = false }()
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
		run(true)
		run(false)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var off, on float64
		if i%2 == 0 {
			off, on = run(true), run(false)
		} else {
			on, off = run(false), run(true)
		}
		ratios = append(ratios, on/off)
		if on < off {
			below++
		}
		fmt.Fprintf(os.Stderr, "[E3-G2] pair %d (load %s): off %.3f ms/token, on %.3f ms/token, on/off %.4f\n", i, strings.TrimSpace(string(la)), off*1e3, on*1e3, on/off)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[E3-G2] median on/off %.4f, %d of 10 pairs below 1\n", (ratios[4]+ratios[5])/2, below)
}
