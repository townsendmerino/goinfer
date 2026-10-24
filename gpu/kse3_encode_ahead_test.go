//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/hostmem"
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

// kse3Sequence runs a fixed mix on a fresh resident: 8 Forward steps, a 4-row ForwardN, 2 ForwardNoLogits, 4 Forward; then,
// with a second KV slot, 4 Forward on slot 1 (its own inputs) and 4 more on slot 0. It returns every logit row in order.
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
			lg, err := rd.Forward(emb(pos+100), pos) // slot 1's own inputs, not slot 0's
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(lg))
			if plant {
				kse3Plant(t, rd.runner)
			}
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
// GOINFER_HEAVY_TESTS=1, the 1.5B's .gguf) at int8int8 on WebGPU with two KV slots, kse3Sequence with encode-ahead off on one
// fresh resident and on on another; every logit byte-equal. GOINFER_KSE3_PLANT=1 swaps in a pending buffer without the logits copy and
// expects red.
func TestKSE3_encodeAheadIdentity(t *testing.T) {
	newOrSkipHW(t).Close()
	paths := []string{"../testdata/llama-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		paths = append(paths, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	}
	plant := os.Getenv("GOINFER_KSE3_PLANT") == "1"
	for _, p := range paths {
		m, err := decoder.Load(p, decoder.Options{Quant: "int8int8", ResidentKVSlots: 2})
		if err != nil {
			t.Fatalf("load %s: %v", p, err)
		}
		rdOff, closeOff := kse3Resident(t, m)
		off := kse3Sequence(t, m, rdOff, true, false)
		closeOff()
		rdOn, closeOn := kse3Resident(t, m)
		on := kse3Sequence(t, m, rdOn, false, plant)
		slots := rdOn.KVSlots()
		closeOn()
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

// TestKSE3_setAdapterDropsPending: llama-tiny at int4 on WebGPU, a fresh resident per arm, 4 Forward steps without an adapter
// (the last one pre-records the next token's buffer against the base steps), then SetAdapter binds a LoRA adapter and 4 more
// Forward steps run. The adapter steps' logits with encode-ahead on equal off. GOINFER_KSE3_PLANT=2 hands the stale buffer back
// to the runner after SetAdapter (as if SetAdapter kept it) and expects red.
func TestKSE3_setAdapterDropsPending(t *testing.T) {
	newOrSkipHW(t).Close()
	const ckpt = "../testdata/llama-tiny"
	adapterDir := buildLlamaTinyLoRAFixtureGPU(t)
	plant := os.Getenv("GOINFER_KSE3_PLANT") == "2"
	arm := func(off bool) [][]float32 {
		prevOff := encodeAheadOffForTest
		encodeAheadOffForTest = off
		defer func() { encodeAheadOffForTest = prevOff }()
		m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err) // the hardware is gated above
		}
		defer m.Close()
		if err := m.LoadAdapter("a", adapterDir); err != nil {
			t.Fatal(err)
		}
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok {
			t.Fatalf("not a WebGPU resident: %s", m.ResidentDecline())
		}
		_, _, _, _, _, _, V := m.Dims()
		var out [][]float32
		for pos := range 8 {
			if pos == 4 {
				stale := rd.runner.pending
				if plant && !off {
					rd.runner.pending = nil // detach it so SetAdapter cannot drop it, then hand it back
				}
				if err := rd.SetAdapter(m.ResidentAdapterLayersForTest("a")); err != nil {
					t.Fatal(err)
				}
				if plant && !off {
					rd.runner.pending = stale
				}
			}
			lg, err := rd.Forward(m.EmbedResidentForTest((pos*47+5)%V), pos)
			if err != nil {
				t.Fatal(err)
			}
			if pos >= 4 {
				out = append(out, slices.Clone(lg))
			}
		}
		return out
	}
	off, on := arm(true), arm(false)
	d := 0
	for s := range off {
		for i := range off[s] {
			if math.Float32bits(off[s][i]) != math.Float32bits(on[s][i]) {
				d++
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[E3-G1] adapter bound after a pre-recorded token (plant %v): %d of %d logits differ, on against off\n", plant, d, 4*len(off[0]))
	if d != 0 {
		t.Errorf("%d logits differ after SetAdapter", d)
	}
}

// TestKSE3_kvSlotWithAdapter: llama-tiny at int4 on WebGPU with two KV slots and a LoRA adapter bound, a fresh resident per
// arm: 4 Forward steps on slot 0, a switch to slot 1 and 4 steps with slot 1's own inputs, then back to slot 0 for 4 more.
// Every logit with encode-ahead on equals off.
func TestKSE3_kvSlotWithAdapter(t *testing.T) {
	adapterDir := buildLlamaTinyLoRAFixtureGPU(t)
	arm := func(off bool) [][]float32 {
		prevOff := encodeAheadOffForTest
		encodeAheadOffForTest = off
		defer func() { encodeAheadOffForTest = prevOff }()
		m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Quant: "int4"}, 2)
		defer m.Close()
		if rd.KVSlots() != 2 {
			t.Fatalf("%d KV slots, want 2", rd.KVSlots())
		}
		if err := m.LoadAdapter("a", adapterDir); err != nil {
			t.Fatal(err)
		}
		if err := rd.SetAdapter(m.ResidentAdapterLayersForTest("a")); err != nil {
			t.Fatal(err)
		}
		_, _, _, _, _, _, V := m.Dims()
		var out [][]float32
		step := func(tok, pos int) {
			lg, err := rd.Forward(m.EmbedResidentForTest(tok%V), pos)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(lg))
		}
		for pos := range 4 {
			step(pos*47+5, pos)
		}
		if err := rd.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		for pos := range 4 {
			step(pos*31+200, pos)
		}
		if err := rd.UseKVSlot(0); err != nil {
			t.Fatal(err)
		}
		for pos := 4; pos < 8; pos++ {
			step(pos*47+5, pos)
		}
		return out
	}
	off, on := arm(true), arm(false)
	d := 0
	for s := range off {
		for i := range off[s] {
			if math.Float32bits(off[s][i]) != math.Float32bits(on[s][i]) {
				d++
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[E3-G1] two KV slots with an adapter bound: %d of %d logits differ, on against off\n", d, len(off)*len(off[0]))
	if d != 0 {
		t.Errorf("%d logits differ across a slot switch with an adapter bound", d)
	}
}

// TestKSE3_pendingDroppedWhereRecorded: a Forward leaves a pending buffer; RunNoLogits, SetAdapter(nil) and the runner's
// release each clear it.
func TestKSE3_pendingDroppedWhereRecorded(t *testing.T) {
	newOrSkipHW(t).Close()
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rd, closeRD := kse3Resident(t, m)
	r := rd.runner
	fwd := func(pos int) {
		if _, err := rd.Forward(m.EmbedResidentForTest(13), pos); err != nil {
			t.Fatal(err)
		}
		if r.pending == nil {
			t.Fatalf("no pending buffer after Forward at %d", pos)
		}
	}
	fwd(0)
	if err := rd.ForwardNoLogits(m.EmbedResidentForTest(13), 1); err != nil {
		t.Fatal(err)
	}
	if r.pending != nil {
		t.Error("RunNoLogits left the pending buffer")
	}
	fwd(2)
	if err := r.SetAdapter(nil); err != nil {
		t.Fatal(err)
	}
	if r.pending != nil {
		t.Error("SetAdapter left the pending buffer")
	}
	fwd(3)
	closeRD()
	if r.pending != nil {
		t.Error("release left the pending buffer")
	}
}

// TestKSE3_decodeSpeed is E3-G2 (and its re-run, docs/tasks/task-kernel-survey-2026-10.md, "E3-G2 re-run"): the 1.5B's .gguf
// at int8int8 on WebGPU, encode-ahead on against off on one resident, 128 greedy decode tokens after a 32-token prompt, A B B A
// over 10 valid pairs after 2 warm runs each. It refuses to run outside the timing lock or under 6 GB reclaimable before the
// load; voidWindow brackets each arm's timed decode (not its prompt), and a pair with an arm over the void rule's line (busy cores
// over half the cores averaged over the arm, or under 2 GiB reclaimable at either end) is void and re-run, up to 15 times. It fails
// when the median on/off is over 0.98 or fewer than 9 of 10 pairs are faster.
func TestKSE3_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSE3_TIMING") == "" {
		t.Skip("set GOINFER_KSE3_TIMING=1 (a timed run under the timing lock)")
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	if r, ok := hostmem.Read(); !ok || r.Reclaimable < 6<<30 {
		t.Fatalf("reclaimable %.2f GiB before the load, under 6 GiB: the cell would be void", float64(r.Reclaimable)/(1<<30))
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
	// run runs the 32-token prompt, then times the 128 decode tokens; with guard set, voidWindow brackets exactly the timed loop.
	run := func(off, guard bool) (sec float64, valid bool, note string) {
		encodeAheadOffForTest = off
		defer func() { encodeAheadOffForTest = false }()
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
	for range 2 {
		run(true, false)
		run(false, false)
	}
	timed := func(off bool) (float64, bool, string) { return run(off, true) }
	var ratios []float64
	below, voids := 0, 0
	for len(ratios) < 10 {
		var off, on float64
		var vOff, vOn bool
		var nOff, nOn string
		if len(ratios)%2 == 0 {
			off, vOff, nOff = timed(true)
			on, vOn, nOn = timed(false)
		} else {
			on, vOn, nOn = timed(false)
			off, vOff, nOff = timed(true)
		}
		if !vOff || !vOn {
			voids++
			fmt.Fprintf(os.Stderr, "[E3-G2] VOID pair (off: %s; on: %s)\n", nOff, nOn)
			if voids > 15 {
				t.Fatalf("over 15 void pairs")
			}
			time.Sleep(30 * time.Second)
			continue
		}
		ratios = append(ratios, on/off)
		if on < off {
			below++
		}
		fmt.Fprintf(os.Stderr, "[E3-G2] pair %d (off: %s; on: %s): off %.3f ms/token, on %.3f ms/token, on/off %.4f\n",
			len(ratios)-1, nOff, nOn, off*1e3, on*1e3, on/off)
	}
	slices.Sort(ratios)
	med := (ratios[4] + ratios[5]) / 2
	fmt.Fprintf(os.Stderr, "[E3-G2] median on/off %.4f, %d of 10 pairs below 1, %d void pairs\n", med, below, voids)
	if med > 0.98 || below < 9 {
		t.Errorf("E3-G2 below its bar: median %.4f, %d of 10 faster", med, below)
	}
}
