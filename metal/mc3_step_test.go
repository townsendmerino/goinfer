//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// mc3Resident is the resident the MC3 identity checks run on, with `slots` resident KV slots at a ctx-token resident
// context: the S1 checkpoint under GOINFER_METAL_MC3=1 (mc3RealResident), and otherwise the generated fixture
// (mc3_fixture_test.go), so the checks run by default (E-G01, docs/audit-metal-2026-09-30.md). A timing calls
// mc3RealResident instead: on the fixture it would time nothing real.
// mc3TestQuant is the quant the MC3 checks load at: "int4", or "int8int8" with nativeInt8 on for the native int8 step
// (int8 slice 3, TestMC3Step_w8BitIdentical).
var mc3TestQuant = "int4"

// mc3TestKV is the --kv the MC3 fixture loads at: "" (f16), or "i8" / "f32" for P3b-1's identity checks (p3b_rows_test.go).
var mc3TestKV = ""

func mc3Resident(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	if os.Getenv("GOINFER_METAL_MC3") == "1" {
		return mc3RealResident(t, slots, ctx)
	}
	m, r := mc3FixtureResident(t, slots, ctx)
	if got := len(r.kvSlotBufs); got < slots || r.batch == nil {
		t.Fatalf("the MC3 fixture built %d KV slots (%d requested) and batched step %v (batchIneligible: %q)", got, slots, r.batch != nil, r.batchIneligible())
	}
	if r.w8 != (mc3TestQuant == "int8int8") {
		t.Fatalf("native int8 %v at quant %s: not the path asked for", r.w8, mc3TestQuant)
	}
	return m, r
}

// mc3LoadCheckpoint loads the checkpoint the MC3 checks run on under GOINFER_METAL_MC3=1 (GOINFER_METAL_MC3_MODEL, else the
// 1.5B in ~/models) with `slots` resident KV slots at a ctx-token resident context, and builds its resident. It skips
// without the variable or the file.
func mc3LoadCheckpoint(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_MC3_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	quant := mc3TestQuant
	switch q := os.Getenv("GOINFER_METAL_MC3_QUANT"); q { // the native int8 step (slice 3) or int4mix (slice 4) on the real checkpoint
	case "int8int8", "int4mix":
		quant = q
		prev, prevMix := nativeInt8, nativeInt4Mix
		nativeInt8, nativeInt4Mix = true, true
		t.Cleanup(func() { nativeInt8, nativeInt4Mix = prev, prevMix })
	}
	m, err := decoder.Load(path, decoder.Options{Quant: quant, ResidentContext: ctx, ResidentKVSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build resident: %v", err)
	}
	if quant == "int4mix" && !r.w8Attn || quant == "int8int8" && !r.w8 {
		t.Fatalf("%s at %s: native int8 %v, int4mix %v: not the path asked for", path, quant, r.w8, r.w8Attn)
	}
	// Close both and hand the heap back before the next load: the fit guard prices live memory, and a second load in the
	// same process otherwise sees the first model's host weights still held and (correctly) refuses.
	t.Cleanup(func() {
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	})
	return m, r
}

// mc3RealResident is mc3LoadCheckpoint for the plain dense W4A8 decode path S1 covers.
func mc3RealResident(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	m, r := mc3LoadCheckpoint(t, slots, ctx)
	if r.moe != nil || r.g4moe != nil || r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm ||
		r.decodeLaneW4F16 || r.nonGatedMLP || r.outBias || r.loraLayers != nil || r.qkNormWhole || r.learnedPos { // per-head QK-norm (Qwen3) is in the step since E-P07
		t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
	}
	for _, L := range r.layers {
		if L.qGate || L.delta != nil || L.geom == nil || L.geom.kEqV {
			t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
		}
	}
	if got := len(r.kvSlotBufs); got < slots {
		t.Fatalf("resident allocated %d KV slots, %d requested (the fit clamp)", got, slots)
	}
	if r.batch == nil {
		t.Fatalf("resident built no batched step (batchIneligible: %q)", r.batchIneligible())
	}
	return m, r
}

// mc3Emb is token id's (scaled) input embedding.
func mc3Emb(r *resident, id int) []float32 {
	e := make([]float32, r.H)
	r.embed.Row(id, e)
	if r.embedScale > 1 {
		for i := range e {
			e[i] *= r.embedScale
		}
	}
	return e
}

// mc3Fill writes history ids at positions 0.. into KV slot `slot` through production's layer-major batch path, in
// chunks of 64.
func mc3Fill(t *testing.T, r *resident, slot int, ids []int) {
	t.Helper()
	if err := r.useKVSlot(slot); err != nil {
		t.Fatal(err)
	}
	for c := 0; c < len(ids); c += 64 {
		var embs [][]float32
		for _, id := range ids[c:min(c+64, len(ids))] {
			embs = append(embs, mc3Emb(r, id))
		}
		if _, err := r.ForwardBatch(embs, c); err != nil {
			t.Fatalf("fill slot %d: %v", slot, err)
		}
	}
}

// mc3Step runs production's batched step (forwardMulti) and fails the test on an error.
func mc3Step(t *testing.T, r *resident, seqs []batchSeq) [][]float32 {
	t.Helper()
	out, _, err := r.forwardMulti(seqs)
	if err != nil {
		t.Fatalf("forwardMulti: %v", err)
	}
	return out
}

// TestMC3Step_bitIdentical is S1's identity check: 4 sequences at different depths, each on its own KV slot, stepped
// together 12 times teacher-forced, against production's single-token forward (ForwardEmb) run per sequence on a TWIN
// slot filled with the same history. Every logit must match bit for bit, every step.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3Step_bitIdentical$' -v ./metal/
func TestMC3Step_bitIdentical(t *testing.T) { mc3Identity(t, []int{5, 23, 40, 300}, 1024) }

// TestMC3Step_qwen3BitIdentical (E-P07): a Qwen3-shaped model (per-head QK-norm) runs the batched step, and its rows equal
// production's single-token forward bit for bit, as TestMC3Step_bitIdentical checks for the qwen2 shape. On the generated
// qwen3 fixture by default; under GOINFER_METAL_MC3=1 the identity suite runs on whatever checkpoint GOINFER_METAL_MC3_MODEL
// names.
// moeStepFixtures are the tracked generic-MoE tiny checkpoints the step's MoE block is checked on: Mixtral (no shared
// expert), Qwen3-MoE at two top-k, and GLM-4.5 (a sigmoid router with an ungated shared expert and a dense first layer).
var moeStepFixtures = []string{"../testdata/mixtral-tiny", "../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3", "../testdata/glm-tiny"}

// moeStepResident builds dir's resident at int4 with eight KV slots, loaded with Options.ResidentMoEStep set to on.
func moeStepResident(t *testing.T, dir string, on bool) *resident {
	t.Helper()
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 8, ResidentMoEStep: on})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build %s's resident: %v", dir, err)
	}
	t.Cleanup(func() { r.Close(); m.Close() })
	return r
}

// TestMC3Step_moeBitIdentical is OP4m-d3's identity gate (docs/tasks/task-forward-architecture-2026-10.md §4.9): with the
// step's MoE block on, each tracked generic-MoE fixture builds the batched step, and four sequences at different depths stepped
// together equal production's single-token forward bit for bit, every logit of every step. With the block off, which is the
// default, none builds it.
// TestOP4mD3_moeStepSpeed is OP4m-d3's D3-3 (docs/tasks/task-forward-architecture-2026-10.md §4.9): on the MoE checkpoint at
// GOINFER_OP4MD3_MODEL, loaded int4 with the step's MoE block on, four slots filled to depth 128, then 64 greedy tokens of all four
// sequences timed two ways: the batched step (device argmax), against the do-nothing arm, each sequence decoded one token at a
// time on its own slot as today (ForwardArgmax). A B B A over 10 pairs after one warm run each; both arms' tokens must be
// equal. The cell records the 1-minute load and the reclaimable memory against the resident's need plus a free-memory margin
// (the void rule's 2 GiB), read after the load and before the resident allocates. A pair voids the cell when the test process itself pages (its own
// major faults rise during the pair) or system swap in use grows; system-wide pageouts are recorded as context only. A
// pair whose starting 1-minute load is above half the cores plus GOMAXPROCS voids the cell. Pass: a median rate ratio, step ÷
// sequential, of at least 1.05 with at least 9 of 10 pairs faster; fail: a median under 1.00; parked otherwise. A fail or a
// park fails the test. A timed run:
// it refuses to run outside the timing lock (GOINFER_OP4MD3_TIMING=1).
func TestOP4mD3_moeStepSpeed(t *testing.T) {
	op4mStepSpeed(t, stepSpeedCell{env: "GOINFER_OP4MD3", label: "[OP4m-d3 D3-3]", want: func(r *resident) bool { return r.moe != nil },
		wantWhat: "a MoE resident"})
}

// TestOP4mD4b_recurrentStepSpeed is OP4m-d4b's D4b-4, D3-3's cell on the Qwen3.5-family checkpoint at GOINFER_OP4MD4B_MODEL
// with the step's recurrent blocks on (GOINFER_OP4MD4B_TIMING=1, under the timing lock).
func TestOP4mD4b_recurrentStepSpeed(t *testing.T) {
	op4mStepSpeed(t, stepSpeedCell{env: "GOINFER_OP4MD4B", label: "[OP4m-d4b D4b-4]", recurrent: true,
		want: func(r *resident) bool { return r.dnet != nil }, wantWhat: "a Qwen3.5-family resident"})
}

// stepSpeedCell is one feature-bar cell's model and arms (op4mStepSpeed).
type stepSpeedCell struct {
	env, label        string // the cell runs on env_MODEL when env_TIMING is set, and logs under label
	recurrent, gemma4 bool   // the step's recurrent or Gemma 4 blocks on (its MoE block always is)
	want              func(*resident) bool
	wantWhat          string
	// rows, when set, is the model's full embedding row for a token (a Gemma 4 E-model's carries its PLE tail). Then both
	// arms take full rows and return logits to the host, as production decodes such a model: the step's full-logits rows,
	// against Forward on each slot, each argmaxed on the host. Without it the arms are the step's device argmax against
	// ForwardArgmax.
	rows func(*decoder.Model) func(id int) []float32
}

// op4mStepSpeed is the feature-bar cell TestOP4mD3_moeStepSpeed describes, on the model and arms c names.
func op4mStepSpeed(t *testing.T, c stepSpeedCell) {
	env, label, want, wantWhat := c.env, c.label, c.want, c.wantWhat
	path := os.Getenv(env + "_MODEL")
	if os.Getenv(env+"_TIMING") == "" || path == "" {
		t.Skipf("set %s_TIMING=1 and %s_MODEL (a timed run under the timing lock)", env, env)
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set: a timing from it measures the disk", path)
	}
	const B, depth, steps, pairs, ctx = 4, 128, 64, 10, 1024
	load1 := func() float64 {
		out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		if err != nil {
			t.Fatal(err)
		}
		var v float64
		if _, err := fmt.Sscanf(strings.Trim(strings.TrimSpace(string(out)), "{} "), "%g", &v); err != nil {
			t.Fatalf("vm.loadavg %q: %v", out, err)
		}
		return v
	}
	// paging is swap in use (MiB, vm.swapusage), whose growth voids a pair, and the system-wide pageouts count (vm_stat),
	// recorded as context. A swap reading that does not parse fails the cell rather than reading as zero.
	paging := func() (swapMiB float64, pageouts int64) {
		out, err := exec.Command("sysctl", "-n", "vm.swapusage").Output()
		if err != nil {
			t.Fatal(err)
		}
		f := strings.Fields(string(out)) // total = X  used = Y  free = Z ...
		parsed := false
		for i := range f {
			if f[i] == "used" && i+2 < len(f) {
				_, err := fmt.Sscanf(strings.TrimSuffix(f[i+2], "M"), "%g", &swapMiB)
				parsed = err == nil
			}
		}
		if !parsed {
			t.Fatalf("vm.swapusage %q: no swap-in-use reading", out)
		}
		vs, err := exec.Command("vm_stat").Output()
		if err != nil {
			t.Fatal(err)
		}
		for line := range strings.SplitSeq(string(vs), "\n") {
			if strings.HasPrefix(line, "Pageouts:") {
				fmt.Sscanf(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(line, "Pageouts:")), "."), "%d", &pageouts)
			}
		}
		return swapMiB, pageouts
	}
	majFaults := func() int64 { // this process's major faults (page-ins from disk), the never-swap rule's own reading
		var ru syscall.Rusage
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
			t.Fatal(err)
		}
		return int64(ru.Majflt)
	}
	l0 := load1()
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: ctx, ResidentKVSlots: B, ResidentMoEStep: true,
		ResidentRecurrentStep: c.recurrent})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	need := m.ResidentNeedBytes("metal", 0, ctx, false, false) + int64(B-1)*metalSlotBytes(m) + metalRecurrentBaseBytes(m, B) // every slot's KV and recurrent state
	const margin = 2048                                                                                                       // MiB, the void rule's registered margin
	r0, ok := probe.Read()                                                                                                    // the fit check: after the load maps the file, before the resident allocates
	if !ok {
		t.Fatal("no memory reading")
	}
	gib := func(b int64) float64 { return float64(b) / (1 << 30) }
	bar := float64(runtime.NumCPU())/2 + float64(runtime.GOMAXPROCS(0))
	memOK := int64(r0.Reclaimable) >= need+margin<<20
	name := filepath.Base(path)
	fmt.Fprintf(os.Stderr, label+" %s: cell start: 1-minute load %.2f; reclaimable %.2f GiB against need %.2f GiB + %d MiB (%s); pair bar %.1f\n",
		name, l0, gib(int64(r0.Reclaimable)), gib(need), margin, map[bool]string{true: "ok", false: "VOID"}[memOK], bar)
	if !memOK {
		t.Fatalf("cell void: reclaimable %.2f GiB under need %.2f GiB + %d MiB", gib(int64(r0.Reclaimable)), gib(need), margin)
	}
	prevG := gemma4StepOn.Swap(c.gemma4)
	r, err := buildResident(m)
	gemma4StepOn.Store(prevG)
	if err != nil {
		t.Fatalf("build the resident: %v", err)
	}
	defer r.Close()
	if scratch := r.stepScratchBytes(); int64(r0.Reclaimable) < need+scratch+margin<<20 {
		t.Fatalf("cell void: reclaimable %.2f GiB under need %.2f GiB, the step's scratch %.1f MiB and %d MiB", gib(int64(r0.Reclaimable)), gib(need),
			float64(scratch)/(1<<20), margin)
	}
	if !want(r) || r.batch == nil || len(r.kvSlotBufs) < B {
		t.Fatalf("%s %v, batched step %v, %d KV slots: want %s with the step and %d slots (%s)", wantWhat, want(r), r.batch != nil,
			len(r.kvSlotBufs), wantWhat, B, r.batchIneligible())
	}
	seed := uint32(128)
	rnd := func() int {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return int(seed % uint32(min(20000, r.V)))
	}
	row := func(id int) []float32 { return mc3Emb(r, id) }
	if c.rows != nil {
		row = c.rows(m)
	}
	for sl := range B {
		ids := make([]int, depth)
		for i := range ids {
			ids[i] = rnd()
		}
		if c.rows == nil {
			mc3Fill(t, r, sl, ids)
			continue
		}
		if err := r.useKVSlot(sl); err != nil {
			t.Fatal(err)
		}
		for i, id := range ids {
			r.ForwardEmb(row(id), i)
		}
	}
	first := make([]int, B)
	for i := range first {
		first[i] = rnd()
	}
	run := func(batched bool) (float64, []int) {
		cur := append([]int(nil), first...)
		var toks []int
		t0 := time.Now()
		for s := range steps {
			if batched {
				seqs := make([]batchSeq, B)
				for i := range seqs {
					seqs[i] = batchSeq{slot: i, pos: depth + s, emb: row(cur[i])}
				}
				logits, ids, err := r.forwardMultiInto(seqs, c.rows == nil)
				if err != nil {
					t.Fatal(err)
				}
				if c.rows != nil {
					ids = make([]int, B)
					for i := range ids {
						ids[i] = argmaxF32(logits[i])
					}
				}
				cur = ids
			} else {
				for i := range B {
					if err := r.useKVSlot(i); err != nil {
						t.Fatal(err)
					}
					if c.rows != nil {
						cur[i] = argmaxF32(r.ForwardEmb(row(cur[i]), depth+s))
					} else {
						cur[i] = int(r.ForwardArgmax(cur[i], depth+s))
					}
				}
			}
			toks = append(toks, cur...)
		}
		return float64(B*steps) / time.Since(t0).Seconds(), toks
	}
	run(true)
	run(false)
	var ratios []float64
	faster := 0
	for i := range pairs {
		lp := load1()
		if lp > bar {
			t.Fatalf("cell void: pair %d started at 1-minute load %.2f, above %.1f; re-run the cell", i, lp, bar)
		}
		sw0, po0 := paging()
		mf0 := majFaults()
		var rb, rs float64
		var tb, ts []int
		if i%2 == 0 {
			rb, tb = run(true)
			rs, ts = run(false)
		} else {
			rs, ts = run(false)
			rb, tb = run(true)
		}
		if len(tb) != len(ts) {
			t.Fatalf("pair %d: %d and %d tokens", i, len(tb), len(ts))
		}
		for j := range tb {
			if tb[j] != ts[j] {
				t.Fatalf("pair %d: the batched step's token %d is %d, sequential decode's %d", i, j, tb[j], ts[j])
			}
		}
		sw1, po1 := paging()
		mf1 := majFaults()
		fmt.Fprintf(os.Stderr, label+" pair %d (load %.2f): step %.1f tok/s, sequential %.1f tok/s, step/sequential %.4f; tokens equal; process major faults %d -> %d; swap %.1f -> %.1f MiB; system pageouts %d -> %d (context)\n",
			i, lp, rb, rs, rb/rs, mf0, mf1, sw0, sw1, po0, po1)
		if mf1 > mf0 || sw1 > sw0 {
			t.Fatalf("cell void: pair %d paged (process major faults %d -> %d, swap %.1f -> %.1f MiB); re-run the cell", i, mf0, mf1, sw0, sw1)
		}
		ratios = append(ratios, rb/rs)
		if rb > rs {
			faster++
		}
	}
	sort.Float64s(ratios)
	med := (ratios[pairs/2-1] + ratios[pairs/2]) / 2
	verdict := "PARKED (under the feature bar, not under 1.00)"
	switch {
	case med >= 1.05 && faster >= 9:
		verdict = "PASS"
	case med < 1.00:
		verdict = "FAIL (median under 1.00)"
	}
	fmt.Fprintf(os.Stderr, label+" %s: median step/sequential %.4f, %d of %d pairs faster, ratios %.4f: %s\n",
		name, med, faster, pairs, ratios, verdict)
	if verdict != "PASS" {
		t.Errorf("%s %s: %s (median %.4f, %d of %d pairs faster)", label, name, verdict, med, faster, pairs)
	}
}

func TestMC3Step_moeBitIdentical(t *testing.T) {
	for _, dir := range moeStepFixtures {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if r := moeStepResident(t, dir, false); r.batch != nil || r.moe == nil {
				t.Fatalf("with the MoE block off: batched step %v, MoE %v; want no step on a MoE model", r.batch != nil, r.moe != nil)
			}
			r := moeStepResident(t, dir, true)
			if r.batch == nil {
				t.Fatalf("with the MoE block on, no batched step: %s", r.batchIneligible())
			}
			mc3IdentityOn(t, r, []int{3, 11, 20, 40}) // the tiny fixtures hold 64 positions
		})
	}
}

// TestMC3Step_moeVerifyBitIdentical: with the step's MoE block on, the speculative loop's argmax-only verify
// (PrefillLastNArgmax) runs a MoE model's draft rows on the step, and each row's id equals the sequential loop's argmax at
// that position, for drafts of 1 to 11 rows (one-row pieces, one full piece, a piece and a remainder).
func TestMC3Step_moeVerifyBitIdentical(t *testing.T) {
	for _, dir := range moeStepFixtures {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			r := moeStepResident(t, dir, true)
			a := &metalResident{r: r, hidden: r.H}
			if a.VerifyCost() == nil {
				t.Fatalf("with the MoE block on, no step verify: %s", r.batchIneligible())
			}
			if (&metalResident{r: moeStepResident(t, dir, false)}).VerifyCost() != nil {
				t.Fatal("with the MoE block off, the MoE resident offers a step verify")
			}
			seed := uint32(97531)
			rnd := func() int {
				seed ^= seed << 13
				seed ^= seed >> 17
				seed ^= seed << 5
				return int(seed % uint32(min(20000, r.V)))
			}
			const start = 9
			for _, n := range []int{1, 3, 8, 11} {
				embs := make([][]float32, start+n)
				for i := range embs {
					embs[i] = mc3Emb(r, rnd())
				}
				if err := r.useKVSlot(0); err != nil {
					t.Fatal(err)
				}
				want := make([]int, 0, n)
				for i, e := range embs {
					if lg := r.ForwardEmb(e, i); i >= start {
						want = append(want, argmaxF32(lg))
					}
				}
				if err := r.useKVSlot(1); err != nil {
					t.Fatal(err)
				}
				for i := range start {
					r.ForwardEmb(embs[i], i)
				}
				got, err := a.PrefillLastNArgmax(embs[start:], start)
				if err != nil {
					t.Fatalf("%d draft rows: PrefillLastNArgmax: %v", n, err)
				}
				if !slices.Equal(got, want) {
					t.Errorf("%d draft rows: the step's verify ids %v, the sequential loop's %v", n, got, want)
				}
			}
		})
	}
}

// TestMC3Step_moeScratchPricedAtTwoSlots: with the step's MoE block on and two KV slots, the step's scratch is charged
// against the memory guard. At a ceiling that holds the weights and both slots' KV but not the scratch, the load is
// admitted with both slots and the step is declined. At that ceiling plus the scratch, the step builds.
func TestMC3Step_moeScratchPricedAtTwoSlots(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	build := func(ceiling func(m *decoder.Model) int64) *metalResident {
		m, err := decoder.Load(moeStepFixtures[1], decoder.Options{Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 2, ResidentMoEStep: true})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		metalMemoryCeilingForTest = ceiling(m)
		rf, ok, err := (&metalBackend{}).BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("ceiling %d: the load was refused: %v", metalMemoryCeilingForTest, err)
		}
		a := rf.(*metalResident)
		t.Cleanup(func() { a.r.Close() })
		return a
	}
	tight := build(func(m *decoder.Model) int64 { return residentNeedBytes(m) + residentKVBytes(m) })
	if len(tight.r.kvSlotBufs) != 2 || tight.r.batch != nil || tight.r.stepDecline == "" {
		t.Fatalf("a ceiling of the weights and two slots' KV: %d slots, step built %v, decline %q; want 2 slots and the step declined",
			len(tight.r.kvSlotBufs), tight.r.batch != nil, tight.r.stepDecline)
	}
	scratch := tight.r.stepScratchBytes()
	roomy := build(func(m *decoder.Model) int64 { return residentNeedBytes(m) + residentKVBytes(m) + scratch })
	if len(roomy.r.kvSlotBufs) != 2 || roomy.r.batch == nil || roomy.r.stepDecline != "" {
		t.Errorf("that ceiling plus the step's %d bytes of scratch: %d slots, step built %v, decline %q; want 2 slots and the step",
			scratch, len(roomy.r.kvSlotBufs), roomy.r.batch != nil, roomy.r.stepDecline)
	}
}

// TestMC3Step_moePromptRouteChangesNoOutput: with the step's MoE block on, a MoE prompt takes the step only where it
// replaces the sequential loop. A prompt ending below the batched pass's floor prefills on the step, and its last-row logits
// equal the sequential loop's. One that reaches the floor takes the pass, even below metalStepPrefillCeiling (a dense
// model's step range), and its logits equal those from the same prompt with the block off. So turning the block on
// changes no prompt's output.
func TestMC3Step_moePromptRouteChangesNoOutput(t *testing.T) {
	for _, dir := range moeStepFixtures {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			on, off := moeStepResident(t, dir, true), moeStepResident(t, dir, false)
			if on.batch == nil {
				t.Fatalf("with the MoE block on, no batched step: %s", on.batchIneligible())
			}
			aOn, aOff := &metalResident{r: on, hidden: on.H}, &metalResident{r: off, hidden: off.H}
			floor := metalFastPrefillFloorFor(on.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
			if floor != 16 || metalStepPrefillCeiling != 32 {
				t.Fatalf("fast-prefill floor %d, step bound %d; the cases below are laid out for 16 and 32", floor, metalStepPrefillCeiling)
			}
			if got := aOn.prefillKernelFloor(); got != floor {
				t.Errorf("the MoE resident's prefill kernel floor is %d with the block on, want the pass's %d", got, floor)
			}
			seed := uint32(13579)
			rnd := func() int {
				seed ^= seed << 13
				seed ^= seed >> 17
				seed ^= seed << 5
				return int(seed % uint32(min(20000, on.V)))
			}
			bits := func(a, b []float32) int {
				if len(a) != len(b) {
					return max(len(a), len(b))
				}
				n := 0
				for j := range a {
					if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
						n++
					}
				}
				return n
			}
			for _, c := range []struct {
				n    int
				step bool
			}{{8, true}, {15, true}, {16, false}, {20, false}, {31, false}} {
				if got := aOn.promptStepOK(c.n, 0, floor); got != c.step {
					t.Errorf("%d tokens: promptStepOK = %v with the block on, want %v", c.n, got, c.step)
				}
				if aOff.promptStepOK(c.n, 0, floor) {
					t.Errorf("%d tokens: promptStepOK with the block off", c.n)
				}
				embs := make([][]float32, c.n)
				for i := range embs {
					embs[i] = mc3Emb(on, rnd())
				}
				got, err := aOn.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("%d tokens: PrefillLast with the block on: %v", c.n, err)
				}
				var ref []float32
				if c.step { // the block off sends this prompt down the sequential loop
					if _, err := aOff.PrefillLast(context.Background(), embs, 0); err == nil {
						t.Errorf("%d tokens: with the block off, PrefillLast took the prompt; want it declined to the sequential loop", c.n)
					}
					if err := off.useKVSlot(0); err != nil {
						t.Fatal(err)
					}
					for i, e := range embs {
						ref = append(ref[:0], off.ForwardEmb(e, i)...)
					}
				} else if ref, err = aOff.PrefillLast(context.Background(), embs, 0); err != nil {
					t.Fatalf("%d tokens: PrefillLast with the block off: %v", c.n, err)
				}
				if d := bits(got, ref); d != 0 {
					t.Errorf("%d tokens: %d of %d last-row logits differ between the block on and off", c.n, d, len(ref))
				}
			}
		})
	}
}

// recurrentFixture is a tracked Qwen3.5-family checkpoint the step's per-row DeltaNet and gated-attention blocks are checked
// on (OP4m-d4b), and whether it is a MoE model, whose MoE block is then on in both arms.
type recurrentFixture struct {
	dir string
	moe bool
}

// recurrentStepFixtures: two MoE models, and the dense qwen3_5 one, which reads program.MetalStepRecurrent without the MoE
// block.
var recurrentStepFixtures = []recurrentFixture{{"../testdata/qwen35-tiny", true}, {"../testdata/qwen3next-tiny", true},
	{"../decoder/testdata/qwen3_5-tiny", false}}

// recurrentStepResidentSlots builds fx's resident at int4 with `slots` KV slots, loaded with Options.ResidentRecurrentStep set
// to on, and its MoE block on for a MoE fixture whatever on is.
func recurrentStepResidentSlots(t *testing.T, fx recurrentFixture, on bool, slots int) *resident {
	t.Helper()
	m, err := decoder.Load(fx.dir, decoder.Options{Quant: "int4", ResidentContext: 1024, ResidentKVSlots: slots, ResidentMoEStep: fx.moe,
		ResidentRecurrentStep: on})
	if err != nil {
		t.Fatalf("load %s: %v", fx.dir, err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build %s's resident: %v", fx.dir, err)
	}
	t.Cleanup(func() { r.Close(); m.Close() })
	return r
}

// recurrentStepResident is recurrentStepResidentSlots at eight slots.
func recurrentStepResident(t *testing.T, fx recurrentFixture, on bool) *resident {
	t.Helper()
	return recurrentStepResidentSlots(t, fx, on, 8)
}

// mc3IdentityDiff is mc3IdentityOn's check counted rather than asserted: the logits of every step's rows that differ from
// production's single-token forward on each sequence's twin slot.
func mc3IdentityDiff(t *testing.T, r *resident, depths []int) int {
	t.Helper()
	B := len(depths)
	seed := uint32(1234567)
	rnd := func() int {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return int(seed % uint32(min(20000, r.V)))
	}
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
		mc3Fill(t, r, B+m, ids)
	}
	diff := 0
	for st := range 12 {
		seqs := make([]batchSeq, B)
		ref := make([][]float32, B)
		for m := range B {
			seqs[m] = batchSeq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			if err := r.useKVSlot(B + m); err != nil {
				t.Fatal(err)
			}
			ref[m] = append([]float32(nil), r.ForwardEmb(seqs[m].emb, seqs[m].pos)...)
		}
		got := mc3Step(t, r, seqs)
		for m := range B {
			for i := range got[m] {
				if math.Float32bits(got[m][i]) != math.Float32bits(ref[m][i]) {
					diff++
				}
			}
		}
	}
	return diff
}

// TestMC3Step_recurrentBitIdentical is OP4m-d4b's D4b-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9): with the
// step's recurrent blocks on, each tracked Qwen3.5-family fixture builds the batched step. Four sequences at different
// depths, each on its own slot with its own DeltaNet state, stepped together 12 times, equal production's single-token
// forward on every logit. With the blocks off, neither builds it.
func TestMC3Step_recurrentBitIdentical(t *testing.T) {
	for _, fx := range recurrentStepFixtures {
		t.Run(filepath.Base(fx.dir), func(t *testing.T) {
			if r := recurrentStepResident(t, fx, false); r.batch != nil {
				t.Fatal("with the recurrent blocks off, a Qwen3.5-family model built the batched step")
			}
			r := recurrentStepResident(t, fx, true)
			if r.batch == nil {
				t.Fatalf("with the recurrent blocks on, no batched step: %s", r.batchIneligible())
			}
			mc3IdentityOn(t, r, []int{3, 11, 20, 40}) // the tiny fixtures hold 64 positions
		})
	}
}

// TestMC3Step_recurrentPlantsGoRed is D4b-1's two plants, each red on each fixture: every row's DeltaNet on slot 0's
// state, and every row's gated attention at row 0's position.
func TestMC3Step_recurrentPlantsGoRed(t *testing.T) {
	for _, plant := range []struct {
		name string
		on   *bool
	}{{"every row's DeltaNet on slot 0's state", &stepDeltaSlot0PlantForTest}, {"every row's attention at row 0's position", &stepAttnRow0PlantForTest}} {
		for _, fx := range recurrentStepFixtures {
			t.Run(filepath.Base(fx.dir), func(t *testing.T) {
				r := recurrentStepResident(t, fx, true)
				if r.batch == nil {
					t.Fatalf("with the recurrent blocks on, no batched step: %s", r.batchIneligible())
				}
				*plant.on = true
				defer func() { *plant.on = false }()
				d := mc3IdentityDiff(t, r, []int{3, 11, 20, 40})
				t.Logf("%s: %d logits differ", plant.name, d)
				if d == 0 {
					t.Errorf("planted %s, and the step still equals production's forward", plant.name)
				}
			})
		}
	}
}

// TestMC3Step_recurrentPromptRouteChangesNoOutput is D4b-2: with the step's recurrent blocks on, a Qwen3.5-family prompt
// takes the step only below the batched pass's floor, where its last-row logits equal the sequential loop's from a reset
// state, even when the slot holds a previous sequence's state. A prompt that reaches the floor takes the pass and equals the
// same prompt with the blocks off. Each case runs on slot 0 and on slot 1, and the slot's recurrent state after the prompt
// equals the reference's too, every element.
func TestMC3Step_recurrentPromptRouteChangesNoOutput(t *testing.T) {
	for _, fx := range recurrentStepFixtures {
		t.Run(filepath.Base(fx.dir), func(t *testing.T) {
			on, off := recurrentStepResident(t, fx, true), recurrentStepResident(t, fx, false)
			if on.batch == nil {
				t.Fatalf("with the recurrent blocks on, no batched step: %s", on.batchIneligible())
			}
			aOn, aOff := &metalResident{r: on, hidden: on.H}, &metalResident{r: off, hidden: off.H}
			floor := metalFastPrefillFloorFor(on.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
			if got := aOn.prefillKernelFloor(); got != floor {
				t.Errorf("the prefill kernel floor is %d with the blocks on, want the pass's %d", got, floor)
			}
			seed := uint32(24680)
			rnd := func() int {
				seed ^= seed << 13
				seed ^= seed >> 17
				seed ^= seed << 5
				return int(seed % uint32(min(20000, on.V)))
			}
			bind := func(r *resident, s int) {
				if err := r.useKVSlot(s); err != nil {
					t.Fatal(err)
				}
			}
			state := func(r *resident) []float32 { // the bound slot's recurrent windows and states, concatenated
				var out []float32
				for _, b := range r.recurrentLive() {
					out = append(out, b.Floats()...)
				}
				return out
			}
			for _, c := range []struct {
				n, slot int
				step    bool
			}{{8, 0, true}, {floor - 1, 0, true}, {floor, 0, false}, {floor + 4, 0, false}, {8, 1, true}, {floor + 4, 1, false}} {
				if got := aOn.promptStepOK(c.n, 0, floor); got != c.step {
					t.Errorf("%d tokens: promptStepOK = %v with the blocks on, want %v", c.n, got, c.step)
				}
				embs := make([][]float32, c.n)
				for i := range embs {
					embs[i] = mc3Emb(on, rnd())
				}
				// The blocks-on slot is left dirty by a sequence of its own: a prompt at position 0 starts a fresh sequence,
				// so PrefillLast must zero the slot's recurrent state itself, as Forward at position 0 does.
				bind(on, c.slot)
				for i := range 3 {
					on.ForwardEmb(mc3Emb(on, rnd()), i)
				}
				got, err := aOn.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("%d tokens: PrefillLast with the blocks on: %v", c.n, err)
				}
				var ref []float32
				bind(off, c.slot)
				off.resetRecurrent()
				if c.step {
					for i, e := range embs {
						ref = append(ref[:0], off.ForwardEmb(e, i)...)
					}
				} else if ref, err = aOff.PrefillLast(context.Background(), embs, 0); err != nil {
					t.Fatalf("%d tokens: PrefillLast with the blocks off: %v", c.n, err)
				}
				d := 0
				for j := range ref {
					if j >= len(got) || math.Float32bits(got[j]) != math.Float32bits(ref[j]) {
						d++
					}
				}
				if d != 0 {
					t.Errorf("slot %d, %d tokens: %d of %d last-row logits differ between the blocks on and off", c.slot, c.n, d, len(ref))
				}
				gs, ws := state(on), state(off)
				sd := 0
				for j := range ws {
					if j >= len(gs) || math.Float32bits(gs[j]) != math.Float32bits(ws[j]) {
						sd++
					}
				}
				if sd != 0 || len(gs) != len(ws) {
					t.Errorf("slot %d, %d tokens: %d of %d recurrent-state elements differ between the blocks on and off", c.slot, c.n, sd, len(ws))
				}
			}
		})
	}
}

// TestMC3Step_recurrentBlocksWithoutDeltaNet: a model whose only recurrent-block layer is gated attention, with no Gated-DeltaNet
// (no tracked fixture has that shape, so the resident is built by hand), runs the step's recurrent blocks. So its step scratch
// is priced at two slots, and its prompts take the step only below the pass's floor, as a DeltaNet model's do.
func TestMC3Step_recurrentBlocksWithoutDeltaNet(t *testing.T) {
	r := &resident{recurrentStep: true, layers: []residLayer{{qGate: true}}, kvSlotBufs: make([]kvSlotBuf, 2)}
	if !r.recurrentBlocks() {
		t.Fatal("a gated-attention model with the recurrent blocks on does not run them")
	}
	if !r.stepPriced() {
		t.Error("a gated-attention model's step scratch is not priced at two slots")
	}
	if got := (&metalResident{r: r}).stepPrefillBound(16); got != 16 {
		t.Errorf("a gated-attention model's prompts take the step below %d, want the floor 16", got)
	}
	if r.recurrentStep = false; r.recurrentBlocks() {
		t.Error("with the recurrent blocks off, the model runs them")
	}
}

// TestMC3Step_recurrentNoAttentionDeclined: a recurrent model with no attention layer at all is declined by the step's
// admission, since its attention scratch is sized from an attention layer's geometry. The resident is qwen3_5-tiny with its
// gated-attention layers rewritten as DeltaNet steps.
// TestMC3Step_recurrentRowAtZeroStartsFresh: a step row at position 0 starts a new sequence, so it runs from a zeroed
// recurrent state even when its slot holds a previous sequence's. On each fixture, slot 0 is dirtied by three tokens of its
// own. Then a step of two rows at position 0, on slot 0 and on clean slot 1, equals each embedding's single-token forward on a
// freshly reset slot, every logit.
func TestMC3Step_recurrentRowAtZeroStartsFresh(t *testing.T) {
	for _, fx := range recurrentStepFixtures {
		t.Run(filepath.Base(fx.dir), func(t *testing.T) {
			r := recurrentStepResident(t, fx, true)
			if r.batch == nil {
				t.Fatalf("with the recurrent blocks on, no batched step: %s", r.batchIneligible())
			}
			bind := func(s int) {
				if err := r.useKVSlot(s); err != nil {
					t.Fatal(err)
				}
			}
			e0, e1 := mc3Emb(r, 11), mc3Emb(r, 23)
			bind(0)
			r.resetRecurrent()
			for i, id := range []int{5, 7, 9} {
				r.ForwardEmb(mc3Emb(r, id), i)
			}
			ref := func(e []float32) []float32 {
				bind(2)
				r.resetRecurrent()
				return append([]float32(nil), r.ForwardEmb(e, 0)...)
			}
			want0, want1 := ref(e0), ref(e1)
			got := mc3Step(t, r, []batchSeq{{slot: 0, pos: 0, emb: e0}, {slot: 1, pos: 0, emb: e1}})
			for m, want := range [][]float32{want0, want1} {
				d := 0
				for i := range want {
					if math.Float32bits(got[m][i]) != math.Float32bits(want[i]) {
						d++
					}
				}
				if d != 0 {
					t.Errorf("row %d (slot %d, position 0): %d of %d logits differ from a fresh sequence's", m, m, d, len(want))
				}
			}
		})
	}
}

func TestMC3Step_recurrentNoAttentionDeclined(t *testing.T) {
	r := recurrentStepResident(t, recurrentStepFixtures[2], true)
	if r.batch == nil {
		t.Fatalf("the unmodified model builds no step: %s", r.batchIneligible())
	}
	for l := range r.layers {
		if L := &r.layers[l]; L.qGate {
			L.qGate, L.geom = false, nil
			L.steps = []layerStep{stepDeltaNet, L.steps[1]}
		}
	}
	if why := r.batchIneligible(); why == "" {
		t.Error("a model with no attention layer is admitted to the step")
	} else {
		t.Logf("declined: %s", why)
	}
}

// TestMC3Step_recurrentScratchPriced: with the recurrent blocks on, the step's scratch is charged against the memory guard,
// every slot's recurrent state counted, at one slot and at two, on the dense qwen3_5 fixture.
//   - At two slots, a ceiling that holds the weights and both slots' KV and state, but not the scratch, keeps both slots and
//     declines the step; that ceiling plus the scratch builds it.
//   - At one slot, a ceiling of the need plus the scratch, without the slot's own recurrent state, declines the step; with it,
//     the step builds.
func TestMC3Step_recurrentScratchPriced(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	fx := recurrentStepFixtures[2]
	build := func(slots int, ceiling func(m *decoder.Model) int64) *metalResident {
		m, err := decoder.Load(fx.dir, decoder.Options{Quant: "int4", ResidentContext: 1024, ResidentKVSlots: slots, ResidentRecurrentStep: true})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		metalMemoryCeilingForTest = ceiling(m)
		rf, ok, err := (&metalBackend{}).BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("ceiling %d: the load was refused: %v", metalMemoryCeilingForTest, err)
		}
		a := rf.(*metalResident)
		t.Cleanup(func() { a.r.Close() })
		return a
	}
	twoSlots := func(m *decoder.Model) int64 {
		return residentNeedBytes(m) + metalRecurrentBaseBytes(m, 2) + metalSlotBytes(m)
	}
	tight := build(2, twoSlots)
	if len(tight.r.kvSlotBufs) != 2 || tight.r.batch != nil || tight.r.stepDecline == "" {
		t.Fatalf("two slots, no room for the scratch: %d slots, step built %v, decline %q; want 2 slots and the step declined",
			len(tight.r.kvSlotBufs), tight.r.batch != nil, tight.r.stepDecline)
	}
	scratch := tight.r.stepScratchBytes()
	if roomy := build(2, func(m *decoder.Model) int64 { return twoSlots(m) + scratch }); roomy.r.batch == nil {
		t.Errorf("two slots and the scratch: no step (%s)", roomy.r.stepDecline)
	}
	one := build(1, func(m *decoder.Model) int64 { return residentNeedBytes(m) + scratch })
	if one.r.batch != nil {
		t.Error("one slot, a ceiling without the slot's recurrent state: the step built")
	}
	if ok := build(1, func(m *decoder.Model) int64 { return residentNeedBytes(m) + metalRecurrentSlotBytes(m) + scratch }); ok.r.batch == nil {
		t.Errorf("one slot, its state and the scratch: no step (%s)", ok.r.stepDecline)
	}
}

func TestMC3Step_qwen3BitIdentical(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") == "1" {
		t.Skip("the fixture form; under GOINFER_METAL_MC3=1 run TestMC3Step_bitIdentical with GOINFER_METAL_MC3_MODEL set to a Qwen3")
	}
	mc3FixtureWriter = writeMC3FixtureQwen3
	t.Cleanup(func() { mc3FixtureWriter = writeMC3Fixture })
	mc3IdentityWith(t, []int{5, 23, 40, 300}, 1024, func(r *resident) {
		if !r.qkNorm || r.qkNormWhole {
			t.Fatalf("the qwen3 fixture loaded with qkNorm %v, qkNormWhole %v: this would check the wrong path", r.qkNorm, r.qkNormWhole)
		}
	})
}

// TestMC3Step_bitIdenticalDeep is the same check where production's single-token step plans attention_fa (at or above
// attnFADepthFloor keys): two sequences past the floor, one crossing it during the 12 steps, one shallow — so the
// batched step must reproduce each sequence's own attention plan, both kinds in one command buffer.
func TestMC3Step_bitIdenticalDeep(t *testing.T) {
	mc3Identity(t, []int{attnFADepthFloor + 400, attnFADepthFloor - 6, 100, attnFADepthFloor + 64}, 2048)
}

// TestMC3Step_faRowsBitIdenticalDeep is the deep identity check with E-P05's multi-row flash attention on (parked off by
// default): two rows past the attention floor and two below, so the multi-row pair serves two rows at once.
func TestMC3Step_faRowsBitIdenticalDeep(t *testing.T) {
	prev := mc3FARowsOn
	mc3FARowsOn = true
	defer func() { mc3FARowsOn = prev }()
	mc3Identity(t, []int{attnFADepthFloor + 400, attnFADepthFloor - 6, 100, attnFADepthFloor + 64}, 2048)
}

func mc3Identity(t *testing.T, depths []int, ctx int) { mc3IdentityWith(t, depths, ctx, nil) }

// mc3IdentityWith is mc3Identity with setup run on the resident before any step (to force a path).
func mc3IdentityWith(t *testing.T, depths []int, ctx int, setup func(r *resident)) {
	_, r := mc3Resident(t, 2*len(depths), ctx)
	if setup != nil {
		setup(r)
	}
	mc3IdentityOn(t, r, depths)
}

// mc3IdentityOn is the identity check on resident r, which holds at least two KV slots per depth: each sequence's history
// fills its slot and a twin, and 12 teacher-forced batched steps must equal production's single-token forward on the twins
// bit for bit. Token ids stay under the vocabulary.
func mc3IdentityOn(t *testing.T, r *resident, depths []int) {
	t.Helper()
	const steps = 12
	B := len(depths)
	seed := uint32(1234567)
	rnd := func() int {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return int(seed % uint32(min(20000, r.V)))
	}
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
		mc3Fill(t, r, B+m, ids)
	}
	totalDiff := 0
	for st := range steps {
		seqs := make([]batchSeq, B)
		ref := make([][]float32, B)
		for m := range B {
			seqs[m] = batchSeq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			if err := r.useKVSlot(B + m); err != nil {
				t.Fatal(err)
			}
			ref[m] = append([]float32(nil), r.ForwardEmb(seqs[m].emb, seqs[m].pos)...)
		}
		got := mc3Step(t, r, seqs)
		rowsLabel := attnDigestRowsLabel
		if rowsLabel == "" {
			rowsLabel = "mc3_attention_rows"
			if strings.Contains(t.Name(), "faRows") { // the fixture's G = 6 takes the block kernel and its rows form past the depth floor
				rowsLabel = "mc3_attention_rows,mc3_attention_fa_blk_rows_g6,mc3_attention_fa_combine_rows"
			}
		}
		for m := range B {
			recordAttnDigest(t, rowsLabel, got[m])
			recordAttnDigest(t, "model-decode", ref[m])
			diff, worst := 0, 0.0
			for i := range got[m] {
				if math.Float32bits(got[m][i]) != math.Float32bits(ref[m][i]) {
					diff++
					worst = math.Max(worst, math.Abs(float64(got[m][i]-ref[m][i])))
				}
			}
			if diff != 0 {
				t.Errorf("step %d sequence %d (pos %d): %d of %d logits differ from production (max |diff| %.3g)", st, m, seqs[m].pos, diff, r.V, worst)
			}
			totalDiff += diff
		}
	}
	t.Logf("%d sequences x %d steps at depths %v: %d logits differ from production's single-token forward", B, steps, depths, totalDiff)
}

// TestMC3Step_rowsPathBitIdentical is S4's identity check: with qkv and gate|up forced onto per-row production GEMVs
// at every batch size, and then forced onto the fragment at every size, a batched step at B = 2, 3 and 4 matches
// production's single-token forward on every logit. Which one calibrateRows picks changes speed only. The per-row case
// runs E-P02's one-dispatch adjacent form (mc3AdjRowsOn), so it is that kernel's identity gate too.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC3Step_rowsPathBitIdentical$' -v ./metal/
func TestMC3Step_rowsPathBitIdentical(t *testing.T) {
	for _, c := range []struct {
		name string
		rows int
	}{{"per-row", batchMaxSeqs}, {"fragment", 0}} {
		for _, depths := range [][]int{{5, 300}, {23, 40, 300}, {5, 23, 40, 300}} {
			t.Run(fmt.Sprintf("%s/B=%d", c.name, len(depths)), func(t *testing.T) {
				mc3IdentityWith(t, depths, 1024, func(r *resident) {
					r.batch.rowsQKV, r.batch.rowsGU = c.rows, c.rows
					// E-P02: the per-row case runs the adjacent-threadgroup form (mc3AdjRowsOn), which needs the rows kernels
					if c.rows > 0 && mc3AdjRowsOn && (r.gemvRows.qkv == 0 || r.gemvRows.gu == 0) {
						t.Fatalf("the fixture's qkv / gate|up rows kernels are %d / %d: the adjacent form would not run", r.gemvRows.qkv, r.gemvRows.gu)
					}
				})
			})
		}
	}
}

// TestMC3Step_throughput is S1's in-sequence cost (exploratory): one batched step of B = 1, 2, 4, 8 sequences, each on
// its own slot at depth D, against production's single-token decode at the same depth, GPU time per command buffer,
// arms interleaved per rep with rotating order. Reports aggregate = B x production / step(B).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_DEPTHS=128,512] go test -count=1 -run '^TestMC3Step_throughput$' -v ./metal/
func TestMC3Step_throughput(t *testing.T) {
	const maxB = 8
	_, r := mc3RealResident(t, maxB+1, 1024)
	depths := []int{128, 512}
	if v := os.Getenv("GOINFER_METAL_MC3_DEPTHS"); v != "" {
		depths = nil
		for f := range strings.SplitSeq(v, ",") {
			var n int
			fmt.Sscan(strings.TrimSpace(f), &n)
			depths = append(depths, n)
		}
	}
	const reps, tokens = 7, 12
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc3-s1 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(7654321)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	for _, D := range depths {
		for sl := 0; sl <= maxB; sl++ {
			ids := make([]int, D)
			for i := range ids {
				ids[i] = rnd()
			}
			mc3Fill(t, r, sl, ids)
		}
		hb("depth %d: %d slots filled", D, maxB+1)
		arms := []int{0, 1, 2, 4, 8} // 0 = production single-token decode on slot maxB
		per := map[int][]float64{}   // arm -> per-rep median GPU ms per step
		for rep := range reps {
			for k := range arms {
				a := arms[(k+rep)%len(arms)]
				var ms []float64
				for range tokens {
					emb := mc3Emb(r, rnd())
					if a == 0 {
						if err := r.useKVSlot(maxB); err != nil {
							t.Fatal(err)
						}
						r.ForwardEmb(emb, D) // rewrites position D each token: the depth stays D
						ms = append(ms, (r.gpuEnd-r.gpuStart)*1e3)
						continue
					}
					seqs := make([]batchSeq, a)
					for m := range a {
						seqs[m] = batchSeq{slot: m, pos: D, emb: emb}
					}
					mc3Step(t, r, seqs)
					ms = append(ms, (r.gpuEnd-r.gpuStart)*1e3)
				}
				per[a] = append(per[a], med(ms))
			}
		}
		line := fmt.Sprintf("depth %d: production %.3f ms/token", D, med(per[0]))
		for _, a := range arms[1:] {
			var ratio []float64
			for i := range per[a] {
				ratio = append(ratio, float64(a)*per[0][i]/per[a][i])
			}
			sort.Float64s(ratio)
			line += fmt.Sprintf("  | B=%d step %.3f ms (%.2fx a token), aggregate %.3fx [%.3f-%.3f]", a, med(per[a]), med(per[a])/med(per[0]), med(ratio), ratio[0], ratio[len(ratio)-1])
		}
		hb("%s", line)
	}
}

// TestMC3Step_drawsMatchForwardSample: rows that carry a device draw return exactly the id production's ForwardSample
// draws on a twin slot for the same (temperature, seed, draw), in steps that mix drawing rows with logits rows — two
// temperatures, and a draw counter that advances every step, teacher-forced 12 steps.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3Step_drawsMatchForwardSample$' -v ./metal/
func TestMC3Step_drawsMatchForwardSample(t *testing.T) {
	const steps = 12
	depths := []int{7, 31, 64, 200}
	temps := []float64{0.8, 0, 1.3, 0} // 0: a logits row
	B := len(depths)
	_, r := mc3Resident(t, 2*B, 1024)
	if !r.SampleAvailable() {
		t.Skip("this resident cannot draw on-device")
	}
	seed := uint32(24681357)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
		mc3Fill(t, r, B+m, ids)
	}
	differ, draws := 0, 0
	for st := range steps {
		seqs := make([]batchSeq, B)
		wantID := make([]int, B)
		wantLogits := make([][]float32, B)
		for m := range B {
			seqs[m] = batchSeq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			if err := r.useKVSlot(B + m); err != nil {
				t.Fatal(err)
			}
			if temps[m] > 0 {
				d := &decoder.ResidentBatchDraw{Temperature: temps[m], Seed: uint64(9000 + m), Draw: uint64(st)}
				seqs[m].draw = d
				id, err := r.ForwardSample(seqs[m].emb, seqs[m].pos, d.Temperature, d.Seed, d.Draw)
				if err != nil {
					t.Fatal(err)
				}
				wantID[m] = id
			} else {
				wantLogits[m] = append([]float32(nil), r.ForwardEmb(seqs[m].emb, seqs[m].pos)...)
			}
		}
		logits, ids, err := r.forwardMulti(seqs)
		if err != nil {
			t.Fatal(err)
		}
		for m := range B {
			if temps[m] > 0 {
				draws++
				if ids[m] != wantID[m] || logits[m] != nil {
					t.Errorf("step %d row %d: drew %d (logits %v), ForwardSample drew %d", st, m, ids[m], logits[m] != nil, wantID[m])
					differ++
				}
				continue
			}
			for i := range logits[m] {
				if math.Float32bits(logits[m][i]) != math.Float32bits(wantLogits[m][i]) {
					t.Errorf("step %d row %d: logits differ from ForwardEmb at %d", st, m, i)
					differ++
					break
				}
			}
			if ids[m] != -1 {
				t.Errorf("step %d row %d: a logits row reported id %d", st, m, ids[m])
			}
		}
	}
	t.Logf("%d steps: %d device draws and %d logits rows checked, %d differ", steps, draws, steps*B-draws, differ)
}

// TestMC3StepBreakdown: EXPLORATORY, for MC3 S3 (step polish). Where a batched step's GPU time goes, in sequence: the
// step is timed whole, then with one category's kernels replaced by an empty kernel (so its dispatches still launch
// but do no work), and with every kernel empty (what is left is dispatch overhead alone). Categories: the batched
// matmuls (mc3_bt / mc3_btd / mc3_lm), attention, and the small per-row kernels (norm+quant, RoPE, KV store, ctx
// quant, SwiGLU, the packers). Median of 15 steps per arm, arms interleaved.
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_DEPTHS=128,512] go test -count=1 -run '^TestMC3StepBreakdown$' -v ./metal/
func TestMC3StepBreakdown(t *testing.T) {
	_, r := mc3RealResident(t, 9, 1024)
	b := r.batch
	lib, err := r.d.CompileLibrary("kernel void mc3_noop() {}", MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	noop, err := r.d.NewComputePipeline(lib, "mc3_noop")
	if err != nil {
		t.Fatal(err)
	}
	cats := map[string][]*Pipeline{
		"matmul":    {&b.bt, &b.btd, &b.lm},
		"attention": {&r.pAttn, &r.pAttnFA, &r.pAttnFACombine, &r.pAttnRows},
		"per-row":   {&r.pRms, &r.pRope2, &r.pKv, &r.pQv, &r.pSw, &b.pack, &b.packLM, &r.pRmsRows, &r.pQvRows, &r.pSwRows, &r.pRope2Rows, &r.pKvRows},
	}
	orig := map[*Pipeline]Pipeline{}
	for _, ps := range cats {
		for _, p := range ps {
			orig[p] = *p
		}
	}
	set := func(names ...string) {
		for p, o := range orig {
			*p = o
		}
		for _, n := range names {
			for _, p := range cats[n] {
				*p = noop
			}
		}
	}
	defer set()
	depths := []int{128}
	if v := os.Getenv("GOINFER_METAL_MC3_DEPTHS"); v != "" {
		depths = nil
		for f := range strings.SplitSeq(v, ",") {
			var n int
			fmt.Sscan(strings.TrimSpace(f), &n)
			depths = append(depths, n)
		}
	}
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	arms := []struct {
		name string
		off  []string
	}{{"full", nil}, {"-matmul", []string{"matmul"}}, {"-attention", []string{"attention"}}, {"-per-row", []string{"per-row"}},
		{"all empty", []string{"matmul", "attention", "per-row"}}}
	for _, D := range depths {
		for sl := range 8 {
			ids := make([]int, D)
			for i := range ids {
				ids[i] = 1000 + 7*i + sl
			}
			set()
			mc3Fill(t, r, sl, ids)
		}
		for _, B := range []int{2, 4, 8} {
			ms := map[string][]float64{}
			for rep := range 15 {
				for k := range arms {
					a := arms[(k+rep)%len(arms)]
					set(a.off...)
					seqs := make([]batchSeq, B)
					for m := range B {
						seqs[m] = batchSeq{slot: m, pos: D, emb: mc3Emb(r, 300+rep)}
					}
					if _, _, err := r.forwardMulti(seqs); err != nil {
						t.Fatal(err)
					}
					ms[a.name] = append(ms[a.name], (r.gpuEnd-r.gpuStart)*1e3)
				}
			}
			full := med(ms["full"])
			line := fmt.Sprintf("depth %d B=%d: step %.2f ms GPU", D, B, full)
			for _, a := range arms[1:4] {
				line += fmt.Sprintf(" | %s %.2f", a.name[1:], full-med(ms[a.name]))
			}
			line += fmt.Sprintf(" | dispatch overhead (all empty) %.2f", med(ms["all empty"]))
			fmt.Fprintf(os.Stderr, "[mc3-s3] %s\n", line)
		}
	}
	set()
}

// TestMC3StepWallVsGPU: EXPLORATORY, for an encode-ahead batched step. How much of a batched step's wall time is not
// GPU time — encoding its ~28 × 13 dispatches, commit and wait, and the host's logits copies — which is the most an
// encode-ahead executor (pre-encoding step n+1 while the GPU runs step n, as production decode does) could hide. In
// sequence, depth 128, median of 20 steps per arm: the batched step at B = 2, 4, 8, and production's single-token
// decode synchronous (ForwardEmb) and through its encode-ahead executor (ForwardEmbPipe).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_MODEL=…] go test -tags goinfer_testhooks -count=1 -run '^TestMC3StepWallVsGPU$' -v ./metal/
func TestMC3StepWallVsGPU(t *testing.T) {
	const maxB, D, steps = 8, 128, 20
	_, r := mc3RealResident(t, maxB+1, 1024)
	seed := uint32(97531)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for sl := 0; sl <= maxB; sl++ {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, sl, ids)
	}
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	report := func(name string, wall, gpu []float64) {
		w, g := med(wall), med(gpu)
		fmt.Fprintf(os.Stderr, "[mc3-wall] %-24s wall %7.3f ms  GPU %7.3f ms  not-GPU %6.3f ms (%.1f%% of wall)\n", name, w, g, w-g, 100*(w-g)/w)
	}
	for _, B := range []int{2, 4, 8} {
		var wall, gpu []float64
		for range steps + 2 {
			seqs := make([]batchSeq, B)
			for m := range B {
				seqs[m] = batchSeq{slot: m, pos: D, emb: mc3Emb(r, rnd())}
			}
			t0 := time.Now()
			mc3Step(t, r, seqs)
			wall = append(wall, time.Since(t0).Seconds()*1e3)
			gpu = append(gpu, (r.gpuEnd-r.gpuStart)*1e3)
		}
		report(fmt.Sprintf("batched step B=%d", B), wall[2:], gpu[2:])
	}
	for _, pipe := range []bool{false, true} {
		if err := r.useKVSlot(maxB); err != nil {
			t.Fatal(err)
		}
		var wall, gpu []float64
		for range steps + 2 {
			emb := mc3Emb(r, rnd())
			t0 := time.Now()
			if pipe {
				r.ForwardEmbPipe(emb, D)
			} else {
				r.ForwardEmb(emb, D)
			}
			wall = append(wall, time.Since(t0).Seconds()*1e3)
			gpu = append(gpu, (r.gpuEnd-r.gpuStart)*1e3)
		}
		r.stopExec()
		name := "production token (sync)"
		if pipe {
			name = "production token (pipe)"
		}
		report(name, wall[2:], gpu[2:])
	}
}

// TestMC3Verify_sameSlotRowsBitIdentical is the first gate of Metal spec verify on the step kernels
// (docs/tasks/task-concurrency-2026-09.md, MC4): a step whose rows are CONSECUTIVE positions of one sequence on one slot
// (a verify round) gives every logit and every K/V element bit-identical to production decoding those tokens one after
// another. Twin slots are filled with the same history; M = 2, 4, 8 rows at depths 128 and 2048, and a run straddling
// attnFADepthFloor, so one step mixes per-head and attention_fa rows.
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_MODEL=…] go test -tags goinfer_testhooks -count=1 -run '^TestMC3Verify_sameSlotRowsBitIdentical$' -v ./metal/
func TestMC3Verify_sameSlotRowsBitIdentical(t *testing.T) {
	_, r := mc3Resident(t, 2, 4096)
	seed := uint32(424242)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	kv := func(slot, from, n int) [][]uint16 {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2) + from*d
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
		}
		return out
	}
	cases := []struct{ D, M int }{{128, 2}, {128, 4}, {128, 8}, {2048, 2}, {2048, 4}, {2048, 8}, {attnFADepthFloor - 4, 8}}
	total := 0
	for _, c := range cases {
		ids := make([]int, c.D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, 0, ids)
		mc3Fill(t, r, 1, ids)
		embs := make([][]float32, c.M)
		for i := range embs {
			embs[i] = mc3Emb(r, rnd())
		}
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		var ref [][]float32
		for i, e := range embs {
			ref = append(ref, append([]float32(nil), r.ForwardEmb(e, c.D+i)...))
		}
		refKV := kv(0, c.D, c.M)
		seqs := make([]batchSeq, c.M)
		for i := range seqs {
			seqs[i] = batchSeq{slot: 1, pos: c.D + i, emb: embs[i]}
		}
		got := mc3Step(t, r, seqs)
		gotKV := kv(1, c.D, c.M)
		lg, kd := 0, 0
		for i := range got {
			for j := range got[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(ref[i][j]) {
					lg++
				}
			}
		}
		for i := range refKV {
			for j := range refKV[i] {
				if refKV[i][j] != gotKV[i][j] {
					kd++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[mc3-verify] depth %d, %d rows on one slot: %d of %d logits differ, %d K/V elements differ\n", c.D, c.M, lg, c.M*r.V, kd)
		if lg != 0 || kd != 0 {
			t.Errorf("depth %d, M = %d: the same-slot step differs from sequential decode (%d logits, %d K/V elements)", c.D, c.M, lg, kd)
		}
		total += lg + kd
	}
	t.Logf("%d differing values over %d cases", total, len(cases))
}

// TestMC3Step_promptInRowsBitIdentical is T1.10's byte comparison (E-P01's premise, docs/audit-metal-2026-09-30.md):
// a fresh prompt fed to the batched step in 8-row pieces from position 0, as a prompt below the prefill floor could be,
// writes the same K/V bytes and gives the same logits, every row, as production's sequential single-token loop.
func TestMC3Step_promptInRowsBitIdentical(t *testing.T) {
	_, r := mc3Resident(t, 2, 1024)
	seed := uint32(10101)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	kv := func(slot, n int) [][]uint16 { // per layer: K rows then V rows, positions 0..n-1
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
		}
		return out
	}
	for _, K := range []int{8, 16, 32, 64} {
		embs := make([][]float32, K)
		for i := range embs {
			embs[i] = mc3Emb(r, rnd())
		}
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		var ref [][]float32
		for i, e := range embs {
			ref = append(ref, append([]float32(nil), r.ForwardEmb(e, i)...))
		}
		refKV := kv(0, K)
		var got [][]float32
		for c := 0; c < K; c += 8 {
			var seqs []batchSeq
			for i := c; i < min(c+8, K); i++ {
				seqs = append(seqs, batchSeq{slot: 1, pos: i, emb: embs[i]})
			}
			for _, row := range mc3Step(t, r, seqs) {
				got = append(got, append([]float32(nil), row...))
			}
		}
		gotKV := kv(1, K)
		lg, kd := 0, 0
		for i := range got {
			for j := range got[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(ref[i][j]) {
					lg++
				}
			}
		}
		for i := range refKV {
			for j := range refKV[i] {
				if refKV[i][j] != gotKV[i][j] {
					kd++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[mc3-prompt] K = %d in 8-row steps from position 0: %d of %d logits differ, %d K/V elements differ\n", K, lg, K*r.V, kd)
		if len(got) != K || lg != 0 || kd != 0 {
			t.Errorf("K = %d: %d rows back, %d logits and %d K/V elements differ from the sequential loop", K, len(got), lg, kd)
		}
	}
}

// TestMC3Verify_rowCost is the second gate: what each extra verified row costs on the step kernels. GPU time of one
// step of M = 1, 2, 4, 8 consecutive rows on one slot, against a production token, at depths 128, 512 and 2048;
// median of 7, arms interleaved. Extra-row cost = (T(M)/T(token) - 1) / (M - 1).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_MODEL=…] go test -tags goinfer_testhooks -count=1 -run '^TestMC3Verify_rowCost$' -v ./metal/
func TestMC3Verify_rowCost(t *testing.T) {
	_, r := mc3RealResident(t, 2, 4096) // the step needs slot buffers, which a one-slot resident does not keep
	seed := uint32(13131)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	for _, D := range []int{128, 512, 2048} {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, 0, ids)
		arms := []int{0, 1, 2, 4, 8} // 0 = production token
		per := map[int][]float64{}
		for rep := range 9 {
			for k := range arms {
				M := arms[(k+rep)%len(arms)]
				if M == 0 {
					r.ForwardEmb(mc3Emb(r, rnd()), D)
				} else {
					seqs := make([]batchSeq, M)
					for i := range seqs {
						seqs[i] = batchSeq{slot: 0, pos: D + i, emb: mc3Emb(r, rnd())}
					}
					mc3Step(t, r, seqs)
				}
				if rep >= 2 {
					per[M] = append(per[M], (r.gpuEnd-r.gpuStart)*1e3)
				}
			}
		}
		tok := med(per[0])
		line := fmt.Sprintf("depth %d: production token %.3f ms", D, tok)
		for _, M := range arms[1:] {
			tm := med(per[M])
			extra := "—"
			if M > 1 {
				extra = fmt.Sprintf("%.3f", (tm/tok-1)/float64(M-1))
			}
			line += fmt.Sprintf("  | M=%d %.3f ms (%.2fx a token, extra row %s)", M, tm, tm/tok, extra)
		}
		fmt.Fprintf(os.Stderr, "[mc3-verify-cost] %s\n", line)
	}
}
