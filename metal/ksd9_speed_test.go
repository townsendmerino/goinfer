//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// TestKSD9_layerMajorSpeed is D9-G2 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D9 gates" and its model amendment): the
// generic MoE at GOINFER_KSD9_MODEL paged on Metal (GOINFER_METAL_MOE_SLOTS slots a layer, 16 unless set), a 512-row
// layer-major prefill (PrefillLast), the batched phase 2 against the per-row one (moeExpertBatchOn) on one resident, A B B A
// over 3 valid pairs (A B, B A, A B) after one warm run each. voidWindow brackets each timed prefill, and a pair with an arm over
// the void rule's line is void and re-run, up to 6 times. Each arm must take the phase 2 it names, and each pair's last logits
// must be byte-equal (the change is bit-identical); the test fails otherwise. A timed run: it refuses to run outside the timing
// lock (GOINFER_KSD9_TIMING=1).
func TestKSD9_layerMajorSpeed(t *testing.T) {
	path := os.Getenv("GOINFER_KSD9_MODEL")
	if os.Getenv("GOINFER_KSD9_TIMING") == "" || path == "" {
		t.Skip("set GOINFER_KSD9_TIMING=1 and GOINFER_KSD9_MODEL (a timed run under the timing lock)")
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	slots := os.Getenv("GOINFER_METAL_MOE_SLOTS")
	if slots == "" {
		slots = "16"
	}
	t.Setenv("GOINFER_METAL_MOE_SLOTS", slots)
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[D9-G2] FIT-REFUSED: %v\n", err)
		}
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || a.r.moe == nil || !a.r.moe.paged {
		t.Fatalf("not a paged generic MoE at %s slots: %s", slots, m.ResidentDecline())
	}
	rng := rand.New(rand.NewSource(97))
	ids := make([]int, 512)
	for i := range ids {
		ids[i] = rng.Intn(min(a.r.V, 32000))
	}
	embs := getEmbs(a.r, ids)
	prev := moeExpertBatchOn
	defer func() { moeExpertBatchOn = prev }()
	run := func(batched bool) (float64, []float32) {
		moeExpertBatchOn = batched
		groups := a.r.moeBatchGroups
		t0 := time.Now()
		lg, err := a.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatal(err)
		}
		sec := time.Since(t0).Seconds()
		if took := a.r.moeBatchGroups > groups; took != batched {
			t.Fatalf("batched phase 2 taken %v on the %v arm", took, map[bool]string{true: "batched", false: "per-row"}[batched])
		}
		return sec, slices.Clone(lg)
	}
	run(false)
	run(true)
	// timed runs one arm inside voidWindow; valid is false if the arm crossed the void rule's line.
	timed := func(batched bool) (sec float64, lg []float32, valid bool, note string) {
		end := voidWindow(t)
		sec, lg = run(batched)
		valid, note = end()
		return sec, lg, valid, note
	}
	var ratios []float64
	faster, voids := 0, 0
	for len(ratios) < 3 {
		var p, b float64
		var lp, lb []float32
		var vp, vb bool
		var np, nb string
		if len(ratios)%2 == 0 {
			p, lp, vp, np = timed(false)
			b, lb, vb, nb = timed(true)
		} else {
			b, lb, vb, nb = timed(true)
			p, lp, vp, np = timed(false)
		}
		d := 0
		for j := range lp {
			if math.Float32bits(lp[j]) != math.Float32bits(lb[j]) {
				d++
			}
		}
		if d != 0 {
			t.Fatalf("%d of %d last logits differ, batched against per-row", d, len(lp))
		}
		if !vp || !vb {
			voids++
			fmt.Fprintf(os.Stderr, "[D9-G2] VOID pair (per-row: %s; batched: %s)\n", np, nb)
			if voids > 6 {
				t.Fatalf("over 6 void pairs")
			}
			time.Sleep(30 * time.Second)
			continue
		}
		ratios = append(ratios, b/p)
		if b < p {
			faster++
		}
		fmt.Fprintf(os.Stderr, "[D9-G2] pair %d (per-row: %s; batched: %s): per-row %.3f s, batched %.3f s, batched/per-row %.4f; 0 of %d last logits differ\n",
			len(ratios)-1, np, nb, p, b, b/p, len(lp))
	}
	slices.Sort(ratios)
	verdict := "PARK"
	if ratios[1] <= 0.98 && faster == 3 {
		verdict = "SHIP"
	}
	fmt.Fprintf(os.Stderr, "[D9-G2] %s at %s slots: median batched/per-row %.4f, %d of 3 pairs faster, %d void pairs: %s\n",
		path, slots, ratios[1], faster, voids, verdict)
}

// TestOP4mD1_prefillStepsSpeed is OP4m-d1's D-4 (docs/tasks/task-forward-architecture-2026-10.md §4.9): on the model at
// GOINFER_OP4MD1_MODEL, loaded int4, a 3,900-token batched prefill (PrefillLast), the layer steps against the body that chose its
// blocks by the layer's kind (prefillLayerSwitchForTest) on one resident, interleaved A B B A over 6 pairs after one warm run
// each. Both arms' last logits must be byte-equal. The cell records the 1-minute load and the reclaimable memory against the
// resident's need plus 2 GiB at its start; a pair whose starting 1-minute load is above half the cores plus GOMAXPROCS voids the
// cell. Pass: a median rate ratio, steps ÷ switch, of at least 0.99. A timed run: it refuses to run outside the timing lock
// (GOINFER_OP4MD1_TIMING=1).
func TestOP4mD1_prefillStepsSpeed(t *testing.T) {
	path := os.Getenv("GOINFER_OP4MD1_MODEL")
	if os.Getenv("GOINFER_OP4MD1_TIMING") == "" || path == "" {
		t.Skip("set GOINFER_OP4MD1_TIMING=1 and GOINFER_OP4MD1_MODEL (a timed run under the timing lock)")
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	const promptLen, pairs, ctx = 3900, 6, 4096
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
	r0, ok := probe.Read()
	if !ok {
		t.Fatal("no memory reading")
	}
	l0 := load1()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: ctx})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not a Metal resident: %s", m.ResidentDecline())
	}
	if !a.r.prefillOK {
		t.Fatalf("%s does not take the batched prefill", filepath.Base(path))
	}
	need := m.ResidentNeedBytes("metal", 0, ctx, false, false)
	gib := func(b int64) float64 { return float64(b) / (1 << 30) }
	bar := float64(runtime.NumCPU())/2 + float64(runtime.GOMAXPROCS(0))
	memOK := int64(r0.Reclaimable) >= need+2<<30
	fmt.Fprintf(os.Stderr, "[OP4m-d1 D-4] %s: cell start: 1-minute load %.2f; reclaimable %.2f GiB against need %.2f + 2 GiB (%s); pair bar %.1f\n",
		filepath.Base(path), l0, gib(int64(r0.Reclaimable)), gib(need), map[bool]string{true: "ok", false: "VOID"}[memOK], bar)
	if !memOK {
		t.Fatalf("cell void: reclaimable %.2f GiB under need %.2f + 2 GiB", gib(int64(r0.Reclaimable)), gib(need))
	}
	rng := rand.New(rand.NewSource(3900))
	ids := make([]int, promptLen)
	for i := range ids {
		ids[i] = rng.Intn(min(a.r.V, 32000))
	}
	embs := getEmbs(a.r, ids)
	defer func() { prefillLayerSwitchForTest = false }()
	run := func(old bool) (float64, []float32) {
		prefillLayerSwitchForTest = old
		t0 := time.Now()
		lg, err := a.PrefillLast(context.Background(), embs, 0)
		el := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		return float64(promptLen) / el.Seconds(), lg
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
		var rs, ro float64
		var ls, lo []float32
		if i%2 == 0 {
			rs, ls = run(false)
			ro, lo = run(true)
		} else {
			ro, lo = run(true)
			rs, ls = run(false)
		}
		for j := range ls {
			if math.Float32bits(ls[j]) != math.Float32bits(lo[j]) {
				t.Fatalf("pair %d: the arms' last logits differ at %d (%g, %g)", i, j, ls[j], lo[j])
			}
		}
		ratios = append(ratios, rs/ro)
		if rs > ro {
			faster++
		}
		fmt.Fprintf(os.Stderr, "[OP4m-d1 D-4] pair %d (load %.2f): steps %.1f tok/s, switch %.1f tok/s, steps/switch %.4f; logits equal\n",
			i, lp, rs, ro, rs/ro)
	}
	slices.Sort(ratios)
	med := (ratios[pairs/2-1] + ratios[pairs/2]) / 2
	verdict := "PASS"
	switch {
	case med >= 0.99:
	case faster == 0:
		verdict = "FAIL (every pair slower, median under 0.99)"
	default:
		verdict = "PARKED (median under 0.99, not resolved)"
	}
	fmt.Fprintf(os.Stderr, "[OP4m-d1 D-4] %s: median steps/switch %.4f, %d of %d pairs faster, ratios %.4f: %s\n",
		filepath.Base(path), med, faster, pairs, ratios, verdict)
	if med < 0.99 {
		t.Errorf("median steps/switch %.4f under 0.99: %s", med, verdict)
	}
}
