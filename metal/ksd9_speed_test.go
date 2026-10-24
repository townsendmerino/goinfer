//go:build darwin && goinfer_testhooks

package metal

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

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// TestKSD9_layerMajorSpeed is D9-G2 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D9 gates" and its model amendment): the
// generic MoE at GOINFER_KSD9_MODEL paged on Metal (GOINFER_METAL_MOE_SLOTS slots a layer, 16 unless set), a 512-row
// layer-major prefill (PrefillLast), the batched phase 2 against the per-row one (moeExpertBatchOn) on one resident, A B B A
// over 3 valid pairs after one warm run each. Each arm reads the void rule's terms (ksd9Guard) just before and just after its
// timed run, and a pair with any read over the line is void and re-run, up to 6 times. Each pair's last logits must be
// byte-equal (the change is bit-identical); the test fails if any differ. A timed run: it refuses to run outside the timing
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
		t0 := time.Now()
		lg, err := a.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds(), slices.Clone(lg)
	}
	run(false)
	run(true)
	// timed runs one arm with the void rule's terms read just before and just after it; valid is false if either read fails.
	timed := func(batched bool) (sec float64, lg []float32, valid bool, note string) {
		ok1, l1, r1 := ksd9Guard(t)
		sec, lg = run(batched)
		ok2, l2, r2 := ksd9Guard(t)
		return sec, lg, ok1 && ok2, fmt.Sprintf("load %.2f→%.2f, reclaimable %.2f→%.2f GiB", l1, l2, float64(r1)/(1<<30), float64(r2)/(1<<30))
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

// ksd9Guard reads the void rule's two terms now: ok iff the 1-minute load is at most half the cores and at least 2 GiB is
// reclaimable. The test fails if either cannot be read.
func ksd9Guard(t *testing.T) (ok bool, load float64, recl int64) {
	t.Helper()
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		t.Fatalf("load average: %v", err)
	}
	text := strings.Trim(strings.TrimSpace(string(out)), "{} ")
	if n, err := fmt.Sscanf(text, "%g", &load); n != 1 || err != nil {
		t.Fatalf("load average %q: %v", text, err)
	}
	r, rok := hostmem.Read()
	if !rok {
		t.Fatal("no memory reading")
	}
	return load <= float64(runtime.NumCPU())/2 && r.Reclaimable >= 2<<30, load, r.Reclaimable
}
