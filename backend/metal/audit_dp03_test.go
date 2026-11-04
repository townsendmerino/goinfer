//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDP03_kSlotsDecode is D-P03's whole-token check (docs/audit-metal-2026-09-30.md, "D-P03"): resident MoE decode
// with the k selected experts in four dispatches (moeKSlotsOn, parked off by default) against the per-slot loop's 3k, on real
// weights, shaped as TestDP04_routeSGDecode. Identity: every logit of every token bit-equal between the arms. Speed:
// the token's GPU time, arms interleaved and alternated rep by rep from the same positions, paired per rep.
// GOINFER_DP04_MODEL picks the checkpoint (default ~/models/qwen15-moe-a27b-l4slice).
//
//	GOINFER_DP04=1 go test -tags goinfer_testhooks -count=1 -run '^TestDP03_kSlotsDecode$' -v ./metal/
func TestDP03_kSlotsDecode(t *testing.T) {
	if os.Getenv("GOINFER_DP04") != "1" {
		t.Skip("set GOINFER_DP04=1 (loads a real MoE checkpoint)")
	}
	path := os.Getenv("GOINFER_DP04_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen15-moe-a27b-l4slice")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	const tokens = 48
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: tokens + 64})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	t0 := time.Now()
	nMoE := 0
	for l := range r.layers {
		if r.layers[l].moe != nil {
			nMoE++
			if r.layers[l].moe.pool != nil {
				t.Fatalf("layer %d is paged: this covers the resident path", l)
			}
		}
	}
	if r.moe == nil || nMoE == 0 {
		t.Fatalf("%s: not a generic resident MoE (moe %v, %d MoE layers)", path, r.moe != nil, nMoE)
	}
	if !r.moe.kSlots {
		t.Fatalf("%s: the k-slot path is not selected", path)
	}
	prev := moeKSlotsOn
	set := func(arm string) {
		moeKSlotsOn = arm != "loop"
		r.stopExec()
	}
	defer func() { moeKSlotsOn = prev; r.stopExec() }()
	embs := auditEmbs(r, tokens, 29)

	// identity, cold from position 0 in each arm
	logits := map[string][][]float32{}
	for _, arm := range []string{"loop", "kslots"} {
		set(arm)
		for i, e := range embs {
			logits[arm] = append(logits[arm], append([]float32(nil), r.ForwardEmb(e, i)...))
		}
	}
	for i := range embs {
		d := 0
		for j, x := range logits["loop"][i] {
			if math.Float32bits(x) != math.Float32bits(logits["kslots"][i][j]) {
				d++
			}
		}
		if d != 0 {
			t.Fatalf("token %d: %d of %d logits differ between the per-slot loop and the k-slot dispatches", i, d, len(logits["kslots"][i]))
		}
	}
	auditHB("d-p03", t0, "%s: %d tokens, %d MoE layers: logits bit-identical between the arms", path, tokens, nMoE)

	// speed: each rep, both arms decode positions 0..tokens-1 (the KV is rewritten each time), GPU time per token
	reps := auditReps(7)
	ms := map[string][]float64{}
	for rep := range reps {
		order := []string{"loop", "kslots"}
		if rep%2 == 1 {
			order = []string{"kslots", "loop"}
		}
		for _, arm := range order {
			set(arm)
			var ks []float64
			for i, e := range embs {
				r.ForwardEmb(e, i)
				ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
			}
			ms[arm] = append(ms[arm], auditMedian(ks))
		}
	}
	ratio, saved := make([]float64, reps), make([]float64, reps)
	for i := range reps {
		ratio[i] = ms["loop"][i] / ms["kslots"][i]
		saved[i] = (ms["loop"][i] - ms["kslots"][i]) * 1e3 / float64(nMoE)
	}
	auditHB("d-p03", t0, "token GPU ms, per rep: loop %s; k-slot %s", auditFmt3(ms["loop"]), auditFmt3(ms["kslots"]))
	auditHB("d-p03", t0, "RESULT %s: token loop/k-slot median %.3f (per rep %s); saved %.1f us per MoE layer (median)",
		path, auditMedian(ratio), auditFmt3(ratio), auditMedian(saved))
}
