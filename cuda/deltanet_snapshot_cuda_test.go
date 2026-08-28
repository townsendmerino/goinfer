//go:build cuda && goinfer_testhooks

// MEASUREMENT ONLY — prices the DeltaNet state snapshot on the RESIDENT CUDA path, which is the
// regime that actually decides the narrow-snapshot question. Builds nothing: specRollbackSafe is
// untouched and no snapshot is wired into any decode path.
//
// WHY THIS AND NOT THE CPU NUMBER. docs/spec/09-mtp-heads.md priced the copy on CPU and recorded
// the direction: the numerator is a fixed 20.2 MiB while the denominator shrinks with every
// quantization and backend improvement, so the fraction grows over time by construction. On CPU
// f32 -> int8 already spanned most of the "cheap" band. This measures the endpoint that matters —
// a resident decode step, where decode is fastest relative to a fixed copy.
//
// AND THE SHAPE OF THE COPY CHANGES HERE, which is the part the CPU figure cannot speak to. On the
// resident path both pieces of state are ALREADY on the device (cuda/resident.go:240 — dnWin, the
// causal-conv ring, and dnState, the recurrent matrix). A snapshot is therefore a device-side
// copy, not a host memcpy, and the 13.9 GB/s host figure has no bearing on it in either direction.
//
//	GOINFER_QWEN35_08B=~/models/qwen3.5-0.8b \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestDeltaNetSnapshotCUDA -v -timeout 30m
package cuda

import (
	"math"
	"os"
	"sort"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

func statsD(d []time.Duration) (lo, med, hi time.Duration) {
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[0], s[len(s)/2], s[len(s)-1]
}

func TestDeltaNetSnapshotCUDA(t *testing.T) {
	ckpt := os.Getenv("GOINFER_QWEN35_08B")
	if ckpt == "" {
		ckpt = os.ExpandEnv("$HOME/models/qwen3.5-0.8b")
	}
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no checkpoint at %s (%v)", ckpt, err)
	}
	quant := os.Getenv("GOINFER_SNAPCOST_QUANT")
	if quant == "" {
		quant = "int4"
	}

	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Skipf("resident load (%s): %v", quant, err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident — this model did not take the resident path")
	}

	// Collect the linear layers' state buffers and their sizes.
	type st struct {
		win, state gpu.Buffer
		nWin, nSt  int
	}
	var states []st
	total := 0
	for l := range rf.layers {
		L := &rf.layers[l]
		if !L.isDeltaNet {
			continue
		}
		s := st{win: L.dnWin, state: L.dnState, nWin: L.dnWin.Len(), nSt: L.dnState.Len()}
		states = append(states, s)
		total += (s.nWin + s.nSt) * 4
	}
	if len(states) == 0 {
		t.Fatal("no DeltaNet layers on the resident path")
	}
	t.Logf("REGIME: qwen3.5-0.8b, quant=%s, RESIDENT CUDA, single sequence. 0.8B trunk — a larger "+
		"trunk has a SLOWER decode step and so a smaller fraction; this is the adverse direction.", quant)
	t.Logf("state: %d DeltaNet layers, %d B total (%.2f MiB) — dnWin %d + dnState %d floats/layer",
		len(states), total, float64(total)/(1<<20), states[0].nWin, states[0].nSt)

	// Prime so the conv ring is at steady state.
	for i := 0; i < 16; i++ {
		if _, err := rf.Forward(mc.EmbedResidentForTest(1+i), i); err != nil {
			t.Fatalf("prime forward %d: %v", i, err)
		}
	}

	// TWO SNAPSHOT PATHS, MEASURED IN THE SAME LOOP so the comparison is paired rather than
	// cross-session.
	//
	// (a) PCIe ROUND TRIP — what was implementable before aikit/gpu v0.31.0. With only Upload and
	//     Download, state that is already on the device has to come back to the host and go out
	//     again. Kept as the control: it is the number the passthrough was justified against.
	//
	// (b) DEVICE-TO-DEVICE via CopyDeviceBatch (v0.31.0). This is the in-situ measurement the
	//     synthetic-buffer probe stood in for — real state, real buffers, real decode between
	//     rounds. The probe reported ~446 us for snapshot+restore; a figure measured through the
	//     primitive on synthetic buffers is not an integration cost, which is why this exists.
	hw := make([][]float32, len(states))
	hs := make([][]float32, len(states))
	shadowWin := make([]gpu.Buffer, len(states))
	shadowSt := make([]gpu.Buffer, len(states))
	for i, s := range states {
		hw[i], hs[i] = make([]float32, s.nWin), make([]float32, s.nSt)
		shadowWin[i] = rf.dev.NewBufferLen(s.nWin)
		shadowSt[i] = rf.dev.NewBufferLen(s.nSt)
	}

	pcieSnapshot := func() {
		for i, s := range states {
			_ = gpu.Download(s.win, hw[i])
			_ = gpu.Download(s.state, hs[i])
		}
	}
	pcieRestore := func() {
		for i, s := range states {
			_ = gpu.Upload(s.win, hw[i])
			_ = gpu.Upload(s.state, hs[i])
		}
	}

	// One batch, one synchronize, for all 36 copies — the form aikit added for exactly this
	// consumer. A loop over CopyDevice would pay 36 synchronizes instead of one.
	toShadow := make([]gpu.DeviceCopy, 0, 2*len(states))
	toLive := make([]gpu.DeviceCopy, 0, 2*len(states))
	for i, s := range states {
		toShadow = append(toShadow,
			gpu.DeviceCopy{Dst: shadowWin[i], Src: s.win, Bytes: s.nWin * 4},
			gpu.DeviceCopy{Dst: shadowSt[i], Src: s.state, Bytes: s.nSt * 4})
		toLive = append(toLive,
			gpu.DeviceCopy{Dst: s.win, Src: shadowWin[i], Bytes: s.nWin * 4},
			gpu.DeviceCopy{Dst: s.state, Src: shadowSt[i], Bytes: s.nSt * 4})
	}
	d2dSnapshot := func() {
		if err := gpu.CopyDeviceBatch(toShadow); err != nil {
			t.Fatalf("CopyDeviceBatch (snapshot): %v", err)
		}
	}
	d2dRestore := func() {
		if err := gpu.CopyDeviceBatch(toLive); err != nil {
			t.Fatalf("CopyDeviceBatch (restore): %v", err)
		}
	}
	snapshot, restore := pcieSnapshot, pcieRestore
	if os.Getenv("GOINFER_SNAP_PCIE") == "" {
		snapshot, restore = d2dSnapshot, d2dRestore
		t.Logf("PATH: device-to-device (aikit/gpu CopyDeviceBatch, 36 copies, one synchronize)")
	} else {
		t.Logf("PATH: PCIe round trip (Download+Upload) — the pre-v0.31.0 control")
	}

	const rounds = 30
	var dec, snap, rest []time.Duration
	for r := 0; r < rounds; r++ {
		t0 := time.Now()
		if _, err := rf.Forward(mc.EmbedResidentForTest(1+(r%64)), 16+r); err != nil {
			t.Fatalf("decode forward: %v", err)
		}
		dec = append(dec, time.Since(t0))

		t1 := time.Now()
		snapshot()
		snap = append(snap, time.Since(t1))

		t2 := time.Now()
		restore()
		rest = append(rest, time.Since(t2))
	}
	dLo, dMed, dHi := statsD(dec)
	sLo, sMed, sHi := statsD(snap)
	rLo, rMed, rHi := statsD(rest)
	t.Logf("IN SITU, %d paired rounds (min / median / max):", rounds)
	t.Logf("  resident decode step  %8v %8v %8v", dLo, dMed, dHi)
	t.Logf("  snapshot (D2H)        %8v %8v %8v", sLo, sMed, sHi)
	t.Logf("  restore  (H2D)        %8v %8v %8v", rLo, rMed, rHi)

	// THE RATIO IS FORMED PER ROUND AND CARRIES ITS OWN SPREAD — not median(cost)/median(decode).
	// A ratio of medians hides the round-to-round covariance, and here that matters: the decode
	// step alone ranges 2.2x within one run, so a single "100.3%" says nothing about whether the
	// figure is 100 +/- 5 or 100 +/- 60. Both terms wander; only the paired form shows whether
	// they wander together.
	ratios := make([]float64, rounds)
	for i := range dec {
		ratios[i] = float64(snap[i]+rest[i]) / float64(dec[i])
	}
	sort.Float64s(ratios)
	rMedian, mean := ratios[len(ratios)/2], 0.0
	for _, r := range ratios {
		mean += r
	}
	mean /= float64(len(ratios))
	sd := 0.0
	for _, r := range ratios {
		sd += (r - mean) * (r - mean)
	}
	sd = math.Sqrt(sd / float64(len(ratios)-1))
	t.Logf("PAIRED RATIO, (snapshot+restore)/decode, formed per round (n=%d):", rounds)
	t.Logf("  min %.1f%%  p50 %.1f%%  max %.1f%%  |  mean %.1f%%  sd %.1f pp",
		100*ratios[0], 100*rMedian, 100*ratios[len(ratios)-1], 100*mean, 100*sd)
	t.Logf("  (ratio-of-medians would have reported %.1f%% — shown only to expose the gap)",
		100*float64(sMed+rMed)/float64(dMed))
	t.Logf("  traffic: %d B read+written twice = %.1f GB/s (TRAFFIC convention: a copy reads n and writes n)", total, float64(total)*2/float64((sMed+rMed).Nanoseconds()))
	for _, K := range []int{4, 7} {
		t.Logf("   p50 %% of a K=%d round IF a round cost K decode steps = %.2f%%", K, 100*rMedian/float64(K))
	}
	t.Logf("NOTE: on a bandwidth-bound resident decode a BATCHED verify of K+1 costs closer to ONE " +
		"decode step than to K, so the K-step framing is the FAVOURABLE reading here, not the neutral one.")
}
