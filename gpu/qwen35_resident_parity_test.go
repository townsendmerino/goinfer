//go:build gpu && goinfer_testhooks

package gpu

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/program"
)

// Whole-model parity for the Gated-DeltaNet hybrid: the resident runner vs the
// CPU program walker, fed the SAME fixed token sequence, logits compared as the run gets long.
//
// Long is the point. Three of every four layers carry a recurrent state that COMPOUNDS, so the
// wiring errors this family invites — a mixer misroute, a conv ring slip, the state left un-reset,
// the value-head → key-head GVA map inverted — do not show as a bad first token. They show as a
// cosine that decays. A single-token check would pass on all of them.
//
// The gated softmax layers are gated here too, and they are the half the residency PLAN did not
// anticipate: q_proj is double width ([query ‖ gate] per head) and the context is scaled by
// sigmoid(gate) before o_proj. Reading that weight as an ordinary q_proj produces plausible logits
// from the wrong tensor, which is exactly what an argmax-only check would wave through.
func TestQwen35ResidentParity(t *testing.T) {
	if os.Getenv("GOINFER_DNET_PARITY") == "" {
		t.Skip("qwen3.5 DeltaNet resident parity (set GOINFER_DNET_PARITY=1)")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	// BOTH siblings, because the resident bridge composes differently for each: the dense one pairs
	// the DeltaNet mixer with a plain SwiGLU, the MoE one with the sparse router + stacked experts +
	// shared expert in the same layer. Mixer+MoE is gated for Mamba-2 (Granite) and the mixer alone
	// by the dense fixture; neither gates THIS pairing.
	//
	// Both fixtures' model.safetensors are gitignored (only config.json is tracked), so on a fresh
	// clone every t.Run below skips, and Go reports a parent whose subtests all skipped as a
	// top-level PASS, which cmd/gate's webgpu-parity cell (TopLevelOnly) counts as real coverage
	// (docs/review-2026-09-04.md, V-06). subTs captures each subtest's *testing.T so the parent can
	// check Skipped() itself after every t.Run returns (inside the closure it would never run: Skip
	// calls runtime.Goexit) and skip itself when nothing ran.
	var subTs []*testing.T
	for _, fx := range []string{"qwen3_5-tiny", "qwen3_5_moe-tiny"} {
		t.Run(fx, func(st *testing.T) {
			subTs = append(subTs, st)
			qwen35ResidentParity(st, "../decoder/testdata/"+fx)
		})
	}
	if ck := os.Getenv("GOINFER_DNET_CKPT"); ck != "" { // a bigger/real checkpoint, opt-in
		t.Run("env", func(st *testing.T) {
			subTs = append(subTs, st)
			qwen35ResidentParity(st, ck)
		})
	}
	ran := false
	for _, st := range subTs {
		if !st.Skipped() {
			ran = true
		}
	}
	if !ran {
		t.Skip("no qwen3_5 fixture anywhere (model.safetensors is gitignored for both — run " +
			"scripts/pin_qwen3_5_forward.py [--moe] or set GOINFER_DNET_CKPT)")
	}
}

func qwen35ResidentParity(t *testing.T, ckpt string) {
	quant := "int8int8"
	if v := os.Getenv("GOINFER_DNET_QUANT"); v != "" {
		quant = v
	}
	qwen35ResidentParityQuant(t, ckpt, quant)
}

func qwen35ResidentParityQuant(t *testing.T, ckpt, quant string) {
	t.Helper()
	p := deltaNetParityWebGPU(t, ckpt, quant)
	// DRIFT is the wiring signal, and it is checked separately from the absolute floor on
	// purpose: a recurrence that is merely coarse sits flat and low, while one that is wired
	// wrong decays. Conflating the two into a single floor loses that distinction — and it is
	// the distinction that says whether to widen a tolerance or go find a bug.
	if p.first-p.worst > 0.02 {
		t.Errorf("cosine DRIFTS over the run: tok1 %.6f → worst %.6f (Δ%.4f). The recurrent state "+
			"compounds; a decaying cosine is a wiring error, not quantization noise", p.first, p.worst, p.first-p.worst)
	}
	if p.worst < 0.95 {
		t.Errorf("worst cosine %.6f < 0.95 — below the resident-path floor the other families hold", p.worst)
	}
	if p.replay < 0.9999999 {
		t.Errorf("replaying the same tokens after Reset gives DIFFERENT logits (worst self-cosine "+
			"%.9f): the recurrent state carried over from the first sequence. residentDecoder.Reset "+
			"must zero {win, dnState}, not just Mamba's {win, ssm}", p.replay)
	}
}

// deltaNetParity is one run of deltaNetParityWebGPU: the worst and first cosines against the CPU, and the replay's worst
// self-cosine.
type deltaNetParity struct{ worst, first, replay float64 }

// failsBar reports whether p is outside TestQwen35ResidentParity's bar.
func (p deltaNetParity) failsBar() bool {
	return p.first-p.worst > 0.02 || p.worst < 0.95 || p.replay < 0.9999999
}

// deltaNetParityWebGPU runs ckpt resident on WebGPU against the CPU at quant and returns the readings, asserting only that the
// build went resident.
func deltaNetParityWebGPU(t *testing.T, ckpt, quant string) deltaNetParity {
	t.Helper()
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		// stat the WEIGHTS, not the directory: the fixture's config.json is committed while
		// model.safetensors is gitignored, so a dir-existence guard would flip this skip into a
		// hard failure on a fresh clone.
		t.Skipf("no fixture at %s: %v", ckpt, err)
	}

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: quant})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		// Non-vacuity: this test is worthless if it silently measures CPU-vs-CPU, and a declined
		// residency is exactly how that happens.
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s", ckpt, mRes.DecodePath(), mRes.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	// The CPU arm runs at the SAME weight quant, so what is being measured is WIRING, not the
	// int8 gap. The resident path still quantizes activations the CPU keeps in f32, which is the
	// floor this cosine sits on — it should be flat, not decaying.
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	ntok := 128
	if v := os.Getenv("GOINFER_DNET_NTOK"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			ntok = n
		}
	}
	_, _, _, _, _, _, vocab := mCPU.Dims()
	checks := map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 64: true, 128: true, 512: true}

	cache := mCPU.NewCache(ntok + 1)
	rf.Reset()
	worst, first, agree := 1.0, 1.0, 0
	var firstRun [][]float32 // the resident's own opening logits, for the replay check below
	for i := range ntok {
		tok := (i*131 + 7) % vocab
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		if len(firstRun) < 16 {
			firstRun = append(firstRun, append([]float32(nil), lr...))
		}
		if argmaxF(lc) == argmaxF(lr) {
			agree++
		}
		cos, maxAbs := cosSim(lc, lr)
		if i == 0 {
			first = cos
		}
		if cos < worst {
			worst = cos
		}
		if checks[i+1] || i+1 == ntok {
			t.Logf("  tok %4d: cosine=%.6f maxAbs=%.4g argmax_match=%v", i+1, cos, maxAbs, argmaxF(lc) == argmaxF(lr))
		}
	}
	t.Logf("  worst cosine over %d tokens: %.6f (tok1 %.6f, drift %.4f); argmax agree %d/%d",
		ntok, worst, first, first-worst, agree, ntok)

	// SECOND GENERATION on the same model: a REPLAY, not another CPU comparison. The recurrent state
	// compounds and is NOT positional, so unlike a KV cache the next sequence cannot overwrite it;
	// it has to be zeroed (audit C-01), and the first loop cannot see that because the state buffers
	// are allocated zeroed. Comparing generation 2 against the CPU does not see it either (deleting
	// the DeltaNet arm of residentDecoder.Reset left the cosine above 0.95: the decay gate shrinks
	// the stale state faster than 16 tokens can notice). So this replays the FIRST run's opening
	// tokens and requires the resident to reproduce ITS OWN logits: same inputs + same (reset) state
	// give the same output to f32 determinism, a bound only leftover state can break and no
	// tolerance argument can absorb.
	const replay = 16
	rf.Reset()
	worstReplay := 1.0
	for i := range replay {
		lr, err := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("resident replay[%d]: %v", i, err)
		}
		if cos, _ := cosSim(firstRun[i], lr); cos < worstReplay {
			worstReplay = cos
		}
	}
	t.Logf("  replay of the first %d tokens after Reset: worst self-cosine %.9f", replay, worstReplay)
	return deltaNetParity{worst: worst, first: first, replay: worstReplay}
}

// TestOlmoHybridResidentParityWebGPU tests Olmo Hybrid resident decode parity on WebGPU against the CPU at int4 over 128 tokens
// (FeatNoPE + FeatPostOnlyNorm + FeatQKNormWhole + DeltaNet NegEigval + ONormEps). With either DeltaNet setting dropped at the
// build, it falls outside the parity bar.
func TestOlmoHybridResidentParityWebGPU(t *testing.T) {
	const fx = "../testdata/olmo_hybrid-tiny"
	if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
		t.Skipf("no olmo_hybrid-tiny weights: %v", err)
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	// olmo_hybrid-tiny's config.json sets max_position_embeddings: 64, which webgpuCtxBeforeSlots clamps ctxCap to.
	// Raise the test floor to 128 so the 128-token recurrent drift run fits the KV cache.
	minCtxCapForTest = 128
	defer func() { minCtxCapForTest = 0 }()

	t.Run("prompts take the sequential loop", func(t *testing.T) {
		m, err := decoder.Load(fx, decoder.Options{Backend: "webgpu", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if m.ResidentForwardForTest() == nil {
			t.Fatalf("not resident: %s", m.ResidentDecline())
		}
		if batched, why := m.PrefillPath(); batched {
			t.Errorf("olmo_hybrid-tiny's prefill is batched on WebGPU; parity run no longer covers its prompts (%s)", why)
		}
	})
	t.Run("as built", func(t *testing.T) { qwen35ResidentParityQuant(t, fx, "int4") })
	for _, plant := range []struct {
		name string
		on   *bool
	}{{"NegEigval dropped", &dnetNegEigvalOffForTest}, {"ONormEps dropped", &dnetONormEpsOffForTest}} {
		t.Run(plant.name, func(t *testing.T) {
			*plant.on = true
			defer func() { *plant.on = false }()
			p := deltaNetParityWebGPU(t, fx, "int4")
			t.Logf("planted (%s): worst %.6f, drift %.4f, replay %.9f", plant.name, p.worst, p.first-p.worst, p.replay)
			if !p.failsBar() {
				t.Errorf("with %s, olmo_hybrid-tiny still holds the bar: the gate does not see the setting", plant.name)
			}
		})
	}
}

// TestDeltaNetOpParams_layersMustAgree: the block holds one set of DeltaNet parameters, so a model whose DeltaNet layers disagree
// on one declines at the build, naming the layer. qwen35-tiny's program is rewritten on one DeltaNet layer, one parameter at a time.
func TestDeltaNetOpParams_layersMustAgree(t *testing.T) {
	const fx = "../testdata/qwen35-tiny"
	if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
		t.Skipf("no qwen35-tiny weights: %v", err)
	}
	m, err := decoder.Load(fx, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, _, err := deltaNetOpParams(m); err != nil {
		t.Fatalf("the unmodified model's layers disagree: %v", err)
	}
	last := func() *program.Op {
		var op *program.Op
		for _, ops := range m.Program().Layers {
			for i := range ops {
				if ops[i].Kind == program.OpDeltaNet {
					op = &ops[i]
				}
			}
		}
		return op
	}
	op := last()
	if op == nil {
		t.Fatal("no DeltaNet layer")
	}
	saved := op.Delta
	defer func() { op.Delta = saved }()
	for _, c := range []struct {
		name   string
		change func(p *program.DeltaNetOpParams)
	}{
		{"NegEigval", func(p *program.DeltaNetOpParams) { p.NegEigval = !p.NegEigval }},
		{"ONormEps", func(p *program.DeltaNetOpParams) { p.ONormEps = 1e-3 }},
	} {
		op.Delta = saved
		c.change(&op.Delta)
		if _, _, err := deltaNetOpParams(m); err == nil || !strings.Contains(err.Error(), "differ") {
			t.Errorf("%s disagreeing: %v, want the decline", c.name, err)
		}
	}
}

// TestParentSelfSkipsWhenNoSubtestRan pins, portably, the anti-vacuity mechanism of
// TestQwen35ResidentParity: a stub stands in for qwen35ResidentParity so this runs on any machine
// whatever fixtures are present, and it asserts on go test's own JSON action rather than hardware
// behaviour. The mechanism: Go reports a parent whose every t.Run subtest skipped as a top-level
// PASS, not SKIP, and cmd/gate's webgpu-parity cell counts top-level results only, so the parent
// tracks each subtest's t.Skipped() (checked from the OUTER test after t.Run returns, since Skip
// calls runtime.Goexit) and skips itself when nothing ran (docs/review-2026-09-04.md, V-06).
func TestParentSelfSkipsWhenNoSubtestRan(t *testing.T) {
	bin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go binary on PATH")
	}
	dir := t.TempDir()
	src := `package p
import "testing"
func TestParent(t *testing.T) {
	var subTs []*testing.T
	for _, name := range []string{"a", "b"} {
		t.Run(name, func(st *testing.T) {
			subTs = append(subTs, st)
			st.Skip("stub: simulated fixture absent")
		})
	}
	ran := false
	for _, st := range subTs {
		if !st.Skipped() {
			ran = true
		}
	}
	if !ran {
		t.Skip("nothing ran")
	}
}
`
	if err := os.WriteFile(filepath.Join(dir, "p_test.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module p\ngo 1.21\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin, "test", "-json", "./...")
	cmd.Dir = dir
	// The stub is its own module: a GOWORK or GOFLAGS inherited from the caller (cmd/gate sets an explicit
	// per-module workspace when the checkout has no go.work) would make the child refuse it.
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=")
	out, runErr := cmd.CombinedOutput()
	if runErr != nil {
		if _, ok := runErr.(*exec.ExitError); !ok {
			t.Fatalf("running stub module: %v\n%s", runErr, out)
		}
	}
	for line := range strings.SplitSeq(string(out), "\n") {
		if !strings.Contains(line, `"Test":"TestParent"`) {
			continue
		}
		var ev struct{ Action string }
		if jerr := json.Unmarshal([]byte(line), &ev); jerr != nil {
			continue
		}
		switch ev.Action {
		case "pass":
			t.Fatal("the all-subtests-skipped parent reported PASS, not SKIP — this is exactly " +
				"the vacuous-coverage bug: a webgpu forward could be broken with nothing ever " +
				"forwarded, and a gate counting top-level results only would still print PASS")
		case "skip":
			return // correct
		}
	}
	t.Fatalf("never saw a terminal action for TestParent — this guard is watching nothing; the child go test printed:\n%s", out)
}
