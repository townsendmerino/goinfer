//go:build gpu && goinfer_testhooks

package webgpu

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
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
			qwen35ResidentParity(st, "../../decoder/testdata/"+fx)
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
	return residentParityWebGPU(t, ckpt, decoder.Options{Quant: quant})
}

// residentParityWebGPU is one such run at options o on both sides.
func residentParityWebGPU(t *testing.T, ckpt string, o decoder.Options) deltaNetParity {
	t.Helper()
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		// stat the WEIGHTS, not the directory: the fixture's config.json is committed while
		// model.safetensors is gitignored, so a dir-existence guard would flip this skip into a
		// hard failure on a fresh clone.
		t.Skipf("no fixture at %s: %v", ckpt, err)
	}

	o.Backend = "webgpu"
	mRes, err := decoder.Load(ckpt, o)
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
	o.Backend = "cpu"
	mCPU, err := decoder.Load(ckpt, o)
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

// TestOlmoHybridResidentParityWebGPU holds the three families the claims admit (FeatNoPE, FeatPostOnlyNorm, FeatQKNormWhole, and
// Olmo Hybrid's NegEigval and ONormEps) to the CPU at the same options over 128 tokens, and requires a plant on each setting to fall
// outside the bar on a fixture that uses it (docs/tasks/task-olmo-hybrid-metal-webgpu-2026-11.md, § WebGPU).
func TestOlmoHybridResidentParityWebGPU(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	minCtxCapForTest = 128 // these fixtures' max_position_embeddings is 64
	defer func() { minCtxCapForTest = 0 }()
	type plant struct {
		name string
		on   *bool
		// minDrop, when set, holds the plant to a clean int4 run read beside it, not the bar: its worst cosine must fall by at least this.
		// NoPE moves a tiny random model's logits too little for the bar to see.
		minDrop float64
	}
	type cell struct {
		name string
		o    decoder.Options
	}
	int4, int8 := cell{"int4", decoder.Options{Quant: "int4"}}, cell{"int8int8", decoder.Options{Quant: "int8int8"}}
	f32, slots := cell{"f32", decoder.Options{Quant: "f32"}}, cell{"int4 on two slots", decoder.Options{Quant: "int4", ResidentKVSlots: 2}}
	kvF16, kvI8 := cell{"int4 kv f16", decoder.Options{Quant: "int4", KVPrecision: "f16"}}, cell{"int4 kv i8", decoder.Options{Quant: "int4", KVPrecision: "i8"}}
	slotsKVI8 := cell{"int4 kv i8 on two slots", decoder.Options{Quant: "int4", KVPrecision: "i8", ResidentKVSlots: 2}}
	for _, c := range []struct {
		fx     string
		cells  []cell
		plants []plant
	}{
		{"olmo_hybrid-tiny", []cell{int4, int8, f32, slots, kvF16, kvI8, slotsKVI8},
			[]plant{{name: "NegEigval dropped", on: &dnetNegEigvalOffForTest}, {name: "ONormEps dropped", on: &dnetONormEpsOffForTest}}},
		{"olmo3-tiny", []cell{int4, int8, f32, slots, kvF16, kvI8},
			[]plant{{name: "whole-vector QK-norm dropped", on: &qkNormWholeOffForTest}, {name: "post-only norm dropped", on: &postOnlyNormOffForTest}}},
		{"smollm3-tiny", []cell{int4, int8, f32, slots, kvF16, kvI8},
			[]plant{{name: "NoPE dropped", on: &noPEOffForTest, minDrop: 5e-5}}},
	} {
		fx := "../../testdata/" + c.fx
		t.Run(c.fx, func(t *testing.T) {
			if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
				t.Skipf("no %s weights: %v", c.fx, err)
			}
			for _, cl := range c.cells {
				t.Run(cl.name, func(t *testing.T) {
					p := residentParityWebGPU(t, fx, cl.o)
					t.Logf("%s %s: worst %.9f, drift %.4f, replay %.9f", c.fx, cl.name, p.worst, p.first-p.worst, p.replay)
					if p.failsBar() {
						t.Errorf("%s at %s is outside the bar: worst %.6f, drift %.4f, replay %.9f", c.fx, cl.name, p.worst, p.first-p.worst, p.replay)
					}
				})
			}
			for _, pl := range c.plants {
				t.Run(pl.name, func(t *testing.T) {
					clean := residentParityWebGPU(t, fx, decoder.Options{Quant: "int4"})
					*pl.on = true
					defer func() { *pl.on = false }()
					p := residentParityWebGPU(t, fx, decoder.Options{Quant: "int4"})
					t.Logf("planted (%s): worst %.9f against the clean run's %.9f, drift %.4f, replay %.9f", pl.name, p.worst, clean.worst, p.first-p.worst, p.replay)
					if pl.minDrop > 0 {
						if drop := clean.worst - p.worst; drop < pl.minDrop {
							t.Errorf("with %s, %s's worst falls by %.9f, under %g: the gate does not see the setting", pl.name, c.fx, drop, pl.minDrop)
						}
						return
					}
					if !p.failsBar() {
						t.Errorf("with %s, %s still holds the bar: the gate does not see the setting", pl.name, c.fx)
					}
				})
			}
		})
	}
}

// olmo3GQAFixture writes olmo3-tiny with half its KV heads: num_key_value_heads, k_proj, v_proj and k_norm cut to their first rows.
// Real Olmo 3 checkpoints are GQA, and whole-vector QK-norm then normalizes Q and K at different widths.
func olmo3GQAFixture(t *testing.T) string {
	t.Helper()
	const src = "../../testdata/olmo3-tiny"
	blob, err := os.ReadFile(src + "/model.safetensors")
	if err != nil {
		t.Skipf("no olmo3-tiny weights: %v", err)
	}
	n := binary.LittleEndian.Uint64(blob)
	body := blob[8+n:]
	var header map[string]json.RawMessage
	if err := json.Unmarshal(blob[8:8+n], &header); err != nil {
		t.Fatal(err)
	}
	type tensor struct {
		DType   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	names := make([]string, 0, len(header))
	for name := range header {
		if name != "__metadata__" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	out := map[string]any{}
	var data []byte
	for _, name := range names {
		var ts tensor
		if err := json.Unmarshal(header[name], &ts); err != nil {
			t.Fatal(err)
		}
		raw := body[ts.Offsets[0]:ts.Offsets[1]]
		if strings.Contains(name, "self_attn.k_proj") || strings.Contains(name, "self_attn.v_proj") || strings.Contains(name, "self_attn.k_norm") {
			ts.Shape = slices.Clone(ts.Shape)
			ts.Shape[0] /= 2
			raw = raw[:len(raw)/2]
		}
		ts.Offsets = [2]int64{int64(len(data)), int64(len(data) + len(raw))}
		data = append(data, raw...)
		out[name] = ts
	}
	head, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	st := binary.LittleEndian.AppendUint64(nil, uint64(len(head)))
	if err := os.WriteFile(dir+"/model.safetensors", slices.Concat(st, head, data), 0o644); err != nil {
		t.Fatal(err)
	}
	cfgBlob, err := os.ReadFile(src + "/config.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(cfgBlob, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["num_key_value_heads"] = cfg["num_attention_heads"].(float64) / 2
	cfgOut, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+"/config.json", cfgOut, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestOlmo3GQAResidentParityWebGPU: admission claims whole-vector QK-norm whatever the model's KV heads, so a GQA Olmo 3 holds the same
// bar as the MHA fixture, and dropping the whole-vector form is red on it.
func TestOlmo3GQAResidentParityWebGPU(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	minCtxCapForTest = 128
	defer func() { minCtxCapForTest = 0 }()
	fx := olmo3GQAFixture(t)
	for _, o := range []decoder.Options{{Quant: "int4"}, {Quant: "int8int8"}, {Quant: "int4", KVPrecision: "i8"}} {
		t.Run(o.Quant+"/"+o.KVPrecision, func(t *testing.T) {
			p := residentParityWebGPU(t, fx, o)
			t.Logf("GQA olmo3 %s %s: worst %.9f, drift %.4f, replay %.9f", o.Quant, o.KVPrecision, p.worst, p.first-p.worst, p.replay)
			if p.failsBar() {
				t.Errorf("outside the bar: worst %.6f, drift %.4f, replay %.9f", p.worst, p.first-p.worst, p.replay)
			}
		})
	}
	t.Run("whole-vector QK-norm dropped", func(t *testing.T) {
		qkNormWholeOffForTest = true
		defer func() { qkNormWholeOffForTest = false }()
		p := residentParityWebGPU(t, fx, decoder.Options{Quant: "int4"})
		t.Logf("planted: worst %.9f, drift %.4f", p.worst, p.first-p.worst)
		if !p.failsBar() {
			t.Error("with the whole-vector form dropped, the GQA model still holds the bar")
		}
	})
}

// TestOlmoPromptRouteWebGPU: the batched pass takes no post-only or whole-vector QK-norm model, so Olmo's prompts run the sequential
// loop, which the parity test measures. SmolLM3 takes the pass, whose last row must equal the sequential decode's at the bar of
// prefillrunner_test.go, and which must read its NoPE layers' table. Every family generates through its route.
func TestOlmoPromptRouteWebGPU(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	const rows = 24
	load := func(t *testing.T, fx, quant string) (*decoder.Model, *residentDecoder, [][]float32) {
		t.Helper()
		if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
			t.Skipf("no weights at %s: %v", fx, err)
		}
		m, err := decoder.Load(fx, decoder.Options{Backend: "webgpu", Quant: quant})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("%s: not resident: %s", fx, m.ResidentDecline())
		}
		_, _, _, _, _, _, V := m.Dims()
		embs := make([][]float32, rows)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*131 + 7) % V)
		}
		return m, rf.(*residentDecoder), embs
	}
	generates := func(t *testing.T, m *decoder.Model, V int) {
		t.Helper()
		prompt := make([]int, rows)
		for i := range prompt {
			prompt[i] = (i*131 + 7) % V
		}
		ch, g := m.Generate(t.Context(), prompt, 4, decoder.SamplingParams{})
		n := 0
		for range ch {
			n++
		}
		if err := g.Err(); err != nil || n == 0 {
			t.Errorf("generate: %d tokens, %v", n, err)
		}
	}
	for _, fx := range []string{"olmo_hybrid-tiny", "olmo3-tiny"} {
		t.Run(fx, func(t *testing.T) {
			m, rd, embs := load(t, "../../testdata/"+fx, "int4")
			if _, err := rd.PrefillLast(t.Context(), embs, 0); err == nil || !strings.Contains(err.Error(), "declines") {
				t.Errorf("the batched pass took %s (%v): the parity test no longer covers its prompts", fx, err)
			}
			_, _, _, _, _, _, V := m.Dims()
			generates(t, m, V)
		})
	}
	for _, quant := range []string{"int4", "int8int8"} {
		t.Run("smollm3-tiny/"+quant, func(t *testing.T) {
			const fx = "../../testdata/smollm3-tiny"
			m, rd, embs := load(t, fx, quant)
			rd.Reset()
			var seq []float32
			for i, e := range embs {
				l, err := rd.Forward(e, i)
				if err != nil {
					t.Fatal(err)
				}
				seq = append(seq[:0], l...)
			}
			got, err := rd.PrefillLast(t.Context(), embs, 0)
			if err != nil {
				t.Fatalf("the batched pass: %v", err)
			}
			cos, maxAbs := cosSim(seq, got)
			t.Logf("last row against the sequential decode: cosine %.8f, maxAbs %.3e", cos, maxAbs)
			if cos < 0.999999 || maxAbs > 1e-3 {
				t.Errorf("the batched pass diverges from the sequential decode: cosine %.8f, maxAbs %.3e", cos, maxAbs)
			}
			_, _, _, _, _, _, V := m.Dims()
			generates(t, m, V)

			noPEOffForTest = true
			defer func() { noPEOffForTest = false }()
			_, planted, plantedEmbs := load(t, fx, quant)
			pl, err := planted.PrefillLast(t.Context(), plantedEmbs, 0)
			if err != nil {
				t.Fatalf("the batched pass with NoPE dropped: %v", err)
			}
			if pcos, pAbs := cosSim(seq, pl); pcos >= 0.999999 && pAbs <= 1e-3 {
				t.Errorf("with NoPE dropped the pass still matches (cosine %.8f, maxAbs %.3e): it does not read the NoPE layers' table", pcos, pAbs)
			}
		})
	}
}

// TestDeltaNetOpParams_layersMustAgree: the block holds one set of DeltaNet parameters, so a model whose DeltaNet layers disagree
// on one declines at the build, naming the layer. qwen35-tiny's program is rewritten on one DeltaNet layer, one parameter at a time.
func TestDeltaNetOpParams_layersMustAgree(t *testing.T) {
	const fx = "../../testdata/qwen35-tiny"
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
