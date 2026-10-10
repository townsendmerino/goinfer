//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDB01_prefillAgreesWithSequential (D-B01): a Gated-DeltaNet hybrid's batched prefill pass against the sequential
// decode loop on the same prompt: the last position's logits (cosine, top-1), every attention layer's K/V and every
// DeltaNet layer's recurrent state after the prompt, as relative L2. The pass is the gated lane (f16 projections), so
// this is a sanity bar, not the grade: the pooled fidelity gate decides. With dnetPrefillOn off the model must decline
// the pass. GOINFER_DB01_MODEL names the checkpoint (default ~/models/qwen3.5-0.8b).
func TestDB01_prefillAgreesWithSequential(t *testing.T) {
	path := os.Getenv("GOINFER_DB01_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.5-0.8b")
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	prevOff := dnetPrefillOn
	dnetPrefillOn = false // the switch's off arm (it is on by default since D-B01's grade)
	off, err := buildResident(m)
	dnetPrefillOn = prevOff
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	declined := !off.prefillOK
	off.Close()
	if !declined || off.dnet == nil {
		t.Fatalf("with dnetPrefillOn off a DeltaNet model must decline the pass (prefillOK %v, dnet %v)", off.prefillOK, off.dnet != nil)
	}
	prevDnet := dnetPrefillOn
	dnetPrefillOn = true
	defer func() { dnetPrefillOn = prevDnet }()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	if !r.prefillOK {
		t.Fatal("with dnetPrefillOn the model still declines the pass")
	}
	tk, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	snap := func(M int) (kv [][]float32, st [][]float32) {
		for l := range r.layers {
			L := &r.layers[l]
			if L.delta != nil {
				st = append(st, append([]float32(nil), L.delta.state.Floats()...))
				continue
			}
			dd := L.geom.kvDim
			var row []float32
			for _, b := range []Buffer{r.kc[l], r.vc[l]} {
				for _, h := range b.U16s()[:M*dd] {
					row = append(row, f16ToF32(h))
				}
			}
			kv = append(kv, row)
		}
		return
	}
	rel := func(a, ref []float32) float64 {
		var n, d float64
		for i := range ref {
			x := float64(a[i]) - float64(ref[i])
			n += x * x
			d += float64(ref[i]) * float64(ref[i])
		}
		return math.Sqrt(n / d)
	}
	// GOINFER_DB01_CPU_REF=1: the CPU backend at f32 with exact attention as a third party, so the pass's distance from
	// the sequential path can be read as whose error it is (decode's int8 activations, or the pass's f16 ones).
	var cpu *decoder.Model
	if os.Getenv("GOINFER_DB01_CPU_REF") == "1" {
		t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
		if cpu, err = decoder.Load(path, decoder.Options{Backend: "cpu", ResidentContext: 320}); err != nil {
			t.Fatalf("cpu reference load: %v", err)
		}
		defer cpu.Close()
	}
	for _, M := range []int{64, 256} {
		ids := decoder.PrefillGateProseIDsForTest(t, tk, files[0], M)[:M]
		r.resetDeltaNet()
		var seq []float32
		for i, id := range ids {
			seq = r.Forward(id, i)
		}
		seq = append([]float32(nil), seq...)
		kvS, stS := snap(M)
		r.resetDeltaNet()
		pass := r.PrefillLast(getEmbs(r, ids), 0)
		if pass == nil {
			t.Fatalf("M=%d: the pass returned no logits: %v", M, r.takeExecErr())
		}
		kvP, stP := snap(M)
		worstKV, worstSt := 0.0, 0.0
		var prof []string
		for i := range kvS {
			x := rel(kvP[i], kvS[i])
			worstKV = math.Max(worstKV, x)
			prof = append(prof, fmt.Sprintf("attn%d %.3f", i, x))
		}
		for i := range stS {
			x := rel(stP[i], stS[i])
			worstSt = math.Max(worstSt, x)
			prof = append(prof, fmt.Sprintf("dn%d %.3f", i, x))
		}
		t.Logf("M=%d per layer: %v", M, prof)
		msg := fmt.Sprintf("M=%d: logits cosine %.6f, top-1 seq %d pass %d; worst attention-layer K/V rel L2 %.3e, worst DeltaNet state rel L2 %.3e",
			M, cosF(seq, pass), argmaxF(seq), argmaxF(pass), worstKV, worstSt)
		t.Log(msg)
		if cpu != nil {
			ref, err := cpu.PrefillLogitsForTest(context.Background(), ids, cpu.NewCache(M))
			if err != nil {
				t.Fatalf("cpu reference: %v", err)
			}
			t.Logf("M=%d against CPU f32: KL(ref || seq) %.4e, KL(ref || pass) %.4e; cosine seq %.6f pass %.6f; top-1 ref %d seq %d pass %d",
				M, klLogits(ref, seq), klLogits(ref, pass), cosF(ref, seq), cosF(ref, pass), argmaxF(ref), argmaxF(seq), argmaxF(pass))
		}
		// Sanity bars, wide on purpose: the two arms differ by the activation lane (decode's int8 per row against the
		// pass's f16), about 5-10% relative L2 per layer from the first layer the model's outlier activations reach
		// (on the 0.8B), and against a CPU f32 reference both sit equally far (GOINFER_DB01_CPU_REF).
		// A defect in the pass's layout or a skipped stage reads far above them; the pooled gate grades the rest.
		if c := cosF(seq, pass); c < 0.99 || argmaxF(seq) != argmaxF(pass) || worstSt > 0.25 || worstKV > 0.25 {
			t.Errorf("%s: the pass is not near the sequential path", msg)
		}
	}
}

// TestDB01_chunkedPrefillMatchesWhole (D-B01): a hybrid's prompt prefilled in chunks (64 then 32 at startPos 64, and 40
// then 56) leaves the same last logits, attention K/V and DeltaNet window and state, bit for bit, as one 96-token pass.
// The decoder's chunked prefill (serve -prefill-chunk) relies on it, as TestMC5_prefillChunkInvariance pins for dense
// models: each chunk's conv window and state continue exactly where the last left them, and the pass's GEMMs and
// attention are per row.
func TestDB01_chunkedPrefillMatchesWhole(t *testing.T) {
	path := os.Getenv("GOINFER_DB01_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.5-0.8b")
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	prevDnet := dnetPrefillOn
	dnetPrefillOn = true
	defer func() { dnetPrefillOn = prevDnet }()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	tk, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	const N = 96
	ids := decoder.PrefillGateProseIDsForTest(t, tk, files[1], N)[:N]
	embs := getEmbs(r, ids)
	state := func() (out [][]uint32) {
		for l := range r.layers {
			L := &r.layers[l]
			if L.delta != nil {
				out = append(out, append([]uint32(nil), L.delta.win.U32s()...), append([]uint32(nil), L.delta.state.U32s()...))
				continue
			}
			n := N * L.geom.kvDim / 2 // halves, read as words
			out = append(out, append([]uint32(nil), r.kc[l].U32s()[:n]...), append([]uint32(nil), r.vc[l].U32s()[:n]...))
		}
		return
	}
	r.resetDeltaNet()
	whole := append([]float32(nil), r.PrefillLast(embs, 0)...)
	wantState := state()
	for _, cut := range []int{64, 40} {
		r.resetDeltaNet()
		r.PrefillLast(embs[:cut], 0)
		got := append([]float32(nil), r.PrefillLast(embs[cut:], cut)...)
		lg := 0
		for i := range whole {
			if math.Float32bits(whole[i]) != math.Float32bits(got[i]) {
				lg++
			}
		}
		st := 0
		for i, w := range state() {
			for j := range w {
				if w[j] != wantState[i][j] {
					st++
				}
			}
		}
		t.Logf("chunks %d + %d: %d of %d logits and %d state/window/K/V words differ from one pass", cut, N-cut, lg, len(whole), st)
		if lg != 0 || st != 0 {
			t.Errorf("chunks %d + %d differ from one %d-token pass (%d logits, %d state words)", cut, N-cut, N, lg, st)
		}
	}
}

// TestDB01_prefillFromZeroResetsState (D-B01): PrefillLast from position 0 is a fresh sequence, so a hybrid's DeltaNet
// window and state start from zero, as Forward(pos 0) makes them. The pass continues whatever state the resident holds
// (right for a continuation), so without the reset a new prompt prefilled after another one started from the previous
// sequence's state (the fidelity gate's pass arm did, after its sequential arm). Through metalResident.PrefillLast, the
// entry point the decoder and the gate call: prompt B after prompt A must equal prompt B on a reset resident, logits and
// state bit for bit.
func TestDB01_prefillFromZeroResetsState(t *testing.T) {
	path := os.Getenv("GOINFER_DB01_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.5-0.8b")
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 1024,
		Knobs: &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"}})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	prevDnet := dnetPrefillOn
	dnetPrefillOn = true
	defer func() { dnetPrefillOn = prevDnet }()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	a := &metalResident{r: r, hidden: r.H}
	tk, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	idsA := decoder.PrefillGateProseIDsForTest(t, tk, files[2], 40)[:40]
	idsB := decoder.PrefillGateProseIDsForTest(t, tk, files[3], 40)[:40]
	state := func() (out []uint32) {
		for l := range r.layers {
			if D := r.layers[l].delta; D != nil {
				out = append(append(out, D.win.U32s()...), D.state.U32s()...)
			}
		}
		return
	}
	r.resetDeltaNet()
	want, err := a.PrefillLast(context.Background(), getEmbs(r, idsB), 0)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	wantState := state()
	if _, err := a.PrefillLast(context.Background(), getEmbs(r, idsA), 0); err != nil {
		t.Fatal(err)
	}
	got, err := a.PrefillLast(context.Background(), getEmbs(r, idsB), 0)
	if err != nil {
		t.Fatal(err)
	}
	lg, st := 0, 0
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			lg++
		}
	}
	for i, w := range state() {
		if w != wantState[i] {
			st++
		}
	}
	t.Logf("prompt B after prompt A: %d of %d logits and %d window/state words differ from prompt B on a reset resident", lg, len(want), st)
	if lg != 0 || st != 0 {
		t.Errorf("PrefillLast from position 0 did not start a fresh sequence (%d logits, %d state words differ)", lg, st)
	}
}
