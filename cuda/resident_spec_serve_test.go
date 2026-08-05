//go:build cuda

package cuda

import (
	"context"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentSpecServe is the SERVE-path gate for D1 on resident CUDA: the exact production entry the
// OpenAI handler now calls for a resident model under --spec ngram —
// Model.GenerateNgramSpeculativeAdaptive — must emit the byte-identical token stream to plain resident
// Model.Generate (greedy). Unlike TestSpecDecode (which drives rf.PrefillLastN by hand), this exercises
// the real decoder loop: genNgramInto's resident branch (resBusy claim + per-token prefill + batched
// ForwardN verify, which now routes through prefillCore) end-to-end. If the resBusy guard, the
// ForwardN→prefillCore routing, or the rollback semantics were wrong, the streams would diverge. Heavy.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestResidentSpecServe -v
func TestResidentSpecServe(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	const path = "/home/francis/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	if !mc.ResidentActive() {
		t.Skip("model did not go resident on this device")
	}
	if !mc.DecodeRunnerEligible() {
		t.Skip("arch not DecodeRunner-eligible")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	// A repetitive prompt (a short cycle) so the n-gram drafter actually hits and the spec loop runs
	// multi-token verify rounds — losslessness must hold regardless, but this exercises accept>0.
	const P = 48
	prompt := make([]int, P)
	cycle := []int{40, 41, 42, 43, 40, 41, 42, 43}
	for i := range prompt {
		prompt[i] = cycle[i%len(cycle)] % vocab
	}
	const maxTok = 96

	drain := func(ch <-chan int) []int {
		out := make([]int, 0, maxTok)
		for id := range ch {
			out = append(out, id)
		}
		return out
	}

	// Ground truth: plain resident greedy decode.
	var sp decoder.SamplingParams // Temperature 0 ⇒ greedy argmax
	gch, gg := mc.Generate(context.Background(), prompt, maxTok, sp)
	gt := drain(gch)
	if e := gg.Err(); e != nil {
		t.Fatalf("plain Generate: %v", e)
	}
	if len(gt) == 0 {
		t.Fatalf("plain Generate produced no tokens")
	}

	// The serve path: n-gram speculative decode with adaptive depth (what openai.go now calls for a
	// resident model under --spec ngram).
	sch, sg, err := mc.GenerateNgramSpeculativeAdaptive(context.Background(), prompt, maxTok,
		&decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, sp)
	if err != nil {
		t.Fatalf("GenerateNgramSpeculativeAdaptive: %v", err)
	}
	spec := drain(sch)
	if e := sg.Err(); e != nil {
		t.Fatalf("spec Generate: %v", e)
	}

	if len(spec) != len(gt) {
		t.Fatalf("length mismatch: spec %d vs greedy %d tokens", len(spec), len(gt))
	}
	for i := range gt {
		if spec[i] != gt[i] {
			t.Fatalf("LOSSLESS VIOLATION at token %d: spec %d vs greedy %d", i, spec[i], gt[i])
		}
	}
	if st := sg.Spec; st != nil {
		t.Logf("resident spec serve OK: %d tokens byte-identical; rounds=%d drafted=%d accepted=%d (accept %.2f)",
			len(spec), st.Rounds, st.Drafted, st.Accepted, st.AcceptanceRate())
	} else {
		t.Logf("resident spec serve OK: %d tokens byte-identical", len(spec))
	}
}
