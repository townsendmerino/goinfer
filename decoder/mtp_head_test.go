package decoder

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// mtpTargetDir is the Gate 1 probe target: qwen3.5-0.8b, the cheapest checkpoint on disk carrying
// an MTP head (docs/spec/09-mtp-heads.md, Gate 0). Overridable for the later, larger candidates.
func mtpTargetDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("GOINFER_MTP_TARGET")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen3.5-0.8b")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no MTP target at %s: %v", dir, err)
	}
	return dir
}

// TestMTP_targetCaptures is the prerequisite for everything in 09: the probe needs the target's
// hidden state at t to draft t+1, and ForwardCapture is the seam that supplies it.
//
// It is asserted rather than assumed because the seam's own comment lists the wired families as
// "qwen3_5_moe, gemma4, gpt-oss" while this target is qwen3_5 DENSE. The capturing loop in
// forward_qwen35.go is shared between the dense and MoE paths (one `cache.captureResidual` after
// either gatedMLP or moeMLP), so dense should capture too — but a docstring narrower than the code
// is exactly the kind of thing that is true until it isn't.
func TestMTP_targetCaptures(t *testing.T) {
	requireHeavyModel(t)
	dir := mtpTargetDir(t)

	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	defer m.Close()
	a := m.w.arch
	t.Logf("target: arch=%s layers=%d hidden=%d kvHeads=%d vocab=%d",
		a.Name, a.NumLayers, a.HiddenDim, a.NumKVHeads, a.VocabSize)

	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, err := tk.Encode("The capital of France is", true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	cache := m.NewCache(len(ids) + 8)
	last := a.NumLayers - 1
	var logits []float32
	var hid [][]float32
	for _, id := range ids {
		logits, hid, err = m.ForwardCapture(id, cache, []int{last})
		if err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
	}
	if len(hid) != 1 || len(hid[0]) != a.HiddenDim {
		t.Fatalf("captured %d row(s), want 1 of width %d — the seam handed back nothing usable",
			len(hid), a.HiddenDim)
	}
	var nonzero int
	for _, v := range hid[0] {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("captured hidden state is all zeros — the loop did not populate it")
	}
	t.Logf("captured last-layer hidden: %d dims, %d non-zero; argmax=%d", len(hid[0]), nonzero, argmax(logits))
}

// TestMTP_loadsHead is the adapter's own gate: the head the existing loaders skip is detected by
// the right route, loads, and has the geometry the target implies. Shapes are asserted so a
// silently-empty head cannot pass as a loaded one.
func TestMTP_loadsHead(t *testing.T) {
	requireHeavyModel(t)
	dir := mtpTargetDir(t)

	ok, how := HasMTPHead(dir)
	if !ok {
		t.Fatalf("HasMTPHead(%s) = false (%s)", dir, how)
	}
	t.Logf("detected via %s", how)

	h, err := LoadMTPHead(dir)
	if err != nil {
		t.Fatalf("LoadMTPHead: %v", err)
	}
	defer h.Close()
	t.Logf("head: hidden=%d nHeads=%d nKV=%d headDim=%d inter=%d",
		h.hidden, h.nHeads, h.nKV, h.headDim, h.inter)

	for _, n := range []struct {
		name string
		v    []float32
		want int
	}{
		{"pre_fc_norm_embedding", h.preFCNormEmbed, h.hidden},
		{"pre_fc_norm_hidden", h.preFCNormHidden, h.hidden},
		{"input_layernorm", h.lw.PreAttnNorm, h.hidden},
		{"post_attention_layernorm", h.lw.PreMLPNorm, h.hidden},
		{"q_norm", h.lw.qattn.qNorm, h.headDim},
		{"k_norm", h.lw.qattn.kNorm, h.headDim},
		{"mtp.norm", h.finalNorm, h.hidden},
	} {
		if len(n.v) != n.want {
			t.Errorf("%s: len %d, want %d", n.name, len(n.v), n.want)
		}
	}
}

// TestMTP_acceptedLength is Gate 1 (docs/spec/09-mtp-heads.md): the accepted-prefix length of the
// MTP head's draft against the target's own greedy continuation, by the same method
// TestEagleAcceptedLength uses for the 05 head — prefill the head over the realized prefix, draft
// K, count the leading run that matches.
//
// READ IT AS 09 PRE-REGISTERED IT. On this 0.8B target a pass is a pass on MECHANISM and NOT on
// economics; Gate 2 is not evaluable at this scale, because the break-even ratio is dominated by a
// c_decode that does not resemble a 27B trunk. 05's ~1.6 is a CROSS-TARGET reference point (it was
// measured on Qwen3-1.7B, which carries no MTP head), so this is a screen, not a paired comparison.
func TestMTP_acceptedLength(t *testing.T) {
	requireHeavyModel(t)
	dir := mtpTargetDir(t)

	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	head, err := LoadMTPHead(dir)
	if err != nil {
		t.Fatalf("LoadMTPHead: %v", err)
	}
	defer head.Close()
	if head.hidden != m.w.arch.HiddenDim {
		t.Fatalf("head hidden %d != target hidden %d", head.hidden, m.w.arch.HiddenDim)
	}

	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	// Gate 1 as pre-registered asks for code / math / chat, so all three run — not the one prompt
	// this test first used. Note for comparability: 05's own 1.60 is a SINGLE prose prompt
	// (TestEagleAcceptedLength, same M=48/K=6 shape); 05:230 lists the suites as future work. The
	// chat wrapper matches 05's, so the shapes line up.
	suites := []struct{ name, text string }{
		{"code", "Write a Go function that reverses a slice of integers in place."},
		{"math", "What is the sum of all integers from 1 to 100? Show your reasoning."},
		{"chat", "Write a short paragraph about the history of computing."},
	}
	for _, sc := range suites {
		t.Run(sc.name, func(t *testing.T) { mtpAccept(t, m, head, tk, sc.text) })
	}
}

func mtpAccept(t *testing.T, m *Model, head *MTPHead, tk *tokenizer.Tokenizer, text string) {
	prompt, err := tk.Encode("<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n", true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	const M, K = 48, 6
	last := m.w.arch.NumLayers - 1
	cache := m.NewCache(len(prompt) + M + 8)
	var toks []int
	var feats [][]float32
	feed := func(tok int) int {
		logits, hid, err := m.ForwardCapture(tok, cache, []int{last})
		if err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
		toks = append(toks, tok)
		feats = append(feats, append([]float32(nil), hid[0]...))
		return argmax(logits)
	}
	var next int
	for _, id := range prompt {
		next = feed(id)
	}
	for range M {
		next = feed(next)
	}

	var sumAcc, n int
	hist := make([]int, K+1)
	for i := len(prompt); i+K < len(toks); i++ {
		st, err := m.MTPPrefill(head, toks[:i], feats[:i], len(toks)+K+8)
		if err != nil {
			t.Fatalf("MTPPrefill: %v", err)
		}
		draft, err := m.MTPDraftFrom(head, st, toks[i], feats[i], i, K, cache)
		if err != nil {
			t.Fatalf("MTPDraftFrom: %v", err)
		}
		acc := 0
		for j := range K {
			if draft[j] == toks[i+1+j] {
				acc++
			} else {
				break
			}
		}
		sumAcc += acc
		hist[acc]++
		n++
	}
	if n == 0 {
		t.Fatal("no draft positions — continuation too short")
	}
	mean := float64(sumAcc) / float64(n)
	t.Logf("MTP accepted length over %d positions, K=%d: mean %.3f → %.3f tok/verify", n, K, mean, mean+1)
	t.Logf("histogram (accepted 0..K): %v", hist)
	t.Logf("05 EAGLE-3 reference (CROSS-TARGET, Qwen3-1.7B): 1.60 tok/verify")
	t.Logf("Gate 1 screen: %.3f tok/verify vs 1.60 → %s", mean+1,
		map[bool]string{true: "ABOVE", false: "AT OR BELOW"}[mean+1 > 1.60])

	// c_head / c_decode, recorded because 09 pre-registered it and labelled NON-TRANSFERABLE: this
	// is a one-block head against a 24-block trunk with a tied 248320 embedding. It will not
	// resemble a 424M head against a 27B dense trunk, and it is NOT a break-even.
	emb := make([]float32, head.hidden)
	m.embedToken(toks[0], emb)
	stT := m.NewMTPState(head, 16)
	for i := range 3 { // warm
		_, _ = m.MTPStep(head, emb, feats[0], i, stT)
		_, _, _ = m.ForwardCapture(toks[0], cache, []int{last})
	}
	const reps = 20
	t0 := time.Now()
	for i := range reps {
		if _, err := m.MTPStep(head, emb, feats[0], i, stT); err != nil {
			t.Fatalf("MTPStep: %v", err)
		}
	}
	cHead := time.Since(t0) / reps
	t1 := time.Now()
	for range reps {
		if _, _, err := m.ForwardCapture(toks[0], cache, []int{last}); err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
	}
	cDecode := time.Since(t1) / reps
	t.Logf("c_head=%v c_decode=%v ratio=%.3f — DIAGNOSTIC ONLY, NOT TRANSFERABLE to a 27B trunk, "+
		"and NOT a break-even (Gate 2 is not evaluable at this scale)",
		cHead.Round(time.Microsecond), cDecode.Round(time.Microsecond),
		float64(cHead)/float64(cDecode))
}
