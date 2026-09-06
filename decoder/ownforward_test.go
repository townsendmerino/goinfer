package decoder

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// THE TABLE MUST NAME EVERY FAMILY FORWARD THAT EXISTS.
//
// ownForwards is only a chokepoint while it is complete: a runLayersXxx wired in with an `if`
// ahead of the dispatch loop is exactly the state this replaced, where "runLayers dispatches here"
// and "the batched path must not touch this" were two hand-written facts and LFM2 was in one of
// them. A source scan is the only check that notices the SECOND list being born.
func TestOwnForward_tableNamesEveryFamilyForward(t *testing.T) {
	named := map[string]bool{}
	for _, f := range ownForwards {
		named[f.Name] = true
	}
	// The table stores function VALUES, so reflection cannot recover their source names. Match on
	// the table's own source text instead — the one place the method names are written down.
	tableSrc, err := os.ReadFile("arch.go")
	if err != nil {
		t.Fatalf("read arch.go: %v", err)
	}

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(files)
	re := regexp.MustCompile(`(?m)^func \(m \*Model\) (runLayers[A-Z][A-Za-z0-9]*)\(id int, cache \*KVCache\) \(\[\]float32, error\) \{`)
	var found []string
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			found = append(found, m[1])
		}
	}
	if len(found) == 0 {
		// NOT a skip: an empty scan is how this assertion silently stops asserting.
		t.Fatal("no runLayersXxx found by the scan — the check is broken, not the tree")
	}
	for _, fn := range found {
		if !strings.Contains(string(tableSrc), "(*Model)."+fn) {
			t.Errorf("%s is a family forward that ownForwards does not name. Whatever dispatches to "+
				"it is a SECOND list, and canBatchN derives its exclusion from the table — so the "+
				"batched prefill will run the generic attention stack over this family's layers.", fn)
		}
	}
	if len(found) != len(ownForwards) {
		t.Errorf("scan found %d family forwards (%v) but the table has %d — a stale entry claims a "+
			"family that no longer exists", len(found), found, len(ownForwards))
	}
}

// C-01, END TO END, ON THE COMMITTED FIXTURE. Every prompt of >=2 tokens reached the generic
// batched stack, whose first act on an LFM2 conv layer is rmsNorm against a QNorm that was never
// loaded — an index-out-of-range in the Generate goroutine, where net/http's handler recover does
// not reach. TestLFM2_textParity stayed green throughout because it drives m.forward one token at
// a time and never calls prefillLogits.
//
// The assertion is not "it does not panic" but "the two paths agree": prefillLogits documents
// itself as bit-identical to the sequential prefill, and a fix that merely stopped the crash while
// taking a different path would satisfy the weaker claim.
func TestLFM2_multiTokenPrefillMatchesSequential(t *testing.T) {
	const ckpt = "../testdata/lfm2-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Fatalf("the committed LFM2 fixture is missing at %s: %v", ckpt, err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	if m.w.arch.lfm2 == nil {
		t.Fatalf("fixture resolved to arch %q, not lfm2", m.w.arch.Name)
	}
	if m.canBatchN(4) {
		t.Error("canBatchN(4) = true for LFM2 — the batched stack has no conv mixer")
	}
	if batched, reason := m.PrefillPath(); batched {
		t.Errorf("PrefillPath() reports batched (%q); serve prints this at startup, so a wrong "+
			"answer here is published as a fact", reason)
	}

	prompt := []int{3, 11, 47, 5}

	seqCache := m.NewCache(len(prompt) + 1)
	var want []float32
	for _, id := range prompt {
		if want, err = m.forward(id, seqCache); err != nil {
			t.Fatalf("sequential forward: %v", err)
		}
	}

	batchCache := m.NewCache(len(prompt) + 1)
	got, err := m.prefillLogits(context.Background(), prompt, batchCache)
	if err != nil {
		t.Fatalf("prefillLogits: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("prefillLogits returned %d logits, sequential %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("prefill logits diverge from sequential at %d: %v vs %v (prefillLogits "+
				"documents itself as bit-identical)", i, got[i], want[i])
		}
	}
	if batchCache.pos != seqCache.pos {
		t.Errorf("cache position %d after prefill, %d after sequential — the whole prompt must be "+
			"in the cache either way", batchCache.pos, seqCache.pos)
	}
}

// Generate is the reported trigger: `serve --model <lfm2>`, first chat request. It runs the
// prefill in its own goroutine, so the panic took the process rather than the request.
func TestLFM2_generateMultiTokenPrompt(t *testing.T) {
	m, err := Load("../testdata/lfm2-tiny", Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	ch, gen := m.Generate(context.Background(), []int{3, 11, 47, 5}, 3, SamplingParams{})
	n := 0
	for range ch {
		n++
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if n == 0 {
		t.Fatal("Generate produced no tokens")
	}
}

// THE OTHER FOUR CONSUMERS OF THE SAME FACT (audit §0 theme 1). Each hand-listed the own-forward
// families; between them they had drifted by two — lfm2 was missing from all four and gpt-oss from
// HiddenLast — so an LFM2 model reached seams documented to refuse it and got nil rows or a pooled
// vector from a path nobody had checked. Driven off the table so a family added there is covered
// here without anyone remembering to.
func TestOwnForward_lifecycleSeamsRefuseEveryOwnForwardFamily(t *testing.T) {
	for _, f := range ownForwards {
		arch := archWithFamily(t, f)
		m := &Model{w: &Weights{arch: arch}}

		if _, _, _, _, err := m.ForwardSubCapture(0, NewKVCache(1, 1, 1, 0, 4)); err == nil {
			t.Errorf("%s: ForwardSubCapture returned no error; it needs runLayersFromEmbed's "+
				"uniform block, which no own-forward loop routes through", f.Name)
		}
		if _, err := m.HiddenLast([]int{0}); err == nil {
			t.Errorf("%s: HiddenLast returned no error for an own-forward family", f.Name)
		}
		if p := newLayerPager(m.w, []byte{0}, 1<<20); p != nil {
			t.Errorf("%s: newLayerPager built a pager; the family's loop never calls enterLayer, "+
				"so the RAM-bound banner promises what it cannot deliver (N-13)", f.Name)
		}
		_, _, err := m.ForwardCapture(0, NewKVCache(1, 1, 1, 0, 4), nil)
		if f.Captures && err != nil && strings.Contains(err.Error(), "seam not wired") {
			t.Errorf("%s: ForwardCapture refuses a family the table marks as capturing", f.Name)
		}
		if !f.Captures && err == nil {
			t.Errorf("%s: ForwardCapture returned no error, but this family's loop does not call "+
				"captureResidual — it would hand back nil rows", f.Name)
		}
	}
}

// The two views of "recurrent" must not disagree: the table's Recurrent bit decides before a cache
// exists (speculative rollback), KVCache.hasRecurrentState decides once one does (truncate,
// snapshot, session reconcile). A family recurrent in one sense and not the other is the state that
// produced C-02, only with the halves swapped.
func TestOwnForward_recurrentBitMatchesTheCacheKinds(t *testing.T) {
	kinds := map[string]func(*KVCache){
		"qwen3_5_moe":      func(c *KVCache) { c.delta = []*deltaState{{}} },
		"granitemoehybrid": func(c *KVCache) { c.mamba = []*mamba2State{{}} },
		"nemotron_h":       func(c *KVCache) { c.mamba = []*mamba2State{{}} },
		"lfm2":             func(c *KVCache) { c.conv = []*shortConvState{{}} },
	}
	for _, f := range ownForwards {
		set, hasKind := kinds[f.Name]
		if f.Recurrent != hasKind {
			t.Errorf("%s: table says Recurrent=%v but this test knows %v about a cache kind for it; "+
				"one of the two is wrong and they are read by different code paths", f.Name, f.Recurrent, hasKind)
			continue
		}
		c := NewKVCache(1, 1, 1, 0, 4)
		if hasKind {
			set(c)
		}
		if got := c.hasRecurrentState(); got != f.Recurrent {
			t.Errorf("%s: hasRecurrentState()=%v, table Recurrent=%v", f.Name, got, f.Recurrent)
		}
	}
}

// archWithFamily builds the minimal Architecture that f.is() accepts. It sets the marker through
// the SAME predicate the table dispatches on, so a family whose marker this helper does not know
// fails loudly rather than being silently skipped by every assertion above.
func archWithFamily(t *testing.T, f ownForwardFamily) *Architecture {
	t.Helper()
	markers := []func(*Architecture){
		func(a *Architecture) { a.gemma4 = &gemma4Params{} },
		func(a *Architecture) { a.qwen35 = &qwen35Params{} },
		func(a *Architecture) { a.lfm2 = &lfm2Params{} },
		func(a *Architecture) { a.granite = &graniteParams{} },
		func(a *Architecture) { a.nemotron = &nemotronParams{} },
		func(a *Architecture) { a.mla = &mlaParams{} },
		func(a *Architecture) { a.kda = &kdaParams{} }, // bailing_hybrid; must precede deepseek_v2/v3 in ownForwards
		func(a *Architecture) { a.llama4 = &llama4Params{} },
		func(a *Architecture) { a.gptoss = &gptOssParams{} },
	}
	for _, set := range markers {
		a := &Architecture{Name: f.Name, NumLayers: 1, HiddenDim: 1, VocabSize: 8}
		set(a)
		if f.is(a) {
			return a
		}
	}
	t.Fatalf("no marker in this helper satisfies ownForwards[%q].is — add it, or every assertion "+
		"over that family silently tests nothing", f.Name)
	return nil
}
