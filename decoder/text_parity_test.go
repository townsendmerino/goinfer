package decoder

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"slices"
	"strings"
	"testing"
)

// textParity is one family's tiny-fixture check against its HF golden: the last logits and the greedy continuation through
// m.forward, and through Generate when generate is set. A missing golden or fixture skips, naming the script that makes it.
type textParity struct {
	golden, ckpt string
	pin          string                          // the script that makes them
	label        string                          // what the log line calls this check
	check        func(t *testing.T, m *Model)    // what the fixture must have resolved to, before any forward
	row          func(t *testing.T, cos float64) // the manifest row, after the forward's checks
	generate     bool                            // Generate's batched prefill must give the same continuation
}

func (c textParity) run(t *testing.T) {
	t.Helper()
	raw, err := os.ReadFile(c.golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run %s", c.pin)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(c.ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture at %s — run %s", c.ckpt, c.pin)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(c.ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", c.ckpt, err)
	}
	defer m.Close()
	if c.check != nil {
		c.check(t, m)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("%s: argmax got=%d want=%d | logit cosine=%.6f", c.label, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d (got %v want %v)", i, got[i], g.ContinuationIDs[i], got, g.ContinuationIDs)
			break
		}
	}
	if c.row != nil {
		c.row(t, cos)
	}
	if !c.generate {
		return
	}

	gm, err := Load(c.ckpt, Options{})
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	defer gm.Close()
	out, _ := gm.Generate(context.Background(), g.PromptIDs, g.NNew, SamplingParams{})
	gen := make([]int, 0, g.NNew)
	for id := range out {
		gen = append(gen, id)
	}
	for i := range g.ContinuationIDs {
		if i >= len(gen) || gen[i] != g.ContinuationIDs[i] {
			t.Errorf("Generate continuation[%d] mismatch: got %v want %v (batched-prefill / canBatchN?)", i, gen, g.ContinuationIDs)
			break
		}
	}
}

// A family's deps_hash covers the test files its record lists, so that a change to its parity check re-stales it. A test that runs
// through textParity keeps that check here, so a record listing such a test file must list this file too.
func TestTextParity_hashedWithItsUsers(t *testing.T) {
	m, err := loadParityManifest()
	if err != nil {
		t.Fatal(err)
	}
	const helper = "decoder/text_parity_test.go"
	users := 0
	for name, fam := range m.Families {
		deps := familyDepFiles(m, fam)
		for _, f := range deps {
			if !strings.HasSuffix(f, "_test.go") || f == helper {
				continue
			}
			src, err := os.ReadFile(repoPath(f))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(src, []byte("textParity{")) {
				continue
			}
			users++
			if !slices.Contains(deps, helper) {
				t.Errorf("%s's deps_hash covers %s, whose parity test runs through textParity, and not %s", name, f, helper)
			}
		}
	}
	if users == 0 {
		t.Error("no parity record lists a test file that uses textParity, so this check matched nothing")
	}
}
