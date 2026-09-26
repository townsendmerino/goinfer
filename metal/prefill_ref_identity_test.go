//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillRefIdentity audits a prompt set's CPU reference files against the prompts the gates feed them. The files
// carry no prompt ids, so each (model, K, prompt) is checked the way runDecodeFidelityGate's identity check does it:
// prefill the snapshot prompt's first K tokens on Metal and compare the reference's prompt-final logits with the
// resident's. A reference generated from the same text lands at the W4A8-vs-CPU level (measured 0.0015–0.072 on
// set B's S-K3900); one generated from different text lands orders of magnitude higher (3–18 on set A's). Found
// 2026-09-25: set A's 2026-09-05 files predate the 2026-09-09 snapshot (metal-decode-attn-r17-2026-09-25.md).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=a go test -tags goinfer_testhooks -run '^TestPrefillRefIdentity$' -v ./metal/
//
// S cells use the 1.5B, D7 cells the 7B (its .int4.metal.giw sidecar); the D7 references are int8-weight CPU runs,
// so their valid level is somewhat higher than S's, still far below a mismatch.
func TestPrefillRefIdentity(t *testing.T) {
	requireHeavyModel(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	label, files := decoder.PrefillGatePromptSet()
	dir := refDirFor(home, label)
	// GOINFER_REF_IDENTITY_DIR audits another machine's reference files: it may hold just each file's header and
	// prompt-final logits (the first 8+4*vocab bytes), which is all this check reads.
	if v := os.Getenv("GOINFER_REF_IDENTITY_DIR"); v != "" {
		dir = v
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Skipf("no reference directory %s: %v", dir, err)
	}
	re := regexp.MustCompile(`^(S|D7)-K(\d+)-p(\d+)\.bin$`)
	type cell struct {
		model string
		K     int
	}
	prompts := map[cell][]int{}
	for _, e := range entries {
		m := re.FindStringSubmatch(e.Name())
		if m == nil {
			continue
		}
		K, _ := strconv.Atoi(m[2])
		p, _ := strconv.Atoi(m[3])
		c := cell{m[1], K}
		prompts[c] = append(prompts[c], p)
	}
	models := map[string]string{
		"S":  "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
		"D7": "$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw",
	}
	tokenizers := map[string]string{
		"S":  "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
		"D7": "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf",
	}
	suspect := 0
	for _, mdl := range []string{"S", "D7"} {
		var cells []cell
		for c := range prompts {
			if c.model == mdl {
				cells = append(cells, c)
			}
		}
		if len(cells) == 0 {
			continue
		}
		sort.Slice(cells, func(i, j int) bool { return cells[i].K < cells[j].K })
		path := os.ExpandEnv(models[mdl])
		if _, err := os.Stat(path); err != nil {
			t.Logf("%s: no checkpoint at %s — skipping its %d cells", mdl, path, len(cells))
			continue
		}
		maxK := cells[len(cells)-1].K
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: maxK + 8})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		rf, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			m.Close()
			t.Fatalf("metal resident not built for %s", path)
		}
		tk, err := tokenizer.LoadGGUF(os.ExpandEnv(tokenizers[mdl]))
		if err != nil {
			m.Close()
			t.Fatalf("tokenizer: %v", err)
		}
		for _, c := range cells {
			ps := prompts[c]
			sort.Ints(ps)
			cellStart := time.Now()
			line := fmt.Sprintf("[ref-identity] set %q %s-K%-5d", label, mdl, c.K)
			for _, p := range ps {
				ref := filepath.Join(dir, fmt.Sprintf("%s-K%d-p%d.bin", mdl, c.K, p))
				seedRef, err := readRefSeed(ref)
				if err != nil {
					t.Fatalf("read %s: %v", ref, err)
				}
				ids := decoder.PrefillGateProseIDsForTest(t, tk, files[p], c.K)[:c.K]
				embs := make([][]float32, c.K)
				for i, id := range ids {
					embs[i] = m.EmbedResidentForTest(id)
				}
				seed, err := rf.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("PrefillLast %s p%d: %v", mdl, p, err)
				}
				kl := decoder.KLDivergenceForTest(seedRef, seed)
				mark := ""
				if kl > 1.0 {
					mark, suspect = "!", suspect+1
				}
				// heartbeat per prompt, as it happens: a long cell (the 7B at K=8000 is minutes per prompt) must not
				// look dead until its summary line
				fmt.Fprintf(os.Stderr, "[ref-identity]   %s-K%d prompt %d (%d of %d) KL=%.4f%s — cell elapsed %s\n",
					mdl, c.K, p+1, sortIndex(ps, p)+1, len(ps), kl, mark, time.Since(cellStart).Round(time.Second))
				line += fmt.Sprintf("  p%d=%.4f%s", p+1, kl, mark)
			}
			fmt.Fprintln(os.Stderr, line)
		}
		m.Close()
	}
	fmt.Fprintf(os.Stderr, "[ref-identity] set %q: %d (cell, prompt) pairs above KL 1.0 (marked !) — prompts numbered 1–10\n", label, suspect)
}

// readRefSeed reads a prefill reference file's header and prompt-final logits only (int32 vocab, int32 n, then vocab
// float32s — decoder.ReadPrefillReferenceForTest's layout), so a truncated copy holding just that prefix works.
func readRefSeed(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var vocab, n int32
	if err := binary.Read(f, binary.LittleEndian, &vocab); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	seed := make([]float32, vocab)
	if err := binary.Read(f, binary.LittleEndian, seed); err != nil {
		return nil, err
	}
	return seed, nil
}

func sortIndex(xs []int, v int) int {
	for i, x := range xs {
		if x == v {
			return i
		}
	}
	return -1
}
