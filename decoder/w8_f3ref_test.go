//go:build goinfer_testhooks

package decoder

import (
	"compress/gzip"
	"encoding/gob"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestW8F3Reference_write builds F3′'s prompts exactly as the metal test does (the first 8 files of the prefill gate's
// prose set A, the first 16 tokens of each), runs the CPU f32 model over them and 16 tokens of its own greedy
// continuation, and writes the result gzip+gob to GOINFER_W8_F3_REF_OUT. GOINFER_W8_GATE_MODEL names the checkpoint.
//
//	GOINFER_W8_F3_REF_OUT=~/f3ref-1.5b.gob.gz GOINFER_W8_GATE_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//	  go test -tags goinfer_testhooks -count=1 -run '^TestW8F3Reference_write$' -v ./decoder/
func TestW8F3Reference_write(t *testing.T) {
	out, path := os.Getenv("GOINFER_W8_F3_REF_OUT"), os.Getenv("GOINFER_W8_GATE_MODEL")
	if out == "" || path == "" {
		t.Skip("set GOINFER_W8_F3_REF_OUT and GOINFER_W8_GATE_MODEL")
	}
	const nPrompts, promptLen, steps = 8, 16, 32
	tk, err := loadTokenizerForTest(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	files := PrefillGatePromptSetFor("a")
	ref := W8F3Ref{Model: path, Arch: runtime.GOARCH, PromptLen: promptLen, Pos: steps}
	for i := range nPrompts {
		ref.Prompts = append(ref.Prompts, PrefillGateProseIDsForTest(t, tk, files[i], promptLen)[:promptLen])
	}
	m, err := Load(path, Options{ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load f32: %v", err)
	}
	defer m.Close()
	_, nL, _, nKV, hd, _, _ := m.Dims()
	for p := range nPrompts {
		cache := NewKVCache(nL, nKV, hd, 0, 1024, nil)
		var toks []int
		var lgs [][]float32
		tok := ref.Prompts[p][0]
		for i := range steps {
			toks = append(toks, tok)
			l, err := m.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatalf("forward prompt %d at %d: %v", p, i, err)
			}
			lgs = append(lgs, append([]float32(nil), l...))
			if i+1 < promptLen {
				tok = ref.Prompts[p][i+1]
			} else {
				best := 0
				for j, v := range l {
					if v > l[best] {
						best = j
					}
				}
				tok = best
			}
		}
		ref.Toks, ref.Logits = append(ref.Toks, toks), append(ref.Logits, lgs)
		t.Logf("prompt %d: %d positions", p, steps)
	}
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	zw := gzip.NewWriter(f)
	if err := gob.NewEncoder(zw).Encode(ref); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s (%s, %d prompts x %d positions)", out, ref.Arch, nPrompts, steps)
}

// loadTokenizerForTest reads a GGUF's tokenizer, or a checkpoint directory's tokenizer.json.
func loadTokenizerForTest(path string) (*tokenizer.Tokenizer, error) {
	if st, err := os.Stat(path); err == nil && st.IsDir() {
		return tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	}
	return tokenizer.LoadGGUF(path)
}
