package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Qwen3-Embedding parity gates (docs/task-decoder-as-embedder.md §5).
//
// These are the SEMANTIC gates the mechanical ones in decoder_embedder_test.go cannot replace: a
// decoder-backed embedder can be perfectly self-consistent and still be wrong (wrong pooled
// position, missing instruction prefix), which shows up only against the sentence-transformers
// reference. Two gates, because they fail independently:
//
//   - vector    — cosine vs the HF reference embedding, bar > 0.9999.
//   - retrieval — top-1 document per query must match the reference ranking. Cosine can be high
//     while ranking is wrong, and ranking is what an embedder is actually FOR.
//
// Both skip until the golden and the embedding-tuned checkpoint are present:
//
//	.venv/bin/python scripts/pin_qwen3_embedding.py     # writes testdata/qwen3_embedding_golden.json
//	# and place Qwen3-Embedding-0.6B GGUF in $HOME/models/
//
// Skipping is deliberate and visible — see TestQwen3Embedding_gatesAreWired, which fails loudly if
// the golden exists but the gates never actually ran.

const qwen3EmbedGolden = "../../testdata/qwen3_embedding_golden.json"

// vectorParityBar is the certified-embedder bar from aikit's encoder work: its embedders all land
// at 1.000000, so anything below this is a real difference, not float noise.
const vectorParityBar = 0.9999

type embedGolden struct {
	Note        string `json:"note"`
	Model       string `json:"model"`
	Hidden      int    `json:"hidden"`
	QueryPrompt string `json:"query_prompt"`
	DocPrompt   string `json:"doc_prompt"`
	Cases       []struct {
		Text      string    `json:"text"`
		IsQuery   bool      `json:"is_query"`
		InputIDs  []int     `json:"input_ids"`
		Embedding []float32 `json:"embedding"`
	} `json:"cases"`
	Retrieval struct {
		Queries   []string `json:"queries"`
		Documents []string `json:"documents"`
		Top1      []int    `json:"top1"`
		Ranking   [][]int  `json:"ranking"`
	} `json:"retrieval"`
}

func loadEmbedGolden(t *testing.T) *embedGolden {
	t.Helper()
	b, err := os.ReadFile(qwen3EmbedGolden)
	if err != nil {
		t.Skipf("no golden at %s — pin it with: .venv/bin/python scripts/pin_qwen3_embedding.py", qwen3EmbedGolden)
	}
	var g embedGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return &g
}

// qwen3EmbeddingCheckpoint finds the embedding-tuned checkpoint. This is NOT the base Qwen3-0.6B
// the mechanical tests use: same architecture, different weights, and only these weights reproduce
// the reference vectors.
//
// Prefers the HF safetensors directory the reference itself was pinned from — identical f32 weights
// on both sides, so any cosine shortfall is a real forward/pooling difference rather than
// quantization error, which at a 0.9999 bar a GGUF quant could plausibly account for on its own.
func qwen3EmbeddingCheckpoint(t *testing.T) string {
	t.Helper()
	for _, pat := range findEmbedCheckpointPatterns() {
		matches, _ := filepath.Glob(os.ExpandEnv(pat))
		if len(matches) > 0 {
			return matches[0]
		}
	}
	t.Skip("no Qwen3-Embedding-0.6B checkpoint (HF dir or GGUF) — the base Qwen3 decoder cannot satisfy this gate")
	return ""
}

// findEmbedCheckpointPatterns lists where the checkpoint may live, safetensors first. The HF cache
// entry is the snapshot the pin script downloaded, so certifying needs no second copy on disk.
func findEmbedCheckpointPatterns() []string {
	return []string{
		"$HOME/models/Qwen3-Embedding-0.6B",
		"$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*",
		"$HOME/models/Qwen3-Embedding-0.6B*.gguf",
		"$HOME/models/qwen3-embedding-0.6b*.gguf",
	}
}

func newQwen3Embedder(t *testing.T, g *embedGolden) *decoderEmbedder {
	t.Helper()
	requireHeavyModel(t)
	path := qwen3EmbeddingCheckpoint(t)
	tk, err := loadDecoderTokenizer(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return newDecoderEmbedder(m, tk, g.QueryPrompt, g.DocPrompt)
}

// pooledFromIDs runs the golden's own input_ids through the pooling seam and L2-normalizes, so this
// gate measures the FORWARD + POOLING only — tokenizer differences are isolated into
// TestQwen3Embedding_tokenizerAgreement rather than being blamed on pooling here.
func pooledFromIDs(t *testing.T, e *decoderEmbedder, ids []int) []float32 {
	t.Helper()
	v, err := e.m.HiddenLast(ids)
	if err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	l2normalize(v)
	return v
}

// TestQwen3Embedding_vectorParity is the primary gate: our pooled vector must match the HF
// sentence-transformers reference to > 0.9999 cosine.
func TestQwen3Embedding_vectorParity(t *testing.T) {
	g := loadEmbedGolden(t)
	e := newQwen3Embedder(t, g)
	if e.HiddenDim() != g.Hidden {
		t.Fatalf("hidden dim %d != golden %d (wrong checkpoint?)", e.HiddenDim(), g.Hidden)
	}
	for i, c := range g.Cases {
		got := pooledFromIDs(t, e, c.InputIDs)
		cos := cosineF32(got, c.Embedding)
		t.Logf("case %d (is_query=%v): cosine %.7f — %q", i, c.IsQuery, cos, truncStr(c.Text, 48))
		if cos < vectorParityBar {
			t.Errorf("case %d cosine %.7f < %v vs the HF reference", i, cos, vectorParityBar)
		}
	}
}

// TestQwen3Embedding_retrievalParity: top-1 ordering must match the reference. Run through the FULL
// path (tokenize + prefix + pool), because ranking is what the end-to-end behavior must preserve.
func TestQwen3Embedding_retrievalParity(t *testing.T) {
	g := loadEmbedGolden(t)
	e := newQwen3Embedder(t, g)
	r := g.Retrieval
	if len(r.Queries) == 0 {
		t.Skip("golden carries no retrieval set")
	}

	docs := make([][]float32, len(r.Documents))
	for i, d := range r.Documents {
		v, err := e.Encode(d, false)
		if err != nil {
			t.Fatalf("encode doc %d: %v", i, err)
		}
		l2normalize(v)
		docs[i] = v
	}
	for qi, q := range r.Queries {
		qv, err := e.Encode(q, true)
		if err != nil {
			t.Fatalf("encode query %d: %v", qi, err)
		}
		l2normalize(qv)
		best, bestSim := -1, -2.0
		for di, dv := range docs {
			if s := cosineF32(qv, dv); s > bestSim {
				best, bestSim = di, s
			}
		}
		t.Logf("query %d top1=%d (want %d, sim %.4f)", qi, best, r.Top1[qi], bestSim)
		if best != r.Top1[qi] {
			t.Errorf("query %d ranked doc %d first, reference says %d — cosine can be high while ranking is wrong",
				qi, best, r.Top1[qi])
		}
	}
}

// TestQwen3Embedding_tokenizerAgreement checks our prefix+tokenization reproduces the exact ids the
// reference ate. Separate from the vector gate so a tokenizer drift reports as itself.
func TestQwen3Embedding_tokenizerAgreement(t *testing.T) {
	g := loadEmbedGolden(t)
	e := newQwen3Embedder(t, g)
	for i, c := range g.Cases {
		got, err := e.tokenize(c.Text, c.IsQuery)
		if err != nil {
			t.Fatalf("tokenize case %d: %v", i, err)
		}
		if len(got) != len(c.InputIDs) {
			t.Errorf("case %d: %d ids, reference has %d (prefix or special-token convention differs)",
				i, len(got), len(c.InputIDs))
			continue
		}
		for j := range got {
			if got[j] != c.InputIDs[j] {
				t.Errorf("case %d: id[%d] = %d, reference %d", i, j, got[j], c.InputIDs[j])
				break
			}
		}
	}
}

// TestQwen3Embedding_breakItFirst proves the vector gate can actually FAIL. A gate that cannot fail
// isn't a gate — the same discipline the parity/NaN gates hold. Two perturbations, each a real
// silent-wrong this task is fenced against:
//
//	(a) pool the WRONG POSITION (drop the final token, so the pool lands on L-2)
//	(b) drop the query Instruct prefix
//
// Both must push cosine below the bar. Nothing is mutated permanently — each perturbation is just a
// different call, so there is nothing to revert.
func TestQwen3Embedding_breakItFirst(t *testing.T) {
	g := loadEmbedGolden(t)
	e := newQwen3Embedder(t, g)

	// (a) wrong pooled position.
	for i, c := range g.Cases {
		if len(c.InputIDs) < 3 {
			continue
		}
		good := cosineF32(pooledFromIDs(t, e, c.InputIDs), c.Embedding)
		bad := cosineF32(pooledFromIDs(t, e, c.InputIDs[:len(c.InputIDs)-1]), c.Embedding)
		t.Logf("case %d: correct pool %.7f, wrong-position pool %.7f", i, good, bad)
		if bad >= vectorParityBar {
			t.Errorf("case %d: pooling the WRONG position still passed the bar (%.7f) — the vector gate is vacuous", i, bad)
		}
	}

	// (b) dropped query prefix — only meaningful on the query cases.
	bare := newDecoderEmbedder(e.m, e.tk, "", g.DocPrompt)
	ran := false
	for i, c := range g.Cases {
		if !c.IsQuery {
			continue
		}
		ran = true
		v, err := bare.Encode(c.Text, true) // "query" with no prefix
		if err != nil {
			t.Fatalf("encode case %d: %v", i, err)
		}
		l2normalize(v)
		cos := cosineF32(v, c.Embedding)
		t.Logf("case %d: prefix-dropped cosine %.7f", i, cos)
		if cos >= vectorParityBar {
			t.Errorf("case %d: dropping the query Instruct prefix still passed the bar (%.7f) — the gate does not see the prefix", i, cos)
		}
	}
	if !ran {
		t.Error("golden has no is_query cases — the prefix perturbation never ran")
	}
}

// TestQwen3Embedding_gatesAreWired keeps the skips honest, in the ONE direction that can't
// false-positive: if the Qwen3-Embedding checkpoint is present but the golden is NOT pinned, the
// parity gates silently skip on the golden-missing path while the model sits right there — so fail
// and say to pin it.
//
// The reverse (golden present, checkpoint absent) is NOT an error: the committed golden is the
// certification RECORD, not a promise the 1.2 GB gated checkpoint is available. CI and most dev
// machines have no checkpoint, so the parity gates legitimately skip there; asserting the checkpoint
// exists whenever the golden does is what turned CI red.
func TestQwen3Embedding_gatesAreWired(t *testing.T) {
	haveCheckpoint := false
	for _, pat := range findEmbedCheckpointPatterns() {
		if m, _ := filepath.Glob(os.ExpandEnv(pat)); len(m) > 0 {
			haveCheckpoint = true
			break
		}
	}
	if !haveCheckpoint {
		t.Skip("no Qwen3-Embedding checkpoint present — parity gates skip (certified at pin time; the committed golden is the record)")
	}
	if _, err := os.Stat(qwen3EmbedGolden); err != nil {
		t.Fatalf("Qwen3-Embedding checkpoint is present but %s is not pinned — run scripts/pin_qwen3_embedding.py so the parity gates actually run, not silently skip", qwen3EmbedGolden)
	}
}

func truncStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
