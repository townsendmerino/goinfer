package serveapp

import (
	"math"
	"os"
	"sync"
	"testing"

	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/decoder"
)

// Decoder-as-embedder gates (docs/completed/task-decoder-as-embedder.md).
//
// These are the MECHANICAL properties — interface fit, instruction prefix, batch/sequential
// agreement, concurrency safety, honest token counting — and they run against a base Qwen3 decoder,
// which shares Qwen3-Embedding-0.6B's architecture and geometry (1024×28). The SEMANTIC gates
// (cosine > 0.9999 vs the HF sentence-transformers reference, retrieval top-k) require the
// embedding-tuned checkpoint plus a pinned golden — see TestQwen3Embedding_vectorParity below and
// scripts/pin_qwen3_embedding.py.

// The whole integration rests on this: a decoder-backed embedder must satisfy the same aikit
// interface the encoder-backed one does, so everything downstream of server.embed is unchanged.
var _ encoder.Encoder = (*decoderEmbedder)(nil)

func newTestDecoderEmbedder(t *testing.T) *decoderEmbedder {
	t.Helper()
	requireHeavyModel(t)
	p := os.ExpandEnv("$HOME/models/Qwen3-0.6B-Q8_0.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no checkpoint at %s", p)
	}
	tk, err := loadDecoderTokenizer(p)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	m, err := decoder.Load(p, decoder.Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return newDecoderEmbedder(m, tk, qwen3EmbedQueryPrompt, qwen3EmbedDocPrompt)
}

// TestDecoderEmbedder_queryPrefix pins the query/document asymmetry: a query is embedded with the
// Instruct preamble PREPENDED AND POOLED (include_prompt: true), a document with nothing. Getting
// this backwards (or stripping the prefix before pooling) is a silent-wrong — the vectors still look
// fine, they just land in the wrong place, which is what the retrieval gate exists to catch.
func TestDecoderEmbedder_queryPrefix(t *testing.T) {
	e := newTestDecoderEmbedder(t)
	const text = "what is the capital of France"

	docIDs, err := e.tokenize(text, false)
	if err != nil {
		t.Fatalf("tokenize doc: %v", err)
	}
	queryIDs, err := e.tokenize(text, true)
	if err != nil {
		t.Fatalf("tokenize query: %v", err)
	}
	if len(queryIDs) <= len(docIDs) {
		t.Fatalf("query (%d ids) must be longer than document (%d) — the Instruct prefix is not being prepended",
			len(queryIDs), len(docIDs))
	}
	// Every sequence ends with the appended EOD — that is the position last-token pooling reads,
	// and sentence-transformers appends it even though no config file mentions it.
	if e.appendID < 0 {
		t.Fatal("no EOD token resolved — pooling would land on the final content token")
	}
	for _, c := range []struct {
		name string
		ids  []int
	}{{"document", docIDs}, {"query", queryIDs}} {
		if got := c.ids[len(c.ids)-1]; got != e.appendID {
			t.Errorf("%s form ends with id %d, want the appended EOD %d", c.name, got, e.appendID)
		}
	}
	// The document form is the bare text + EOD (Qwen3-Embedding's document prompt is "").
	bare, err := e.tk.Encode(text, false)
	if err != nil {
		t.Fatalf("encode bare: %v", err)
	}
	if len(bare)+1 != len(docIDs) {
		t.Errorf("document form is not bare text + EOD: %d ids vs %d+1", len(docIDs), len(bare))
	}
	// And the query form is exactly prefix+text + EOD.
	want, err := e.tk.Encode(qwen3EmbedQueryPrompt+text, false)
	if err != nil {
		t.Fatalf("encode prefixed: %v", err)
	}
	if len(want)+1 != len(queryIDs) {
		t.Fatalf("query form is not prefix+text + EOD: %d ids vs %d+1", len(queryIDs), len(want))
	}
	// Different prompts must produce different vectors — proof the prefix reaches the model.
	qv, err := e.Encode(text, true)
	if err != nil {
		t.Fatalf("encode query: %v", err)
	}
	dv, err := e.Encode(text, false)
	if err != nil {
		t.Fatalf("encode doc: %v", err)
	}
	if cosineF32(qv, dv) > 0.9999 {
		t.Errorf("query and document embeddings are identical (cos %.6f) — the prefix is not affecting the pooled vector",
			cosineF32(qv, dv))
	}
}

// TestDecoderEmbedder_batchMatchesSequential: EncodeBatch must be exactly N Encodes. If a stale KV
// cache leaked between inputs, later vectors in a batch would differ from their standalone values.
func TestDecoderEmbedder_batchMatchesSequential(t *testing.T) {
	e := newTestDecoderEmbedder(t)
	texts := []string{"the cat sat on the mat", "quantum chromodynamics", "Paris is in France"}
	flags := []bool{false, true, false}

	batch, err := e.EncodeBatch(texts, flags, 0)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	if len(batch) != len(texts) {
		t.Fatalf("got %d vectors, want %d", len(batch), len(texts))
	}
	for i, text := range texts {
		one, err := e.Encode(text, flags[i])
		if err != nil {
			t.Fatalf("Encode %d: %v", i, err)
		}
		for j := range one {
			if batch[i][j] != one[j] {
				t.Fatalf("batch[%d] differs from standalone Encode at %d (%v vs %v) — KV state leaked between batch items",
					i, j, batch[i][j], one[j])
			}
		}
	}
	// Mismatched flag length is rejected rather than silently mis-prefixing.
	if _, err := e.EncodeBatch(texts, []bool{true}, 0); err == nil {
		t.Error("EncodeBatch must reject a isQueries length mismatch")
	}
}

// The embedBatchCounter capability (audit R-07) is what lets the /v1/embeddings handler skip a
// second, count-only tokenize pass over every input — it must actually be satisfied, or the
// handler silently falls back to the two-pass path and the fix does nothing.
var _ embedBatchCounter = (*decoderEmbedder)(nil)

// TestDecoderEmbedder_encodeBatchCountedMatchesSeparateCalls pins EncodeBatchCounted (audit R-07)
// against the two calls it replaces: its vectors must be bit-identical to EncodeBatch's (same
// tokenize, same forward — nothing about the actual encoding changes), and its counts must equal
// CountTokens's per-input counts (same ids, just not thrown away and re-derived from scratch).
func TestDecoderEmbedder_encodeBatchCountedMatchesSeparateCalls(t *testing.T) {
	e := newTestDecoderEmbedder(t)
	texts := []string{"the cat sat on the mat", "quantum chromodynamics", "Paris is in France"}
	flags := []bool{false, true, false}

	vecs, counts, err := e.EncodeBatchCounted(texts, flags, 0)
	if err != nil {
		t.Fatalf("EncodeBatchCounted: %v", err)
	}
	if len(vecs) != len(texts) || len(counts) != len(texts) {
		t.Fatalf("got %d vecs, %d counts, want %d of each", len(vecs), len(counts), len(texts))
	}

	batch, err := e.EncodeBatch(texts, flags, 0)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	for i := range texts {
		for j := range vecs[i] {
			if vecs[i][j] != batch[i][j] {
				t.Fatalf("EncodeBatchCounted vecs[%d] differs from EncodeBatch at %d (%v vs %v)",
					i, j, vecs[i][j], batch[i][j])
			}
		}
		want := e.CountTokens(texts[i], flags[i])
		if counts[i] != want {
			t.Errorf("EncodeBatchCounted counts[%d] = %d, want %d (CountTokens on the same input)",
				i, counts[i], want)
		}
		if counts[i] <= 0 {
			t.Errorf("counts[%d] = %d, want > 0", i, counts[i])
		}
	}

	// Mismatched flag length is rejected rather than silently mis-prefixing (same contract as
	// EncodeBatch).
	if _, _, err := e.EncodeBatchCounted(texts, []bool{true}, 0); err == nil {
		t.Error("EncodeBatchCounted must reject a isQueries length mismatch")
	}
}

// TestDecoderEmbedder_concurrentEncode is trap 1: /v1/embeddings runs WITHOUT the server mutex
// because an aikit encoder is goroutine-safe for concurrent Encode. A decoder is not — one KV cache,
// one scratch. This must be safe under -race AND must still return the correct vector; interleaved
// decoding would corrupt state and yield plausible-but-wrong results.
func TestDecoderEmbedder_concurrentEncode(t *testing.T) {
	e := newTestDecoderEmbedder(t)
	const text = "concurrent access must not corrupt the shared decoder"
	want, err := e.Encode(text, false)
	if err != nil {
		t.Fatalf("baseline Encode: %v", err)
	}

	const workers = 8
	var wg sync.WaitGroup
	errs := make([]error, workers)
	got := make([][]float32, workers)
	for i := range workers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got[i], errs[i] = e.Encode(text, false)
		}(i)
	}
	wg.Wait()
	for i := range workers {
		if errs[i] != nil {
			t.Fatalf("worker %d: %v", i, errs[i])
		}
		for j := range want {
			if got[i][j] != want[j] {
				t.Fatalf("worker %d diverged from the serial baseline at %d (%v vs %v) — concurrent Encode corrupted the shared decoder",
					i, j, got[i][j], want[j])
			}
		}
	}
}

// TestDecoderEmbedder_usageTokens is trap 2: countEmbedTokens reads s.embedTok (an aikit
// embed.Tokenizer), which is nil for a decoder-backed embedder — so usage.prompt_tokens silently
// reported 0 on every response. The embedTokenCounter seam must make it real, and must count the
// instruction prefix a query actually pays for.
func TestDecoderEmbedder_usageTokens(t *testing.T) {
	e := newTestDecoderEmbedder(t)
	s := &server{embed: e, embedDim: e.HiddenDim(), embedID: "qwen3-embed"}

	got := s.countEmbedTokens([]string{"hello world"}, false)
	if got <= 0 {
		t.Fatalf("prompt_tokens = %d for a decoder-backed embedder — the count is silently zero", got)
	}
	// Two inputs cost more than one.
	two := s.countEmbedTokens([]string{"hello world", "hello world"}, false)
	if two != 2*got {
		t.Errorf("two identical inputs counted %d, want %d", two, 2*got)
	}
	// A query pays for the Instruct prefix.
	q := s.countEmbedTokens([]string{"hello world"}, true)
	if q <= got {
		t.Errorf("query count %d must exceed document count %d (the prefix is tokenized too)", q, got)
	}
}

// TestDecoderEmbedder_emptyInput: an input that tokenizes to nothing has no position to pool. It
// must error rather than return a zero vector that would look like a legitimate embedding.
func TestDecoderEmbedder_emptyInput(t *testing.T) {
	e := newTestDecoderEmbedder(t)
	if _, err := e.Encode("", false); err == nil {
		t.Error("empty document input must error, not yield a zero vector")
	}
}

func cosineF32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
