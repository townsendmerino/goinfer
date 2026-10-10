package serveapp

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"slices"

	"github.com/townsendmerino/aikit/encoder"
)

// embedReq is the subset of the OpenAI /v1/embeddings request we honor, plus the
// input_type extension (Cohere/Voyage convention) for this encoder's asymmetric
// query/document encoding.
type embedReq struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`           // string | []string
	EncodingFormat string          `json:"encoding_format"` // "float" (default) | "base64"
	Dimensions     *int            `json:"dimensions"`      // optional: truncate (then renormalize)
	InputType      string          `json:"input_type"`      // extension: "query" | "document" (default)
	Task           string          `json:"task"`            // extension: a named task prompt of a model that has them, or "none"
	User           string          `json:"user"`            // accepted, ignored
}

// taskEmbedder is the optional capability of an embedder whose quality depends on a named task prompt (EmbeddingGemma
// 2's prompts, from its config_sentence_transformers.json). The handler then chooses the prompt (embedTask) instead
// of the query/document flag, and echoes the one it applied.
type taskEmbedder interface {
	PromptNames() []string
	PromptText(name string) (string, error)
	EncodeTasks(texts []string, prompt string) ([][]float32, []int, error)
}

// embedTask picks the prompt a taskEmbedder applies (docs/tasks/task-embeddinggemma2.md): an explicit task names one
// of the model's prompts ("none" for none); else input_type maps onto the model's own "query" and "document" prompts;
// else no prompt, which is what sentence-transformers applies by default, so a client that knows nothing about
// prompts gets the reference's own output.
func embedTask(te taskEmbedder, task, inputType string) (string, error) {
	if task != "" {
		if task == "none" {
			return "", nil
		}
		if _, err := te.PromptText(task); err != nil {
			return "", err
		}
		return task, nil
	}
	if inputType == "" {
		return "", nil
	}
	isQuery, err := parseInputType(inputType)
	if err != nil {
		return "", err
	}
	if isQuery {
		return "query", nil
	}
	return "document", nil
}

// Embedding request bounds. /v1/embeddings is deliberately un-queued (the encoder is goroutine-safe and parallelizes
// internally), so without a per-request cap one body drives an unbounded allocation and N concurrent requests
// multiply it: a 4 MiB body of empty strings is ~2M inputs, and the response builder would materialize 2M maps and 2M
// vectors before writing a byte. maxEmbedInputs matches OpenAI's per-request batch cap; maxEmbedInputBytes bounds a
// single input to text sizes (an embedding input is a query or passage, not a document dump).
const (
	maxEmbedInputs     = 2048
	maxEmbedInputBytes = 1 << 20 // 1 MiB
)

// checkEmbedInputBounds rejects a request whose input count or any single input exceeds the caps above,
// so the handler's allocation is bounded before EncodeBatch runs.
func checkEmbedInputBounds(inputs []string) error {
	if len(inputs) > maxEmbedInputs {
		return fmt.Errorf("too many inputs: %d (max %d per request)", len(inputs), maxEmbedInputs)
	}
	for i, in := range inputs {
		if len(in) > maxEmbedInputBytes {
			return fmt.Errorf("input %d is %d bytes, exceeds the %d-byte per-input limit", i, len(in), maxEmbedInputBytes)
		}
	}
	return nil
}

// handleEmbeddings serves POST /v1/embeddings. Vectors are L2-normalized (so cosine
// is a dot product, matching OpenAI's unit-length outputs); an optional dimensions
// field truncates each vector and renormalizes (Matryoshka-style). encoding_format
// "base64" returns little-endian float32 bytes, else a JSON number array.
func (s *server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	// The route is registered unconditionally so an unconfigured server gives a JSON error naming the constraint and
	// the flag that fixes it, not Go's default text/plain "404 page not found", which SDKs surface as NotFoundError
	// (reading as a wrong URL rather than an unconfigured server). 501 Not Implemented: the endpoint exists, this
	// server just has no embedding model loaded.
	if s.embed == nil {
		writeErr(w, http.StatusNotImplemented, "no embedding model is loaded; start the server with -embed-model <path> to enable /v1/embeddings")
		return
	}
	var req embedReq
	if !decodeJSON(w, r, &req) {
		return
	}
	inputs, err := parseEmbedInput(req.Input)
	var mediaItems []embedItem // set only when the request carries an image or audio
	if err != nil {
		items, ierr := parseEmbedItems(req.Input)
		if ierr != nil {
			writeErr(w, http.StatusBadRequest, ierr.Error())
			return
		}
		if ni, na := embedMedia(items); ni+na > 0 {
			_, okT := s.embed.(taskEmbedder)
			_, okI := s.embed.(imageEmbedder)
			_, okA := s.embed.(audioEmbedder)
			switch {
			case ni > 0 && (!okI || !okT):
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("model %q takes no image input", s.embedID))
				return
			case na > 0 && (!okA || !okT):
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("model %q takes no audio input", s.embedID))
				return
			case ni > maxEmbedImages:
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("too many images: %d (max %d per request)", ni, maxEmbedImages))
				return
			case na > maxEmbedAudio:
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("too many audio clips: %d (max %d per request)", na, maxEmbedAudio))
				return
			}
			mediaItems = items
		}
		inputs = make([]string, len(items))
		for i, it := range items {
			inputs[i] = it.text
		}
	}
	if len(inputs) == 0 {
		writeErr(w, http.StatusBadRequest, "input is required (a string or array of strings)")
		return
	}
	if err := checkEmbedInputBounds(inputs); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	isQuery, err := parseInputType(req.InputType)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	te, hasTasks := s.embed.(taskEmbedder)
	var task string
	if hasTasks {
		if task, err = embedTask(te, req.Task, req.InputType); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
	} else if req.Task != "" {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("model %q has no named task prompts; omit task (input_type selects query or document)", s.embedID))
		return
	}
	dims, err := s.resolveDimensions(req.Dimensions)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	base64Out, err := wantsBase64(req.EncodingFormat)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	// One isQuery flag for the whole request (OpenAI sends a homogeneous batch).
	isQueries := make([]bool, len(inputs))
	for i := range isQueries {
		isQueries[i] = isQuery
	}
	// The encoder is goroutine-safe and EncodeBatch parallelizes internally, so there is no server mutex (that guards
	// only the single shared decoder). embedBatchCounter reports each input's token count as a byproduct of the
	// tokenize pass EncodeBatch already makes, for encoders that can (the decoder-backed embedder); the rest (the
	// aikit-embed.Tokenizer path, an external interface this server does not control) fall back to EncodeBatch plus a
	// second, count-only tokenize pass (countEmbedTokens).
	var vecs [][]float32
	var promptTokens int
	if mediaItems != nil {
		ie, _ := s.embed.(imageEmbedder)
		ae, _ := s.embed.(audioEmbedder)
		vecs, promptTokens, err = encodeMediaItems(ie, ae, te, mediaItems, task)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "encode: "+err.Error())
			return
		}
	} else if hasTasks {
		var counts []int
		vecs, counts, err = te.EncodeTasks(inputs, task)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "encode: "+err.Error())
			return
		}
		for _, c := range counts {
			promptTokens += c
		}
	} else if bc, ok := s.embed.(embedBatchCounter); ok {
		var counts []int
		var err error
		vecs, counts, err = bc.EncodeBatchCounted(inputs, isQueries, 0)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, "encode: "+err.Error())
			return
		}
		for _, c := range counts {
			promptTokens += c
		}
	} else {
		var err error
		vecs, err = s.embed.EncodeBatch(inputs, isQueries, 0)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, "encode: "+err.Error())
			return
		}
		promptTokens = s.countEmbedTokens(inputs, isQuery)
	}

	data := make([]map[string]any, len(vecs))
	for i, v := range vecs {
		v = postprocess(v, dims) // L2-normalize, then truncate+renormalize if dims set
		emb := any(v)
		if base64Out {
			emb = float32sToBase64(v)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": emb}
	}
	resp := map[string]any{
		"object": "list",
		"data":   data,
		"model":  s.embedID,
		"usage":  map[string]any{"prompt_tokens": promptTokens, "total_tokens": promptTokens},
	}
	if hasTasks {
		// The prompt actually applied, so an index built under one prompt and queried under another can be seen:
		// that mismatch degrades retrieval quietly and is very hard to diagnose afterwards.
		name, text := task, ""
		if name == "" {
			name = "none"
		} else {
			text, _ = te.PromptText(task)
		}
		resp["goinfer_task"] = map[string]any{"name": name, "prompt": text}
		w.Header().Set("X-Goinfer-Embedding-Task", name)
	}
	writeJSON(w, http.StatusOK, resp)
}

// parseEmbedInput reads "input" as a string or []string. (OpenAI also allows
// token-id arrays; this server takes text only.)
func parseEmbedInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil {
		return many, nil
	}
	return nil, fmt.Errorf("input must be a string or an array of strings")
}

// parseInputType maps the input_type extension to the encoder's query/doc
// asymmetry. Default (empty) is document — the symmetric OpenAI behavior.
func parseInputType(t string) (isQuery bool, err error) {
	switch t {
	case "", "document", "search_document", "passage", "doc":
		return false, nil
	case "query", "search_query":
		return true, nil
	default:
		return false, fmt.Errorf("invalid input_type %q (want \"query\" or \"document\")", t)
	}
}

// resolveDimensions validates the optional output-dimension override against the model's native width; nil or 0 means
// full width.
//
// Truncation is legitimate only for models trained with Matryoshka Representation Learning: slicing any other
// embedder returns a unit-length, plausible vector that simply retrieves worse, a silent wrong. So a dimensions
// request below the model's floor, or any dimensions for a non-MRL model, is a 400, not a quietly degraded vector.
// s.embedMRLMin carries that floor (0 = not truncatable), resolved at load from aikit's exported registry, the source
// of its published Truncatable column; s.embedWidths, when set, is the exact set of allowed widths instead.
func (s *server) resolveDimensions(d *int) (int, error) {
	if d == nil || *d == 0 || *d == s.embedDim {
		return s.embedDim, nil // unset, or explicitly the native width: nothing to truncate
	}
	if *d < 1 || *d > s.embedDim {
		return 0, fmt.Errorf("dimensions must be between 1 and %d", s.embedDim)
	}
	if len(s.embedWidths) > 0 {
		if slices.Contains(s.embedWidths, *d) {
			return *d, nil
		}
		return 0, fmt.Errorf("dimensions %d is not a width %q was trained to be truncated to: dimensions must be one of %v", *d, s.embedID, s.embedWidths)
	}
	if s.embedMRLMin <= 0 {
		return 0, fmt.Errorf("model %q does not support the dimensions parameter: it was not trained "+
			"with Matryoshka Representation Learning, so a shortened vector would look valid but retrieve "+
			"worse. Omit dimensions, or pass %d", s.embedID, s.embedDim)
	}
	if *d < s.embedMRLMin {
		return 0, fmt.Errorf("dimensions %d is below the smallest supported width for %q: dimensions "+
			"must be between %d and %d", *d, s.embedID, s.embedMRLMin, s.embedDim)
	}
	return *d, nil
}

func wantsBase64(format string) (bool, error) {
	switch format {
	case "", "float":
		return false, nil
	case "base64":
		return true, nil
	default:
		return false, fmt.Errorf("invalid encoding_format %q (want \"float\" or \"base64\")", format)
	}
}

// postprocess L2-normalizes v (the encoder returns raw CLS vectors), then — if
// dims < len(v) — truncates and renormalizes so the shortened vector is still
// unit length. Operates on a copy when truncating so the original isn't aliased.
func postprocess(v []float32, dims int) []float32 {
	l2normalize(v)
	if dims < len(v) {
		v = append([]float32(nil), v[:dims]...)
		l2normalize(v)
	}
	return v
}

// l2normalize scales v in place to unit length; a zero vector is left as-is.
func l2normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// float32sToBase64 encodes v as OpenAI does: little-endian float32 bytes, base64
// (standard alphabet).
func float32sToBase64(v []float32) string {
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

// embedTokenCounter is implemented by embedders that carry their own tokenizer instead of an aikit embed.Tokenizer:
// the decoder-backed embedder (docs/completed/task-decoder-as-embedder.md). s.embedTok is nil for those, so without
// this prompt_tokens would silently report 0 on every response.
type embedTokenCounter interface {
	CountTokens(text string, isQuery bool) int
}

// embedBatchCounter is the optional capability of an embedder that can report each input's token count as a byproduct
// of the tokenize pass EncodeBatch already makes, so the handler need not tokenize every input twice
// (embedTokenCounter and countEmbedTokens below). The decoder-backed embedder implements it; the
// aikit-embed.Tokenizer path does not (an external interface this server does not control), so it keeps the two-pass
// fallback.
type embedBatchCounter interface {
	EncodeBatchCounted(texts []string, isQueries []bool, concurrency int) (vecs [][]float32, tokenCounts []int, err error)
}

// countEmbedTokens sums the wrapped token counts the encoder actually sees
// ([CLS]+prefix+text+[SEP], truncated to the model's max length), reusing the
// encoder's own tokenizer + query/doc rule. Tokenize-only, no forward pass.
func (s *server) countEmbedTokens(inputs []string, isQuery bool) int {
	if c, ok := s.embed.(embedTokenCounter); ok {
		total := 0
		for _, text := range inputs {
			total += c.CountTokens(text, isQuery)
		}
		return total
	}
	if s.embedTok == nil {
		return 0 // no tokenizer (e.g. a stub encoder in tests): skip the count
	}
	total := 0
	for _, text := range inputs {
		var (
			ids []int32
			err error
		)
		if isQuery {
			ids, err = encoder.EncodeQuery(s.embedTok, text, 0)
		} else {
			ids, err = encoder.EncodeDoc(s.embedTok, text, 0)
		}
		if err == nil {
			total += len(ids)
		}
	}
	return total
}
