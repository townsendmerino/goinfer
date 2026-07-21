package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"

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
	User           string          `json:"user"`            // accepted, ignored
}

// handleEmbeddings serves POST /v1/embeddings. Vectors are L2-normalized (so cosine
// is a dot product, matching OpenAI's unit-length outputs); an optional dimensions
// field truncates each vector and renormalizes (Matryoshka-style). encoding_format
// "base64" returns little-endian float32 bytes, else a JSON number array.
func (s *server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req embedReq
	if !decodeJSON(w, r, &req) {
		return
	}
	inputs, err := parseEmbedInput(req.Input)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(inputs) == 0 {
		writeErr(w, http.StatusBadRequest, "input is required (a string or array of strings)")
		return
	}
	isQuery, err := parseInputType(req.InputType)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
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
	// Encoder is goroutine-safe and EncodeBatch parallelizes internally, so no
	// s.mu (that guards only the single shared decoder).
	vecs, err := s.embed.EncodeBatch(inputs, isQueries, 0)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "encode: "+err.Error())
		return
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

	promptTokens := s.countEmbedTokens(inputs, isQuery)
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  s.embedID,
		"usage":  map[string]any{"prompt_tokens": promptTokens, "total_tokens": promptTokens},
	})
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

// resolveDimensions validates the optional output-dimension override against the
// model's native width. nil/0 means full width.
func (s *server) resolveDimensions(d *int) (int, error) {
	if d == nil || *d == 0 {
		return s.embedDim, nil
	}
	if *d < 1 || *d > s.embedDim {
		return 0, fmt.Errorf("dimensions must be between 1 and %d", s.embedDim)
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

// embedTokenCounter is implemented by embedders that carry their OWN tokenizer instead of an aikit
// embed.Tokenizer — the decoder-backed embedder (docs/task-decoder-as-embedder.md). s.embedTok is
// nil for those, so without this the count below would silently report prompt_tokens: 0 on every
// response. Preferring this interface keeps usage honest for both embedder kinds.
type embedTokenCounter interface {
	CountTokens(text string, isQuery bool) int
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
