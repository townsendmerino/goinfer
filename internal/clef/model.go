package clef

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/decoder"
)

// Model is the whole Clef decision pipeline: the record encoder, the backbone (a plain qwen3_5 model: the release's adapters are merged,
// docs/measurements/decisions-d10-clef-2026-10-02.md), and the joint head. One backbone pass scores every question of the request.
type Model struct {
	backbone  *decoder.Model
	head      *Head
	tokenize  Tokenize
	maxLength int
}

// NewModel joins a loaded backbone, a loaded head and a tokenizer callback (see Tokenize for the special-token rule). It refuses a head and a
// backbone whose widths disagree, so a mismatched pair fails at construction and not as a wrong number.
func NewModel(backbone *decoder.Model, head *Head, tok Tokenize, maxLength int) (*Model, error) {
	if backbone == nil || head == nil || tok == nil {
		return nil, fmt.Errorf("clef: NewModel needs a backbone, a head and a tokenizer")
	}
	if backbone.HiddenSize() != head.Cfg.HiddenSize {
		return nil, fmt.Errorf("clef: the backbone is %d wide and the head expects %d", backbone.HiddenSize(), head.Cfg.HiddenSize)
	}
	return &Model{backbone: backbone, head: head, tokenize: tok, maxLength: maxLength}, nil
}

// Answer is one question's result: its id and type, and its options in the head's order with the softmax probability of each.
type Answer struct {
	ID      string
	Type    int
	Options []string
	Probs   []float64
}

// Result is a request's answers (in the order the questions were sent) and the number of tokens the backbone read.
type Result struct {
	Answers     []Answer
	InputTokens int
	// StateTruncated is the number of state tokens cut to fit the context (see Encoded).
	StateTruncated int
}

// RequestError is an error in the request itself (malformed, an unsupported question, a schema that does not fit), as opposed to a failure of the model: a
// server answers it with a 4xx and the rest with a 5xx.
type RequestError struct{ Err error }

func (e *RequestError) Error() string { return e.Err.Error() }
func (e *RequestError) Unwrap() error { return e.Err }

// Decide encodes the request, runs the backbone once over the whole sequence, and scores every question with the head. A problem with the request itself is a
// *RequestError.
func (m *Model) Decide(ctx context.Context, request []byte) (*Result, error) {
	enc, err := Encode(request, m.tokenize, m.maxLength)
	if err != nil {
		return nil, &RequestError{err}
	}
	hidden, err := m.backbone.PromptHiddenAll(ctx, enc.InputIDs)
	if err != nil {
		return nil, err
	}
	logits, err := m.head.Forward(hidden, enc.InputIDs, enc.Questions, func(id int) ([]float32, error) {
		row := make([]float32, m.backbone.HiddenSize())
		return row, m.backbone.OutputEmbeddingRow(id, row)
	})
	if err != nil {
		return nil, err
	}
	res := &Result{InputTokens: len(enc.InputIDs), StateTruncated: enc.StateTruncated}
	for i, q := range enc.Questions {
		res.Answers = append(res.Answers, Answer{ID: q.ID, Type: q.Type, Options: q.OptionIDs, Probs: Softmax(logits[i])})
	}
	return res, nil
}
