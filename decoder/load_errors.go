package decoder

import (
	"errors"
	"fmt"
)

// The kinds of error Load refuses a model with, for errors.Is, so a caller can tell them apart without reading the text.
var (
	// ErrUnsupportedModel is a checkpoint of a family this build does not implement: config.json's model_type, or a GGUF's
	// general.architecture.
	ErrUnsupportedModel = errors.New("decoder: unsupported model")
	// ErrUnknownQuant is an Options.Quant that names no quantization.
	ErrUnknownQuant = errors.New("decoder: unknown quant")
	// ErrUnknownBackend is an Options.Backend that names no backend. A backend this build does not link is not this error: it
	// falls back to the CPU with a note (NewBackend).
	ErrUnknownBackend = errors.New("decoder: unknown backend")
)

// kindError is an error with its own text that answers errors.Is for a kind. The kind's text stays out of the message, which harnesses
// and people read.
type kindError struct {
	msg  string
	kind error
}

func (e *kindError) Error() string { return e.msg }
func (e *kindError) Unwrap() error { return e.kind }

func errorOfKind(kind error, format string, args ...any) error {
	return &kindError{msg: fmt.Sprintf(format, args...), kind: kind}
}
