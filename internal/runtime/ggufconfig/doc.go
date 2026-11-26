// Package ggufconfig reads a GGUF file's metadata into a checkpoint Config, one reader per general.architecture, with
// the file's recommended sampling. It imports decoder/config and aikit's GGUF parser and nothing of decoder, which calls
// Read and builds the weights from the same file.
//
// Every reader turns hostile metadata into an error, never a panic: Read recovers a reader's panic into one, and a reader
// that allocates per layer bounds block_count first (ggufLayerCount). FuzzGGUFConfig holds that contract.
package ggufconfig

import (
	"errors"
	"fmt"

	"github.com/townsendmerino/aikit/embed"

	"github.com/townsendmerino/goinfer/decoder/config"
)

// Config and RecommendedSampling are decoder/config's types, named here as decoder names them.
type (
	Config              = config.Config
	RecommendedSampling = config.RecommendedSampling
)

// ErrUnsupported is the kind of error Read returns for a file this build cannot read: an architecture no reader handles,
// or weights it cannot run (Hadamard-rotated). decoder reports it as ErrUnsupportedModel with the same text.
var ErrUnsupported = errors.New("ggufconfig: unsupported model")

// Read synthesizes a Config from a GGUF file's metadata, dispatching on general.architecture.
func Read(g *embed.GGUFFile) (*Config, error) { return ggufConfig(g) }

// kindError is an error with its own text that answers errors.Is for a kind; the kind's text stays out of the message.
type kindError struct {
	msg  string
	kind error
}

func (e *kindError) Error() string { return e.msg }
func (e *kindError) Unwrap() error { return e.kind }

func unsupported(format string, args ...any) error {
	return &kindError{msg: fmt.Sprintf(format, args...), kind: ErrUnsupported}
}
