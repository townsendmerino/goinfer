package decoder

import (
	"errors"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/families"
)

// registry aliases the family adapter table in internal/runtime/families.
var registry = families.Registry

// resolveArchitecture delegates to internal/runtime/families.Resolve and maps an
// unsupported model_type onto ErrUnsupportedModel with the original message.
func resolveArchitecture(cfg *Config) (*darch.Architecture, *archparts.TensorSchema, error) {
	a, s, err := families.Resolve(cfg)
	if errors.Is(err, families.ErrUnsupportedModel) {
		return nil, nil, errorOfKind(ErrUnsupportedModel, "decoder: unsupported model_type %q (have: %s)", cfg.ModelType, families.KnownModelTypes())
	}
	return a, s, err
}
