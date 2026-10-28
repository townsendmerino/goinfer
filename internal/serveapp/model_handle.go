package serveapp

import (
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/engine"
)

// modelHandle is a loaded model as the serving shell calls it, engine.Model. Assign one only through handleOf, so a
// nil model stays a nil handle (an interface holding a typed nil is not nil, and the shell tests model == nil).
type modelHandle = engine.Model

// blockDrafter is an attached pretrained block drafter, engine.BlockDrafter. Assign one only through drafterOf.
type blockDrafter = engine.BlockDrafter

// handleOf is m as a handle: nil for a nil m.
func handleOf(m *decoder.Model) modelHandle { return engine.HandleOf(m) }

// drafterOf is b as a blockDrafter: nil for a nil b.
func drafterOf(b *decoder.BlockSpec) blockDrafter { return engine.DrafterOf(b) }

// nilHandle reports whether h is nil or holds a nil pointer (engine.IsNil).
func nilHandle(h modelHandle) bool { return engine.IsNil(h) }

// mdl is the model to call methods on (engine.Call): lm.model, or a nil *decoder.Model when the entry has none, so a
// model-less entry answers as it always has. Compare lm.model itself with nil.
func (lm *loadedModel) mdl() modelHandle { return engine.Call(lm.model) }

// asDecoder is the *decoder.Model behind h, for the calls that take the concrete model (the session LRU, the decision
// heads); nil for a nil or fake handle.
func asDecoder(h modelHandle) *decoder.Model { return engine.AsDecoder(h) }
