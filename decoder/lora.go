package decoder

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/loraload"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

// LoRA adapters (PEFT safetensors) are read, validated and merged by internal/runtime/loraload: merged into the base
// weights at load (Options.LoRA), or applied at compute time (Model.LoadAdapter). Only the safetensors base path
// supports merge — PEFT adapters target HF module names; a GGUF base + LoRA is rejected (see Load).

type loraDelta = cpuengine.LoRADelta

type loraAdapter = loraload.Adapter

// --- Compute-time LoRA: apply the low-rank delta in the forward instead of merging it into the base weight. The base
// stays immutable and zero-copy, so N adapters of one base share a single resident transformer (base plus N small
// deltas). Opt-in: merged LoRA stays the faster, simpler default; this trades a little decode speed for density. Only
// the generic dense forward (the per-token walker: projQKVToken, gatedMLPAct) is wired; Model.LoadAdapter rejects the
// special-forward, MoE and non-gated archs.

// loraLayerDelta holds one transformer layer's per-projection deltas; a nil entry
// is a projection the adapter does not target (the apply is a no-op).
type loraLayerDelta = cpuengine.LoRALayerDelta

// loraRuntime is a loaded adapter indexed for the forward: layer → projection
// deltas. The deltas alias the adapter's mmapped tensors (read-only), so it costs
// only the low-rank A/B bytes on top of the shared base.
type loraRuntime struct {
	name   string
	layers []loraLayerDelta
	a      *loraAdapter // retains the mmap so the aliased delta data stays valid
	// kvID is this runtime's identity in a KV key (kvAdapterID): 0 until first asked for.
	kvID atomic.Uint64
}

func (rt *loraRuntime) close() {
	if rt != nil {
		rt.a.Close()
	}
}

// buildLoraRuntime indexes a loaded adapter by (layer, projection) for compute-time
// application, mapping each schema suffix to the adapter's HF-named delta.
func buildLoraRuntime(name string, a *loraAdapter, numLayers int, s *archparts.TensorSchema) *loraRuntime {
	return &loraRuntime{name: name, layers: a.LayerDeltas(numLayers, s), a: a}
}

// LoadAdapter loads a PEFT LoRA adapter for compute-time application and registers it under name. Unlike the
// merge-at-load path (Options.Lora) the base stays immutable, so many adapters of one base share its resident weights.
// Only the generic dense forward is wired: MoE, non-gated MLP and every family with a layer outside the generic operations
// (Architecture.genericKindsOnly) are rejected, as is a GGUF or serialized base (the adapter is HF-named and needs the
// safetensors schema).
//
// The rejection must derive from the operations, not a hand-written family list: a family missing from a copy would load and
// validate cleanly and then silently apply nothing (a short convolution takes no LoRA delta).
// History: docs/code-notes/decoder.md#Model.LoadAdapter.
func (m *Model) LoadAdapter(name, dir string) error {
	_, done, err := m.enterCall(context.Background())
	if err != nil {
		return err
	}
	defer done()
	return m.closeErr(context.Background(), m.loadAdapter(name, dir))
}

// loadAdapter is LoadAdapter without the lifecycle guard (lifecycle.go).
func (m *Model) loadAdapter(name, dir string) error {
	arch := m.w.Arch
	switch {
	case m.w.Schema == nil:
		return fmt.Errorf("decoder: compute-time LoRA needs a safetensors base (HF tensor names)")
	case arch.OwnLoopFamily() || !arch.GenericKindsOnly():
		return fmt.Errorf("decoder: compute-time LoRA unsupported for %s (own forward path)", arch.Name)
	case arch.MoE != nil:
		return fmt.Errorf("decoder: compute-time LoRA unsupported for MoE (expert FFN not wired)")
	case arch.NonGatedMLP:
		return fmt.Errorf("decoder: compute-time LoRA unsupported for the non-gated MLP layout (%s)", arch.Name)
	}
	a, err := loraload.Load(dir)
	if err != nil {
		return err
	}
	// Compute-time application (buildLoraRuntime) indexes deltas by the bare wbuild.TensorName, so validate
	// against that — this path is prefix-agnostic (it applies by layer/projection, not physical name).
	if err := a.ValidateTargets(arch.NumLayers, m.w.Schema, wbuild.TensorName); err != nil {
		a.Close()
		return err
	}
	// ValidateTargets checks names only; every resident backend's SetAdapter trusts In/Out from the checkpoint, so a
	// mismatched adapter is caught here, once, before it reaches any of them.
	if err := a.ValidateComputeTimeDims(arch.NumLayers, m.w.Layers, m.w.Schema, wbuild.TensorName); err != nil {
		a.Close()
		return err
	}
	m.registerAdapter(name, buildLoraRuntime(name, a, arch.NumLayers, m.w.Schema))
	return nil
}

// registerAdapter installs rt under name, retiring any runtime it displaces. A live Session may still hold the displaced
// runtime via cache.lora and read its mmap'd deltas mid-generation, so it is retired (released at Model.Close), never
// munmap'd here. The registry's mutex guards this against concurrent UseAdapter/HasAdapter reads. The registry is
// allocated on first use (adapters load at setup, before serving), so the pointer field is only ever set here.
func (m *Model) registerAdapter(name string, rt *loraRuntime) {
	if m.adapters == nil {
		m.adapters = newAdapterRegistry()
	}
	reg := m.adapters
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if old := reg.byName[name]; old != nil {
		reg.retired = append(reg.retired, old)
	}
	reg.byName[name] = rt
}

// HasAdapter reports whether a compute-time adapter is registered under name.
func (m *Model) HasAdapter(name string) bool {
	reg := m.adapters
	if reg == nil {
		return false
	}
	reg.mu.Lock()
	defer reg.mu.Unlock()
	_, ok := reg.byName[name]
	return ok
}

// adapter returns the runtime registered under name (nil if absent), under the registry lock.
func (m *Model) adapter(name string) *loraRuntime {
	reg := m.adapters
	if reg == nil {
		return nil
	}
	reg.mu.Lock()
	defer reg.mu.Unlock()
	return reg.byName[name]
}
