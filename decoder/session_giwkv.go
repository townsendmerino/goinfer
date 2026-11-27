package decoder

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/giwkv"
)

// A Session's KV cache and token list persist as a ".giw-kv" snapshot (format and guards in internal/runtime/giwkv), so a
// prefilled conversation survives a server restart or moves between processes instead of re-prefilling. cmd/serve uses it
// to checkpoint its SessionLRU.

// SnapshotError is returned by Model.LoadSession on any magic/version/geometry/
// CRC mismatch — distinct so callers can skip a stale or foreign snapshot and
// fall back to a cold prefill rather than treating it as fatal.
type SnapshotError = giwkv.SnapshotError

// Snapshot serializes the session's KV cache and token sequence to a portable
// blob. id is an opaque identity string (e.g. the model name or a conversation
// key) stored for tooling. The blob is self-describing and CRC-guarded; load it
// with Model.LoadSession on a model of the same architecture.
func (s *Session) Snapshot(id string) []byte {
	return s.snapshot(id, false)
}

func (s *Session) snapshot(id string, f16ByteAsF32 bool) []byte {
	c := s.cache
	// Refuse the state this format does not carry: every kind whose snapshot cell in the cache-state grid (cachestate.go) is
	// "refused", i.e. recurrent state (DeltaNet, Mamba-2, LFM2's conv window, KDA), the MLA latent, and the multimodal image blocks
	// and m-RoPE positions. Restoring any of them would start from zeroed or empty state and continue silently wrong, so Snapshot
	// returns nil (the caller skips and cold-prefills). The grid is read, not a list re-typed here: a hand-listed set once missed
	// LFM2 and a session was restored warm with empty conv windows. All other families serialize fully (giwkv.Encode refuses a
	// cache with non-uniform global-layer widths), including ring (windowed) and int8 caches.
	if c.holdsStateHandled(lcSnapshot, hRefused) {
		return nil
	}
	// The adapter is recorded by its registry name, the only identity that survives a restart. An
	// adapter runtime without a name cannot be looked up again on load, so refuse (cold prefill).
	adapterName := ""
	if c.lora != nil {
		if c.lora.name == "" {
			return nil
		}
		adapterName = c.lora.name
	}
	return giwkv.Encode(&c.c, id, adapterName, s.tokens, f16ByteAsF32)
}

// LoadSession reconstructs a *Session from a Snapshot blob, validating its KV
// geometry against this model (cache layout must match) before trusting any
// lengths. The returned session is ready to extend via Generate. On any magic/
// version/geometry/CRC mismatch it returns a *SnapshotError so the caller can
// skip it and start cold.
//
// wantID, when non-empty, must equal the id the snapshot was written with (the
// id arg to Snapshot) or the load is rejected. The geometry guard only catches a
// different *architecture*; an identity guard is what catches a same-shaped but
// different model (a different finetune, a different checkpoint file) whose KV
// would silently produce garbage. Pass "" to skip it (the id stays opaque to the
// decoder — equality is the only test).
func (m *Model) LoadSession(data []byte, wantID string) (*Session, error) {
	r, h, err := giwkv.ReadHeader(data)
	if err != nil {
		return nil, err
	}
	// Identity guard (post-CRC, so the id is trustworthy): reject KV written for a
	// different model even if the architecture happens to match.
	if wantID != "" && h.ID != wantID {
		return nil, &SnapshotError{Reason: fmt.Sprintf("model identity mismatch: snapshot %q, want %q", h.ID, wantID)}
	}
	// Adapter guard (v3): the KV was projected through this adapter, so the restored session is
	// bound to the same one. If this model has no adapter by that name the KV cannot be continued
	// correctly, so it is skipped like any other stale snapshot.
	var lora *loraRuntime
	if h.Adapter != "" {
		if lora = m.adapter(h.Adapter); lora == nil {
			return nil, &SnapshotError{Reason: fmt.Sprintf("built under adapter %q, which this model has not loaded", h.Adapter)}
		}
	}
	var ref *KVCache
	tokens, err := giwkv.Decode(r, h, func(capHint int) *cpuengine.Cache {
		ref = m.NewCache(capHint)
		return &ref.c
	})
	if err != nil {
		return nil, err
	}
	ref.lora = lora
	return &Session{m: m, cache: ref, tokens: tokens, kvAdapter: lora}, nil
}
