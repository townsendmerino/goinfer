package serveapp

import (
	"net/http"
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
)

// Model liveness + the drain-based unload path. See docs/completed/task-admin-unload-drain.md for the full
// design and the reciprocal comments at handleAdminUnload and resident.Close.
//
// The problem this solves: freeing a model's native memory (purego has no ARC, no finalizers) on
// unload is a use-after-free unless every in-flight request that holds the model has finished first.
// The request preamble (pick → tokenize → prepare) touches the model with no lock, so the old
// mu.TryLock could not see it. The fix: a per-*decoder.Model read/write lock held by every request
// from resolution to completion (rw), plus a count of registry entries backed by the model (refs).
// Unload drains rw before Close; refs decides when Close is the LAST owner's to make.

// modelLiveness is shared by every registry entry backed by one *decoder.Model (a base and its
// compute-time adapters share one). Guarded for mutation by server.regMu; rw is taken directly.
type modelLiveness struct {
	rw   sync.RWMutex // request holders take RLock (resolution→completion); unload takes Lock to drain
	refs int          // registry entries backed by this model; Close is safe only at refs==0
}

// retainLocked records one more registry entry backed by m. CALLER HOLDS regMu.Lock. Called at every
// publish site (startup --model / --adapter, admin load) so refs always matches the live entry count.
func (s *server) retainLocked(m *decoder.Model) {
	if m == nil {
		return
	}
	if s.liveness == nil {
		s.liveness = map[*decoder.Model]*modelLiveness{}
	}
	ml := s.liveness[m]
	if ml == nil {
		ml = &modelLiveness{}
		s.liveness[m] = ml
	}
	ml.refs++
}

// releaseLocked drops one entry backed by m and reports (its liveness, whether this was the LAST
// owner). CALLER HOLDS regMu.Lock and MUST HAVE ALREADY deleted the entry from s.models — the
// delete-before-decide invariant: it makes "am I the last owner?" a question about a registry that
// no longer contains me, so of two concurrent sibling unloads the second to run necessarily sees
// refs==0 and closes. Reverse it and both can read refs>0 and both decline, orphaning the model
// with no entry left to retry — and double-decline is unrecoverable where double-Close is idempotent.
func (s *server) releaseLocked(m *decoder.Model) (*modelLiveness, bool) {
	ml := s.liveness[m]
	if ml == nil {
		return nil, false // never retained (defensive): treat as not-closable, leak-not-crash
	}
	ml.refs--
	last := ml.refs <= 0
	if last {
		delete(s.liveness, m)
	}
	return ml, last
}

// resolveAndLock resolves a name to an entry and takes the model's liveness RLock UNDER regMu —
// atomic with any unload's decision, which runs under regMu.Lock — so a request either holds the
// RLock before unload decides (unload's drain then waits for it) or finds the model already gone. It
// returns the entry and a release func (a no-op when nothing was locked). PRIVATE: the release
// discipline lives only in the withModel wrappers below, which defer it around fn — no handler ever
// sees the release, so none can skip it. This is the sole place a request resolves a model.
func (s *server) resolveAndLock(name string) (*loadedModel, func()) {
	s.regMu.RLock()
	lm := s.lookupLocked(name)
	var ml *modelLiveness
	if lm != nil {
		if ml = s.liveness[lm.model]; ml != nil {
			ml.rw.RLock()
		}
	}
	s.regMu.RUnlock()
	if lm == nil {
		return nil, nil
	}
	if ml == nil {
		return lm, func() {} // defensive: model never retained → no lock to release
	}
	return lm, ml.rw.RUnlock
}

// withModel is the ONE route from a request's model name to a *loadedModel on the OpenAI surfaces.
// The RLock is held for the whole handler body fn — spanning the preamble AND the generation, the
// window the old per-request mu missed — and the single deferred release covers every exit route
// (early return, panic, disconnect), so no handler can leak the read-lock (which would hang unload's
// drain forever). pick was removed; this wrapper (and withModelAnthropic) is why handlers cannot skip
// the lock.
func (s *server) withModel(w http.ResponseWriter, name string, fn func(*loadedModel)) {
	lm, release := s.resolveAndLock(name)
	if lm == nil {
		s.modelNotFound(w, name)
		return
	}
	defer release()
	preamblePark() // test-only seam (goinfer_testhooks): fires INSIDE the window, holding the RLock
	fn(lm)
}

// withModelAnthropic is withModel with the Anthropic-dialect not-found error shape (/v1/messages,
// /v1/messages/count_tokens). Same lock discipline.
func (s *server) withModelAnthropic(w http.ResponseWriter, name string, fn func(*loadedModel)) {
	lm, release := s.resolveAndLock(name)
	if lm == nil {
		s.anthropicModelNotFound(w, name)
		return
	}
	defer release()
	preamblePark()
	fn(lm)
}

// closeEntryNatives releases this entry's PRIVATE native resources — the vision tower — which are not
// shared with any other registry entry (adapters do not copy them) and would otherwise leak on
// unload just like the model's weights. Idempotent (nils each). The shared *decoder.Model is NOT
// closed here; that is refs-gated in the unload path.
func (lm *loadedModel) closeEntryNatives() {
	if lm.venc != nil {
		lm.venc.Close()
		lm.venc = nil
	}
	if lm.qwenEnc != nil {
		lm.qwenEnc.Close()
		lm.qwenEnc = nil
	}
	lm.vproj = nil // no native Close (weights); drop the reference
}

// startDrain runs the DETACHED phase of unload. The entry is already unpublished (Phase 1). On a bare
// goroutine — NOT tied to the admin request's context or the shutdown path, so a client disconnect
// or process shutdown cannot abort the free — it: marks name draining (surfaced by /health); drains
// every in-flight holder of the model via rw.Lock (guaranteed to terminate: no new RLock can appear
// once unpublished); checkpoints the now-settled KV before its buffers go; closes the entry's private
// natives; closes the model iff last owner; clears the draining mark. The returned channel closes
// when the native memory is actually freed, so the handler can answer 200 (freed) vs 202 (still
// draining). ml may be nil only if the model was never retained (defensive) — then there is nothing
// to drain and we skip Close (leak-not-crash).
func (s *server) startDrain(lm *loadedModel, ml *modelLiveness, last bool) <-chan struct{} {
	done := make(chan struct{})
	s.regMu.Lock()
	if s.draining == nil {
		s.draining = map[string]struct{}{}
	}
	s.draining[lm.name] = struct{}{}
	s.regMu.Unlock()

	go func() {
		defer close(done)
		if ml != nil {
			ml.rw.Lock() // DRAIN: waits out every in-flight request holding this model (all siblings)
		}
		// KV is settled now (no in-flight generation). Checkpoint before the buffers it lives in are
		// freed — for a resident backend the KV is inside the model's device memory.
		if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {
			_ = lm.sessions.save(sessionSubdir(s.cfg.sessionDir, lm.fp))
		}
		lm.closeEntryNatives()
		if last && lm.model != nil {
			_ = lm.model.Close()
		}
		if ml != nil {
			ml.rw.Unlock()
		}
		s.regMu.Lock()
		delete(s.draining, lm.name)
		s.regMu.Unlock()
	}()
	return done
}
