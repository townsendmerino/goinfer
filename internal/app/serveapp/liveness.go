package serveapp

import (
	"fmt"
	"io"
	"net/http"
	"reflect"
	"sync"
)

// Model liveness and the drain-based unload path ( has the design; reciprocal
// comments at handleAdminUnload and resident.Close).
//
// Freeing a model's native memory on unload (purego has no ARC or finalizers) is a use-after-free unless every in-flight
// request that holds the model has finished. The request preamble (pick -> tokenize -> prepare) touches the model with no
// lock, so a TryLock on the decode mutex cannot see it. So there is a per-*decoder.Model read/write lock held by every request
// from resolution to completion (rw), plus a count of registry entries backed by the model (refs). Unload drains rw before
// Close; refs decides when Close is the LAST owner's to make.

// modelLiveness is shared by every registry entry backed by one *decoder.Model (a base and its
// compute-time adapters share one). Guarded for mutation by server.regMu; rw is taken directly.
type modelLiveness struct {
	rw   sync.RWMutex // request holders take RLock (resolution→completion); unload takes Lock to drain
	refs int          // registry entries backed by this model; Close is safe only at refs==0
}

// nilCloser reports whether c is nil or holds a typed nil pointer (which c == nil does not catch).
func nilCloser(c io.Closer) bool {
	if c == nil {
		return true
	}
	v := reflect.ValueOf(c)
	return v.Kind() == reflect.Pointer && v.IsNil()
}

// livenessKey is this entry's model identity in server.liveness and its unload Close target: the concrete model when
// loaded, else a test-supplied closer; nil when the entry holds neither.
func (lm *loadedModel) livenessKey() io.Closer {
	if lm.model != nil {
		return lm.model
	}
	if nilCloser(lm.closer) {
		return nil
	}
	return lm.closer
}

// retainLocked records one more registry entry backed by c. CALLER HOLDS regMu.Lock. Called at every
// publish site (startup --model / --adapter, admin load) so refs always matches the live entry count.
func (s *server) retainLocked(c io.Closer) {
	if nilCloser(c) {
		return
	}
	if s.liveness == nil {
		s.liveness = map[io.Closer]*modelLiveness{}
	}
	ml := s.liveness[c]
	if ml == nil {
		ml = &modelLiveness{}
		s.liveness[c] = ml
	}
	ml.refs++
}

// releaseLocked drops one entry backed by c and reports (its liveness, whether this was the LAST
// owner). CALLER HOLDS regMu.Lock and MUST HAVE ALREADY deleted the entry from s.models — the
// delete-before-decide invariant: it makes "am I the last owner?" a question about a registry that
// no longer contains me, so of two concurrent sibling unloads the second to run necessarily sees
// refs==0 and closes. Reverse it and both can read refs>0 and both decline, orphaning the model
// with no entry left to retry — and double-decline is unrecoverable where double-Close is idempotent.
func (s *server) releaseLocked(c io.Closer) (*modelLiveness, bool) {
	if nilCloser(c) {
		return nil, false
	}
	ml := s.liveness[c]
	if ml == nil {
		return nil, false // never retained (defensive): treat as not-closable, leak-not-crash
	}
	ml.refs--
	last := ml.refs <= 0
	if last {
		delete(s.liveness, c)
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
		if k := lm.livenessKey(); k != nil {
			if ml = s.liveness[k]; ml != nil {
				ml.rw.RLock()
			}
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

// withModel is the ONE route from a request's model name to a *loadedModel on the OpenAI surfaces. The RLock is held for the
// whole handler body fn, spanning the preamble AND the generation, and the single deferred release covers every exit route
// (early return, panic, disconnect), so no handler can leak the read-lock (which would hang unload's drain forever) or skip
// it.
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

// closeEntryNatives releases this entry's native resources — the vision and audio towers — which would otherwise leak on
// unload just like the model's weights. Idempotent (nils each). The audio encoders load per entry and are always closed;
// the startup vision tower is shared by every startup entry, adapters included, so it is closed only by the last holder
// (visionShare). The shared *decoder.Model is NOT closed here; that is refs-gated in the unload path.
func (lm *loadedModel) closeEntryNatives() {
	last := lm.visionShare.release()
	lm.visionShare = nil
	if last && lm.vision != nil {
		lm.vision.close()
	}
	lm.vision = nil
	if lm.audio != nil {
		lm.audio.close()
	}
}

// visionShare counts the registry entries holding one startup vision tower. release reports whether the caller was the
// last holder; each holder's unload drains it first, so the last release runs with no encode in flight on any of them.
// A nil *visionShare is an entry nobody shares a tower with: its release is always the last.
type visionShare struct {
	mu sync.Mutex
	n  int
}

func (v *visionShare) release() bool {
	if v == nil {
		return true
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	v.n--
	return v.n <= 0
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
		defer recoverDrain(lm.name)
		if ml != nil {
			ml.rw.Lock() // DRAIN: waits out every in-flight request holding this model (all siblings)
		}
		// KV is settled now (no in-flight generation). Checkpoint before the buffers it lives in are
		// freed — for a resident backend the KV is inside the model's device memory.
		if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {
			// lm.sessMu is the sessionLRU's guard (sessions.go documents the LRU as NOT goroutine-safe), and demoteLoop
			// mutates it on its own ticker. The drain waited out in-flight REQUESTS, which is different: the idle-demote
			// goroutine is not a request and does not hold ml.rw. The window is narrow (-kv-idle-demote + -session-dir + an
			// unload landing on a tick), but it is a map mutated concurrently, a process crash rather than a wrong answer.
			// admin.go takes lm.sessMu around the load-side restore for the same reason.
			lm.sessMu.Lock()
			if err := lm.sessions.save(sessionSubdir(s.cfg.sessionDir, lm.fp)); err != nil {
				fmt.Fprintf(sessionNoticeOut, "unload: model %q: session checkpoint not written (%v); its sessions will not restore\n", lm.name, err)
			}
			lm.sessMu.Unlock()
		}
		lm.closeEntryNatives()
		if last {
			if c := lm.livenessKey(); c != nil {
				_ = c.Close()
			}
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
