package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"time"
)

// K2 (docs/task-halt-2026-09.md): a global, no-restart halt. The model stays loaded; resume is
// instant. Distinct from graceful shutdown (SIGINT/SIGTERM, main.go) — that one drains and
// exits; this one just stops taking and running work until told otherwise.

// haltInfo is the current halt state, or nil (not halted). Read via server.haltInfo, written via
// server.halt/server.resume. atomic.Pointer gives every inference request a lock-free read on
// its hot path (the whole point of putting the check ahead of inf — it must not itself become a
// bottleneck), at the cost of a small allocation per halt/resume transition, which are rare.
type haltInfo struct {
	reason  string
	at      time.Time
	trigger string // "admin" | "halt file" | "SIGUSR1" | "SIGUSR2" (resume only writes nil, not a haltInfo)
}

// haltState reports the current halt state (nil = running normally).
func (s *server) haltState() *haltInfo { return s.halted.Load() }

// halt sets the halted state, cancels every in-flight generation (K1's registry) with reason,
// waits (bounded) for them to actually stop, and — if cfg.haltExitCode is nonzero — exits the
// process with that code once they have. Called synchronously from every trigger (the admin
// handler, the halt-file poller, SIGUSR1) so each one's own caller sees (and, for the HTTP
// route, can report) the real quiescence time, not just "halt requested".
//
// Idempotent in effect: calling it again while already halted just overwrites reason/trigger/at
// and re-runs cancelAll (a no-op on anything already stopped) — "the latest halt call wins" is
// the simplest rule that does not need a queue of halt reasons.
func (s *server) halt(reason, trigger string) (quiescedIn time.Duration) {
	s.halted.Store(&haltInfo{reason: reason, at: time.Now(), trigger: trigger})
	n := s.gens.cancelAll(reason)
	quiescedIn = s.gens.waitEmpty(30 * time.Second)
	fmt.Fprintf(os.Stderr, "halt: %s (trigger=%s reason=%q, cancelled %d generation(s), quiesced in %s)\n",
		trigger, trigger, reason, n, quiescedIn)
	if s.cfg.haltExitCode != 0 {
		fmt.Fprintf(os.Stderr, "halt: -halt-exit-code %d set — exiting after quiescence\n", s.cfg.haltExitCode)
		os.Exit(s.cfg.haltExitCode)
	}
	return quiescedIn
}

// resume clears the halted state unconditionally, regardless of which trigger asked — the
// simplest composable rule for multiple independent triggers (admin, halt file, signals) with
// no single owner. This DOES mean an admin resume while -halt-file still exists on disk gets
// re-halted on the poller's very next tick (≤250ms) — the file, while it exists, is the
// standing authority for the halt-file trigger specifically, not a one-shot request. Documented
// here rather than hidden behind extra state, per this doc's own "every halt/cancel is loud"
// rule extended to this edge case.
func (s *server) resume(trigger string) {
	s.halted.Store(nil)
	fmt.Fprintf(os.Stderr, "resume: trigger=%s\n", trigger)
}

// haltGate is a chain-level wrapper (alongside auth/inf, main.go) — inference routes only.
// Checked AFTER auth (a bad key is still rejected during a halt — halting does not change
// what "authenticated" means) and BEFORE inf (a halt must not have to wait for an inflight
// slot: docs/task-halt-2026-09.md's own K2 gate, "a halt that has to wait for a slot is not a
// halt"). /admin/* and /health are never wrapped in this — an operator must always be able to
// resume a halted server and check its status.
func (s *server) haltGate(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if hi := s.haltState(); hi != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "halted", "reason": hi.reason})
			return
		}
		h(w, r)
	}
}

type adminHaltReq struct {
	Reason string `json:"reason"`
}

// handleAdminHalt is POST /admin/halt {"reason"}. Blocks until quiescence (bounded 30s) so the
// response itself is proof the halt landed, not just that it was requested — and so a caller
// (an operator, or the K2 gate test) gets the real time-to-quiescence back directly.
func (s *server) handleAdminHalt(w http.ResponseWriter, r *http.Request) {
	if !s.adminEnabled(w) {
		return
	}
	var req adminHaltReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Reason == "" {
		writeErr(w, http.StatusBadRequest, "reason is required")
		return
	}
	quiescedIn := s.halt(req.Reason, "admin")
	writeJSON(w, http.StatusOK, map[string]any{
		"halted": true, "reason": req.Reason, "quiesced_in_ms": quiescedIn.Milliseconds(),
	})
}

// handleAdminResume is POST /admin/resume.
func (s *server) handleAdminResume(w http.ResponseWriter, r *http.Request) {
	if !s.adminEnabled(w) {
		return
	}
	s.resume("admin")
	writeJSON(w, http.StatusOK, map[string]any{"halted": false})
}

// haltFilePoller watches cfg.haltFile every 250ms (docs/task-halt-2026-09.md's own interval) —
// present ⇒ halted, absent ⇒ resumed — so a supervisor can halt goinfer with `touch`/`rm` and no
// HTTP call, socket, or signal. Only acts on the RISING/FALLING edge of the file's presence (not
// every tick) so a long halt doesn't re-run cancelAll (a no-op, but a noisy log line) every
// 250ms. Runs until stop is closed; started from main only when -halt-file is set.
func haltFilePoller(s *server, path string, stop <-chan struct{}) {
	present := false
	t := time.NewTicker(250 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			_, err := os.Stat(path)
			now := err == nil
			if now == present {
				continue
			}
			present = now
			if present {
				s.halt("halt file", "halt file")
			} else {
				s.resume("halt file")
			}
		}
	}
}
