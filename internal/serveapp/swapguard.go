package serveapp

import (
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/swapguard"
)

// Both halves of the swap tripwire (docs/tasks/task-never-swap-2026-09.md).
//
// The SERVING half (armSwapGuard, armed once for the process's life): the callback flips the admission gate to refuse new
// requests, lets in-flight generations finish, logs once, and re-opens when swap-used is back within threshold of the baseline
// for 30 s (hysteresis, so it does not flap). It reuses haltGate's chokepoint (right before inf, on every generation route)
// rather than adding another wrapper: swapGuardTripped is a second condition haltGate checks next to the halted pointer
// (halt.go). Unlike halt(), tripping the guard does NOT cancel anything already running; it only refuses new admissions.
//
// The LOAD-TIME half (internal/swapguard's ArmLoad, shared with goinfer-chat; armed and torn down around exactly one
// decoder.Load call): the callback cancels the load's context. The plumbing that stops the load lives in decoder
// (Options.LoadAbort, checked by parallelLayers); this file arms and tears down the watch around loadDecoder's decoder.Load
// call sites and turns a plain ErrLoadAborted into a priced message ("swap grew 0.9 GB during load: resident weights 12.6 GB +
// mapped source 12.1 GB on a 16 GB machine; use the sidecar path / a smaller quant"). Scoped to the GGUF direct-build resident
// path only; see Options.LoadAbort for why the .giw/streaming path is unaffected.
//
// Both halves share swapguard.ThresholdBytes so GOINFER_SWAP_GUARD cannot quote two different numbers for the same env var.

// startSwapGuard arms the swap watch, if enabled, and returns once the goroutine is running (it
// does not block on a first reading). Called once from newServer. GOINFER_SWAP_GUARD sets the
// threshold in MB (default 512); the literal value "off" disables the guard entirely (the watch
// is not started at all — no per-request branch cost, matching StartSwapWatch's own nil-Read
// no-op shape). An unparseable non-"off" value falls back to the default and says so, rather
// than silently guarding at 0 (which would trip on the first tick) or not guarding at all.
func (s *server) startSwapGuard() {
	// Under `go test`, do not arm a REAL background watch against the machine's actual swap-used unless a test explicitly
	// asked for one (GOINFER_SWAP_GUARD set): many tests construct a real *server via newServer for unrelated reasons, and
	// each would start its own goroutine polling `sysctl -n vm.swapusage` every 2s for the rest of the test binary's life,
	// risking a real swap excursion tripping some OTHER test's handler through haltGate for no coverage gain
	// (swapguard_test.go exercises the mechanism through armSwapGuard with a scripted reader). testing.Testing() reports false
	// in a real binary, so production serving is unaffected. An explicit GOINFER_SWAP_GUARD (a number, or "off") always wins.
	if testing.Testing() && os.Getenv("GOINFER_SWAP_GUARD") == "" {
		return
	}
	own, why := decoder.SelfSwapReader()
	s.armSwapGuardSources(own, why, decoder.SwapUsedBytes)
}

// armSwapGuard is armSwapGuardSources with no per-process reader: the system-wide figure alone at GOINFER_SWAP_GUARD, the guard as it was before the
// per-process reader. Tests drive the trip -> haltGate -> resume path through it with a scripted reader.
func (s *server) armSwapGuard(read decoder.SwapReader) {
	s.armSwapGuardSources(nil, "no per-process reader was supplied", read)
}

// armSwapGuardSources arms the swap watch over the sources swapguard.Select picks (see there for the two-trip rule and the fallback), logging
// which reader is armed and why. The guard is tripped while any one source is past its threshold and re-opens when every tripped source has been
// back within its threshold for 30 s.
func (s *server) armSwapGuardSources(own decoder.SwapReader, ownWhy string, sys decoder.SwapReader) {
	srcs, banner, on := swapguard.Select(own, ownWhy, sys)
	if !on {
		fmt.Fprintln(os.Stderr, "swap guard: disabled (GOINFER_SWAP_GUARD=off)")
		return
	}
	fmt.Fprintf(os.Stderr, "swap guard: armed — %s\n", banner)

	var mu sync.Mutex
	tripped := map[string]bool{}
	s.swapWatch = swapguard.Start(srcs, swapguard.ResumeAfter,
		func(src swapguard.Source, used, delta int64) {
			mu.Lock()
			tripped[src.Name] = true
			mu.Unlock()
			s.swapGuardTripped.Store(true)
			fmt.Fprintf(os.Stderr, "swap guard: TRIPPED — %s grew %.2f GB over baseline (%.2f GB used now); refusing new requests until it recovers\n", src.Name, float64(delta)/1e9, float64(used)/1e9)
		},
		func(src swapguard.Source) {
			mu.Lock()
			delete(tripped, src.Name)
			clear := len(tripped) == 0
			mu.Unlock()
			if clear {
				s.swapGuardTripped.Store(false)
				fmt.Fprintf(os.Stderr, "swap guard: recovered — %s back within threshold of baseline for %s; admitting new requests again\n", src.Name, swapguard.ResumeAfter)
			}
		})
}
