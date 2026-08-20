//go:build cuda

package cuda

import (
	"fmt"
	"os"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// CUDA graph replay (r.graphs) is ~1.4–1.7× faster but is BIT-EXACT to live launch only under
// EXCLUSIVE_PROCESS device tenancy or active CUDA MPS. Under DEFAULT compute mode (time-sliced
// multi-context sharing) it silently mis-runs on this Turing box — proven by an MPS A/B (MPS-off
// diverges, MPS-on bit-exact ×10; docs/cuda-graphs-investigation.md §5.1). The backend must be
// "byte-identical or decline, never silently mis-run", so graphs are admitted only under a
// driver-enforced safe condition and then confirmed with a startup self-test.

// CU_COMPUTEMODE_* (cuda.h).
const (
	computeModeDefault          = 0
	computeModeExclusive        = 1 // deprecated single-context-exclusive
	computeModeProhibited       = 2
	computeModeExclusiveProcess = 3
)

// graphsTenancySafe reports whether graph replay is safe to enable, with a human reason either way.
// Safe = EXCLUSIVE_PROCESS compute mode (driver-enforced: no other context can share the device) OR
// active MPS (every client shares one server context, so there is no inter-context time-slicing).
func (r *cudaResident) graphsTenancySafe() (reason string, ok bool) {
	// MPS first: the operator started a server and pointed clients at its pipe directory. That removes
	// the time-slicing that corrupts baked replay regardless of the per-device compute mode.
	if dir := os.Getenv("CUDA_MPS_PIPE_DIRECTORY"); dir != "" {
		if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
			return "CUDA MPS active (CUDA_MPS_PIPE_DIRECTORY=" + dir + ")", true
		}
	}
	mode, err := r.dev.Context().Device().Attribute(gc.DeviceAttributeComputeMode)
	if err != nil {
		return fmt.Sprintf("could not read device compute mode (%v)", err), false
	}
	if mode == computeModeExclusiveProcess {
		return "compute mode EXCLUSIVE_PROCESS", true
	}
	names := map[int]string{computeModeDefault: "DEFAULT", computeModeExclusive: "EXCLUSIVE",
		computeModeProhibited: "PROHIBITED"}
	name, okn := names[mode]
	if !okn {
		name = fmt.Sprintf("mode %d", mode)
	}
	return fmt.Sprintf("compute mode %s and no active MPS — replay is bit-exact only under "+
		"EXCLUSIVE_PROCESS tenancy or MPS (docs/cuda-graphs-investigation.md §5.1)", name), false
}

// graphsSelfTest runs one forward twice — live-issue then graph-replay — on a fixed input and requires
// the logits to be BIT-EXACT. Runs on the executor thread. This is a CAPTURE-CORRECTNESS backstop (it
// catches a broken/stale capture); it CANNOT catch the tenancy hazard, which needs concurrent-context
// churn absent at startup — that is graphsTenancySafe's job. Leaves r.graphs=true on success; the
// caller disables graphs on error. The pos-0 K/V it writes is overwritten by the first real prefill.
func (r *cudaResident) graphsSelfTest() error {
	emb := make([]float32, r.hidden)
	for i := range emb {
		emb[i] = float32((i%13)-6) * 0.05 // deterministic, non-trivial
	}
	read := func(useGraphs bool) ([]float32, error) {
		r.graphs = useGraphs
		if e := r.launchToken(emb, 0, true); e != nil {
			return nil, e
		}
		if e := r.stream.Sync(); e != nil {
			return nil, e
		}
		out := make([]float32, r.vocab)
		if e := gpu.Download(r.logits, out); e != nil {
			return nil, e
		}
		return out, nil
	}
	live, e := read(false)
	if e != nil {
		return e
	}
	rep, e := read(true)
	if e != nil {
		return e
	}
	for i := range live {
		if live[i] != rep[i] {
			return fmt.Errorf("graph replay diverged from live at logit %d (%v != %v)", i, rep[i], live[i])
		}
	}
	r.graphs = true
	return nil
}

// admitGraphs applies the safe-gate: it is the ONLY place r.graphs is promoted from "requested" to
// "on". Order: tenancy gate (decline under DEFAULT even on an idle box) → capture → self-test.
// GOINFER_CUDA_GRAPHS_UNSAFE bypasses the tenancy gate for the on-box benchmark/investigation only,
// with a loud warning; the self-test still runs. Returns a fatal error only if capture itself fails
// (a half-captured chain is never run); a failed tenancy check or self-test just leaves graphs off.
func (r *cudaResident) admitGraphs() error {
	if !r.graphs {
		return nil
	}
	if r.dnet != nil {
		// A Gated-DeltaNet layer's mixer runs LIVE — it is not one of the three captured segments,
		// and the buffers it touches ARE the per-token recurrent state. Capturing around it would
		// leave the family's 3-in-4 linear layers outside the graph anyway, so the whole benefit
		// is gone; declining is honest rather than half-capturing. Graphs measured 1.01× on this
		// backend regardless (they are a safety feature, not a speed one), so nothing is lost.
		fmt.Fprintf(os.Stderr, "[cuda] CUDA graphs DECLINED: Gated-DeltaNet mixer layers run live (recurrent state)\n")
		r.graphs = false
		return nil
	}
	unsafe := os.Getenv("GOINFER_CUDA_GRAPHS_UNSAFE") != ""
	reason, ok := r.graphsTenancySafe()
	switch {
	case ok:
		fmt.Fprintf(os.Stderr, "[cuda] CUDA graphs enabled: %s\n", reason)
	case unsafe:
		fmt.Fprintf(os.Stderr, "[cuda] CUDA graphs FORCED (GOINFER_CUDA_GRAPHS_UNSAFE) despite: %s — "+
			"replay may not be bit-exact under context contention\n", reason)
	default:
		fmt.Fprintf(os.Stderr, "[cuda] CUDA graphs requested but DECLINED: %s\n", reason)
		r.graphs = false
		return nil
	}
	if e := r.do(func() error { return r.captureGraphs() }); e != nil {
		return fmt.Errorf("graph capture: %w", e)
	}
	if e := r.do(func() error { return r.graphsSelfTest() }); e != nil {
		fmt.Fprintf(os.Stderr, "[cuda] CUDA graphs DECLINED: startup bit-exactness self-test failed: %v\n", e)
		r.graphs = false
	}
	return nil
}
