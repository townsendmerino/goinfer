//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// Foreign CUDA contexts — asking the real question.
//
// Several gates in this package assert against a PINNED device allocation floor, and that floor is not a
// property of the code: it moves when another process holds a CUDA context on the device (a desktop
// compositor's context raised it by exactly 16 MiB on the measuring box). The kernel's requirement is
// unchanged (the demand identity still closes to the byte); only the environment-dependent component moved.
// Figures: docs/code-notes/cuda.md#foreign_context_test.header.
//
// WHY THIS HELPER EXISTS AT ALL. moe_route_demand_test.go used to discriminate the two regimes with
// `warm := freeBefore < pinnedDeviceFloor`, a heuristic that infers device state from a number and goes red
// claiming a broken identity once the floor moves. This asks the device.
//
// nvidia-smi rather than NVML bindings: this is test-only, runs once, and adding a library dependency to answer
// a question a shipped tool already answers would be a worse trade. If nvidia-smi is absent the caller is told
// "unknown" and must decide, never silently "none", which would restore the very failure mode this replaces.
type foreignCtx struct {
	pid   int
	name  string
	bytes int64
}

// foreignCUDAContexts lists processes OTHER than this one holding a CUDA context
// on the device. ok=false means the question could not be answered (no
// nvidia-smi, unparsable output) — which is NOT the same as "none", and callers
// must not collapse the two.
func foreignCUDAContexts() (out []foreignCtx, ok bool) {
	cmd := exec.Command("nvidia-smi",
		"--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
	b, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	self := os.Getpid()
	for line := range strings.SplitSeq(strings.TrimSpace(string(b)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		pid, perr := strconv.Atoi(strings.TrimSpace(f[0]))
		if perr != nil {
			continue
		}
		if pid == self {
			continue
		}
		mib, _ := strconv.ParseInt(strings.TrimSpace(f[2]), 10, 64)
		out = append(out, foreignCtx{pid: pid, name: strings.TrimSpace(f[1]), bytes: mib << 20})
	}
	return out, true
}

// describeForeign renders the contexts for a failure or skip message. The point
// is that a reader sees WHICH process and HOW MUCH without running anything.
func describeForeign(cs []foreignCtx) string {
	if len(cs) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(cs))
	for _, c := range cs {
		parts = append(parts, fmt.Sprintf("pid %d %s (%.0f MiB)", c.pid, c.name, float64(c.bytes)/(1<<20)))
	}
	return strings.Join(parts, ", ")
}
