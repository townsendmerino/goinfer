package gatecore

import (
	"fmt"
	"maps"
	"strings"
)

var backendMoveMap = map[string]string{
	"backend/cuda":   "cuda",
	"backend/metal":  "metal",
	"backend/webgpu": "gpu",
}

var internalMoveMap = map[string]string{
	"internal/runtime/adapter":    "internal/adapter",
	"internal/app/chatapp":        "internal/chatapp",
	"internal/app/cliutil":        "internal/cliutil",
	"internal/app/connect":        "internal/connect",
	"internal/app/decidecmd":      "internal/decidecmd",
	"internal/runtime/diskfree":   "internal/diskfree",
	"internal/runtime/f16":        "internal/f16",
	"internal/app/fitcmd":         "internal/fitcmd",
	"internal/runtime/giw":        "internal/giw",
	"internal/diag/hwcensus":      "internal/hwcensus",
	"internal/diag/hwreport":      "internal/hwreport",
	"internal/runtime/hybridattn": "internal/hybridattn",
	"internal/app/inproc":         "internal/inproc",
	"internal/diag/kernelprint":   "internal/kernelprint",
	"internal/runtime/kvmgr":      "internal/kvmgr",
	"internal/app/loadflags":      "internal/loadflags",
	"internal/runtime/mamba2":     "internal/mamba2",
	"internal/runtime/mem":        "internal/mem",
	"internal/guard/memgate":      "internal/memgate",
	"internal/runtime/modelload":  "internal/modelload",
	"internal/guard/movedpaths":   "internal/movedpaths",
	"internal/guard/paritykey":    "internal/paritykey",
	"internal/runtime/prequant":   "internal/prequant",
	"internal/guard/prezerolint":  "internal/prezerolint",
	"internal/runtime/program":    "internal/program",
	"internal/app/pullcmd":        "internal/pullcmd",
	"internal/runtime/sampling":   "internal/sampling",
	"internal/diag/sb2census":     "internal/sb2census",
	"internal/runtime/scheduler":  "internal/scheduler",
	"internal/app/serveapp":       "internal/serveapp",
	"internal/guard/servecheck":   "internal/servecheck",
	"internal/guard/swapguard":    "internal/swapguard",
	"internal/diag/towerid":       "internal/towerid",
	"internal/runtime/whisper":    "internal/whisper",
}

var allMoveMap = mergeMoveMaps(backendMoveMap, internalMoveMap)

func mergeMoveMaps(ms ...map[string]string) map[string]string {
	out := map[string]string{}
	for _, m := range ms {
		maps.Copy(out, m)
	}
	return out
}

// PreMovePath returns p with its moved backend or internal prefix replaced by the pre-move path, and whether it matched.
func PreMovePath(p string) (string, bool) {
	for nw, old := range allMoveMap {
		if p == nw {
			return old, true
		}
		if strings.HasPrefix(p, nw+"/") {
			return old + p[len(nw):], true
		}
	}
	return p, false
}

// CanonicalPath maps a pre-move backend path ("cuda", "metal", "gpu") to its current "backend/..." name.
func CanonicalPath(p string) string {
	for nw, old := range backendMoveMap {
		if p == old {
			return nw
		}
		if strings.HasPrefix(p, old+"/") {
			return nw + p[len(old):]
		}
	}
	return p
}

// AbsentFromRev reports whether err indicates git found no such path at a revision.
func AbsentFromRev(err error) bool {
	return err != nil && (strings.Contains(err.Error(), "does not exist") || strings.Contains(err.Error(), "but not in"))
}

// TreeAt returns the git tree SHA of dir at rev through lookup, falling back once to its pre-move path when absent.
func TreeAt(lookup func(rev, dir string) (string, error), rev, dir string) (string, error) {
	id, err := lookup(rev, dir)
	if err == nil {
		return id, nil
	}
	old, moved := PreMovePath(dir)
	if !moved || !AbsentFromRev(err) {
		return id, err
	}
	return lookup(rev, old)
}

// ObjectAt returns the git object SHA of path at rev through lookup, falling back once to its pre-move path when absent.
func ObjectAt(lookup func(rev, path string) (string, error), rev, path string) (string, error) {
	id, err := lookup(rev, path)
	if err == nil && id != "" {
		return id, nil
	}
	old, moved := PreMovePath(path)
	if !moved || (err != nil && !AbsentFromRev(err)) {
		return id, err
	}
	return lookup(rev, old)
}

// AbsentBeforeMove synthesizes a missing-path lookup response when a golden case holds only the pre-move twin.
func AbsentBeforeMove(kind string, args []string, has func(kind string, args ...string) bool) (string, error, bool) {
	old, moved := mapMovedArgs(args)
	if !moved || !has(kind, old...) {
		return "", nil, false
	}
	if kind == "object" || (kind == "git" && len(args) > 0 && args[0] == "ls-tree") {
		return "", nil, true
	}
	return "", fmt.Errorf("fatal: path does not exist"), true
}

func mapMovedArgs(args []string) ([]string, bool) {
	out := make([]string, len(args))
	moved := false
	for i, a := range args {
		if rev, p, ok := strings.Cut(a, ":"); ok {
			if q, m := PreMovePath(p); m {
				out[i], moved = rev+":"+q, true
				continue
			}
		}
		if q, m := PreMovePath(a); m {
			out[i], moved = q, true
			continue
		}
		out[i] = a
	}
	return out, moved
}

// Short8 truncates a commit SHA to 8 characters when longer.
func Short8(sha string) string {
	if len(sha) > 8 {
		return sha[:8]
	}
	return sha
}
