package decoder

import (
	"maps"
	"sort"
	"sync"
)

// Hardware information for `check --hardware` (docs/tasks/task-hardware-coverage-2026-10.md, H3). A backend registers a function that returns the device facts a bug report needs, one line each
// ("device: NVIDIA GeForce RTX 2070 SUPER", "compute capability: 7.5"), the same register-from-init shape as RegisterBackend and RegisterMemoryProbe so decoder stays free of the GPU packages.
var (
	hwInfoMu sync.RWMutex
	hwInfo   = map[string]func() []string{}
)

// RegisterHardwareInfo registers name's device-facts function. It is called at report time, not at registration, so a missing driver costs nothing until someone asks.
func RegisterHardwareInfo(name string, info func() []string) {
	hwInfoMu.Lock()
	defer hwInfoMu.Unlock()
	hwInfo[name] = info
}

// HardwareInfo returns each registered backend's device facts, keyed by backend name.
func HardwareInfo() map[string][]string {
	hwInfoMu.RLock()
	fns := make(map[string]func() []string, len(hwInfo))
	maps.Copy(fns, hwInfo)
	hwInfoMu.RUnlock()
	out := make(map[string][]string, len(fns))
	for k, f := range fns {
		out[k] = f()
	}
	return out
}

// RegisteredBackends lists the backend names linked into this binary (cpu is always present), sorted.
func RegisteredBackends() []string {
	backendMu.RLock()
	names := []string{"cpu"}
	for n := range backendRegistry {
		if n != "cpu" {
			names = append(names, n)
		}
	}
	backendMu.RUnlock()
	sort.Strings(names)
	return names
}

// Backend self-tests (H2). A GPU backend registers the function that runs its own self-test on its device, once, the same register-from-init shape as the memory probe; the backend itself calls it at
// BuildResident, and a hardware report calls RunSelfTests to run them all on demand.
var (
	selfTestFnMu sync.RWMutex
	selfTestFns  = map[string]func() SelfTestResult{}
)

// RegisterSelfTest registers backend name's self-test.
func RegisterSelfTest(name string, fn func() SelfTestResult) {
	selfTestFnMu.Lock()
	defer selfTestFnMu.Unlock()
	selfTestFns[name] = fn
}

// RunSelfTests runs the CPU self-test (once per process) and every registered backend's, records the results, and returns them all sorted by backend. It is what `check --hardware` calls; a model
// load runs the CPU test itself and each backend runs its own at BuildResident.
func RunSelfTests() []SelfTestResult {
	ensureCPUSelfTest()
	if !SelfTestsSkipped() {
		selfTestFnMu.RLock()
		fns := make(map[string]func() SelfTestResult, len(selfTestFns))
		maps.Copy(fns, selfTestFns)
		selfTestFnMu.RUnlock()
		for _, f := range fns {
			RecordSelfTest(f())
		}
		// Every linked GPU backend runs the shared resident probe (selftest_gpu.go) unless it registered its own.
		for _, b := range RegisteredBackends() {
			if _, own := fns[b]; b != "cpu" && !own {
				probeStandalone(b)
			}
		}
	}
	return SelfTestResults()
}
