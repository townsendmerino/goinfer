//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// voidWindow brackets one timed arm with the void rule's two terms (docs/measurements/noise-registry.md §8): the machine's busy
// cores averaged over exactly the arm (hostmem.ReadCPU at both ends, so a spike inside it counts in full) and reclaimable memory at
// both ends. Call it just before the timer starts and call the returned end just after it stops. end reports ok iff the busy cores
// were at most half the cores and at least 2 GiB was reclaimable at both reads; a CPU sample pair that spans no interval voids the
// arm (it counts toward the caller's void cap). The test fails if a reading cannot be taken at all.
func voidWindow(t *testing.T) (end func() (ok bool, note string)) {
	t.Helper()
	read := func() (hostmem.CPUTicks, hostmem.Reading) {
		c, cok := hostmem.ReadCPU()
		r, rok := hostmem.Read()
		if !cok || !rok {
			t.Fatalf("no CPU or memory reading (cpu %v, memory %v)", cok, rok)
		}
		return c, r
	}
	c0, r0 := read()
	return func() (bool, string) {
		c1, r1 := read()
		gib := func(r hostmem.Reading) float64 { return float64(r.Reclaimable) / (1 << 30) }
		busy, ok := hostmem.BusyCores(c0, c1, runtime.NumCPU())
		if !ok {
			return false, fmt.Sprintf("CPU samples span no interval, reclaimable %.2f→%.2f GiB", gib(r0), gib(r1))
		}
		valid := busy <= float64(runtime.NumCPU())/2 && min(r0.Reclaimable, r1.Reclaimable) >= 2<<30
		return valid, fmt.Sprintf("busy cores %.2f, reclaimable %.2f→%.2f GiB", busy, gib(r0), gib(r1))
	}
}
