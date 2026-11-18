//go:build darwin

package swapguard

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/mem/probe"
)

// TestSelect_darwinTwoSources: on macOS the guard watches the server's own compressed pages at the threshold and system swap at
// BackstopMB, the shape Linux has with VmSwap (D1, Francis, 2026-11-18). The sources are the ones Select picks over this platform's
// probe (probe.SelfSwapReader, probe.SwapUsed); each is then read from a script, and the serving watch (Start) is run over them.
//   - The apps-open D1 cell: system swap grows by 985 MiB while the server's own compressed pages stay at 150 MiB. It must not trip.
//   - The server's own compressed pages grow past 512 MB. It must trip, on the server's source.
//   - System swap grows past 4,096 MB with the server's own pages flat. It must trip, on the backstop.
func TestSelect_darwinTwoSources(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "")
	own, why := probe.SelfSwapReader()
	srcs, banner, on := Select(own, why, probe.SwapUsed)
	if !on || len(srcs) != 2 || srcs[0].Threshold != 512*mb || srcs[1].Threshold != BackstopMB*mb {
		var got []string
		for _, s := range srcs {
			got = append(got, s.Name+":"+itoa(s.Threshold/mb))
		}
		t.Fatalf("darwin sources %v (banner %q); want the server's own pages at 512 MB and system swap at %d MB", got, banner, BackstopMB)
	}

	saved := PollInterval
	PollInterval = 5 * time.Millisecond
	t.Cleanup(func() { PollInterval = saved })
	const base = int64(1175) * mb
	for _, c := range []struct {
		name          string
		ownEnd, sysTo int64 // the server's own compressed bytes and system swap after the growth
		want          string
	}{
		{"apps open: system swap +985 MiB, the server's own pages at 150 MiB", 150 * mb, base + 985*mb, ""},
		{"the server's own pages past the threshold", 600 * mb, base, srcs[0].Name},
		{"system swap past the backstop", 150 * mb, base + (BackstopMB+100)*mb, srcs[1].Name},
	} {
		t.Run(c.name, func(t *testing.T) {
			var ticks atomic.Int64
			script := func(start, end int64) func() (int64, bool) {
				var n atomic.Int64
				return func() (int64, bool) {
					ticks.Add(1)
					if n.Add(1) <= 2 {
						return start, true // the baseline and one more before the growth
					}
					return end, true
				}
			}
			run := []Source{srcs[0], srcs[1]}
			run[0].Read, run[1].Read = script(0, c.ownEnd), script(base, c.sysTo)
			var tripped atomic.Value
			w := Start(run, 0, func(src Source, _, _ int64) { tripped.CompareAndSwap(nil, src.Name) }, nil)
			deadline := time.Now().Add(2 * time.Second)
			for time.Now().Before(deadline) && tripped.Load() == nil && ticks.Load() < 200 {
				time.Sleep(5 * time.Millisecond)
			}
			w.Stop()
			got, _ := tripped.Load().(string)
			if got != c.want {
				t.Errorf("tripped on %q after %d readings; want %q", got, ticks.Load(), c.want)
			}
		})
	}
}
