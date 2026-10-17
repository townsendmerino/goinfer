package swapguard

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// BackstopMB is the system-wide swap growth that trips the guard on a machine where the per-process figure is read (CLAUDE.md, "guard
// thresholds"; Francis's decision of 2026-10-17). It is a fixed number, not an env read: a larger GOINFER_SWAP_GUARD raises it with
// the per-process threshold, so the backstop is never the lower of the two.
const BackstopMB = 4096

// PollInterval and ResumeAfter are the watch's sampling period and the serving half's resume hysteresis; variables so a test can shorten them.
var (
	PollInterval = 2 * time.Second
	ResumeAfter  = 30 * time.Second
)

// Source is one swap figure the guard watches and the growth over its baseline that trips it.
type Source struct {
	Name      string // what a trip message calls it: "server swap" or "system swap"
	Read      decoder.SwapReader
	Threshold int64 // bytes of growth over the first reading
}

// Select picks what the guard watches. own is the per-process reader (nil, with ownWhy, where the platform has none); sys is the
// system-wide one. Where own reads, the guard has two sources: the server's own swap at the GOINFER_SWAP_GUARD threshold (512 MB by
// default), and system swap at BackstopMB. Where it does not (no per-process figure, or this kernel reports no VmSwap), it has one: system
// swap at the GOINFER_SWAP_GUARD threshold, as before, and banner says why. ok is false for GOINFER_SWAP_GUARD=off.
func Select(own decoder.SwapReader, ownWhy string, sys decoder.SwapReader) (srcs []Source, banner string, ok bool) {
	threshold, mb, on := ThresholdBytes()
	if !on {
		return nil, "", false
	}
	if own != nil {
		if _, readable := own(); !readable {
			own, ownWhy = nil, "this process's VmSwap is not readable (a kernel without swap accounting?)"
		}
	}
	if own == nil {
		return []Source{{"system swap", sys, threshold}},
			fmt.Sprintf("system swap +%d MB over baseline (the per-process reader is not armed: %s)", mb, ownWhy), true
	}
	back := max(int64(BackstopMB), mb)
	return []Source{{"server swap", own, threshold}, {"system swap", sys, back * 1024 * 1024}},
		fmt.Sprintf("server swap (VmSwap of this process and its descendants) +%d MB over baseline, system swap +%d MB as a backstop", mb, back), true
}

// Watch is the running samplers of a guard, one per Source.
type Watch struct{ ws []*decoder.SwapWatch }

// Stop ends every sampler and blocks until each has exited. Safe on a nil receiver.
func (w *Watch) Stop() {
	if w == nil {
		return
	}
	for _, x := range w.ws {
		x.Stop()
	}
}

// Start runs one SwapWatch per source. onTrip fires when a source first passes its threshold; onResume (nil for none) when that source has
// been back within it for resumeAfter. Each source keeps its own baseline, printed once as it is first read.
func Start(srcs []Source, resumeAfter time.Duration, onTrip func(src Source, used, delta int64), onResume func(src Source)) *Watch {
	w := &Watch{}
	for _, src := range srcs {
		var once sync.Once
		w.ws = append(w.ws, decoder.StartSwapWatch(context.Background(), decoder.SwapWatchOptions{
			Read: func() (int64, bool) {
				used, ok := src.Read()
				once.Do(func() {
					if ok {
						fmt.Fprintf(os.Stderr, "swap guard: baseline %.2f GB (%s)\n", float64(used)/1e9, src.Name)
					} else {
						fmt.Fprintf(os.Stderr, "swap guard: the %s probe reports unknown on the first sample; that source will not trip until it reports a real reading\n", src.Name)
					}
				})
				return used, ok
			},
			PollInterval: PollInterval,
			Threshold:    src.Threshold,
			ResumeAfter:  resumeAfter,
			OnTrip:       func(used, delta int64) { onTrip(src, used, delta) },
			OnResume: func() {
				if onResume != nil {
					onResume(src)
				}
			},
		}))
	}
	return w
}
