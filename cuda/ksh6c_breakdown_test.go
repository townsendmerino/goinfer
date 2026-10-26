//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSH6C_postPoolBreakdown is a reading for KS-H6's build (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build"), not a
// gate: one CUDA build of KSH6C_MODEL at KSH6C_CTX, served as serve loads it, with free VRAM read through the primary context at every
// requested allocation. It lists every allocation after the last K/V pool buffer (bytes, and the card's drop across it) and the
// drop no allocation accounts for, so the post-pool pricing can mirror what the build does.
func TestKSH6C_postPoolBreakdown(t *testing.T) {
	path := os.Getenv("KSH6C_MODEL")
	if path == "" {
		t.Skip("set KSH6C_MODEL")
	}
	ctx, _ := strconv.Atoi(os.Getenv("KSH6C_CTX"))
	kvF16Auto.Store(true)
	defer kvF16Auto.Store(false)
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	free := func() int64 {
		f, _, err := probe.Context().MemInfo()
		if err != nil {
			t.Errorf("MemInfo: %v", err)
		}
		return int64(f)
	}
	type event struct {
		kv          bool
		bytes       int
		freeBefore  int64
		freeAfterOf int64 // filled from the next event's freeBefore
	}
	var (
		mu     sync.Mutex
		events []event
	)
	allocSeenForTest = func(b int) {
		f := free()
		mu.Lock()
		events = append(events, event{bytes: b, freeBefore: f})
		mu.Unlock()
	}
	kvAllocSeenForTest = func(_ *cudaResident, b int) {
		mu.Lock()
		events = append(events, event{kv: true, bytes: b})
		mu.Unlock()
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctx})
	allocSeenForTest, kvAllocSeenForTest = nil, nil
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	after := free()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	lastKV := -1
	for i, e := range events {
		if e.kv {
			lastKV = i
		}
	}
	if lastKV < 0 {
		t.Fatal("no K/V pool allocation seen")
	}
	// The allocSeen event of the last pool buffer follows (f32) or precedes (f16) its kv event; start after both.
	start := lastKV + 1
	if start < len(events) && !events[start].kv && events[start].bytes == events[lastKV].bytes {
		start++
	}
	var post []event
	for i := start; i < len(events); i++ {
		if events[i].kv {
			continue
		}
		e := events[i]
		e.freeAfterOf = after
		for j := i + 1; j < len(events); j++ {
			if !events[j].kv {
				e.freeAfterOf = events[j].freeBefore
				break
			}
		}
		post = append(post, e)
	}
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	var req, charged int64
	type row struct {
		bytes int
		drop  int64
	}
	var rows []row
	for _, e := range post {
		req += int64(e.bytes)
		charged += int64(e.bytes) + allocRoundSlack(int64(e.bytes))
		rows = append(rows, row{e.bytes, e.freeBefore - e.freeAfterOf})
	}
	var startFree int64 = after
	if len(post) > 0 {
		startFree = post[0].freeBefore
	}
	fmt.Fprintf(os.Stderr, "[H6C] %s ctx %d kvF16 %v: %d allocations after the pools, requested %.1f MiB, charged by allocRoundSlack %.1f MiB; the card's drop from the first of them to the end of the load %.1f MiB; unaccounted %.1f MiB\n",
		path, r.ctxCap, r.kvF16, len(post), mib(req), mib(charged), mib(startFree-after), mib(startFree-after-charged))
	sort.Slice(rows, func(i, j int) bool { return rows[i].drop > rows[j].drop })
	for i, x := range rows {
		if i >= 25 {
			break
		}
		fmt.Fprintf(os.Stderr, "[H6C]   %10d B requested, the card's drop across it %.2f MiB\n", x.bytes, mib(x.drop))
	}
}
