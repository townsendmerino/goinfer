//go:build cuda && goinfer_testhooks

package cuda

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSH6C_postPoolBreakdown is a reading for KS-H6's build: one CUDA
// build of KSH6C_MODEL at KSH6C_CTX, served as serve loads it, with free VRAM read at every requested allocation. It lists every
// allocation after the last K/V pool buffer (bytes, and the card's drop across it) and the drop no allocation accounts for.
// KSH6C_NO_SELFTEST skips the decoder's startup self-test, whose fixtures build before this one. KSH6C_S4=1 makes it H6C-S4's
// gate: the card's drop from the first post-pool allocation to the end of the load must be at most 2 MiB.
func TestKSH6C_postPoolBreakdown(t *testing.T) {
	path := os.Getenv("KSH6C_MODEL")
	if path == "" {
		t.Skip("set KSH6C_MODEL")
	}
	ctx, _ := strconv.Atoi(os.Getenv("KSH6C_CTX"))
	if os.Getenv("KSH6C_NO_SELFTEST") != "" {
		decoder.SkipSelfTests() // the probe's fixtures build before this build
	}
	kvF16Auto.Store(true)
	defer kvF16Auto.Store(false)
	// Free VRAM is read through a short-lived device, as decoder.FreeBytesFor reads it: a handle held across the load would keep
	// the primary context alive between the self-test's fixture builds, and with it a local-memory pool serve does not keep.
	requireCUDADevice(t)
	free := func() int64 {
		f, ok := decoder.FreeBytesFor("cuda")
		if !ok {
			t.Errorf("no free-VRAM reading")
		}
		return f
	}
	type event struct {
		kv          bool
		bytes       int
		freeBefore  int64
		freeAfterOf int64 // filled from the next event's freeBefore
		site        string
		next        string // the next allocation's site: the drop falls between the two
	}
	var (
		mu     sync.Mutex
		events []event
	)
	site := func() string {
		pc := make([]uintptr, 24)
		n := runtime.Callers(2, pc)
		fr := runtime.CallersFrames(pc[:n])
		var parts []string
		for {
			f, more := fr.Next()
			if strings.Contains(f.Function, "goinfer/backend/cuda.") {
				parts = append(parts, f.Function[strings.Index(f.Function, "cuda.")+5:]+" "+f.File[strings.LastIndex(f.File, "/")+1:]+":"+strconv.Itoa(f.Line))
			}
			if !more || len(parts) == 6 {
				break
			}
		}
		return strings.Join(parts, " < ")
	}
	allocSeenForTest = func(b int) {
		f := free()
		st := site()
		mu.Lock()
		events = append(events, event{bytes: b, freeBefore: f, site: st})
		mu.Unlock()
	}
	kvAllocSeenForTest = func(_ *cudaResident, b int) {
		mu.Lock()
		events = append(events, event{kv: true, bytes: b})
		mu.Unlock()
	}
	fitByR := map[*cudaResident]int64{} // each build's first fit-check reading (checkKVFits): the self-test's fixtures build first
	saved := cudaFreeVRAM
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, e := saved(r)
		mu.Lock()
		if _, seen := fitByR[r]; !seen {
			fitByR[r] = int64(f)
		}
		mu.Unlock()
		return f, e
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctx})
	allocSeenForTest, kvAllocSeenForTest, cudaFreeVRAM = nil, nil, saved
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
				e.next = events[j].site
				break
			}
		}
		post = append(post, e)
	}
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	var req, charged int64
	type row struct {
		bytes      int
		drop       int64
		site, next string
	}
	var rows []row
	for _, e := range post {
		req += int64(e.bytes)
		charged += int64(e.bytes) + allocRoundSlack(int64(e.bytes))
		rows = append(rows, row{e.bytes, e.freeBefore - e.freeAfterOf, e.site, e.next})
	}
	var startFree int64 = after
	if len(post) > 0 {
		startFree = post[0].freeBefore
	}
	fmt.Fprintf(os.Stderr, "[H6C] %s ctx %d kvF16 %v: %d allocations after the pools, requested %.1f MiB, charged by allocRoundSlack %.1f MiB; the card's drop from the first of them to the end of the load %.1f MiB; unaccounted %.1f MiB\n",
		path, r.ctxCap, r.kvF16, len(post), mib(req), mib(charged), mib(startFree-after), mib(startFree-after-charged))
	fmt.Fprintf(os.Stderr, "[H6C] %s self-test skipped %v: the fit check read %.1f MiB free\n", path, os.Getenv("KSH6C_NO_SELFTEST") != "", mib(fitByR[r]))
	if os.Getenv("KSH6C_S4") != "" && startFree-after > 2<<20 {
		t.Errorf("H6C-S4: the card dropped %.1f MiB after the pools (want <= 2): something the margin does not price came after them", mib(startFree-after))
	}
	slices.SortFunc(rows, func(a, b row) int { return cmp.Compare(b.drop, a.drop) })
	for i, x := range rows {
		if i >= 25 {
			break
		}
		fmt.Fprintf(os.Stderr, "[H6C]   %10d B requested, the card's drop across it %.2f MiB\n", x.bytes, mib(x.drop))
		if x.drop > 1<<20 {
			fmt.Fprintf(os.Stderr, "[H6C]     at %s\n[H6C]     next allocation at %s\n", x.site, x.next)
		}
	}
}

// TestKV3R_allocSequence loads KV3R_MODEL (a path under ~/models) as serve does and writes the build's device allocation sequence,
// with its frees and the card's free VRAM after each, to KV3R_OUT (tab-separated: A|F, id, requested bytes, on-device, plan line,
// free bytes; "S" rows carry a site marker: kv pools done, post-pool done, the fit-check reading). A reading, not a gate.
func TestKV3R_allocSequence(t *testing.T) {
	rel, out := os.Getenv("KV3R_MODEL"), os.Getenv("KV3R_OUT")
	if rel == "" || out == "" {
		t.Skip("set KV3R_MODEL and KV3R_OUT")
	}
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", rel)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	free := func() int64 {
		f, ok := decoder.FreeBytesFor("cuda")
		if !ok {
			t.Errorf("no free-VRAM reading")
		}
		return f
	}
	kvF16Auto.Store(true)
	defer kvF16Auto.Store(false)
	if w, err := decoder.Load(filepath.Join("../../testdata/llama-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"}); err == nil {
		_ = w.Close() // the startup self-test runs here, before the sequence below
	}
	var (
		mu  sync.Mutex
		log []string
		n   int
	)
	mark := func(s string) {
		mu.Lock()
		log = append(log, fmt.Sprintf("S\t%s\t%d", s, free()))
		mu.Unlock()
	}
	leaseSeenForTest = func(onDevice bool, line string, requested int64, release func()) func() {
		mu.Lock()
		id := n
		n++
		log = append(log, fmt.Sprintf("A\t%d\t%d\t%d\t%s\t%d", id, requested, b2i(onDevice), line, free()))
		mu.Unlock()
		return func() {
			release()
			mu.Lock()
			log = append(log, fmt.Sprintf("F\t%d\t%d\t%d\t%s\t%d", id, requested, b2i(onDevice), line, free()))
			mu.Unlock()
		}
	}
	saved := cudaFreeVRAM
	gotFit := false
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, e := saved(r)
		if !gotFit {
			gotFit = true
			mu.Lock()
			log = append(log, fmt.Sprintf("S\tfit\t%d", f))
			mu.Unlock()
		}
		return f, e
	}
	kvPoolsDoneForTest = func() { mark("kvpools") }
	postPoolDoneForTest = func() { mark("postpool") }
	restore := func() {
		leaseSeenForTest, cudaFreeVRAM, kvPoolsDoneForTest, postPoolDoneForTest = nil, saved, nil, nil
	}
	defer restore()
	mark("start")
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	restore()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
	}
	mu.Lock()
	log = append(log, fmt.Sprintf("S\tdone\t%d", free()))
	body := ""
	for _, l := range log {
		body += l + "\n"
	}
	mu.Unlock()
	if err := os.WriteFile(out, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "[KV3R] %s: %d events, ctx planned %d cap %d, written to %s\n", rel, len(log), r.ctxPlanned, r.ctxCap, out)
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
