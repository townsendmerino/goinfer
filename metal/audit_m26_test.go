//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAuditM26_pagedProbe serves T1.8 (C-P01) and T1.9 (D-P02, M-11) of docs/tasks/task-metal-audit-2026-10.md on
// the Gemma 4 26B (M26), one configuration per process, pre-registered in that doc before the night run. The run
// script sets the model and slot count, arms scripts/swap_killwatch.sh on this process's PID, and, for T1.8, takes
// vmmap and footprint from outside while this process holds at token 32. The load goes through the production Metal
// path (Backend "metal", every guard on), and nothing here forks: a fork while pages are GPU-wired is what collapsed
// M26 on 2026-09-24.
//
//	GOINFER_METAL_AUDIT_M26=1 GOINFER_AUDIT_MODEL=<.giw> GOINFER_AUDIT_SLOTS=<N> [GOINFER_AUDIT_HOLD=<file>]
//	[GOINFER_AUDIT_TOKENS=40] [GOINFER_MOE_PROF_SPLIT=1] ./metal-<rev>.test -test.run '^TestAuditM26_pagedProbe$' -test.v
func TestAuditM26_pagedProbe(t *testing.T) {
	if os.Getenv("GOINFER_METAL_AUDIT_M26") != "1" {
		t.Skip("set GOINFER_METAL_AUDIT_M26=1: loads the 26B MoE (owner-gated, night-only)")
	}
	path := os.Getenv("GOINFER_AUDIT_MODEL")
	slots, _ := strconv.Atoi(os.Getenv("GOINFER_AUDIT_SLOTS"))
	if path == "" || slots <= 0 {
		t.Fatal("GOINFER_AUDIT_MODEL and GOINFER_AUDIT_SLOTS are required")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	tokens := 40
	if n, err := strconv.Atoi(os.Getenv("GOINFER_AUDIT_TOKENS")); err == nil && n > 32 {
		tokens = n
	}
	hold := os.Getenv("GOINFER_AUDIT_HOLD")
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[m26 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	hb("pid %d, model %s, %d slots per layer, %d tokens, hold %q", os.Getpid(), path, slots, tokens, hold)

	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: 512,
		MoECacheExperts: true, MoECacheSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || a.r.g4moe == nil || !a.r.g4moe.paged {
		t.Fatalf("no paged Gemma 4 MoE resident on Metal (decode path %q)", m.DecodePath())
	}
	r := a.r
	// C-P01 (2026-10-02): the pager stages scales from the mapping, so there is no scale cache to size. The old
	// binary's figure (experts × per-expert scale words × 2 bytes) was 1361.2 MB; the heap line at token 32 is the
	// measurement now.
	hb("built in %.1f s: %d paged MoE layers", time.Since(t0).Seconds(), len(moeLayerIdx(r)))

	id := 1000 % r.V // any valid token; greedy argmax from here on
	var timed time.Duration
	var prof0 pagedProfile
	for i := range tokens + 1 {
		st := time.Now()
		lg, err := a.Forward(m.EmbedResidentForTest(id), i)
		if err != nil {
			t.Fatalf("token %d: %v", i, err)
		}
		if i == 0 {
			prof0 = r.PagedProfile() // token 0 warms paths and pages; the window is tokens 1..
		} else {
			timed += time.Since(st)
		}
		id = argmaxF32(lg)
		if (i+1)%8 == 0 {
			hb("token %d/%d, %.2f tok/s so far", i+1, tokens, float64(i)/timed.Seconds())
		}
		if i == 32 && hold != "" {
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			hb("HOLD at token 32: Go heap in use %.1f MB (heap sys %.1f, sys %.1f)",
				float64(ms.HeapInuse)/(1<<20), float64(ms.HeapSys)/(1<<20), float64(ms.Sys)/(1<<20))
			if err := os.WriteFile(hold, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
				t.Fatalf("hold file: %v", err)
			}
			for w := 0; w < 600; w++ { // the run script removes the file once vmmap and footprint are taken
				if _, err := os.Stat(hold); os.IsNotExist(err) {
					break
				}
				time.Sleep(time.Second)
			}
			hb("released")
		}
	}
	n := tokens
	pf := r.PagedProfile()
	msTok := func(a, b int64) float64 { return float64(a-b) / 1e6 / float64(n) }
	gpu := msTok(pf.p1GpuNanos, prof0.p1GpuNanos) + msTok(pf.p2GpuNanos, prof0.p2GpuNanos) + msTok(pf.denseGpuNanos, prof0.denseGpuNanos)
	wall := msTok(pf.p1WallNanos, prof0.p1WallNanos) + msTok(pf.p2WallNanos, prof0.p2WallNanos) + msTok(pf.denseWallNanos, prof0.denseWallNanos)
	sub := msTok(pf.p1SubNanos, prof0.p1SubNanos) + msTok(pf.p2SubNanos, prof0.p2SubNanos)
	enc := msTok(pf.p1EncNanos, prof0.p1EncNanos) + msTok(pf.p2EncNanos, prof0.p2EncNanos)
	stage := msTok(pf.stageWallNanos, prof0.stageWallNanos)
	tokMs := timed.Seconds() * 1e3 / float64(n)
	hb("DECOMP/tok: token %.1f ms | GPU-busy %.1f | phase wall %.1f | encode %.1f | submit+wait %.1f | stage %.1f | idx-coord %.1f",
		tokMs, gpu, wall, enc, sub, stage, msTok(pf.idxCoordNanos, prof0.idxCoordNanos))
	line := fmt.Sprintf("RESULT %d slots: %.2f tok/s (%.1f ms per token over %d tokens); per-CB round trip (submit+wait - GPU-busy) %.1f ms per token = %.3f of the token",
		slots, 1e3/tokMs, tokMs, n, sub-gpu, (sub-gpu)/tokMs)
	if os.Getenv("GOINFER_MOE_PROF_SPLIT") == "1" {
		commit := msTok(pf.p1CommitNanos, prof0.p1CommitNanos) + msTok(pf.p2CommitNanos, prof0.p2CommitNanos)
		wait := msTok(pf.p1WaitNanos, prof0.p1WaitNanos) + msTok(pf.p2WaitNanos, prof0.p2WaitNanos)
		line += fmt.Sprintf("; split: commit %.1f ms, waitUntilCompleted %.1f ms, GPU idle in wait %.1f ms per token", commit, wait, wait-gpu)
	}
	hb("%s", line)
}
