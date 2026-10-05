//go:build darwin && goinfer_testhooks

package metal

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestG4LayerMajor_M26AB is 4b's grade (docs/tasks/task-m26-mac-2026-10.md, pre-registered 2026-10-04): on M26, a fresh
// prose prompt's wall time through the sequential loop (ForwardNoLogits, then Forward for the last row, as the decoder
// runs it) and through prefillG4Paged, at each M in GOINFER_G4LM_MS (default 128,512), one warm-up then 5 reps per arm,
// the order alternating rep by rep, in one process. Each rep compares the two arms' last-row logits bit for bit and a
// SHA-256 of their K/V over the prompt. The slot count is the auto-sizer's (MoECacheExperts) unless GOINFER_AUDIT_SLOTS
// sets one. Heartbeat on stderr after every arm.
//
//	GOINFER_G4LM_M26=1 GOINFER_AUDIT_MODEL=<.giw> [GOINFER_AUDIT_SLOTS=N] [GOINFER_G4LM_MS=128,512] [GOINFER_G4LM_REPS=5] ./metal-<rev>.test \
//	  -test.run '^TestG4LayerMajor_M26AB$' -test.v -test.timeout 60m
func TestG4LayerMajor_M26AB(t *testing.T) {
	if os.Getenv("GOINFER_G4LM_M26") != "1" {
		t.Skip("set GOINFER_G4LM_M26=1: loads M26 and times its prefill (night, or by day on the owner's word)")
	}
	path := os.Getenv("GOINFER_AUDIT_MODEL")
	if path == "" || strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("GOINFER_AUDIT_MODEL %q: a local-disk .giw is required (CLAUDE.md)", path)
	}
	slots, _ := strconv.Atoi(os.Getenv("GOINFER_AUDIT_SLOTS"))
	Ms := []int{128, 512}
	if v := os.Getenv("GOINFER_G4LM_MS"); v != "" {
		Ms = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				Ms = append(Ms, n)
			}
		}
	}
	reps := 5 // the grade's; GOINFER_G4LM_REPS lowers it for a profile read
	if n, err := strconv.Atoi(os.Getenv("GOINFER_G4LM_REPS")); err == nil && n > 0 {
		reps = n
	}
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[g4lm %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	tk, err := modelload.Tokenizer(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	maxM := 0
	for _, M := range Ms {
		maxM = max(maxM, M)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: maxM + 64,
		MoECacheExperts: true, MoECacheSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || !((a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged)) {
		t.Fatalf("no paged MoE resident on Metal (decode path %q)", m.DecodePath())
	}
	r := a.r
	layerMajor := r.prefillG4Paged // the generic pager's twin when the model is not Gemma 4 (prefillMoEPaged)
	if r.g4moe == nil {
		layerMajor = r.prefillMoEPaged
	}
	pool := func(l int) *expertPool {
		if gl := r.layers[l].g4moe; gl != nil {
			return gl.pool
		}
		if ml := r.layers[l].moe; ml != nil {
			return ml.pool
		}
		return nil
	}
	nSlots := 0
	for l := range r.layers {
		if p := pool(l); p != nil {
			nSlots = len(p.slotExpert)
			break
		}
	}
	hb("built: %s, %d slots a layer, pid %d", m.DecodePath(), nSlots, os.Getpid())
	files := decoder.PrefillGatePromptSetFor("a")
	kvHash := func(n int) [32]byte {
		h := sha256.New()
		for l := range r.layers {
			if r.layers[l].delta != nil || r.kc[l] == (Buffer{}) {
				continue
			}
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			for _, b := range []Buffer{r.kc[l], r.vc[l]} {
				u := b.U16s()[o : o+n*d]
				h.Write(unsafe.Slice((*byte)(unsafe.Pointer(&u[0])), 2*len(u)))
			}
		}
		var s [32]byte
		copy(s[:], h.Sum(nil))
		return s
	}
	stagesAndSubmits := func() int {
		n := 0
		for l := range r.layers {
			if p := pool(l); p != nil {
				n += p.stages + p.prefetched
			}
		}
		return n
	}
	for _, M := range Ms {
		ids := decoder.PrefillGateProseIDsForTest(t, tk, files[0], M)[:M]
		embs := make([][]float32, M)
		for i, id := range ids {
			embs[i] = m.EmbedResidentForTest(id)
		}
		// The arm's time split: GPU-busy over every command buffer, the host's staging wall, and the
		// rest of the wall (encode, submit and wait gaps, the route reads), from r.prof, a prompt token's share in ms.
		var prof0 pagedProfile
		profLine := func(arm string, wallMs float64) string {
			p := r.PagedProfile()
			gpu := float64(p.p1GpuNanos-prof0.p1GpuNanos+p.p2GpuNanos-prof0.p2GpuNanos+p.denseGpuNanos-prof0.denseGpuNanos) / 1e6
			p2 := float64(p.p2GpuNanos-prof0.p2GpuNanos) / 1e6
			stage := float64(p.stageWallNanos-prof0.stageWallNanos) / 1e6
			f := float64(M)
			return fmt.Sprintf("PROF M=%d %s, ms a prompt token: wall %.2f = GPU-busy %.2f (of it phase 2, the experts, %.2f) + staging %.2f + other %.2f",
				M, arm, wallMs/f, gpu/f, p2/f, stage/f, (wallMs-gpu-stage)/f)
		}
		seq := func() ([]float32, [32]byte, float64, int) {
			a.Reset()
			s0 := stagesAndSubmits()
			prof0 = r.PagedProfile()
			st := time.Now()
			for i := 0; i < M-1; i++ {
				if err := a.ForwardNoLogits(embs[i], i); err != nil {
					t.Fatalf("sequential %d: %v", i, err)
				}
			}
			lg, err := a.Forward(embs[M-1], M-1)
			if err != nil {
				t.Fatalf("sequential last: %v", err)
			}
			ms := float64(time.Since(st).Microseconds()) / 1e3
			return append([]float32(nil), lg...), kvHash(M), ms, stagesAndSubmits() - s0
		}
		lm := func() ([]float32, [32]byte, float64, int) {
			a.Reset()
			s0 := stagesAndSubmits()
			prof0 = r.PagedProfile()
			st := time.Now()
			lg := layerMajor(embs, 0, true)
			if err := r.takeExecErr(); err != nil {
				t.Fatalf("layer-major: %v", err)
			}
			ms := float64(time.Since(st).Microseconds()) / 1e3
			return append([]float32(nil), lg...), kvHash(M), ms, stagesAndSubmits() - s0
		}
		var ratios, seqMs, lmMs []float64
		var seqSt, lmSt []int
		for rep := range reps + 1 {
			order := []bool{false, true}
			if rep%2 == 1 {
				order = []bool{true, false}
			}
			var lgs [2][]float32
			var hs [2][32]byte
			var tm [2]float64
			for _, isLM := range order {
				k := 0
				f := seq
				if isLM {
					k, f = 1, lm
				}
				var st int
				lgs[k], hs[k], tm[k], st = f()
				if rep > 0 {
					if isLM {
						lmMs, lmSt = append(lmMs, tm[k]), append(lmSt, st)
					} else {
						seqMs, seqSt = append(seqMs, tm[k]), append(seqSt, st)
					}
				}
				hb("M=%d rep %d %s: %.0f ms, %d experts staged", M, rep, map[bool]string{false: "sequential", true: "layer-major"}[isLM], tm[k], st)
				hb("%s", profLine(map[bool]string{false: "sequential", true: "layer-major"}[isLM], tm[k]))
			}
			for j := range lgs[0] {
				if math.Float32bits(lgs[0][j]) != math.Float32bits(lgs[1][j]) {
					t.Fatalf("M=%d rep %d: last-row logit %d differs (layer-major %v, sequential %v)", M, rep, j, lgs[1][j], lgs[0][j])
				}
			}
			if hs[0] != hs[1] {
				t.Fatalf("M=%d rep %d: K/V differ", M, rep)
			}
			if rep > 0 {
				ratios = append(ratios, tm[0]/tm[1])
			}
		}
		med := func(v []float64) float64 {
			s := append([]float64(nil), v...)
			sort.Float64s(s)
			return s[len(s)/2]
		}
		above := 0
		for _, q := range ratios {
			if q > 1 {
				above++
			}
		}
		medInt := func(v []int) int {
			s := append([]int(nil), v...)
			sort.Ints(s)
			return s[len(s)/2]
		}
		hb("RESULT M=%d, %d slots: sequential %.0f ms, layer-major %.0f ms (medians of %d); sequential / layer-major median %.3f, %d of %d reps above 1 (per rep %v); experts staged a prompt: sequential %d, layer-major %d; logits and K/V equal in every rep",
			M, nSlots, med(seqMs), med(lmMs), reps, med(ratios), above, reps, auditFmt3(ratios), medInt(seqSt), medInt(lmSt))
	}
}
