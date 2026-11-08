//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/loadflags"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
)

// TestQ4KLaneResidentQwen is the q4k lane's by-day wiring smoke (docs/tasks/task-metal-q4k-2026-10.md), not G-Q2: a
// Qwen2.5-Coder q4_k_m at --quant q4k, resident on Metal, against the CPU at q4k on the same file, teacher-forced over
// 32 positions. The two files cover the lane's shapes between them:
//   - 0.5B: ffn_down Q4_K in half the layers and Q6_K (int8) in the rest, as Phi-3's; everything else int8 (Q5_0/Q8_0);
//     q/k/v biases; an int8 LM head. It fits by day.
//   - 1.5B: also Q4_K QKV, gate|up and o-proj, with Q4_K and int8 mixed inside the fused QKV (attn_v is Q6_K in half the
//     layers), so a projection runs as two segments. It loads about 3 GB, so it runs under GOINFER_HEAVY_TESTS=1.
//
// The engines load one at a time. A fit-guard refusal on this machine's free memory is a SKIP naming it, never a pass.
//
// The two paths differ by arithmetic only (f32 activations here against the CPU's per-32 int8 ones), so the bar, set
// before the first run, is a per-position last-row logits cosine of at least 0.99 and no NaN, on a decode path that
// reads metal-resident (q4k). A wiring defect (a wrong segment offset, a missing bias, the wrong kind's kernel) puts a
// position far below it. GOINFER_QWEN05_GGUF / GOINFER_QWEN15_GGUF override the default ~/models paths.
func TestQ4KLaneResidentQwen(t *testing.T) {
	for _, c := range []struct {
		name, env, file string
		heavy           bool
	}{
		{"0.5B", "GOINFER_QWEN05_GGUF", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", false},
		{"1.5B", "GOINFER_QWEN15_GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			if c.heavy {
				requireHeavyModel(t)
			}
			path := os.Getenv(c.env)
			if path == "" {
				home, err := os.UserHomeDir()
				if err != nil {
					t.Skipf("no home directory: %v", err)
				}
				path = filepath.Join(home, "models", c.file)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no model at %s", path)
			}
			testQ4KLaneResident(t, path)
		})
	}
}

func testQ4KLaneResident(t *testing.T, path string) {
	const positions = 32
	tok := func(p int) int { return 100 + 37*p%5000 }

	load := func(backend string) *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Backend: backend, Quant: "q4k", ResidentContext: 256})
		if err != nil && strings.Contains(err.Error(), "was NOT loaded") {
			t.Skipf("SKIP, not a pass: the fit guard refused the %s load on this machine's free memory: %v", backend, err)
		}
		if err != nil {
			t.Fatalf("load %s: %v", backend, err)
		}
		return m
	}
	mm := load("metal")
	dp := mm.DecodePath()
	if !strings.HasPrefix(dp, "metal-resident (q4k") {
		decline := mm.ResidentDecline()
		mm.Close()
		t.Fatalf("decode path %q, want metal-resident (q4k); decline: %s", dp, decline)
	}
	// The decoder's price for the lane's decode scratch is the build's own list, with the chain's buffers, which the build holds once
	// a chain opens (none has yet).
	if a, ok := mm.ResidentForwardForTest().(*metalResident); !ok {
		t.Errorf("the resident is %T, not Metal's: its decode list is not read", mm.ResidentForwardForTest())
	} else {
		chain := int64(0)
		for _, sb := range a.r.chainBuffers(true, true) {
			chain += int64(sb.bytes())
		}
		if got, want := mm.MetalDecodeScratchBytes(metalMoESlotsFromEnv(mm)), a.r.decode.listBytes(a.r)+chain; got != want {
			t.Errorf("q4k lane: the decoder prices decode's scratch at %d bytes, the build's list with the chain's at %d (%+d)", got, want, got-want)
		}
	}
	rf := mm.ResidentForwardForTest()
	rf.Reset()
	var got [][]float32
	for p := range positions {
		lg, err := rf.Forward(mm.EmbedResidentForTest(tok(p)), p)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, append([]float32(nil), lg...))
	}
	mm.Close()

	mc := load("cpu")
	defer mc.Close()
	cache := mc.NewCache(positions)
	worst, agree := 1.0, 0
	for p := range positions {
		want, err := mc.ForwardForTest(tok(p), cache)
		if err != nil {
			t.Fatal(err)
		}
		var dot, na, nb float64
		ga, wa := 0, 0
		for i := range want {
			g, w := float64(got[p][i]), float64(want[i])
			if math.IsNaN(g) || math.IsInf(g, 0) {
				t.Fatalf("position %d logit %d is %v", p, i, g)
			}
			dot, na, nb = dot+g*w, na+g*g, nb+w*w
			if got[p][i] > got[p][ga] {
				ga = i
			}
			if want[i] > want[wa] {
				wa = i
			}
		}
		c := dot / math.Sqrt(na*nb)
		worst = math.Min(worst, c)
		if ga == wa {
			agree++
		}
	}
	fmt.Printf("[q4k lane] %s, Metal vs CPU over %d positions: worst cosine %.6f, argmax agree %d/%d (%s)\n",
		filepath.Base(path), positions, worst, agree, positions, dp)
	if worst < 0.99 {
		t.Errorf("worst per-position cosine %.6f, bar 0.99", worst)
	}
}

// phi3DefaultLoad loads decoder.WritePhi3Q4KTinyGGUFForTest's Phi-3 as a serve with its defaults does on this Mac: -backend auto and no --quant,
// through loadflags.Resolve and modelload.Plan (the per-family default), opening the sidecar Plan names, built here first as the
// load builds it. It returns the model and the copied source.
func phi3DefaultLoad(t *testing.T) (*decoder.Model, string) {
	t.Helper()
	src := filepath.Join(t.TempDir(), "phi3.gguf")
	if err := decoder.WritePhi3Q4KTinyGGUFForTest(src); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(src, old, old); err != nil {
		t.Fatal(err)
	}
	auto := decoder.AutoBackend()
	if auto.Backend != "metal" {
		t.Fatalf("-backend auto picks %q on this machine (%s), not metal", auto.Backend, auto.Reason)
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	f := loadflags.Register(fs, loadflags.Serve)
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	opts := loadflags.Resolve(*f, loadflags.Machine{Auto: auto}, loadflags.Overrides{Serve: &loadflags.ServeLoad{KVSessions: 4}})
	pl, err := modelload.Plan(modelload.Request{Spec: src, Opts: opts})
	if err != nil {
		t.Fatal(err)
	}
	if pl.Build != "" {
		if err := decoder.TranscodeGIW(context.Background(), src, pl.Build, pl.Opts.Quant, pl.Opts.EmbedInt4, decoder.GIWTargetForBackend(pl.Opts.Backend)); err != nil {
			t.Fatal(err)
		}
		pl.LoadPath = pl.Build
	}
	m, err := decoder.Load(pl.LoadPath, pl.Opts)
	if err != nil {
		t.Fatal(err)
	}
	return m, src
}

// TestPhi3Default_autoRunsResidentOnTheQ4KLane: a Phi-3 .gguf served with the defaults (-backend auto, no --quant) on Apple
// silicon loads at q4k and runs resident on Metal's q4k lane, which keeps the file's Q4_K blocks and quantizes no activation.
func TestPhi3Default_autoRunsResidentOnTheQ4KLane(t *testing.T) {
	m, _ := phi3DefaultLoad(t)
	defer m.Close()
	if q, dp := m.Quant(), m.DecodePath(); q != "q4k" || !strings.HasPrefix(dp, "metal-resident (q4k") {
		t.Errorf("the default load is %s, decode path %q; want q4k, metal-resident (q4k)", q, dp)
	}
}

// TestPhi3Default_tracksTheCPUReferenceAtLeastAsCloselyAsTheOldDefault is the default's fidelity gate, tier 0, on
// decoder.WritePhi3Q4KTinyGGUFForTest's Phi-3, teacher-forced over 32 positions. The reference is the CPU at f32 on the same file (every Q4_K
// block dequantized exactly). The default (phi3DefaultLoad: q4k, resident on Metal) is held to three bars:
//   - against the CPU at q4k on the same file, worst per-position logits cosine at least 0.99: the q4k lane's bar
//     (testQ4KLaneResident), which compares the same quant on both sides;
//   - against the f32 reference, worst cosine at least that of the old default (int8int8 with per-32 activation scales, which
//     Metal declines to the CPU) and of the CPU's own q4k;
//   - against the f32 reference, argmax agreement at least the old default's.
//
// No absolute bar is set against f32: the q4k quant's int8 parts move every path that takes it away from the reference, the CPU's
// own q4k included, by an amount the fixture's random weights set, so the bars compare paths. Random blocks carry none of Phi-3's
// activation outliers, so this gates the default's arithmetic, not the outlier hazard; that is G-Q2's, on the real file
// (TestQ4KLane_realPhi3NonInferiority).
func TestPhi3Default_tracksTheCPUReferenceAtLeastAsCloselyAsTheOldDefault(t *testing.T) {
	const positions = 32
	tok := func(p int) int { return (7 + 11*p) % 64 }
	def, src := phi3DefaultLoad(t)
	if dp := def.DecodePath(); !strings.HasPrefix(dp, "metal-resident (q4k") {
		def.Close()
		t.Fatalf("the default load's decode path is %q, want metal-resident (q4k)", dp)
	}
	rf := def.ResidentForwardForTest()
	rf.Reset()
	var got [][]float32
	for p := range positions {
		lg, err := rf.Forward(def.EmbedResidentForTest(tok(p)), p)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, append([]float32(nil), lg...))
	}
	def.Close()

	cpuLogits := func(name string, o decoder.Options, wantPath string) [][]float32 {
		m, err := decoder.Load(src, o)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		defer m.Close()
		if dp := m.DecodePath(); !strings.HasPrefix(dp, wantPath) {
			t.Fatalf("%s: decode path %q, want %s", name, dp, wantPath)
		}
		cache := m.NewCache(positions)
		var out [][]float32
		for p := range positions {
			lg, err := m.ForwardForTest(tok(p), cache)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	ref := cpuLogits("the f32 reference", decoder.Options{Backend: "cpu", Quant: "", DirectLoad: true}, "cpu")
	cpuQ4K := cpuLogits("the CPU at q4k", decoder.Options{Backend: "cpu", Quant: "q4k", DirectLoad: true}, "cpu")
	oldDef := cpuLogits("the old default", decoder.Options{Backend: "metal", BackendAuto: true, Quant: "int8int8", ActQuantGroup: 32, DirectLoad: true}, "cpu")

	score := func(arm, against [][]float32) (worst float64, agree int) {
		worst = 1
		for p := range positions {
			var dot, na, nb float64
			ga, wa := 0, 0
			for i, w := range against[p] {
				g := float64(arm[p][i])
				if math.IsNaN(g) || math.IsInf(g, 0) {
					t.Fatalf("position %d logit %d is %v", p, i, g)
				}
				dot, na, nb = dot+g*float64(w), na+g*g, nb+float64(w)*float64(w)
				if arm[p][i] > arm[p][ga] {
					ga = i
				}
				if w > against[p][wa] {
					wa = i
				}
			}
			worst = math.Min(worst, dot/math.Sqrt(na*nb))
			if ga == wa {
				agree++
			}
		}
		return worst, agree
	}
	lane, _ := score(got, cpuQ4K)
	dw, da := score(got, ref)
	cw, _ := score(cpuQ4K, ref)
	ow, oa := score(oldDef, ref)
	t.Logf("over %d positions: the default (q4k, Metal) against the CPU at q4k, worst cosine %.6f; against the f32 reference: the default %.6f, argmax %d; the CPU at q4k %.6f; the old default (int8int8 per-32, CPU) %.6f, argmax %d",
		positions, lane, dw, da, cw, ow, oa)
	if lane < 0.99 {
		t.Errorf("the default against the CPU at q4k: worst cosine %.6f, bar 0.99", lane)
	}
	if dw < ow || dw < cw || da < oa {
		t.Errorf("against the f32 reference the default reads worst cosine %.6f, argmax %d; bars: the old default's %.6f and %d, the CPU q4k's %.6f",
			dw, da, ow, oa, cw)
	}
}
