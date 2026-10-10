//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestS9EModelBatched_tiny is G-S9c of S9 step 2 (docs/tasks/task-multimodal-support-2026-10.md, registered before any
// code): on gemma4-emodel-tiny (6 layers; head dims 32 and 64; 2 KV-shared layers with their own sources and double FFN
// width; PLE P=32; layer scalars; v_norm), the f16 batched pass against the S9 layer-major pass (the validated path,
// bit-identical to sequential) on one Metal resident. Compared: the last prompt row and 8 decode steps teacher-forced
// along the layer-major pass's own greedy tokens; the bar is cosine >= 0.98 with every argmax equal or an R10 near-tie
// (the owner amended it from the registered 0.9999, which the pass missed with no non-tie
// argmax difference: the task doc's G-S9c record has the mechanism and the readings; 0.98 sits between the pass's own and the nearest planted defect's).
// Two prompts: the golden 12-token prompt and a synthetic 96-token one, which runs every local layer past its window of 4
// and both shared layers over long sources. The batched pass is called directly, so a silent fallback cannot stand in for
// it. Each planted defect (emodelBatchDefect) must go red on at least one prompt.
func TestS9EModelBatched_tiny(t *testing.T) {
	const steps, bar = 8, 0.98
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_emodel_tiny.py", eModelDir)
	}
	raw, err := os.ReadFile(eModelGolden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_gemma4_emodel_tiny.py", err)
	}
	var g struct {
		PromptIDs []int `json:"prompt_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(eModelDir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatalf("load: %v (the fixture is per-machine: scripts/pin_gemma4_emodel_tiny.py)", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	emodelBatchedOn = true
	defer func() { emodelBatchedOn, emodelBatchDefect = false, 0 }()
	if !a.emodelBatched() {
		t.Fatal("the batched pass does not claim this E-model")
	}
	long := make([]int, 96)
	for i := range long {
		long[i] = 3 + (i*37)%200
	}
	defects := []string{"none", "(1) a shared layer over an empty cache", "(2) the PLE block skipped", "(3) PLE inputs from the next row",
		"(4) the layer scalar dropped", "(5) v_norm dropped", "(6) every FFN at the first layer's width"}
	caught := make([]bool, len(defects))
	for _, p := range []struct {
		name string
		ids  []int
	}{{"golden 12-token", g.PromptIDs}, {"synthetic 96-token", long}} {
		n := len(p.ids)
		rows := make([][]float32, n)
		for i, id := range p.ids {
			rows[i] = m.EmbedResidentForTest(id)
		}
		decode := func(first []float32, forced []int) ([][]float32, []int) {
			out, toks := [][]float32{append([]float32(nil), first...)}, []int{argmaxF(first)}
			for k := range steps {
				tok := toks[k]
				if forced != nil {
					tok = forced[k]
				}
				lg, err := a.Forward(m.EmbedResidentForTest(tok), n+k)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), lg...))
				toks = append(toks, argmaxF(lg))
			}
			return out, toks
		}
		a.Reset()
		refLast := a.r.prefillEModel(rows, 0, true)
		if err := a.r.takeExecErr(); err != nil {
			t.Fatal(err)
		}
		ref, teacher := decode(refLast, nil)
		for di, name := range defects {
			emodelBatchDefect = di
			a.Reset()
			newLast, err := a.batchedPrefill(rows, 0, 0, nil, nil)
			emodelBatchDefect = 0
			if err != nil {
				t.Fatalf("%s %s: the batched pass declined: %v", p.name, name, err)
			}
			got, _ := decode(newLast, teacher)
			worst, real := 1.0, 0
			for k := range ref {
				worst = math.Min(worst, cosF(ref[k], got[k]))
				if ra, ga := argmaxF(ref[k]), argmaxF(got[k]); ra != ga {
					if lp := logSoftmaxF(ref[k]); math.Exp(lp[ga]) < math.Exp(lp[ra])/2 {
						real++
					}
				}
			}
			pass := worst >= bar && real == 0
			fmt.Printf("[S9 G-S9c] %s %s: worst cosine %.7f over the last row and %d steps, %d non-tie argmax differences\n", p.name, name, worst, steps, real)
			if di == 0 && !pass {
				t.Errorf("%s: the batched pass misses the bar (worst %.7f, %d non-tie differences)", p.name, worst, real)
			}
			if di > 0 && !pass {
				caught[di] = true
			}
		}
	}
	for di := 1; di < len(defects); di++ {
		if !caught[di] {
			t.Errorf("planted defect %s left the bar green on both prompts", defects[di])
		}
	}
}

// TestS9EModelBatched_realE2B is G-S9d of S9 step 2 (docs/tasks/task-multimodal-support-2026-10.md, registered before any
// code): S1's G3 shape on the batched prefill, real Gemma 4 E2B (its int4 Metal sidecar). For each of G3's eight prompts,
// the prompt is prefilled by the pass under test and then decoded on Metal: the free run's 32 greedy tokens against the
// CPU's by G3's rule (identical, or the first divergence at a CPU logit gap under 3%), and the CPU's sequence
// teacher-forced through Metal decode after that prefill, argmax agreement over the last prompt row and the generated
// positions. The reference is the same procedure with the S9 layer-major pass (the shipped path), in this process,
// against the same CPU reference. PASS: agreement >= the reference's - 2.0 points and free-run passes >= the reference's
// - 1; 2.0-4.0 points below is parked; worse fails. One load serves the CPU reference and both Metal arms.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -timeout 30m -run '^TestS9EModelBatched_realE2B$' -v ./metal/
func TestS9EModelBatched_realE2B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it")
	if _, err := os.Stat(base + ".int4.metal.giw"); err != nil {
		t.Skipf("no sidecar: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(base + ".gguf")
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(base+".int4.metal.giw", decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: 512})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	defer func() { emodelBatchedOn = false }()
	emodelBatchedOn = true
	if !a.emodelBatched() {
		t.Fatal("the batched pass does not claim E2B")
	}
	gap := func(cpu []float32, other int) float64 {
		ca := argmaxF(cpu)
		return (float64(cpu[ca]) - float64(cpu[other])) / (math.Abs(float64(cpu[ca])) + 1e-30)
	}
	type tally struct{ pass, agree, n int }
	var arms [2]tally // 0: layer-major (the reference), 1: batched
	for pi, text := range g3Prompts {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		np := len(ids)
		n := np + g3NewTokens
		// The CPU reference: its free run, and its logits at every position from the last prompt row on.
		cc := m.NewCache(n)
		cpuSeq := append([]int(nil), ids...)
		cpuL := map[int][]float32{}
		for i := 0; i < n-1; i++ {
			l, err := m.ForwardForTest(cpuSeq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			if i >= np-1 {
				cpuL[i] = append([]float32(nil), l...)
				cpuSeq = append(cpuSeq, argmaxF(l))
			}
		}
		rows := make([][]float32, np)
		for i, id := range ids {
			rows[i] = m.EmbedResidentForTest(id)
		}
		for arm := range 2 {
			prefill := func() []float32 {
				a.Reset()
				if arm == 0 {
					lg := append([]float32(nil), a.r.prefillEModel(rows, 0, true)...)
					if err := a.r.takeExecErr(); err != nil {
						t.Fatal(err)
					}
					return lg
				}
				lg, err := a.batchedPrefill(rows, 0, 0, nil, nil)
				if err != nil {
					t.Fatalf("batched pass declined: %v", err)
				}
				return lg
			}
			// Free run.
			l := prefill()
			gpuSeq := append(append([]int(nil), ids...), argmaxF(l))
			for i := np; i < n-1; i++ {
				lg, err := a.Forward(m.EmbedResidentForTest(gpuSeq[i]), i)
				if err != nil {
					t.Fatal(err)
				}
				gpuSeq = append(gpuSeq, argmaxF(lg))
			}
			first, firstGap := -1, 0.0
			for i := np; i < n; i++ {
				if gpuSeq[i] != cpuSeq[i] {
					first, firstGap = i-np, gap(cpuL[i-1], gpuSeq[i])
					break
				}
			}
			if first < 0 || firstGap < 0.03 {
				arms[arm].pass++
			}
			// Teacher-forced: the CPU's sequence after this prefill.
			l = prefill()
			if argmaxF(l) == argmaxF(cpuL[np-1]) {
				arms[arm].agree++
			}
			arms[arm].n++
			for i := np; i < n-1; i++ {
				lg, err := a.Forward(m.EmbedResidentForTest(cpuSeq[i]), i)
				if err != nil {
					t.Fatal(err)
				}
				if argmaxF(lg) == argmaxF(cpuL[i]) {
					arms[arm].agree++
				}
				arms[arm].n++
			}
			fmt.Fprintf(os.Stderr, "[S9 G-S9d] prompt %d (%d tokens) %s: first divergence %d (CPU gap %.2f%%)\n", pi+1, np,
				[]string{"layer-major", "batched"}[arm], first, firstGap*100)
		}
	}
	pct := func(x tally) float64 { return 100 * float64(x.agree) / float64(x.n) }
	delta := pct(arms[1]) - pct(arms[0])
	verdict := "PASS"
	switch {
	case delta < -4.0 || arms[1].pass <= arms[0].pass-2:
		verdict = "FAIL"
	case delta < -2.0:
		verdict = "PARKED"
	}
	fmt.Fprintf(os.Stderr, "[S9 G-S9d] batched %.2f%% (%d/%d, %d/%d free-run passes) against layer-major %.2f%% (%d/%d, %d/%d): delta %+.2f points: %s\n",
		pct(arms[1]), arms[1].agree, arms[1].n, arms[1].pass, len(g3Prompts), pct(arms[0]), arms[0].agree, arms[0].n, arms[0].pass, len(g3Prompts), delta, verdict)
	if verdict != "PASS" {
		t.Errorf("G-S9d %s: delta %+.2f points, free-run %d against %d", verdict, delta, arms[1].pass, arms[0].pass)
	}
}
