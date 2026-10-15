//go:build darwin && goinfer_testhooks

package metal

import (
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G2-R in two processes (docs/tasks/task-forward-architecture-2026-10.md §4.6, "G2-R amendment, 2026-10-15"):
// TestG2R_cpuReference loads only CPU models and saves everything the arms read on the CPU to $GOINFER_G2R_REF;
// TestG2R_registered then loads only Metal models and compares against that file. Each process's loads meet the fit
// guard and Metal's memory guard alone. $GOINFER_G2R_COMMIT names the build both processes run; a file made at another
// commit, or from other checkpoints, voids the run.

// g2rArm is one arm's CPU side: the token fed at each step (the seed, then the CPU's greedy continuation) and the CPU's
// logits at that step.
type g2rArm struct {
	Inputs []int
	Logits [][]float32
}

// g2rPStats is what arm P records on the CPU, on the tokens P feeds: the scores the cap saw, those past |15|, the largest,
// and the worst per-position cosine between the capped and the uncapped logits.
type g2rPStats struct {
	N, Over int64
	MaxAbs  float64
	Worst   float64
}

type g2rRef struct {
	Commit      string
	SHA256      map[string]string // the .gguf checkpoints, which the Metal process loads too
	GIW         map[string]string // the CPU .giw bundles the CPU process loaded, by path
	HookChanged bool
	Arms        map[string]g2rArm // "B/<quant>", "G/<quant>", "control", "gemma3"
	PStats      map[string]g2rPStats
}

func g2rPaths() (g2, g3, dense string) {
	return os.ExpandEnv("$HOME/models/gemma-2-2b-it-Q8_0.gguf"),
		os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf"),
		os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
}

func g2rHashes(t *testing.T, paths ...string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			t.Fatalf("VOID: a checkpoint the session needs is missing: %v", err)
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		f.Close()
		if err != nil {
			t.Fatalf("VOID: reading %s: %v", p, err)
		}
		out[p] = hex.EncodeToString(h.Sum(nil))
	}
	return out
}

// g2rCPUPath is the CPU .giw the CPU process loads for a checkpoint at a quant (amendment 2): $GOINFER_G2R_GIW_DIR holds
// <checkpoint base>.<quant>.cpu.giw, built from that .gguf by the night job. A missing bundle voids the run.
func g2rCPUPath(t *testing.T, gguf, quant string) string {
	t.Helper()
	dir := os.Getenv("GOINFER_G2R_GIW_DIR")
	if dir == "" {
		t.Fatalf("VOID: GOINFER_G2R_GIW_DIR is not set: the CPU arms load .giw bundles (amendment 2)")
	}
	base := filepath.Base(gguf)
	p := filepath.Join(dir, strings.TrimSuffix(base, ".gguf")+"."+quant+".cpu.giw")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("VOID: the CPU bundle for %s at %s is missing: %v", base, quant, err)
	}
	return p
}

func g2rRefEnv(t *testing.T) (path, commit string) {
	t.Helper()
	path, commit = os.Getenv("GOINFER_G2R_REF"), os.Getenv("GOINFER_G2R_COMMIT")
	if path == "" || commit == "" {
		t.Skip("set GOINFER_G2R_REF and GOINFER_G2R_COMMIT: G2-R runs as the night job's two processes (run-g2r-split-night.sh)")
	}
	return path, commit
}

// g2rCPUArm runs the CPU model on the seed and its greedy continuation, recording the input and the logits at each step.
// wantCap checks the model carries the attention softcap the arm expects (Gemma 2 only).
func g2rCPUArm(t *testing.T, path, quant string, seed []int, steps int, softcapOff, wantCap bool) g2rArm {
	t.Helper()
	return g2rCPUArmFrom(t, g2rCPUPath(t, path, quant), quant, seed, steps, softcapOff, wantCap)
}

// g2rCPUArmFrom is g2rCPUArm loading exactly load (a .giw, or the .gguf itself for the identity precondition).
func g2rCPUArmFrom(t *testing.T, load, quant string, seed []int, steps int, softcapOff, wantCap bool) g2rArm {
	t.Helper()
	was := decoder.SetAttnSoftcapOffForTest(softcapOff)
	m, err := decoder.Load(load, decoder.Options{Quant: quant})
	decoder.SetAttnSoftcapOffForTest(was)
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer m.Close()
	if got := m.AttnLogitSoftcapResident(); wantCap && (got == 0) != softcapOff {
		t.Fatalf("VOID: the CPU model's cap is %v with the softcap removed = %v", got, softcapOff)
	}
	_, nL, _, nKV, hd, _, _ := m.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
	arm := g2rArm{}
	tok := seed[0]
	for i := range steps {
		l, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		arm.Inputs = append(arm.Inputs, tok)
		arm.Logits = append(arm.Logits, append([]float32(nil), l...))
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = argmaxF(l)
		}
	}
	return arm
}

// TestG2R_cpuReference is the CPU process: the hook check, then every arm's CPU side, saved to $GOINFER_G2R_REF.
func TestG2R_cpuReference(t *testing.T) {
	requireHeavyModel(t)
	out, commit := g2rRefEnv(t)
	g2, g3, dense := g2rPaths()
	ref := g2rRef{Commit: commit, SHA256: g2rHashes(t, g2, g3, dense), Arms: map[string]g2rArm{}, PStats: map[string]g2rPStats{}}
	ref.GIW = g2rHashes(t, g2rCPUPath(t, g2, "int8int8"), g2rCPUPath(t, g2, "int4"), g2rCPUPath(t, g3, "int8int8"),
		g2rCPUPath(t, dense, "int8int8"), g2rCPUPath(t, dense, "int4"))

	// The precondition (amendment 2): the .giw route is the direct .gguf route, bit for bit, on the dense 1.5B (whose
	// .gguf the fit guard admits directly), at both quants, on the gate's prompt and 24 steps. A difference voids the run.
	dseed := seedPrompt(t, dense, probeText)
	for _, q := range []string{"int8int8", "int4"} {
		direct := g2rCPUArmFrom(t, dense, q, dseed, 24, false, false)
		viaGIW := g2rCPUArmFrom(t, g2rCPUPath(t, dense, q), q, dseed, 24, false, false)
		for i := range direct.Logits {
			for j := range direct.Logits[i] {
				if math.Float32bits(direct.Logits[i][j]) != math.Float32bits(viaGIW.Logits[i][j]) {
					t.Fatalf("VOID (amendment 2's precondition): the dense 1.5B at %s through its .giw differs from its direct .gguf load at step %d, logit %d", q, i, j)
				}
			}
		}
		t.Logf("precondition: the dense 1.5B at %s gives the same logits through its .giw and its .gguf, every step, bit for bit", q)
	}
	seed := seedPrompt(t, g2, probeText)

	tokens := g2rTokens(t, g2, "int8int8", seed, 24, false)
	with := g2rCPULogits(t, g2, "int8int8", tokens, true)
	without := g2rCPULogits(t, g2, "int8int8", tokens, false)
	for i := range with {
		for j := range with[i] {
			ref.HookChanged = ref.HookChanged || with[i][j] != without[i][j]
		}
	}
	for _, q := range []string{"int8int8", "int4"} {
		ref.Arms["B/"+q] = g2rCPUArm(t, g2, q, seed, 24, true, true)
		g := g2rCPUArm(t, g2, q, seed, 24, false, true)
		ref.Arms["G/"+q] = g
		// P feeds G's tokens: the seed, then the capped CPU's greedy continuation.
		stopStats := decoder.StartAttnScoreStatsForTest()
		capped := g2rCPULogits(t, g2, q, g.Inputs, false)
		n, over, maxAbs := stopStats()
		uncapped := g2rCPULogits(t, g2, q, g.Inputs, true)
		worst := 1.0
		for i := 2; i < len(capped); i++ {
			worst = min(worst, cosF(capped[i], uncapped[i]))
		}
		ref.PStats[q] = g2rPStats{N: n, Over: over, MaxAbs: maxAbs, Worst: worst}
	}
	ref.Arms["control"] = g2rCPUArm(t, dense, "int8int8", seedPrompt(t, dense, probeText), 24, false, false)
	ref.Arms["gemma3"] = g2rCPUArm(t, g3, "int8int8", seedPrompt(t, g3, probeText), 24, false, false)

	tmp := out + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		t.Fatalf("create %s: %v", tmp, err)
	}
	if err := gob.NewEncoder(f).Encode(&ref); err != nil {
		f.Close()
		t.Fatalf("encode: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := os.Rename(tmp, out); err != nil {
		t.Fatalf("rename: %v", err)
	}
	t.Logf("G2-R CPU reference: %d arms, hook changed %v, at %s, written to %s", len(ref.Arms), ref.HookChanged, commit, out)
}

// g2rLoadRef reads the CPU process's file and applies the amendment's voids: the file exists, was made at this build
// from these checkpoints, and holds every arm.
func g2rLoadRef(t *testing.T) g2rRef {
	t.Helper()
	path, commit := g2rRefEnv(t)
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("VOID: the CPU reference is missing: %v", err)
	}
	defer f.Close()
	var ref g2rRef
	if err := gob.NewDecoder(f).Decode(&ref); err != nil {
		t.Fatalf("VOID: the CPU reference does not decode: %v", err)
	}
	if ref.Commit != commit {
		t.Fatalf("VOID: the CPU reference was made at %s, this process runs %s", ref.Commit, commit)
	}
	g2, g3, dense := g2rPaths()
	for p, h := range g2rHashes(t, g2, g3, dense) {
		if ref.SHA256[p] != h {
			t.Fatalf("VOID: %s is not the checkpoint the CPU reference was made from", p)
		}
	}
	for _, k := range []string{"B/int8int8", "B/int4", "G/int8int8", "G/int4", "control", "gemma3"} {
		if a, ok := ref.Arms[k]; !ok || len(a.Inputs) == 0 || len(a.Inputs) != len(a.Logits) {
			t.Fatalf("VOID: the CPU reference has no arm %s", k)
		}
	}
	for _, q := range []string{"int8int8", "int4"} {
		if _, ok := ref.PStats[q]; !ok {
			t.Fatalf("VOID: the CPU reference has no P statistics at %s", q)
		}
	}
	return ref
}

// g2rMetalAgainst is residentParityAt's measurement with the CPU side read from the reference: the Metal resident is fed
// the arm's inputs in lockstep and its logits are compared with the saved CPU logits, step for step.
func g2rMetalAgainst(t *testing.T, path, quant string, arm g2rArm) parityStats {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	mg, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		skipIfMemoryDeclined(t, mg)
		t.Fatalf("metal resident DECLINED (%s) — admission says it should be admitted", mg.ResidentDecline())
	}
	st := parityStats{steps: len(arm.Inputs), minCos: 1}
	for i, tok := range arm.Inputs {
		cpuL := arm.Logits[i]
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("gpu forward: %v", err)
		}
		if i >= 2 { // residentParityAt's metric: the <bos> sink positions are skipped
			st.observeCos(cosF(cpuL, gpuL))
		}
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			st.exact++
			continue
		}
		lo, hi := cpuL[0], cpuL[0]
		for _, v := range cpuL {
			lo, hi = min(lo, v), max(hi, v)
		}
		gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
		st.worstTie = max(st.worstTie, gap)
		if gap > 0.03 {
			st.hard++
		}
	}
	t.Logf("%s at %s (%s): %d/%d argmax-exact, worst near-tie %.3f%%, %d gaps >3%%, min cosine %.6f",
		path, quant, mg.DecodePath(), st.exact, st.steps, st.worstTie*100, st.hard, st.minCos)
	return st
}
