//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDG01_expertMajorMoEPrefill is D-G01's gate (docs/tasks/task-metal-audit-2026-10.md, "D-G01: pre-registration"):
// expert-major MoE prefill against the f16 lane's own MoE baseline, on real weights. Three arms per prompt, each writing
// positions 0..M-1 of the one KV slot:
//   - seq: the sequential decode loop, its routing captured per token and layer (resident.moeCap, slot l);
//   - row: the batched pass, MoE row by row through decode's MoE kernels (GOINFER_MOE_EXPERT_MAJOR=0), routing per
//     (layer, row);
//   - major: the batched pass, expert-major, routing from the host copy the path makes anyway.
//
// It reads, per M and pooled over the prompts: flips (token, layer pairs whose expert set differs from seq's), each
// layer's K/V relative L2 against seq's, and the last position's KL(seq ‖ arm). The gate and its bars are the
// pre-registration's. GOINFER_DG01_MUTATION plants one of dg01Mutation's defects, which must fail it.
//
//	GOINFER_DG01=1 go test -tags goinfer_testhooks -count=1 -run '^TestDG01_expertMajorMoEPrefill$' -v ./metal/
func TestDG01_expertMajorMoEPrefill(t *testing.T) {
	if os.Getenv("GOINFER_DG01") != "1" {
		t.Skip("set GOINFER_DG01=1 (loads a real MoE checkpoint: D-G01's gate, a night run)")
	}
	path := os.Getenv("GOINFER_DG01_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen15-moe-a27b-l4slice")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	Ms := []int{64, 512}
	if v := os.Getenv("GOINFER_DG01_MS"); v != "" {
		Ms = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				Ms = append(Ms, n)
			}
		}
	}
	nPrompts := 10
	if n, err := strconv.Atoi(os.Getenv("GOINFER_DG01_PROMPTS")); err == nil && n > 0 {
		nPrompts = n
	}
	dg01Mutation = os.Getenv("GOINFER_DG01_MUTATION")
	t.Cleanup(func() { dg01Mutation = "" })
	maxM := slices.Max(Ms)

	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: maxM + 64})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	if r.moe == nil || !r.prefillOK {
		t.Fatalf("%s: moe %v, prefillOK %v: the gate would test nothing", path, r.moe != nil, r.prefillOK)
	}
	for l := range r.layers {
		if r.layers[l].moe != nil && r.layers[l].moe.pool != nil {
			t.Fatalf("layer %d is paged: this gate covers the resident expert-major path", l)
		}
	}
	tk, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	if len(files) < nPrompts {
		t.Fatalf("prompt set has %d files, want %d", len(files), nPrompts)
	}
	nL, k := r.nL, r.moe.k
	r.moeCap.idx = NewBufferUint32s(r.d, make([]uint32, nL*maxM*k))
	r.moeCap.wgt = NewBufferFloats(r.d, make([]float32, nL*maxM*k))
	t.Cleanup(func() { r.moeCap.idx, r.moeCap.wgt, r.moeCap.rows, r.moeCap.major = Buffer{}, Buffer{}, 0, nil })
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[dg01 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	hb("%s: %d layers, %d experts, top-%d, shared %d (gated %v); mutation %q; M %v; %d prompts",
		filepath.Base(path), nL, r.moe.nE, k, r.moe.sharedInter, !r.moe.sharedUngated, dg01Mutation, Ms, nPrompts)

	type arm struct {
		route  [][][]uint32 // [layer][position][k]
		kv     [][]float32  // per layer: K then V of positions 0..M-1
		logits []float32
	}
	kvSnap := func(M int) [][]float32 {
		out := make([][]float32, nL)
		for l := range r.layers {
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			row := make([]float32, 0, 2*M*d)
			for _, b := range []Buffer{r.kc[l], r.vc[l]} {
				for _, h := range b.U16s()[o : o+M*d] {
					row = append(row, f16ToF32(h))
				}
			}
			out[l] = row
		}
		return out
	}
	newRoute := func(M int) [][][]uint32 {
		rt := make([][][]uint32, nL)
		for l := range rt {
			rt[l] = make([][]uint32, M)
		}
		return rt
	}
	runSeq := func(ids []int) arm {
		M := len(ids)
		a := arm{route: newRoute(M)}
		r.moeCap.rows = 0
		for i, id := range ids {
			lg := r.Forward(id, i)
			if i == M-1 {
				a.logits = append([]float32(nil), lg...)
			}
			cap := r.moeCap.idx.U32s()
			for l := range nL {
				a.route[l][i] = slices.Clone(cap[l*k : (l+1)*k])
			}
		}
		a.kv = kvSnap(M)
		return a
	}
	runPass := func(ids []int, major bool) arm {
		M := len(ids)
		a := arm{route: newRoute(M)}
		if major {
			setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "1")
			r.moeCap.major = func(l int, idx []uint32, _ []float32) {
				for p := range M {
					a.route[l][p] = slices.Clone(idx[p*k : (p+1)*k])
				}
			}
		} else {
			setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "0")
			r.moeCap.rows = M
		}
		lg := r.PrefillLast(getEmbs(r, ids), 0)
		if lg == nil {
			t.Fatalf("PrefillLast (major %v) returned no logits: %v", major, r.takeExecErr())
		}
		a.logits = append([]float32(nil), lg...)
		r.moeCap.major, r.moeCap.rows = nil, 0
		if !major {
			cap := r.moeCap.idx.U32s()
			for l := range nL {
				for p := range M {
					a.route[l][p] = slices.Clone(cap[(l*M+p)*k : (l*M+p+1)*k])
				}
			}
		}
		a.kv = kvSnap(M)
		return a
	}
	sameSet := func(a, b []uint32) bool {
		x, y := slices.Clone(a), slices.Clone(b)
		slices.Sort(x)
		slices.Sort(y)
		return slices.Equal(x, y)
	}
	relL2 := func(a, ref []float32) (num, den float64) {
		for i := range ref {
			d := float64(a[i]) - float64(ref[i])
			num += d * d
			den += float64(ref[i]) * float64(ref[i])
		}
		return
	}

	pass := true
	for _, M := range Ms {
		flipsRow, flipsMajor, pairs := 0, 0, 0
		numRow, numMajor, den := make([]float64, nL), make([]float64, nL), make([]float64, nL)
		var klRow, klMajor float64
		top1Row, top1Major := 0, 0
		for pi := range nPrompts {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, files[pi], M)[:M]
			seq := runSeq(ids)
			row := runPass(ids, false)
			major := runPass(ids, true)
			// Preconditions: the instrument itself.
			if !slices.Equal(row.kv[0], major.kv[0]) {
				t.Fatalf("M=%d prompt %d: layer 0's K/V differs between row and major, which share everything before the first MoE: the arms are not what this gate assumes", M, pi+1)
			}
			for l := range nL {
				for p := range M {
					s := seq.route[l][p]
					if len(s) != k || slices.ContainsFunc(s, func(e uint32) bool { return int(e) >= r.moe.nE }) || len(slices.Compact(slices.Sorted(slices.Values(s)))) != k {
						t.Fatalf("M=%d prompt %d layer %d position %d: seq routing capture %v is not %d distinct experts", M, pi+1, l, p, s, k)
					}
					pairs++
					if !sameSet(row.route[l][p], s) {
						flipsRow++
					}
					if !sameSet(major.route[l][p], s) {
						flipsMajor++
					}
				}
			}
			for l := 1; l < nL; l++ {
				n1, d1 := relL2(row.kv[l], seq.kv[l])
				n2, _ := relL2(major.kv[l], seq.kv[l])
				numRow[l] += n1
				numMajor[l] += n2
				den[l] += d1
			}
			klRow += klLogits(seq.logits, row.logits)
			klMajor += klLogits(seq.logits, major.logits)
			if argmaxF(row.logits) == argmaxF(seq.logits) {
				top1Row++
			}
			if argmaxF(major.logits) == argmaxF(seq.logits) {
				top1Major++
			}
			hb("M=%d prompt %d/%d: flips row %d major %d (cumulative of %d)", M, pi+1, nPrompts, flipsRow, flipsMajor, pairs)
		}
		klRow /= float64(nPrompts)
		klMajor /= float64(nPrompts)
		flipBound := float64(flipsRow) + 2*math.Sqrt(math.Max(float64(flipsRow), 1))
		flipsOK := float64(flipsMajor) <= flipBound
		klRatio := klMajor / math.Max(klRow, 1e-12)
		worst := klRatio
		var layers []string
		for l := 1; l < nL; l++ {
			rr, rm := math.Sqrt(numRow[l]/den[l]), math.Sqrt(numMajor[l]/den[l])
			ratio := rm / math.Max(rr, 1e-12)
			worst = math.Max(worst, ratio)
			layers = append(layers, fmt.Sprintf("L%d row %.3e major %.3e (%.3fx)", l, rr, rm, ratio))
		}
		verdict := "PASSES"
		switch {
		case !flipsOK || worst > 1.5:
			verdict, pass = "FAILS", false
		case worst > 1.25:
			verdict, pass = "PARKED (a ratio in (1.25, 1.5])", false
		}
		hb("RESULT M=%d: flips row %d major %d of %d (%.4f / %.4f; bound %.1f, ok %v); K/V %s; KL row %.4e major %.4e (%.3fx); top-1 vs seq row %d/%d major %d/%d -> %s",
			M, flipsRow, flipsMajor, pairs, float64(flipsRow)/float64(pairs), float64(flipsMajor)/float64(pairs), flipBound, flipsOK,
			strings.Join(layers, ", "), klRow, klMajor, klRatio, top1Row, nPrompts, top1Major, nPrompts, verdict)
	}
	hb("VERDICT mutation %q: %s", dg01Mutation, map[bool]string{true: "the gate PASSES", false: "the gate does NOT pass"}[pass])
	if !pass {
		t.Errorf("D-G01 gate does not pass (mutation %q)", dg01Mutation)
	}
}
