//go:build darwin

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// OP6's Metal half, tier 0 (docs/tasks/task-forward-architecture-2026-10.md §4.7): the q4k lane on MC3's batched step.
// Every check is bit for bit against the lane's own decode. The model is Qwen2.5-Coder-0.5B q4_k_m, whose attn_v is Q6_K
// (int8 on the lane) in half the layers, so both rows GEMVs and the multi-segment projections run.

// q4kStepResident loads the 0.5B at --quant q4k (q4kStepLoad).
func q4kStepResident(t *testing.T, slots, ctx int) *metalResident {
	t.Helper()
	return q4kStepLoad(t, "GOINFER_QWEN05_GGUF", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", slots, ctx)
}

// q4kStepLoad loads the GGUF named by env (else ~/models/<file>) at --quant q4k with `slots` resident KV slots and builds
// its resident, which must be on the lane with the batched step built. It skips without the file.
func q4kStepLoad(t *testing.T, env, file string, slots, ctx int) *metalResident {
	t.Helper()
	path := os.Getenv(env)
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", file)
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "q4k", ResidentContext: ctx, ResidentKVSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build resident: %v", err)
	}
	t.Cleanup(func() {
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	})
	if !r.q4kLane || r.batch == nil || len(r.kvSlotBufs) < slots {
		t.Fatalf("q4k lane %v, batched step %v (batchIneligible: %q), %d KV slots: not the path under test",
			r.q4kLane, r.batch != nil, r.batchIneligible(), len(r.kvSlotBufs))
	}
	return &metalResident{r: r, hidden: r.H}
}

// q4kStepIDs is a deterministic token sequence.
func q4kStepIDs(seed uint32, n int) []int {
	ids := make([]int, n)
	for i := range ids {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		ids[i] = int(seed % 20000)
	}
	return ids
}

// q4kStepKV copies slot's K and V for positions [from, from+n) of every layer.
func q4kStepKV(t *testing.T, r *resident, slot, from, n int) [][]uint16 {
	t.Helper()
	if err := r.useKVSlot(slot); err != nil {
		t.Fatal(err)
	}
	var out [][]uint16
	for l := range r.layers {
		d := r.layers[l].geom.kvDim
		o := r.kvHostOff(l, 2) + from*d
		out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
	}
	return out
}

// q4kStepDiff counts the logits and K/V elements that differ bit for bit.
func q4kStepDiff(gotLg, wantLg []float32, gotKV, wantKV [][]uint16) (lg, kv int) {
	for j := range wantLg {
		if math.Float32bits(gotLg[j]) != math.Float32bits(wantLg[j]) {
			lg++
		}
	}
	for i := range wantKV {
		for j := range wantKV[i] {
			if gotKV[i][j] != wantKV[i][j] {
				kv++
			}
		}
	}
	return lg, kv
}

// q4kStepPrompt runs the gate (a) case: a prompt of n tokens from startPos through PrefillLast on slot 1 (the step) and
// through decode's single-token loop on slot 0, after the same prefix on both. It returns the differing logits of the
// last row and the differing K/V elements of the prompt's positions.
func q4kStepPrompt(t *testing.T, a *metalResident, ids []int, startPos int) (lg, kv int) {
	t.Helper()
	r := a.r
	embs := make([][]float32, len(ids))
	for i, id := range ids {
		embs[i] = mc3Emb(r, id)
	}
	var want []float32
	for slot := range 2 {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		for i, e := range embs[:startPos] { // the shared prefix, by decode
			r.ForwardEmb(e, i)
		}
	}
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	for i, e := range embs[startPos:] {
		want = append(want[:0], r.ForwardEmb(e, startPos+i)...)
	}
	wantKV := q4kStepKV(t, r, 0, startPos, len(ids)-startPos)
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if !a.promptStepOK(len(ids)-startPos, startPos, metalFastPrefillFloorFor("")) {
		t.Fatalf("a %d-token prompt at %d does not take the step route", len(ids)-startPos, startPos)
	}
	got, err := a.PrefillLast(context.Background(), embs[startPos:], startPos)
	if err != nil {
		t.Fatalf("PrefillLast(%d tokens at %d): %v", len(ids)-startPos, startPos, err)
	}
	return q4kStepDiff(got, want, q4kStepKV(t, r, 1, startPos, len(ids)-startPos), wantKV)
}

// TestQ4KStep_promptMatchesSequential is gate (a): a q4k prompt on the step route gives the last row's logits and every K/V
// byte that decode's single-token loop gives, at lengths that leave a one-row tail (9, 17), end on a full step (8, 64), or
// fit in one step (2, 7), and for a continuation. It also pins PrefillPath's report of the route.
func TestQ4KStep_promptMatchesSequential(t *testing.T) {
	a := q4kStepResident(t, 2, 1024)
	if ok, why := a.PrefillPath(); !ok || !strings.Contains(why, "batched step") {
		t.Errorf("PrefillPath() = %v, %q: want the batched step route", ok, why)
	}
	cases := []struct{ n, start int }{{2, 0}, {7, 0}, {8, 0}, {9, 0}, {17, 0}, {64, 0}, {26, 17}}
	for i, c := range cases {
		lg, kv := q4kStepPrompt(t, a, q4kStepIDs(uint32(9001+i), c.n), c.start)
		t.Logf("%d tokens from %d: %d of %d last-row logits and %d K/V elements differ", c.n-c.start, c.start, lg, a.r.V, kv)
		if lg != 0 || kv != 0 {
			t.Errorf("%d tokens from %d: the step differs from sequential decode (%d logits, %d K/V elements)", c.n-c.start, c.start, lg, kv)
		}
	}
}

// TestQ4KStep_plantIsRed is gate (a)'s planted defect: the min term dropped from the q4k rows kernel alone (decode keeps
// it) must make the step differ from sequential decode.
func TestQ4KStep_plantIsRed(t *testing.T) {
	const good = "acc[r] += d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);"
	if strings.Count(mc3RowsKernels, good) != 1 {
		t.Fatal("the plant's target is not in the q4k rows kernel")
	}
	prev := mc3RowsKernels
	mc3RowsKernels = strings.Replace(prev, good, "acc[r] += d*sc0*dlo + d*sc1*dhi;", 1)
	t.Cleanup(func() { mc3RowsKernels = prev })
	a := q4kStepResident(t, 2, 1024)
	lg, kv := q4kStepPrompt(t, a, q4kStepIDs(9004, 17), 0)
	t.Logf("planted: %d logits, %d K/V elements differ", lg, kv)
	if lg == 0 && kv == 0 {
		t.Error("the planted rows kernel still matches sequential decode: gate (a) cannot see the defect")
	}
}

// TestQ4KStep_verifyRowsMatchDecode is gate (b): M consecutive rows of one sequence on one slot, as a speculative verify
// runs them, against decode at depths 128 and 2048. Both slots take the same history once per depth, through PrefillLast
// (gate (a) shows it writes decode's bytes); each case's rows overwrite their positions before attention reads them.
func TestQ4KStep_verifyRowsMatchDecode(t *testing.T) {
	a := q4kStepResident(t, 2, 4096)
	r := a.r
	filled := 0
	for i, c := range []struct{ D, M int }{{128, 2}, {128, 4}, {128, 8}, {2048, 2}, {2048, 4}, {2048, 8}} {
		if c.D != filled {
			var hist [][]float32
			for _, id := range q4kStepIDs(uint32(500+c.D), c.D) {
				hist = append(hist, mc3Emb(r, id))
			}
			for slot := range 2 {
				if err := r.useKVSlot(slot); err != nil {
					t.Fatal(err)
				}
				if _, err := a.PrefillLast(context.Background(), hist, 0); err != nil {
					t.Fatalf("history of %d: %v", c.D, err)
				}
			}
			filled = c.D
		}
		embs := make([][]float32, c.M)
		for j, id := range q4kStepIDs(uint32(700+i), c.M) {
			embs[j] = mc3Emb(r, id)
		}
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		var ref [][]float32
		for j, e := range embs {
			ref = append(ref, append([]float32(nil), r.ForwardEmb(e, c.D+j)...))
		}
		seqs := make([]batchSeq, c.M)
		for j := range seqs {
			seqs[j] = batchSeq{slot: 1, pos: c.D + j, emb: embs[j]}
		}
		got := mc3Step(t, r, seqs)
		lg, kv := 0, 0
		for j := range got {
			l, _ := q4kStepDiff(got[j], ref[j], nil, nil)
			lg += l
		}
		_, kv = q4kStepDiff(nil, nil, q4kStepKV(t, r, 1, c.D, c.M), q4kStepKV(t, r, 0, c.D, c.M))
		t.Logf("depth %d, %d rows: %d logits and %d K/V elements differ", c.D, c.M, lg, kv)
		if lg != 0 || kv != 0 {
			t.Errorf("depth %d, M = %d: the step's rows differ from decode (%d logits, %d K/V elements)", c.D, c.M, lg, kv)
		}
	}
}

// TestQ4KStep_twoSequencesMatchAlone is gate (c): two sequences at different depths on their own slots, stepped together
// for 8 teacher-forced steps, each row against that sequence's decode alone.
func TestQ4KStep_twoSequencesMatchAlone(t *testing.T) {
	a := q4kStepResident(t, 2, 1024)
	r := a.r
	const steps = 8
	depth := [2]int{40, 70}
	var hist, next [2][]int
	for s := range 2 {
		hist[s], next[s] = q4kStepIDs(uint32(31+s), depth[s]), q4kStepIDs(uint32(61+s), steps)
	}
	var alone [2][][]float32
	for s := range 2 {
		mc3Fill(t, r, s, hist[s])
		for k := range steps {
			alone[s] = append(alone[s], append([]float32(nil), r.ForwardEmb(mc3Emb(r, next[s][k]), depth[s]+k)...))
		}
	}
	for s := range 2 { // the same history again; the steps overwrite each position before attention reads it
		mc3Fill(t, r, s, hist[s])
	}
	diff := 0
	for k := range steps {
		seqs := []batchSeq{
			{slot: 0, pos: depth[0] + k, emb: mc3Emb(r, next[0][k])},
			{slot: 1, pos: depth[1] + k, emb: mc3Emb(r, next[1][k])},
		}
		got := mc3Step(t, r, seqs)
		for s := range 2 {
			l, _ := q4kStepDiff(got[s], alone[s][k], nil, nil)
			diff += l
		}
	}
	t.Logf("%d steps of two sequences: %d logits differ from decode alone", steps, diff)
	if diff != 0 {
		t.Errorf("two sequences stepped together differ from decode alone in %d logits", diff)
	}
}
