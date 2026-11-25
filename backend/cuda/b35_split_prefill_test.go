//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

// TestB35SplitPrefill is B35's first question on CUDA: does a Gated-DeltaNet prompt
// prefilled in two PrefillLast calls, [0,b) then [b,n), leave what one call over [0,n) leaves? It compares the last row's logits,
// every DeltaNet layer's conv ring and recurrent state, and every full-attention layer's K/V rows, bit for bit.
//
// A pass under 16 rows runs decode's GEMV, a longer one the batched GEMM (TestCUDADeltaNetPrefill_matchesPerToken), so:
//   - gated: a split whose passes are all 16 rows or more must be bit-identical. A restore at b followed by the rest equals a cold
//     prefill only if a row's GEMM result does not depend on where its pass starts.
//   - reported, not gated: a split with a pass under 16 rows (a 2-row head, a 4-token tail, B35's empty think block). These are
//     expected to differ; B35 makes every cold prefill cut at the tail too, so the restore and the cold prompt take the same classes.
func TestB35SplitPrefill(t *testing.T) {
	requireCUDADevice(t)
	path := filepath.Join("../../testdata/qwen35-tiny")
	const n = 61
	type arm struct {
		logits      []float32
		win, state  [][]float32
		kRows, vRow [][]float32
	}
	run := func(zeroBetween bool, cuts ...int) arm {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
		}
		_, _, _, _, _, _, vocab := m.Dims()
		emb := func(lo, hi int) [][]float32 {
			rows := make([][]float32, 0, hi-lo)
			for i := lo; i < hi; i++ {
				rows = append(rows, m.EmbedResidentForTest((i*37+5)%vocab))
			}
			return rows
		}
		var a arm
		from := 0
		for _, to := range append(cuts, n) {
			lg, err := r.PrefillLast(context.Background(), emb(from, to), from)
			if err != nil {
				t.Fatalf("PrefillLast [%d,%d): %v", from, to, err)
			}
			a.logits = slices.Clone(lg)
			from = to
			if zeroBetween && from < n {
				for l := range r.layers {
					if Ly := &r.layers[l]; Ly.isDeltaNet() {
						if err := gpu.Upload(Ly.dnState, make([]float32, Ly.dnState.Len())); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
		}
		dl := func(b gpu.Buffer, n int) []float32 {
			out := make([]float32, n)
			if err := gpu.Download(b.At(0), out); err != nil {
				t.Fatalf("download: %v", err)
			}
			return out
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			switch {
			case Ly.isDeltaNet():
				a.win = append(a.win, dl(Ly.dnWin, Ly.dnWin.Len()))
				a.state = append(a.state, dl(Ly.dnState, Ly.dnState.Len()))
			case Ly.kvHalf || Ly.kvShared || Ly.kvDim == 0:
				// f16 or aliased K/V: the logits and the later layers' state already depend on it.
			default:
				k, v := r.readKVForTest(l, n)
				a.kRows = append(a.kRows, k)
				a.vRow = append(a.vRow, v)
			}
		}
		return a
	}
	diff := func(a, b [][]float32) (int, float64) {
		first, worst := -1, 0.0
		for i := range a {
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					if first < 0 {
						first = i
					}
					worst = max(worst, math.Abs(float64(a[i][j]-b[i][j])))
				}
			}
		}
		return first, worst
	}
	ref := run(false)
	if len(ref.state) == 0 || len(ref.kRows) == 0 {
		t.Fatalf("compared %d DeltaNet and %d f32 attention layers: the fixture must have both", len(ref.state), len(ref.kRows))
	}
	for _, c := range []struct {
		name  string
		cuts  []int
		gated bool
		plant bool // the recurrent state zeroed between the passes: must differ, or the comparison sees nothing
	}{
		{"passes of 30 and 31 rows", []int{30}, true, false},
		{"passes of 16, 16 and 29 rows", []int{16, 32}, true, false},
		{"a 2-row head pass", []int{2}, false, false},
		{"a 4-token tail (B35's shape)", []int{n - 4}, false, false},
		{"plant: passes of 30 and 31 rows, state zeroed between", []int{30}, false, true},
	} {
		got := run(c.plant, c.cuts...)
		parts := []struct {
			what string
			a, b [][]float32
		}{
			{"logits", [][]float32{ref.logits}, [][]float32{got.logits}},
			{"conv ring", ref.win, got.win},
			{"recurrent state", ref.state, got.state},
			{"K rows", ref.kRows, got.kRows},
			{"V rows", ref.vRow, got.vRow},
		}
		same := true
		for _, p := range parts {
			if first, worst := diff(p.a, p.b); first >= 0 {
				same = false
				t.Logf("%s: %s differs (first at layer index %d, max |diff| %.3g)", c.name, p.what, first, worst)
			}
		}
		t.Logf("%s: bit-identical to one pass: %v", c.name, same)
		if c.gated && !same {
			t.Errorf("%s: a split whose passes all take the batched GEMM must be bit-identical to one pass", c.name)
		}
		if c.plant && same {
			t.Errorf("%s: zeroing the state between the passes changed nothing, so the comparison cannot see a difference", c.name)
		}
	}
}

// G-R1 (docs/measurements/kv8-owed-2026-11/README.md): on a real checkpoint, a prefill cut at a snapshot point, and a restore from a
// snapshot taken at that point, leave what one uninterrupted prefill leaves: the last row's logits, the logits of the decode steps
// after it, every DeltaNet layer's conv ring and state, and every f32 attention layer's K and V rows.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run '^TestKV8Real_cutEqualsUninterrupted$' -v -timeout 20m ./backend/cuda/
//
// KV8_REAL_CKPT names another checkpoint directory (default ~/models/qwen3.5-0.8b).

// realCutDump is what one arm leaves behind.
type realCutDump struct {
	logits   []float32   // the last prompt row's
	steps    [][]float32 // the logits of each greedy decode step after the prompt
	win, st  [][]float32 // each DeltaNet layer's conv ring and recurrent state
	kv       [][]float32 // each f32 attention layer's K then V rows [0, n)
	argmaxes []int       // the arg-max of logits, then of every step
}

// realCutDiff counts the float32 words of a and b whose bits differ and the largest absolute difference among them.
func realCutDiff(a, b [][]float32) (words int, worst float64) {
	if len(a) != len(b) {
		return math.MaxInt, math.Inf(1)
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return math.MaxInt, math.Inf(1)
		}
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				words++
				worst = max(worst, math.Abs(float64(a[i][j])-float64(b[i][j])))
			}
		}
	}
	return words, worst
}

func realCutArgmax(v []float32) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// realCutRig is the loaded checkpoint, its prompt, and the prompt's embeddings.
type realCutRig struct {
	m    *decoder.Model
	r    *cudaResident
	ids  []int
	emb  [][]float32
	ctx  context.Context
	n    int
	pass int
}

func (g *realCutRig) embOf(ids []int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, x := range ids {
		rows[i] = g.m.EmbedResidentForTest(x)
	}
	return rows
}

// dump reads the slot's state and K/V rows for the first n positions, then runs steps greedy decode steps from the prompt's last row.
func (g *realCutRig) dump(t *testing.T, last []float32, steps int) realCutDump {
	t.Helper()
	d := realCutDump{logits: slices.Clone(last)}
	d.argmaxes = append(d.argmaxes, realCutArgmax(last))
	r := g.r
	if err := r.do(func() error {
		if e := r.stream.Sync(); e != nil {
			return e
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			switch {
			case Ly.isDeltaNet():
				w, s := make([]float32, Ly.dnWin.Len()), make([]float32, Ly.dnState.Len())
				if e := gpu.Download(Ly.dnWin, w); e != nil {
					return e
				}
				if e := gpu.Download(Ly.dnState, s); e != nil {
					return e
				}
				d.win, d.st = append(d.win, w), append(d.st, s)
			case Ly.noKV() || Ly.kvShared || Ly.kvHalf || Ly.kvDim == 0:
			default:
				k, v, e := r.kvRowsForTest(r.kvSlot, l, 0, g.n)
				if e != nil {
					return e
				}
				d.kv = append(d.kv, k, v)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	next := d.argmaxes[0]
	for i := range steps {
		lg, err := r.Forward(g.m.EmbedResidentForTest(next), g.n+i)
		if err != nil {
			t.Fatalf("decode step %d: %v", i, err)
		}
		d.steps = append(d.steps, slices.Clone(lg))
		next = realCutArgmax(lg)
		d.argmaxes = append(d.argmaxes, next)
	}
	return d
}

// prefillCut prefills the prompt in calls cut at cuts, calling between(from) after each call that does not end the prompt.
func (g *realCutRig) prefillCut(t *testing.T, between func(at int), cuts ...int) []float32 {
	t.Helper()
	var lg []float32
	from := 0
	for _, to := range append(slicesClone(cuts), g.n) {
		var err error
		if lg, err = g.r.PrefillLast(g.ctx, g.emb[from:to], from); err != nil {
			t.Fatalf("PrefillLast [%d,%d): %v", from, to, err)
		}
		from = to
		if between != nil && from < g.n {
			between(from)
		}
	}
	return lg
}

func slicesClone(a []int) []int { return slices.Clone(a) }

// minPass is the shortest call of a cut over n rows.
func realCutMinPass(n int, cuts []int) int {
	m, from := n, 0
	for _, to := range append(slicesClone(cuts), n) {
		m = min(m, to-from)
		from = to
	}
	return m
}

func TestKV8Real_cutEqualsUninterrupted(t *testing.T) {
	requireCUDADevice(t)
	testgate.RequireHeavy(t)
	ckpt := os.Getenv("KV8_REAL_CKPT")
	if ckpt == "" {
		ckpt = os.ExpandEnv("$HOME/models/qwen3.5-0.8b")
	}
	if _, err := os.Stat(filepath.Join(ckpt, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s: %v", ckpt, err)
	}
	const n, steps = 3000, 8
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096, KVHostBytes: 1 << 30})
	if err != nil {
		t.Fatalf("load %s: %v", ckpt, err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s is not CUDA-resident: %s", ckpt, m.ResidentDecline())
	}
	if !m.KeepsRecurrentSnapshots() {
		t.Fatalf("%s: snapshots do not carry the recurrent state", ckpt)
	}
	tk, err := modelload.Tokenizer(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	text, err := os.ReadFile("../../CLAUDE.md")
	if err != nil {
		t.Fatal(err)
	}
	all, err := tk.Encode(string(text), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < n+400 {
		t.Fatalf("the prompt text has %d ids, need %d", len(all), n+400)
	}
	g := &realCutRig{m: m, r: r, ids: all[:n], ctx: context.Background(), n: n, pass: int(r.PrefillMarkAlign())}
	g.emb = g.embOf(g.ids)
	other := g.embOf(all[n : n+300]) // another conversation, which takes the slot between a snapshot and its restore
	t.Logf("%s: %d ids, passes of %d rows, %d slots", ckpt, n, g.pass, r.KVSlots())

	ref := g.dump(t, g.prefillCut(t, nil), steps)
	if len(ref.st) == 0 || len(ref.kv) == 0 {
		t.Fatalf("compared %d DeltaNet and %d f32 attention buffers: the checkpoint must have both", len(ref.st), len(ref.kv)/2)
	}
	u2 := g.dump(t, g.prefillCut(t, nil), steps)

	// compare grades one arm against U1 and returns whether it is bit-identical and whether it is behaviour-preserving.
	compare := func(name string, got realCutDump) (bitSame, behaves bool) {
		parts := []struct {
			what string
			a, b [][]float32
		}{
			{"last-row logits", [][]float32{ref.logits}, [][]float32{got.logits}},
			{"decode-step logits", ref.steps, got.steps},
			{"conv rings", ref.win, got.win},
			{"recurrent states", ref.st, got.st},
			{"K/V rows", ref.kv, got.kv},
		}
		bitSame, behaves = true, true
		for _, p := range parts {
			words, worst := realCutDiff(p.a, p.b)
			if words == 0 {
				continue
			}
			bitSame = false
			t.Logf("  %s: %s: %d words differ, max |diff| %.3g", name, p.what, words, worst)
			if p.what == "last-row logits" || p.what == "decode-step logits" {
				if worst > 1e-3 {
					behaves = false
				}
			}
		}
		for i := range ref.argmaxes {
			if ref.argmaxes[i] != got.argmaxes[i] {
				behaves = false
				t.Logf("  %s: the arg-max differs at row %d: %d against %d", name, i, ref.argmaxes[i], got.argmaxes[i])
				break
			}
		}
		return bitSame, behaves
	}

	// The instrument: U1 against U2 must be bit-identical, or no equality below means anything.
	if same, _ := compare("U1 against U2", u2); !same {
		t.Fatalf("VOID: two uninterrupted prefills of the same prompt differ, so the comparison cannot grade equality on this card")
	}

	var parked, failed []string
	grade := func(name string, gated bool, got realCutDump) {
		same, behaves := compare(name, got)
		t.Logf("%s: bit-identical to U1: %v (gated %v)", name, same, gated)
		switch {
		case !gated || same:
		case behaves:
			parked = append(parked, name)
		default:
			failed = append(failed, name)
		}
	}

	// A plant must differ, or the comparison sees nothing.
	zero := func(at int) {
		if err := r.do(func() error {
			for l := range r.layers {
				if Ly := &r.layers[l]; Ly.isDeltaNet() {
					if e := gpu.Upload(Ly.dnState, make([]float32, Ly.dnState.Len())); e != nil {
						return e
					}
				}
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if same, _ := compare("plant: state zeroed between the calls", g.dump(t, g.prefillCut(t, zero, 1152), steps)); same {
		t.Fatalf("VOID: zeroing the recurrent state between the two calls changed nothing, so the comparison cannot see a difference")
	}

	cuts := [][]int{{64}, {960}, {1152}, {2048}, {2368}, {960, 2368}, {64, 1152, 2368}}
	for _, c := range cuts {
		gated := realCutMinPass(n, c) >= 16
		name := fmt.Sprintf("cut %v", c)
		grade(name, gated, g.dump(t, g.prefillCut(t, nil, c...), steps))

		// A restore at the last point of the cut: the snapshot is taken between the calls, another conversation takes the slot,
		// and the snapshot is put back before the suffix.
		q := c[len(c)-1]
		for _, plant := range []bool{false, true} {
			var snap decoder.ResidentKVSnapshot
			g.prefillCut(t, func(at int) {
				if at != q {
					return
				}
				s, e := r.SnapshotKV(0, at)
				if e != nil {
					t.Fatalf("SnapshotKV(0, %d): %v", at, e)
				}
				snap = s
			}, c...)
			// the prefill above ran the whole prompt; run the other conversation over the slot, then restore the snapshot
			if _, err := r.PrefillLast(g.ctx, other, 0); err != nil {
				t.Fatal(err)
			}
			if plant {
				snap.(*cudaKVSnapshot).rec = nil // PLANT: the rows without the state
			}
			if err := r.RestoreKV(snap, 0, q); err != nil {
				t.Fatalf("RestoreKV(%d): %v", q, err)
			}
			lg, err := r.PrefillLast(g.ctx, g.emb[q:], q)
			if err != nil {
				t.Fatal(err)
			}
			got := g.dump(t, lg, steps)
			snap.Release()
			rn := fmt.Sprintf("restore at %d (cut %v)", q, c)
			if plant {
				if same, _ := compare("plant: "+rn+" without the state", got); same {
					t.Fatalf("VOID: restoring the rows without the state changed nothing, so the comparison cannot see the state")
				}
				continue
			}
			grade(rn, realCutMinPass(n, []int{q}) >= 16 && gated, got)
		}
	}
	if len(failed) > 0 {
		t.Errorf("FAIL: the cut changes a real model's numbers beyond 1e-3 or its arg-max in: %v", failed)
	}
	if len(parked) > 0 {
		t.Errorf("AMBIGUOUS (parked for Francis): behaviour-preserving but not bit-identical in: %v", parked)
	}
	if len(failed) == 0 && len(parked) == 0 {
		t.Logf("PASS: every gated cut and restore is bit-identical to the uninterrupted prefill")
	}
}
