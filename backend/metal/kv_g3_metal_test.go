//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/loadflags"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// g3MetalNotBuilt are the Metal cells whose fixture builds no Metal resident (a decline outside KV), so the model runs on the CPU
// cache; each is read back against what the model reports delivering (docs/measurements/kv-precision-2026-10/g3.md).
var g3MetalNotBuilt = []string{
	"phi3-tiny/auto", "phi3-tiny/f32", "phi3-tiny/f16", "phi3-tiny/i8",
	"lfm2-tiny/auto", "lfm2-tiny/f32",
	"gemma4-moe-kv-tiny/auto", "gemma4-moe-kv-tiny/f32",
}

// g3MetalMatrixDiffers are the no-resident cells whose CPU cache is not what the matrix's Metal cell says: the matrix reads the
// family's Metal answer, the load runs the CPU's (G3's record).
var g3MetalMatrixDiffers = []string{
	"phi3-tiny/auto", "lfm2-tiny/auto", "gemma4-moe-kv-tiny/auto",
}

// TestKVDeliveredBytes_Metal is G3's Metal half (docs/tasks/task-kv-precision-everywhere-2026-10.md §4): every Metal cell the
// matrix delivers, loaded with serve's options for --kv and a 100-position resident context, its K/V buffers read back against the
// delivered precision: f16 2 B and f32 4 B per element, int8 1 B plus one f32 scale per position and head for each of K and V; an
// MLA layer's latent at 4 B. A cell whose resident declines is read back on the CPU cache against the model's KVDelivered.
func TestKVDeliveredBytes_Metal(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	fixtures := []string{"llama-tiny", "phi3-tiny", "qwen3moe-tiny", "gemma2-tiny", "cohere-tiny", "qwen25vl-tiny", "deepseek-tiny",
		"qwen35-tiny", "qwen3next-tiny", "nemotron-tiny", "lfm2-tiny", "gemma4-moe-kv-tiny"}
	var notBuilt, matrixDiffers []string
	checked := 0
	for _, f := range fixtures {
		for _, v := range []string{"auto", "f32", "f16", "i8"} {
			cell := f + "/" + v
			kv := v
			if v == "auto" {
				kv = ""
			}
			m, err := decoder.Load(filepath.Join("../../testdata", f), decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, KVQuant: loadflags.CPUKV("", kv), ResidentContext: 100})
			var r *decoder.KVRefusal
			if errors.As(err, &r) {
				continue // a refused cell allocates nothing
			}
			if err != nil {
				t.Fatalf("%s: Load: %v", cell, err)
			}
			want := m.KVDeliveredFor("metal")
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				notBuilt = append(notBuilt, cell)
				got := m.KVDelivered()
				n32, n16, n8, nScale, err := m.KVReadbackForTest(12)
				if err != nil {
					t.Fatalf("%s: CPU readback: %v", cell, err)
				}
				held := map[[4]bool]string{{true, false, false, false}: "f32", {false, true, false, false}: "f16", {false, false, true, true}: "i8"}[[4]bool{n32 > 0, n16 > 0, n8 > 0, nScale > 0}]
				_, _, _, _, hd, _, _ := m.Dims()
				if held != got || (held == "i8" && nScale*int64(hd) != n8) {
					t.Errorf("%s: no resident; the CPU cache reports %s and holds f32 %d, f16 %d, i8 %d, scales %d", cell, got, n32, n16, n8, nScale)
				}
				if held != want {
					matrixDiffers = append(matrixDiffers, cell)
				}
				fmt.Fprintf(os.Stderr, "[G3 metal] %-24s no resident: runs on CPU, holds %-3s, the matrix's Metal cell says %q\n", cell, held, want)
				m.Close()
				continue
			}
			res := a.r
			padded := int64((1 + kvBlocksFor(res.ctxCap)*max(1, len(res.kvSlotBufs))) * decoder.KVBlockRows) // the null block, and every slot's blocks
			why := ""
			for l := range res.kc {
				L := res.layers[l]
				if L.kvShared || res.kc[l].Len() == 0 {
					continue
				}
				if res.mla != nil { // the latent rows, f32 (metal/mla.go); V aliases them
					if got, latent := int64(res.kc[l].Len()), padded*int64(res.mla.latDim)*4; got != latent {
						why = fmt.Sprintf("layer %d holds %d B of latent, not %d B at f32", l, got, latent)
					} else if want != "f32" {
						why = fmt.Sprintf("layer %d holds an f32 latent, the matrix says %s", l, want)
					}
					continue
				}
				elems := padded * int64(L.geom.kvDim)
				per := map[string]int64{"f16": 2, "f32": 4, "i8": 1}[want]
				if got := int64(res.kc[l].Len() + res.vc[l].Len()); got != 2*elems*per {
					why = fmt.Sprintf("layer %d holds %d B of K/V for %d elements, not %d B each (%s)", l, got, elems, per, want)
				}
				nScale := int64(0)
				if l < len(res.ks) {
					nScale = int64(res.ks[l].Len() + res.vs[l].Len())
				}
				if wantScale := map[bool]int64{true: 2 * padded * int64(L.geom.nKV), false: 0}[want == "i8"]; nScale != wantScale {
					why = fmt.Sprintf("layer %d holds %d scales, want %d (%s)", l, nScale, wantScale, want)
				}
			}
			fmt.Fprintf(os.Stderr, "[G3 metal] %-24s delivered %-3s %s\n", cell, want, map[bool]string{true: "ok", false: "MISMATCH: " + why}[why == ""])
			if why != "" {
				t.Errorf("%s: %s", cell, why)
			}
			checked++
			m.Close()
		}
	}
	if !slices.Equal(matrixDiffers, g3MetalMatrixDiffers) {
		t.Errorf("the no-resident cells whose CPU cache differs from the matrix's Metal cell are %v, the list says %v", matrixDiffers, g3MetalMatrixDiffers)
	}
	if !slices.Equal(notBuilt, g3MetalNotBuilt) {
		t.Errorf("the cells that build no Metal resident are %v, the list says %v", notBuilt, g3MetalNotBuilt)
	}
	if checked < 20 {
		t.Fatalf("only %d Metal cells were read back", checked)
	}
	fmt.Fprintf(os.Stderr, "[G3 metal] %d cells read back, %d build no resident\n", checked, len(notBuilt))
}

// TestKVPages_everyResidentFixture is KV5's G1a over every tiny fixture Metal builds a resident for (KV5-4): free blocks handed out
// reversed and interleaved across slots give the in-order tables' logits bit for bit. Each slot prefills a 300-token prompt (the pass
// crosses block edges and takes its multi-row kernels) and decodes 64 tokens; then each takes a second turn's prefill from inside a
// block and a 4-row verify; then a batched step runs over up to three slots where the build has one. Both arms must launch the same
// prefill attention kernels (prefillAttnRanForTest), so a case asserts which kernels it ran. A fixture with no Metal resident is
// listed, not counted; a case a build refuses (an error) must be refused by both arms.
func TestKVPages_everyResidentFixture(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	var fixtures []string
	for _, root := range []string{"../../testdata", "../../decoder/testdata"} {
		dirs, _ := filepath.Glob(filepath.Join(root, "*-tiny"))
		for _, d := range dirs {
			ws, _ := filepath.Glob(filepath.Join(d, "*.safetensors"))
			if _, err := os.Stat(filepath.Join(d, "config.json")); err == nil && len(ws) > 0 {
				fixtures = append(fixtures, d)
			}
		}
	}
	if len(fixtures) == 0 {
		t.Skip("no tiny fixtures")
	}
	const ctx, prompt, decodeN, turn2, verify = 1024, 300, 64, 70, 4
	long := func(dir string) string { return kvLongFixture(t, dir) }
	type arm struct {
		logits  [][]float32
		refused []string
		ran     map[string]int
		slots   int
		step    bool
		// stepRows is how many of logits' last rows are the batched step's.
		stepRows int
	}
	run := func(dir string, scramble, decodeRef bool) (*arm, string) {
		kvScrambleForTest = scramble
		defer func() { kvScrambleForTest = false }()
		m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: ctx, ResidentKVSlots: 3,
			ResidentMoEStep: true, ResidentRecurrentStep: true, ResidentGemma4Step: true}) // every step the build can take
		if err != nil {
			return nil, "load: " + err.Error()
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			return nil, "no Metal resident: " + m.ResidentDecline()
		}
		r := a.r
		if r.kvTabBlocks == 0 {
			return nil, "no KV pool (no attention layer)"
		}
		_, _, _, _, _, _, vocab := m.Dims()
		emb := func(seed, i int) []float32 { return m.EmbedResidentForTest((i*97 + seed*13 + 3) % vocab) }
		embs := func(seed, n int) [][]float32 {
			out := make([][]float32, n)
			for i := range out {
				out[i] = emb(seed, i)
			}
			return out
		}
		out := &arm{}
		prefillAttnRanForTest = map[string]int{}
		defer func() { prefillAttnRanForTest = nil }()
		add := func(what string, l []float32, err error) {
			if err != nil {
				out.refused = append(out.refused, what+": "+err.Error())
				return
			}
			out.logits = append(out.logits, slices.Clone(l))
		}
		slots := r.kvSlotCount()
		for s := range slots {
			if err := r.useKVSlot(s); err != nil {
				return nil, err.Error()
			}
			l, err := a.PrefillLast(context.Background(), embs(s, prompt), 0)
			add(fmt.Sprintf("slot %d prefill", s), l, err)
			for i := range decodeN {
				l, err := a.Forward(emb(s+5, i), prompt+i)
				add(fmt.Sprintf("slot %d decode %d", s, i), l, err)
			}
		}
		at := prompt + decodeN
		for s := range slots {
			if err := r.useKVSlot(s); err != nil {
				return nil, err.Error()
			}
			l, err := a.PrefillLast(context.Background(), embs(s+9, turn2), at)
			add(fmt.Sprintf("slot %d second turn", s), l, err)
			rows, err := a.ForwardN(embs(s+11, verify), at+turn2)
			if err != nil {
				add(fmt.Sprintf("slot %d verify", s), nil, err)
			}
			for _, l := range rows {
				out.logits = append(out.logits, slices.Clone(l))
			}
		}
		if r.batch != nil && slots >= 2 {
			var seqs []batchSeq
			for s := range min(3, slots) {
				seqs = append(seqs, batchSeq{slot: s, pos: at + turn2 + verify, emb: emb(s+13, 0)})
			}
			var rows [][]float32
			var err error
			if !decodeRef {
				rows, _, err = r.forwardMulti(seqs)
			} else { // the reference: each row decoded on its own slot instead, from the same history (no step first: it would advance a recurrent state)
				for _, q := range seqs {
					if err := r.useKVSlot(q.slot); err != nil {
						return nil, err.Error()
					}
					l, e := a.Forward(q.emb, q.pos)
					if e != nil {
						err = e
						break
					}
					rows = append(rows, slices.Clone(l)) // Forward's slice is the resident's logits buffer, overwritten by the next
				}
			}
			if err != nil {
				add("batched step", nil, err)
			}
			for _, l := range rows {
				out.logits = append(out.logits, slices.Clone(l))
			}
			out.stepRows = len(rows)
		}
		out.ran, out.slots, out.step = maps.Clone(prefillAttnRanForTest), slots, r.batch != nil && slots >= 2
		return out, ""
	}
	start, covered, compared := time.Now(), map[string]bool{}, 0
	for i, dir := range fixtures {
		name := filepath.Base(dir)
		dir = long(dir)
		in, why := run(dir, false, false)
		if in == nil {
			fmt.Fprintf(os.Stderr, "[G1a] %d/%d %-28s NOT RUN: %s (elapsed %s)\n", i+1, len(fixtures), name, why, time.Since(start).Round(time.Second))
			continue
		}
		sc, why := run(dir, true, false)
		if sc == nil {
			t.Errorf("%s: the in-order arm built and the scrambled one did not: %s", name, why)
			continue
		}
		compared++
		switch {
		case len(in.logits) == 0:
			t.Errorf("%s builds a resident and ran none of its cases (first refusal: %q): the comparison would be vacuous", name, in.refused[0])
		case !slices.Equal(in.refused, sc.refused):
			t.Errorf("%s: the arms refused different cases: in order %q, scrambled %q", name, in.refused, sc.refused)
		case !maps.Equal(in.ran, sc.ran):
			t.Errorf("%s: the arms launched different prefill kernels: in order %v, scrambled %v", name, in.ran, sc.ran)
		case len(in.logits) != len(sc.logits):
			t.Errorf("%s: %d logit rows in order, %d scrambled", name, len(in.logits), len(sc.logits))
		default:
			n := 0
			for j := range in.logits {
				for k := range in.logits[j] {
					if math.Float32bits(in.logits[j][k]) != math.Float32bits(sc.logits[j][k]) {
						n++
					}
				}
			}
			if n != 0 {
				t.Errorf("%s: %d logits differ between scrambled and in-order blocks", name, n)
			}
		}
		// A row bound to another slot's table is a mistake both arms share, so each step row is also held to its own slot's decode.
		if in.stepRows > 0 {
			ref, why := run(dir, false, true)
			if ref == nil || len(ref.logits) != len(in.logits) || ref.stepRows != in.stepRows {
				t.Errorf("%s: the decode reference did not run as the step did: %s", name, why)
			} else {
				n := 0
				for j := len(in.logits) - in.stepRows; j < len(in.logits); j++ {
					for k := range in.logits[j] {
						if math.Float32bits(in.logits[j][k]) != math.Float32bits(ref.logits[j][k]) {
							n++
						}
					}
				}
				if n != 0 {
					t.Errorf("%s: %d of the batched step's logits differ from each row decoded on its own slot", name, n)
				}
			}
		}
		for k := range in.ran {
			covered[k] = true
		}
		fmt.Fprintf(os.Stderr, "[G1a] %d/%d %-28s %d logit rows, %d slots, step %v, prefill kernels %v, refused %d (elapsed %s)\n", i+1, len(fixtures), name,
			len(in.logits), in.slots, in.step, in.ran, len(in.refused), time.Since(start).Round(time.Second))
	}
	if compared == 0 {
		t.Fatal("no fixture built a Metal resident")
	}
	fmt.Fprintf(os.Stderr, "[G1a] %d of %d fixtures compared; prefill kernels covered %v\n", compared, len(fixtures), slices.Sorted(maps.Keys(covered)))
}

// kvLongFixture copies a fixture with its window raised to 2,048 positions, so a prompt of a few hundred rows fits a tiny model's cache;
// the weights are hard-linked where the filesystem allows it, else copied.
func kvLongFixture(t *testing.T, dir string) string {
	out := filepath.Join(t.TempDir(), filepath.Base(dir))
	if err := os.Mkdir(out, 0o755); err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		src, dst := filepath.Join(dir, e.Name()), filepath.Join(out, e.Name())
		b, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		if e.Name() == "config.json" {
			var raw map[string]any
			if err := json.Unmarshal(b, &raw); err != nil {
				t.Fatal(err)
			}
			raw["max_position_embeddings"] = 2048
			if tc, ok := raw["text_config"].(map[string]any); ok {
				tc["max_position_embeddings"] = 2048
			}
			if b, err = json.Marshal(raw); err != nil {
				t.Fatal(err)
			}
		} else if os.Link(src, dst) == nil {
			continue
		}
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// TestKVPages_shareByReferenceLayerMajor is TestKVPages_shareByReference's write site check for the two layer-major prefills, which run
// only on their own families: Gemma 4's E-model prefill (gemma4-emodel-tiny) and the paged generic MoE's (mixtral-tiny, paged at 3 slots).
// Slot 0 prefills a 128-row preamble and slot 1 takes it by CopyKV; slot 1 prefills a suffix; slot 0 rewinds by a prefill from 100,
// inside the shared block 1. Slot 1's K/V over its history, slot 0's over rows 0 to 119 (its copy of block 1 kept rows 64 to 99) and slot
// 0's rewind logits equal the reference where slot 1 prefilled the preamble itself; with the copy on write planted off slot 1's K/V moves,
// and with it keeping one row too few slot 0's does. The layer-major pass's own counter shows it ran.
func TestKVPages_shareByReferenceLayerMajor(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, c := range []struct{ name, dir, moeSlots string }{
		{"emodel", "../../testdata/gemma4-emodel-tiny", ""}, {"paged-moe", "../../testdata/mixtral-tiny", "3"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(c.dir); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			if c.moeSlots != "" {
				t.Setenv("GOINFER_METAL_MOE_SLOTS", c.moeSlots)
			} else {
				t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "100000") // the E-model's batched pass declines, so its layer-major pass runs
			}
			dir := kvLongFixture(t, c.dir)
			// run returns slot 1's K/V over its history, slot 0's over rows 0 to 119 after its rewind, and slot 0's rewind logits.
			// plant is "", "cow-off" (kvOwnOffForTest) or "keep-1" (kvKeepShortForTest).
			run := func(share bool, plant string) (kv1, kv0 [][]byte, rewind []float32) {
				kvOwnOffForTest, kvKeepShortForTest = plant == "cow-off", plant == "keep-1"
				defer func() { kvOwnOffForTest, kvKeepShortForTest = false, false }()
				m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 512, ResidentKVSlots: 2})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				a, ok := m.ResidentForwardForTest().(*metalResident)
				if !ok || a.r.kvSlotCount() < 2 {
					t.Fatalf("no two-slot Metal resident: %s", m.ResidentDecline())
				}
				r := a.r
				lm := func() int { return r.emodelLayerMajorRuns + r.moeLayerMajorRuns }
				_, _, _, _, _, _, vocab := m.Dims()
				embs := func(seed, n int) [][]float32 {
					out := make([][]float32, n)
					for i := range out {
						out[i] = m.EmbedResidentForTest((i*97 + seed*13 + 3) % vocab)
					}
					return out
				}
				prefill := func(s int, e [][]float32, at int) []float32 {
					t.Helper()
					if err := r.useKVSlot(s); err != nil {
						t.Fatal(err)
					}
					before := lm()
					l, err := a.PrefillLast(context.Background(), e, at)
					if err != nil {
						t.Fatal(err)
					}
					if lm() == before {
						t.Fatalf("slot %d's prefill at %d did not take the layer-major pass", s, at)
					}
					return slices.Clone(l)
				}
				preamble := embs(1, 128)
				prefill(0, preamble, 0)
				if share {
					if err := a.CopyKV(0, 1, 0, 128); err != nil {
						t.Fatal(err)
					}
				} else {
					prefill(1, preamble, 0)
				}
				prefill(1, embs(2, 20), 128)
				// Slot 0 rewinds into block 1, which slot 1 shares: its copy of the block must keep rows 64 to 99 (the rows below the write).
				rewind = prefill(0, embs(3, 20), 100)
				return kv2Slot(t, r, 1, 148), kv2Slot(t, r, 0, 120), rewind
			}
			eq := func(a, b [][]byte) bool { return slices.EqualFunc(a, b, slices.Equal) }
			ref1, ref0, refRewind := run(false, "")
			got1, got0, gotRewind := run(true, "")
			if !eq(ref1, got1) {
				t.Error("slot 1's K/V differs between the shared preamble and its own")
			}
			if !eq(ref0, got0) {
				t.Error("slot 0's K/V after its rewind into the shared block differs from its own preamble's: the copy on write lost rows")
			}
			if !slices.Equal(refRewind, gotRewind) {
				t.Error("slot 0's rewind logits differ between the shared preamble and its own")
			}
			if p1, _, _ := run(true, "cow-off"); eq(ref1, p1) {
				t.Error("with kvOwn off, slot 0's rewind into the shared block left slot 1's K/V as it was: the test cannot see a missing copy on write")
			}
			if _, p0, _ := run(true, "keep-1"); eq(ref0, p0) {
				t.Error("with the copy on write keeping one row too few, slot 0's K/V was as it should be: the test cannot see a short copy")
			}
		})
	}
}

// kvCapLoad is the MC3 fixture, which runs the batched step, with its window at 131,072 (wider than any pool here) and slots KV slots at
// precision kv ("" for the default), under a memory figure that holds positions positions of KV: the fit sizes each slot's share from it,
// and the pool, not the window, bounds one conversation.
func kvCapLoad(t *testing.T, slots, positions int, kv string) (*decoder.Model, *metalResident) {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := writeMC3Fixture(t, 131072)
	o := decoder.Options{Quant: mc3TestQuant, Backend: "metal", KVPrecision: kv}
	probe, err := decoder.Load(dir, o)
	if err != nil {
		t.Fatal(err)
	}
	budget := fitBudget(t, probe, positions)
	probe.Close()
	live := metalLiveAvailable
	metalLiveAvailable = func() int64 { return budget } // the guard's budget and the fit's: both read it
	t.Cleanup(func() { metalLiveAvailable = live })
	o.ResidentKVSlots = slots
	m, err := decoder.Load(dir, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("the build declined: %s", m.ResidentDecline())
	}
	if got := a.r.kvSlotCount(); got != slots {
		t.Fatalf("built %d KV slots, want %d", got, slots)
	}
	return m, a
}

// kvCapIds is n token ids, fixed by salt.
func kvCapIds(n, salt, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*31 + salt*17) % vocab
	}
	return ids
}

// kvCapPrefill binds slot s and prefills ids from position from in 1,024-row passes, returning a copy of the last logits or the first error.
func kvCapPrefill(m *decoder.Model, a *metalResident, s int, ids []int, from int) ([]float32, error) {
	if err := a.r.useKVSlot(s); err != nil {
		return nil, err
	}
	var lg []float32
	for c := 0; c < len(ids); c += 1024 {
		var e [][]float32
		for _, id := range ids[c:min(c+1024, len(ids))] {
			e = append(e, slices.Clone(m.EmbedResidentForTest(id)))
		}
		var err error
		if lg, err = a.PrefillLast(context.Background(), e, from+c); err != nil {
			return nil, err
		}
	}
	return slices.Clone(lg), nil
}

// kvNullZero reports whether every layer's block 0, K and V, still reads zero: a row written through an unmapped entry lands there.
func kvNullZero(t *testing.T, r *resident) bool {
	t.Helper()
	for l := range r.layers {
		if r.layers[l].kvShared || r.kvSlotBytes[l] == 0 {
			continue
		}
		for _, b := range []Buffer{r.kc[l], r.vc[l]} {
			part := make([]byte, r.kvNullBytes[l])
			if err := gpu.Download(b.At(0), part); err != nil {
				t.Fatal(err)
			}
			if slices.ContainsFunc(part, func(x byte) bool { return x != 0 }) {
				return false
			}
		}
	}
	return true
}

// kvSameBits is the first index where x and y differ bit for bit, or -1.
func kvSameBits(x, y []float32) int {
	if len(x) != len(y) {
		return 0
	}
	for i := range x {
		if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
			return i
		}
	}
	return -1
}

// TestKVPages_oneConversationUsesThePool is KV5-6's G3a: two slots, the other idle and empty, and one conversation that prefills and
// decodes until it stops, through the decoder's own generation. Pass: its cap is the pool's positions, past its slot's share; it reaches
// at least 90% of them; and it stops as at the context cap (finish_reason "length": the budget clamped, no error).
func TestKVPages_oneConversationUsesThePool(t *testing.T) {
	m, a := kvCapLoad(t, 2, 20000, "")
	r, pool := a.r, a.KVPool().Positions
	t.Logf("share %d positions, pool %d, one conversation's cap %d", r.kvShareCtx, pool, r.ctxCap)
	if r.ctxCap != min(pool, metalCtxCapMax) || r.ctxCap <= r.kvShareCtx {
		t.Fatalf("one conversation's cap is %d; want the pool's %d, past the slot's %d-position share", r.ctxCap, pool, r.kvShareCtx)
	}
	prompt := kvCapIds(r.ctxCap-1024, 1, m.Config().VocabSize)
	out, gen := m.Generate(context.Background(), prompt, pool, decoder.SamplingParams{Temperature: 0})
	n := 0
	for range out {
		n++
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("the conversation ended with an error, not as at the cap: %v", err)
	}
	reached := len(prompt) + n
	t.Logf("reached %d positions (%.1f%% of the pool), budget %d clamped %v", reached, 100*float64(reached)/float64(pool), gen.Budget, gen.BudgetClamped)
	if reached < pool*9/10 {
		t.Errorf("reached %d positions, under 90%% of the pool's %d", reached, pool)
	}
	if !gen.BudgetClamped || n != gen.Budget {
		t.Errorf("emitted %d with budget %d (clamped %v): the turn did not end as at the context cap", n, gen.Budget, gen.BudgetClamped)
	}
}

// kvPressureIdle is G3d's first step, on a fresh load of four slots: slots 1, 2 and 3, idle, hold conversations a little under their
// share, written in that order; slot 0, running, prefills past its share by every free block and half an idle slot's, so it needs one
// idle slot's blocks and not two. It returns the load, the idle conversations' length and their logits.
func kvPressureIdle(t *testing.T) (*decoder.Model, *metalResident, int, map[int][]float32) {
	t.Helper()
	m, a := kvCapLoad(t, 4, 4*decoder.MetalCtxDefault, "")
	vocab, share := m.Config().VocabSize, a.r.kvShareBlocks*decoder.KVBlockRows
	conv := share - 1024
	first := map[int][]float32{}
	for _, s := range []int{1, 2, 3} {
		lg, err := kvCapPrefill(m, a, s, kvCapIds(conv, s, vocab), 0)
		if err != nil {
			t.Fatalf("slot %d: %v", s, err)
		}
		first[s] = lg
	}
	a.SetKVSlotBusy(0, true)
	if need := 4*share - 3*conv + conv/2; need >= 4*share {
		t.Fatalf("the borrower's %d positions leave no room to grow in a %d-position pool", need, 4*share)
	} else if _, err := kvCapPrefill(m, a, 0, kvCapIds(need, 9, vocab), 0); err != nil {
		t.Fatalf("slot 0 borrowing to %d: %v", need, err)
	}
	return m, a, conv, first
}

// TestKVPages_pressureFollowsD5 is KV5-6's G3d, D5's order on Metal: four slots; three idle ones hold conversations a little under their
// share; the fourth, running, grows past its share. Pass: (1) the idle slots are released least recently written first, each then
// answers 0 to ReusableKVSlot for its old prompt and declines as a CopyKV source, as does a range past what a slot wrote, and that prompt
// prefilled again gives the logits it gave before; (2) the borrower, with no idle slot left and another slot running within its share,
// gets ErrKVPoolFull; (3) a running conversation within its share whose first token arrives in a batched step beside the borrower's
// takes the borrower's blocks back, and the step launches nothing (ErrKVPoolFull, block 0 still zero); its prompt then gets the logits
// it gets alone, and the borrower answers no reuse, declines as a CopyKV source and gets ErrKVPoolFull on its next write. With
// kvEvictOrderPlantForTest (most recently written first), (1) goes red.
func TestKVPages_pressureFollowsD5(t *testing.T) {
	func() {
		kvEvictOrderPlantForTest = true
		defer func() { kvEvictOrderPlantForTest = false }()
		_, pa, conv, _ := kvPressureIdle(t)
		if pa.ReusableKVSlot(1, conv) != conv || pa.ReusableKVSlot(3, conv) != 0 {
			t.Errorf("planted: slot 1 answers %d, slot 3 %d; the plant should release the most recently written slot, 3",
				pa.ReusableKVSlot(1, conv), pa.ReusableKVSlot(3, conv))
		}
	}()
	m, a, conv, first := kvPressureIdle(t)
	r, vocab := a.r, m.Config().VocabSize
	share := r.kvShareBlocks * decoder.KVBlockRows
	for s, w := range map[int]int{1: 0, 2: conv, 3: conv} {
		if got := a.ReusableKVSlot(s, conv); got != w {
			t.Errorf("slot %d answers %d for its %d-token prompt, want %d (released least recently written first)", s, got, conv, w)
		}
	}
	if r.kvEvictions != 1 {
		t.Errorf("%d idle slots released, want 1", r.kvEvictions)
	}
	var d *decoder.KVCopyDecline
	if err := a.CopyKV(1, 3, 0, decoder.KVBlockRows); !errors.As(err, &d) {
		t.Errorf("a copy from the released slot 1: %v, want a KVCopyDecline", err)
	}
	if err := a.CopyKV(2, 3, 0, share); !errors.As(err, &d) {
		t.Errorf("a copy of slot 2's rows to %d, past the %d it wrote: %v, want a KVCopyDecline", share, conv, err)
	}
	if got := a.ReusableKVSlot(3, conv); got != conv {
		t.Errorf("after two declined copies slot 3 answers %d for its %d rows", got, conv)
	}
	if lg, err := kvCapPrefill(m, a, 1, kvCapIds(conv, 1, vocab), 0); err != nil {
		t.Errorf("slot 1's prompt again: %v", err)
	} else if j := kvSameBits(lg, first[1]); j >= 0 {
		t.Errorf("slot 1's prompt prefilled again differs at %d from its first run", j)
	}
	// (2) slot 3 starts running (within its share, so nothing can be taken from it); the borrower grows to the whole pool, which it can
	// only get by releasing every idle slot, and then not.
	a.SetKVSlotBusy(3, true)
	need := 4*share - 3*conv + conv/2
	grow := kvCapIds(4*share, 9, vocab)
	if _, err := kvCapPrefill(m, a, 0, grow[need:], need); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Fatalf("the borrower past every idle slot: got %v, want ErrKVPoolFull", err)
	}
	// (3) a conversation within its share arrives while the borrower holds the pool, its first token in a step beside the borrower's.
	held, at := r.kvHeld(0), r.kvSlotSt[0].valid
	a.SetKVSlotBusy(2, true)
	emb := func(id int) []float32 { return slices.Clone(m.EmbedResidentForTest(id)) }
	if why := r.batchIneligible(); why != "" {
		t.Fatalf("this resident runs no batched step (%s): the reclaim inside a step is not exercised", why)
	}
	if _, _, err := r.forwardMulti([]batchSeq{{slot: 0, pos: at, emb: emb(5)}, {slot: 2, pos: 0, emb: emb(6)}}); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Errorf("a step whose second row takes the first row's blocks back: got %v, want ErrKVPoolFull", err)
	}
	if !kvNullZero(t, r) {
		t.Error("the step launched with a row whose blocks were taken back: block 0 was written")
	}
	alone := kvCapIds(conv, 7, vocab)
	got, err := kvCapPrefill(m, a, 2, alone, 0)
	if err != nil {
		t.Fatalf("a conversation within its share failed while the borrower held %d blocks: %v", held, err)
	}
	mr, ar := kvCapLoad(t, 4, 4*decoder.MetalCtxDefault, "")
	ref, err := kvCapPrefill(mr, ar, 0, alone, 0)
	if err != nil {
		t.Fatal(err)
	}
	if j := kvSameBits(got, ref); j >= 0 {
		t.Errorf("the conversation within its share differs at %d from the same prompt alone", j)
	}
	if n := a.ReusableKVSlot(0, decoder.KVBlockRows); n != 0 {
		t.Errorf("the stopped borrower answers %d for its first block, want 0", n)
	}
	if err := a.CopyKV(0, 1, 0, decoder.KVBlockRows); !errors.As(err, &d) {
		t.Errorf("a copy from the stopped borrower's first block: %v, want a KVCopyDecline", err)
	}
	if _, err := kvCapPrefill(m, a, 0, grow[need:need+64], need); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Errorf("the stopped borrower's next write: got %v, want ErrKVPoolFull", err)
	}
	t.Logf("share %d positions; the borrower held %d blocks when the conversation within its share arrived", share, held)
}

// TestKVPages_poolDetail: KVPool, which the serve banner prints, names this resident's pool at f16 and int8 KV: every slot's share as
// positions, the share each slot is guaranteed, 64-row blocks in the one full group at the cache's precision, and the bytes the kv ledger
// line holds, within its page rounding. The model reports the same (decoder.Model.ResidentKVPool) and can share rows across
// conversations (KVShareState "on"), which serve check's shared-preamble row needs.
func TestKVPages_poolDetail(t *testing.T) {
	for _, kv := range []string{"", "i8"} {
		t.Run("kv="+kv, func(t *testing.T) {
			m, a := kvCapLoad(t, 2, 20000, kv)
			r, p := a.r, m.ResidentKVPool()
			want := map[string]string{"": "f16", "i8": "i8"}[kv]
			if p.Positions != 2*r.kvShareBlocks*decoder.KVBlockRows || p.Guaranteed != r.kvShareCtx || p.BlockRows != decoder.KVBlockRows ||
				p.WindowRows != 0 || !slices.Equal(p.Groups, []string{"full"}) || p.Precision != want {
				t.Errorf("ResidentKVPool = %+v; want %d positions, %d guaranteed, 64-row blocks, no window, the full group, %s", p,
					2*r.kvShareBlocks*decoder.KVBlockRows, r.kvShareCtx, want)
			}
			nL := m.Config().NumLayers
			charged := metalLedgerOf(r.d).Usage().ByLine[mem.LineKV]
			if over := charged - p.Bytes; p.Bytes <= 0 || over < 0 || over > int64(2*nL+1)*16<<10 {
				t.Errorf("the pool names %d B; the kv ledger line holds %d B", p.Bytes, charged)
			}
			if st, why := m.KVShareState(); st != "on" {
				t.Errorf("KVShareState = %q (%s), want on", st, why)
			}
		})
	}
}
