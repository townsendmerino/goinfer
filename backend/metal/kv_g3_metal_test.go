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
		// prefill is PrefillLast, or for a family without a batched prefill the rows decoded one at a time, as the decoder fills them: a
		// write must start where the slot's sequence ends, so a declined prefill cannot be skipped.
		prefill := func(s int, e [][]float32, at int, what string) {
			l, err := a.PrefillLast(context.Background(), e, at)
			if err != nil {
				out.refused = append(out.refused, what+": "+err.Error())
				for i, x := range e {
					if l, err = a.Forward(x, at+i); err != nil {
						break
					}
				}
				what += " decoded"
			}
			add(what, l, err)
		}
		slots := r.kvSlotCount()
		for s := range slots {
			if err := r.useKVSlot(s); err != nil {
				return nil, err.Error()
			}
			prefill(s, embs(s, prompt), 0, fmt.Sprintf("slot %d prefill", s))
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
			prefill(s, embs(s+9, turn2), at, fmt.Sprintf("slot %d second turn", s))
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

// TestKVPages_shareByReferenceLayerMajor is TestKVPages_shareByReference's write site check for the three layer-major prefills, which run
// only on their own families: Gemma 4's E-model prefill (gemma4-emodel-tiny), the paged generic MoE's (mixtral-tiny, paged at 3 slots) and
// the paged Gemma 4 MoE's (gemma4-moe-tiny, paged at 3 slots).
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
		{"paged-g4", "../../testdata/gemma4-moe-tiny", "3"},
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
				lm := func() int { return r.emodelLayerMajorRuns + r.moeLayerMajorRuns + r.g4LayerMajorRuns }
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
				t.Error("with the copy on write off, slot 0's rewind into the shared block left slot 1's K/V as it was: the test cannot see a missing copy on write")
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
// it gets alone, and the borrower answers no reuse, declines as a CopyKV source and gets ErrKVPoolFull on its next write, a prefill or
// any decode entry point. With kvEvictOrderPlantForTest (most recently written first), (1) goes red.
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
	// The refused pass gave back the blocks it took. Refill the pool to mid-block, so the borrower's next row needs no block.
	at := r.kvSlotSt[0].valid
	fill := (len(r.kvRefs)-1-r.kvBlocksInUse())*decoder.KVBlockRows - 32
	if _, err := kvCapPrefill(m, a, 0, grow[at:at+fill], at); err != nil {
		t.Fatalf("the borrower filling the pool: %v", err)
	}
	at += fill
	// (3) a conversation within its share arrives while the borrower holds the pool, its first token in a step beside the borrower's.
	held := r.kvHeld(0)
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
	if held, n := r.kvHeld(2), a.ReusableKVSlot(2, 1); held != 0 || n != 0 {
		t.Errorf("after the step launched nothing slot 2 holds %d blocks and answers %d for its first row; want 0 and 0", held, n)
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
	// Decode's entry points map before they encode, so the borrower's next token gets ErrKVPoolFull from each of them.
	if _, err := a.Forward(emb(5), need); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Errorf("the stopped borrower's Forward: got %v, want ErrKVPoolFull", err)
	}
	if err := a.ForwardNoLogits(emb(5), need); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Errorf("the stopped borrower's ForwardNoLogits: got %v, want ErrKVPoolFull", err)
	}
	if _, err := a.ForwardSample(emb(5), need, 0, 1, 0); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Errorf("the stopped borrower's ForwardSample: got %v, want ErrKVPoolFull", err)
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

// TestKVPages_failedWriteHoldsNoBlocks: a write past 0 that the pool refuses partway (a borrower past its share, the other slot running)
// leaves the pool as it was: the blocks it took before the refusal go back, and the slot keeps the sequence it had. A copy on write whose
// copy fails gives back the block it took, and the shared block stays mapped.
func TestKVPages_failedWriteHoldsNoBlocks(t *testing.T) {
	m, a := kvCapLoad(t, 2, 20000, "")
	r, vocab := a.r, m.Config().VocabSize
	share := r.kvShareBlocks * decoder.KVBlockRows
	if _, err := kvCapPrefill(m, a, 1, kvCapIds(share-5*decoder.KVBlockRows, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	a.SetKVSlotBusy(1, true)
	ids := kvCapIds(share+10*decoder.KVBlockRows, 2, vocab)
	if _, err := kvCapPrefill(m, a, 0, ids[:share], 0); err != nil {
		t.Fatal(err)
	}
	before := r.kvBlocksInUse()
	// One pass of ten blocks from the share's end: five are free, and then nothing is.
	if _, err := kvCapPrefill(m, a, 0, ids[share:], share); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Fatalf("the write past every free block: got %v, want ErrKVPoolFull", err)
	}
	if got, held := r.kvBlocksInUse(), r.kvHeld(0); got != before || held != r.kvShareBlocks {
		t.Errorf("after the refused write the pool holds %d blocks (%d before) and slot 0 %d (want %d)", got, before, held, r.kvShareBlocks)
	}
	if got := a.ReusableKVSlot(0, share); got != share {
		t.Errorf("slot 0 answers %d for the %d rows it held before the refused write", got, share)
	}

	// The copy on write's failure, on a block slot 1 shares with slot 0.
	m2, a2 := kvCapLoad(t, 2, 20000, "")
	r2 := a2.r
	pre := kvCapIds(2*decoder.KVBlockRows, 3, vocab)
	if _, err := kvCapPrefill(m2, a2, 0, pre, 0); err != nil {
		t.Fatal(err)
	}
	if err := a2.CopyKV(0, 1, 0, 2*decoder.KVBlockRows); err != nil {
		t.Fatal(err)
	}
	before, shared := r2.kvBlocksInUse(), *r2.kvRow(1, 1)
	kvCopyFailForTest = true
	_, err := kvCapPrefill(m2, a2, 1, kvCapIds(20, 4, vocab), 100)
	kvCopyFailForTest = false
	if err == nil {
		t.Fatal("a copy on write whose copy failed: no error")
	}
	if got := r2.kvBlocksInUse(); got != before || *r2.kvRow(1, 1) != shared {
		t.Errorf("after the failed copy the pool holds %d blocks (%d before), and slot 1's block 1 is %d (the shared block %d)",
			got, before, *r2.kvRow(1, 1), shared)
	}
}

// TestKVPages_refusedWriteLeavesTheSlot: a refused write leaves the slot as it was, its sequence, its blocks and what it shares, however
// far it had got: (a) a rewind into a shared block whose copy on write finds no block; (b) a write from inside a shared block through the
// next one and past, whose first copy is taken and whose second take is refused.
func TestKVPages_refusedWriteLeavesTheSlot(t *testing.T) {
	a, r := kv2Resident(t, false)
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 128, 1), 0); err != nil {
		t.Fatal(err)
	}
	a.SetKVSlotBusy(0, true) // running, so a refused take cannot release it: a release a mapping causes stands
	defer a.SetKVSlotBusy(0, false)
	same := func(what string, tab [2]uint32, valid, used int) {
		t.Helper()
		if got := [2]uint32{*r.kvRow(1, 0), *r.kvRow(1, 1)}; got != tab || r.kvSlotSt[1].valid != valid || r.kvBlocksInUse() != used {
			t.Errorf("%s: slot 1 maps %v with %d rows and the pool holds %d blocks; before the refused write %v, %d and %d", what, got,
				r.kvSlotSt[1].valid, r.kvBlocksInUse(), tab, valid, used)
		}
	}
	// (a) slot 1 shares slot 0's block 0 and holds its own block 1.
	if err := a.CopyKV(0, 1, 0, 64); err != nil {
		t.Fatal(err)
	}
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 64, 2), 64); err != nil {
		t.Fatal(err)
	}
	tab, used := [2]uint32{*r.kvRow(1, 0), *r.kvRow(1, 1)}, r.kvBlocksInUse()
	if tab[0] != *r.kvRow(0, 0) || tab[1] == 0 || r.kvSlotSt[1].valid != 128 {
		t.Fatalf("slot 1 maps %v with %d rows; the test needs slot 0's block 0, a block of its own and 128 rows", tab, r.kvSlotSt[1].valid)
	}
	kvTakeFailForTest = true
	err := r.kvWrite(1, 10, 11)
	kvTakeFailForTest = false
	if err == nil {
		t.Fatal("(a) a rewind into the shared block with no block to copy into was accepted")
	}
	same("(a) the refused rewind", tab, 128, used)
	// (b) slot 1 shares both of slot 0's blocks.
	if err := a.CopyKV(0, 1, 0, 128); err != nil {
		t.Fatal(err)
	}
	tab, used = [2]uint32{*r.kvRow(1, 0), *r.kvRow(1, 1)}, r.kvBlocksInUse()
	kvTakeFailAfterForTest = 1
	err = r.kvWrite(1, 10, 200)
	kvTakeFailAfterForTest = -1
	if err == nil {
		t.Fatal("(b) a write whose second take was refused was accepted")
	}
	same("(b) the write refused after its first copy", tab, 128, used)
}

// TestKVPages_abortedStepLeavesItsSlots: a batched step that cannot map every row leaves each of its slots as it was, a slot that rewinds
// in it included: its sequence and its blocks are those it had before the step.
func TestKVPages_abortedStepLeavesItsSlots(t *testing.T) {
	a, r := kv2Resident(t, false)
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 128, 1), 0); err != nil {
		t.Fatal(err)
	}
	tab, used := [2]uint32{*r.kvRow(0, 0), *r.kvRow(0, 1)}, r.kvBlocksInUse()
	for _, s := range []int{0, 1} { // the step's slots, which the decoder marks busy
		a.SetKVSlotBusy(s, true)
		defer a.SetKVSlotBusy(s, false)
	}
	kvTakeFailForTest = true // slot 0's row needs no block; slot 1's, its first, finds none
	err := r.kvMapStep([]batchSeq{{slot: 0, pos: 10}, {slot: 1, pos: 0}})
	kvTakeFailForTest = false
	if !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Fatalf("the step whose row found no block: got %v, want ErrKVPoolFull", err)
	}
	if got := [2]uint32{*r.kvRow(0, 0), *r.kvRow(0, 1)}; got != tab || r.kvSlotSt[0].valid != 128 || r.kvBlocksInUse() != used || r.kvHeld(1) != 0 {
		t.Errorf("after the aborted step slot 0 maps %v with %d rows, slot 1 holds %d blocks and the pool %d; before it %v, 128, 0 and %d",
			got, r.kvSlotSt[0].valid, r.kvHeld(1), r.kvBlocksInUse(), tab, used)
	}
}

// TestKVPages_stepReleasesAnIdleBoundSlot: a batched step writes rows for the slots it names, not the bound one, so the bound slot is
// released like any other when it is idle and the least recently written. Three slots of 4,096: slot 2 (bound, idle) and slot 1 (idle)
// hold 3,072 rows each, slot 2 written first; slot 0, running, holds the rest; its next row needs a block, and slot 2's go.
func TestKVPages_stepReleasesAnIdleBoundSlot(t *testing.T) {
	m, a := kvCapLoad(t, 3, 3*decoder.MetalCtxDefault, "")
	r, vocab := a.r, m.Config().VocabSize
	share := r.kvShareBlocks * decoder.KVBlockRows
	conv := share - 1024
	for _, s := range []int{2, 1} {
		if _, err := kvCapPrefill(m, a, s, kvCapIds(conv, s, vocab), 0); err != nil {
			t.Fatalf("slot %d: %v", s, err)
		}
	}
	rest := (len(r.kvRefs)-1)*decoder.KVBlockRows - 2*conv
	if _, err := kvCapPrefill(m, a, 0, kvCapIds(rest, 9, vocab), 0); err != nil {
		t.Fatalf("slot 0 to the pool's end: %v", err)
	}
	a.SetKVSlotBusy(0, true)
	if err := r.useKVSlot(2); err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.forwardMulti([]batchSeq{{slot: 0, pos: rest, emb: slices.Clone(m.EmbedResidentForTest(5))}}); err != nil {
		t.Fatalf("slot 0's step row past the pool's end: %v", err)
	}
	if got1, got2 := a.ReusableKVSlot(1, conv), a.ReusableKVSlot(2, conv); got1 != conv || got2 != 0 {
		t.Errorf("slot 1 answers %d and slot 2 %d for their %d rows; want %d and 0 (slot 2, bound and idle, was written first)", got1, got2, conv, conv)
	}
}

// TestKVPages_truncationReleasesTheTail: a slot maps no block past its sequence. A write past 0 that ends a longer sequence (a prefix
// reuse) gives back the blocks past its end; a copy that ends a longer destination does the same; a copy declines rows its source does not
// hold any more, and a start its destination does not reach.
func TestKVPages_truncationReleasesTheTail(t *testing.T) {
	a, r := kv2Resident(t, false)
	bind := func(s int) {
		t.Helper()
		if err := r.useKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	bind(0)
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 640, 1), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 20, 2), 100); err != nil {
		t.Fatal(err)
	}
	if held, used := r.kvHeld(0), r.kvBlocksInUse(); held != 2 || used != 2 {
		t.Errorf("after a write ends slot 0's 640 rows at 120 it holds %d blocks and the pool %d; want 2 and 2", held, used)
	}
	var d *decoder.KVCopyDecline
	if err := a.CopyKV(0, 1, 0, 640); !errors.As(err, &d) {
		t.Errorf("a copy of 640 rows from a slot that holds 120: %v, want a KVCopyDecline", err)
	}
	if err := a.CopyKV(0, 1, 64, 128); !errors.As(err, &d) {
		t.Errorf("a copy from 64 into a slot that holds nothing: %v, want a KVCopyDecline", err)
	}
	bind(2)
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 640, 3), 0); err != nil {
		t.Fatal(err)
	}
	if err := a.CopyKV(0, 2, 0, 128); err != nil {
		t.Fatal(err)
	}
	if held, used := r.kvHeld(2), r.kvBlocksInUse(); held != 2 || used != 2 {
		t.Errorf("after a copy of 120 rows ends slot 2's 640 it holds %d blocks and the pool %d; want 2 and 2 (shared with slot 0)", held, used)
	}
}

// TestKVPages_stepMapsGuaranteedRowsFirst: a batched step maps the rows that write within their slot's share before the rows that borrow,
// so it gives the pool's blocks as running the rows alone would: one free block, a borrower's row and a guaranteed row each needing one.
// The guaranteed row takes it, the borrower's finds none, and the step launches nothing (ErrKVPoolFull) with the pool as it was and the
// borrower's sequence kept; mapped the other way, the guaranteed row would have reclaimed the borrower's blocks and stopped it.
func TestKVPages_stepMapsGuaranteedRowsFirst(t *testing.T) {
	m, a := kvCapLoad(t, 2, 20000, "")
	r, vocab := a.r, m.Config().VocabSize
	share := r.kvShareBlocks * decoder.KVBlockRows
	in := share - 5*decoder.KVBlockRows                                      // slot 1 within its share, at a block edge
	past := (len(r.kvRefs)-1)*decoder.KVBlockRows - in - decoder.KVBlockRows // slot 0 past its share, leaving one free block
	if _, err := kvCapPrefill(m, a, 1, kvCapIds(in, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := kvCapPrefill(m, a, 0, kvCapIds(past, 2, vocab), 0); err != nil {
		t.Fatal(err)
	}
	a.SetKVSlotBusy(0, true)
	a.SetKVSlotBusy(1, true)
	used := r.kvBlocksInUse()
	emb := slices.Clone(m.EmbedResidentForTest(5))
	_, _, err := r.forwardMulti([]batchSeq{{slot: 0, pos: past, emb: emb}, {slot: 1, pos: in, emb: emb}})
	if !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Fatalf("the step with one free block for two rows: got %v, want ErrKVPoolFull", err)
	}
	if got, n := r.kvBlocksInUse(), a.ReusableKVSlot(0, past); got != used || n != past {
		t.Errorf("after the step the pool holds %d blocks (%d before) and the borrower answers %d for its %d rows; want them unchanged", got, used, n, past)
	}
}

// TestMetalKVBytes_pricesTheRaisedTables: decoder.Model.MetalKVBytes prices what a load of several slots builds, the block tables at one
// conversation's raised cap included, exactly: two and four slots at f16. An int8 cache does not raise the cap (its prefill dequantizes
// the whole context into a scratch the fit does not price), and its KV is priced exactly too.
func TestMetalKVBytes_pricesTheRaisedTables(t *testing.T) {
	for _, c := range []struct {
		slots, positions int
		kv               string
		raised           bool
	}{{2, 20000, "", true}, {4, 4 * decoder.MetalCtxDefault, "", true}, {2, 20000, "i8", false}} {
		t.Run(fmt.Sprintf("slots=%d/kv=%s", c.slots, c.kv), func(t *testing.T) {
			m, a := kvCapLoad(t, c.slots, c.positions, c.kv)
			r := a.r
			if raised := r.ctxCap > r.kvShareCtx; raised != c.raised {
				t.Fatalf("cap %d over a %d-position share: raised %v, want %v", r.ctxCap, r.kvShareCtx, raised, c.raised)
			}
			if got, want := a.KVPool().Bytes, m.MetalKVBytes(c.slots, r.kvShareCtx); got != want {
				t.Errorf("the pools and tables take %d B; MetalKVBytes(%d, %d) prices %d B", got, c.slots, r.kvShareCtx, want)
			}
		})
	}
}

// TestKVPages_stepReclaimOutsideItLaunches: a batched step whose guaranteed row reclaims from a borrower the step does not name launches:
// only a reclaim of one of the step's own slots leaves a row without its block. Three slots of 4,096: slot 2, running but not in the
// step, borrows 40 blocks past its share; slot 1's row needs none; slot 0's row, within its share, needs one, and the pool is full.
func TestKVPages_stepReclaimOutsideItLaunches(t *testing.T) {
	m, a := kvCapLoad(t, 3, 3*decoder.MetalCtxDefault, "")
	r, vocab := a.r, m.Config().VocabSize
	B := decoder.KVBlockRows
	if r.kvShareBlocks != 64 {
		t.Skipf("the fixture's share is %d blocks; the case is laid out for 64", r.kvShareBlocks)
	}
	for _, c := range []struct{ slot, rows int }{{2, 104 * B}, {1, 60*B - 10}, {0, 28 * B}} {
		if _, err := kvCapPrefill(m, a, c.slot, kvCapIds(c.rows, c.slot, vocab), 0); err != nil {
			t.Fatalf("slot %d: %v", c.slot, err)
		}
	}
	if free := len(r.kvRefs) - 1 - r.kvBlocksInUse(); free != 0 {
		t.Fatalf("%d blocks free; the case needs a full pool", free)
	}
	for s := range 3 {
		a.SetKVSlotBusy(s, true)
	}
	emb := slices.Clone(m.EmbedResidentForTest(5))
	if _, _, err := r.forwardMulti([]batchSeq{{slot: 0, pos: 28 * B, emb: emb}, {slot: 1, pos: 60*B - 10, emb: emb}}); err != nil {
		t.Fatalf("a step whose row reclaimed from slot 2, which it does not name: %v, want it to launch", err)
	}
	if n := a.ReusableKVSlot(2, B); n != 0 {
		t.Errorf("the reclaimed borrower answers %d for its first block, want 0 (stopped)", n)
	}
}

// TestKVPages_stepRewindTrimsFirst: a slot whose step row rewinds into its own share gives its borrowed tail back before another row looks
// for a block, as its own write would alone: the other row takes a block that tail held, and nothing is reclaimed. Two slots of 9,984:
// slot 1 borrows 44 blocks past its share and rewinds to 5,000; slot 0, within its share, needs a block; the pool is full.
func TestKVPages_stepRewindTrimsFirst(t *testing.T) {
	m, a := kvCapLoad(t, 2, 20000, "")
	r, vocab := a.r, m.Config().VocabSize
	B := decoder.KVBlockRows
	borrow := (r.kvShareBlocks + 44) * B
	rest := (len(r.kvRefs)-1)*B - borrow
	if _, err := kvCapPrefill(m, a, 1, kvCapIds(borrow, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := kvCapPrefill(m, a, 0, kvCapIds(rest, 2, vocab), 0); err != nil {
		t.Fatal(err)
	}
	a.SetKVSlotBusy(0, true)
	a.SetKVSlotBusy(1, true)
	emb := slices.Clone(m.EmbedResidentForTest(5))
	if _, _, err := r.forwardMulti([]batchSeq{{slot: 0, pos: rest, emb: emb}, {slot: 1, pos: 5000, emb: emb}}); err != nil {
		t.Fatalf("a step with slot 1 rewinding into its share: %v, want it to launch", err)
	}
	if n := a.ReusableKVSlot(1, 5001); n != 5001 {
		t.Errorf("slot 1 answers %d for the 5,001 rows its rewind left, want 5001 (not stopped)", n)
	}
}

// TestKVPages_writePastTheSequenceRefused: a write that starts past a slot's sequence is refused, since the rows between would read block
// 0 or stale rows; so is an upload outside a pass (TestKVPages_uploadPass has the pass, where a windowed layer may start late).
func TestKVPages_writePastTheSequenceRefused(t *testing.T) {
	a, r := kv2Resident(t, false)
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 64, 1), 0); err != nil {
		t.Fatal(err)
	}
	used := r.kvBlocksInUse()
	if _, err := a.PrefillLast(context.Background(), kv2Embs(r, 40, 2), 200); err == nil {
		t.Error("a prefill from 200 into a slot that holds 64 rows was accepted")
	}
	if got, n := r.kvBlocksInUse(), a.ReusableKVSlot(0, 64); got != used || n != 64 {
		t.Errorf("after the refused prefill the pool holds %d blocks (%d before) and slot 0 answers %d for its 64 rows", got, used, n)
	}
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	used = r.kvBlocksInUse()
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil && !r.layers[l].kvShared {
			if err := a.UploadKV(l, 70, make([]float32, g.kvDim), make([]float32, g.kvDim)); err == nil {
				t.Errorf("layer %d: an upload at 70 into an empty slot, outside a pass, was accepted", l)
			}
		}
	}
	if got, n := r.kvBlocksInUse(), a.ReusableKVSlot(1, 71); got != used || n != 0 {
		t.Errorf("after the refused uploads the pool holds %d blocks (%d before) and slot 1 answers %d for 71 rows; want no block taken and 0", got, used, n)
	}
}

// kvUploadLayers is every layer of r that owns K/V: the layers an upload pass must land.
func kvUploadLayers(r *resident) []int {
	var ls []int
	for l := range r.layers {
		if r.layers[l].geom != nil && !r.layers[l].kvShared && r.kc[l] != (Buffer{}) {
			ls = append(ls, l)
		}
	}
	return ls
}

// TestKVPages_uploadPass: the UploadKV calls between BeginKVUpload and EndKVUpload are one write of the bound slot from position 0
// (decoder.ResidentKVUploadPass). On the MC3 fixture, every layer full attention: the slot holds the rows once every layer has landed the same
// end; a full-attention layer that starts past 0, a layer uploaded twice, or one ending elsewhere is refused at its call; a pass a layer
// missed, or one the caller fails, leaves the slot holding no block and answering 0. TestUploadKV_slidingLayerStartsLate has a windowed
// layer starting late through the decoder's own pass.
func TestKVPages_uploadPass(t *testing.T) {
	a, r := kv2Resident(t, false)
	p, ok := any(a).(decoder.ResidentKVUploadPass)
	if !ok {
		t.Fatal("Metal's resident has no upload pass (decoder.ResidentKVUploadPass)")
	}
	ls := kvUploadLayers(r)
	if len(ls) < 2 {
		t.Fatalf("the fixture has %d layers that own K/V; the test needs 2", len(ls))
	}
	rows := func(l, n int) []float32 { return make([]float32, n*r.layers[l].geom.kvDim) }
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	empty := func(what string) {
		t.Helper()
		if held, n := r.kvHeld(1), a.ReusableKVSlot(1, 71); held != 0 || n != 0 {
			t.Errorf("%s: slot 1 holds %d blocks and answers %d for 71 rows; want none and 0", what, held, n)
		}
	}
	// A whole pass.
	if err := p.BeginKVUpload(); err != nil {
		t.Fatal(err)
	}
	for _, l := range ls {
		if err := a.UploadKV(l, 0, rows(l, 71), rows(l, 71)); err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
	}
	if err := p.EndKVUpload(false); err != nil {
		t.Fatalf("a whole pass: %v", err)
	}
	if n := a.ReusableKVSlot(1, 71); n != 71 || r.kvHeld(1) != 2 {
		t.Errorf("after a whole pass slot 1 answers %d for 71 rows in %d blocks; want 71 in 2", n, r.kvHeld(1))
	}
	// Each refusal at its call, then the caller fails the pass.
	for _, c := range []struct {
		name string
		call func() error
	}{
		{"a full-attention layer from 64", func() error { return a.UploadKV(ls[1], 64, rows(ls[1], 7), rows(ls[1], 7)) }},
		{"a layer twice", func() error { return a.UploadKV(ls[0], 0, rows(ls[0], 71), rows(ls[0], 71)) }},
		{"a layer ending elsewhere", func() error { return a.UploadKV(ls[1], 0, rows(ls[1], 80), rows(ls[1], 80)) }},
	} {
		if err := p.BeginKVUpload(); err != nil {
			t.Fatal(err)
		}
		if err := a.UploadKV(ls[0], 0, rows(ls[0], 71), rows(ls[0], 71)); err != nil {
			t.Fatalf("%s: the first layer: %v", c.name, err)
		}
		if err := c.call(); err == nil {
			t.Errorf("%s: accepted", c.name)
		}
		if err := p.EndKVUpload(true); err != nil {
			t.Errorf("%s: a failed pass's end: %v", c.name, err)
		}
		empty(c.name)
	}
	// A pass a layer missed.
	if err := p.BeginKVUpload(); err != nil {
		t.Fatal(err)
	}
	if err := a.UploadKV(ls[0], 0, rows(ls[0], 71), rows(ls[0], 71)); err != nil {
		t.Fatal(err)
	}
	if err := p.EndKVUpload(false); err == nil {
		t.Error("a pass that left layers out ended without an error")
	}
	empty("a pass a layer missed")
}

// TestKVPages_setPosPastTheSequencePanics: setPos's own write (kvWriteMust) refuses to start past the slot's sequence, as any unmapped write
// does: production reaches setPos only after an entry point mapped the row. setPosAheadForTest maps first, for a test that picks the depth.
func TestKVPages_setPosPastTheSequencePanics(t *testing.T) {
	_, r := kv2Resident(t, false)
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("setPos(200) on an empty slot mapped the rows to 200 instead of panicking")
			}
		}()
		r.setPos(200)
	}()
	if held, st := r.kvHeld(0), &r.kvSlotSt[0]; held != 0 || st.valid != 0 {
		t.Errorf("after the refused setPos slot 0 holds %d blocks and %d rows; want none", held, st.valid)
	}
	r.setPosAheadForTest(200)
	if held, st := r.kvHeld(0), &r.kvSlotSt[0]; held != 4 || st.valid != 201 {
		t.Errorf("setPosAheadForTest(200) left slot 0 %d blocks and %d rows; want 4 and 201", held, st.valid)
	}
}

// TestKVPages_chainEndsAtTheLastTokenTaken: a greedy turn decoded through the chain ends the slot's sequence where the decoder's does (the
// prompt and every token it emitted, each fed: residentCommitIDs), and holds no block past it. A 62-token prompt and 2 tokens: the decoder's
// sequence is positions 0 to 63, and the chain's look-ahead at 64, queued past the last token taken, opens block 1 until the turn stops it.
func TestKVPages_chainEndsAtTheLastTokenTaken(t *testing.T) {
	m, a := kvCapLoad(t, 2, 2*decoder.MetalCtxDefault, "")
	r := a.r
	if why := r.greedyChainWhyNot(); why != "" {
		t.Skipf("no greedy chain on this resident: %s", why)
	}
	served := r.chainServed
	out, gen := m.Generate(context.Background(), kvCapIds(62, 5, m.Config().VocabSize), 2, decoder.SamplingParams{Temperature: 0})
	n := 0
	for range out {
		n++
	}
	if err := gen.Err(); err != nil {
		t.Fatal(err)
	}
	if n != 2 || r.chainServed == served {
		t.Fatalf("emitted %d tokens, %d of them through the chain; want 2, some through it, or the test reads nothing", n, r.chainServed-served)
	}
	s := r.kvSlot
	if held, st := r.kvHeld(s), &r.kvSlotSt[s]; held != 1 || st.valid != 64 {
		t.Errorf("slot %d holds %d blocks and %d rows after the turn; want 1 and 64 (the decoder's sequence, positions 0 to 63)", s, held, st.valid)
	}
}
