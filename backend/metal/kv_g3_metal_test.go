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

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/loadflags"
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
	// long copies a fixture with its window raised to 2,048 positions, so a 300-token prompt fits a tiny model's cache; the weights are
	// hard-linked where the filesystem allows it, else copied.
	long := func(dir string) string {
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
	type arm struct {
		logits  [][]float32
		refused []string
		ran     map[string]int
		slots   int
		step    bool
		// stepRows is how many of logits' last rows are the batched step's.
		stepRows  int
		recurrent bool
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
			rows, _, err := r.forwardMulti(seqs)
			if decodeRef { // the reference: each row decoded on its own slot instead
				rows, err = nil, nil
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
		out.ran, out.slots, out.step, out.recurrent = maps.Clone(prefillAttnRanForTest), slots, r.batch != nil && slots >= 2, r.recurrentModel()
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
		// Not on a recurrent model: there a step row continuing a decode-built history differs from that slot's decode on main too
		// (docs/measurements/kv5-metal-2026-11/kv5-4.log, "Open"), and its step is held to decode by TestMC3Step_recurrentBitIdentical.
		if in.stepRows > 0 && !in.recurrent {
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
