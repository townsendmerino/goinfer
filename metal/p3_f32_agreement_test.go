//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// p3Fixtures are the matrix's families that build a Metal resident (docs/tasks/task-kv-precision-everywhere-2026-10.md, P3 gates).
var p3Fixtures = []string{"llama-tiny", "qwen3moe-tiny", "gemma2-tiny", "cohere-tiny", "qwen25vl-tiny", "qwen35-tiny", "qwen3next-tiny",
	"nemotron-tiny", "deepseek-tiny"}

func p3FixtureDir(name string) string {
	dir := filepath.Join("..", "testdata", name)
	if _, err := os.Stat(dir); err != nil {
		return filepath.Join("..", "decoder", "testdata", name)
	}
	return dir
}

// TestP3_metalF32AgreesWithCPU is P3-G2 as amended (D6, docs/tasks/task-kv-precision-everywhere-2026-10.md §6): on every resident
// fixture, Metal at --kv f32 against the CPU at f32, teacher-forced over 32 positions, has a mean of the per-position max |logit
// diff| no larger than Metal at f16 (the arm that ships) has, within 5%, and a top-1 agreement no lower.
func TestP3_metalF32AgreesWithCPU(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const n = 32
	for _, name := range p3Fixtures {
		t.Run(name, func(t *testing.T) {
			dir := p3FixtureDir(name)
			cpu, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatal(err)
			}
			defer cpu.Close()
			_, _, _, _, _, _, vocab := cpu.Dims()
			ids := make([]int, n)
			for i := range ids {
				ids[i] = (i*37 + 11) % vocab
			}
			cache := cpu.NewCache(n + 8)
			var ref [][]float32
			for _, tok := range ids {
				l, err := cpu.ForwardForTest(tok, cache)
				if err != nil {
					t.Fatal(err)
				}
				ref = append(ref, append([]float32(nil), l...))
			}
			arm := func(kv string) (maxDiff, meanDiff float64, agree int) {
				m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, ResidentContext: 64})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				a, ok := m.ResidentForwardForTest().(*metalResident)
				if !ok {
					t.Fatalf("kv %q: not Metal-resident: %s", kv, m.ResidentDecline())
				}
				if a.r.kvF32 != (kv == "f32" && a.r.mla == nil) { // an MLA latent is f32 without the GQA flag (D5)
					t.Fatalf("kv %q: the resident's cache is f32 %v", kv, a.r.kvF32)
				}
				a.Reset()
				for i, tok := range ids {
					l, err := a.Forward(m.EmbedResidentForTest(tok), i)
					if err != nil {
						t.Fatal(err)
					}
					pos := 0.0
					for j := range l {
						pos = math.Max(pos, math.Abs(float64(l[j]-ref[i][j])))
					}
					maxDiff = math.Max(maxDiff, pos)
					meanDiff += pos / n
					if argmaxF(l) == argmaxF(ref[i]) {
						agree++
					}
				}
				return maxDiff, meanDiff, agree
			}
			d32, m32, a32 := arm("f32")
			d16, m16, a16 := arm("")
			fmt.Fprintf(os.Stderr, "[P3-G2] %-15s against the CPU at f32: Metal f32 mean %.6g (max %.6g), top-1 %d/%d; Metal f16 mean %.6g (max %.6g), %d/%d\n",
				name, m32, d32, a32, n, m16, d16, a16, n)
			if m32 > 1.05*m16 || a32 < a16 {
				t.Errorf("Metal at f32 is further from the CPU than at f16: mean per-position max|diff| %.6g against %.6g, top-1 %d against %d", m32, m16, a32, a16)
			}
		})
	}
}

// TestP3_metalF32BytesAndPaths is P3-G3 and P3-G4: on every resident fixture at --kv f32, the K/V buffers read back hold 4 B per
// element and equal ResidentKVBytes("metal", …); and each fast path declines naming the f32 cache.
func TestP3_metalF32BytesAndPaths(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const ctx = 100 // not a multiple of the 8-position padding
	for _, name := range p3Fixtures {
		t.Run(name, func(t *testing.T) {
			m, err := decoder.Load(p3FixtureDir(name), decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: "f32", ResidentContext: ctx})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
			}
			r := a.r
			if r.kvF32 != (r.mla == nil) || m.ResidentKVPrecision() != "f32" { // an MLA latent is f32 without the GQA flag (D5)
				t.Fatalf("the resident's cache is f32 %v, reported %q", r.kvF32, m.ResidentKVPrecision())
			}
			var got, elems int64
			for l := range r.kc {
				if r.layers[l].kvShared {
					continue
				}
				got += int64(r.kc[l].Len())
				if r.mla == nil { // an MLA layer's V aliases its latent buffer (metal/mla.go): count it once
					got += int64(r.vc[l].Len())
				}
				if g := r.layers[l].geom; g != nil && r.kc[l].Len() > 0 {
					elems += 2 * int64((r.ctxCap+7)/8*8) * int64(g.kvDim)
				}
			}
			if want := m.ResidentKVBytes("metal", r.ctxCap, false, false); got != want {
				t.Errorf("allocated %d B of K/V, ResidentKVBytes prices %d B", got, want)
			}
			if r.mla == nil && got != 4*elems {
				t.Errorf("allocated %d B for %d elements: not 4 B each", got, elems)
			}
			fmt.Fprintf(os.Stderr, "[P3-G3] %-15s f32: %d B of K/V (%d elements) = priced\n", name, got, elems)
			if r.mla != nil {
				return // MLA's own paths read the f32 latent; the f16 fast paths below never see it
			}
			if ok, why := a.PrefillPath(); ok && r.prefillOK {
				t.Errorf("PrefillPath batched (%s) under an f32 cache", why)
			}
			if B := a.PrefillBlockRows(); B != 0 {
				t.Errorf("PrefillBlockRows = %d under an f32 cache", B)
			}
			if why := r.batchIneligible(); why == "" {
				t.Error("batched decode eligible under an f32 cache")
			}
			if r.canUseAttnFAAt(0, 4096) {
				t.Error("flash-attention decode eligible under an f32 cache")
			}
		})
	}
}
