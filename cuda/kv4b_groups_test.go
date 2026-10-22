//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// kv4bGroups lists, per KV4b family group, the tiny fixtures its tier-0 gates run on (docs/measurements/kv4-cuda-2026-10-21.md §2, G5:
// G1's store identity and G3's delivered bytes; G1a's f16 arm is TestKV4_scrambledTables). A group joins this table in the commit that
// delivers its f16 cache.
var kv4bGroups = []struct {
	group    string
	fixtures []string // a name under testdata/, or a path from the repo root
	opts     decoder.Options
	windowed bool // windowed KV on, with the small test slack so the window group releases blocks
}{
	{"moe", []string{"qwen3moe-tiny", "qwen3moe-tiny-k3", "mixtral-tiny"}, decoder.Options{}, false},
	{"paged-experts", []string{"qwen3moe-tiny"}, decoder.Options{MoECacheExperts: true}, false},
	{"sliding-window", []string{"mistral-tiny-window", "gemma2-tiny", "gemma3-vl-tiny", "decoder/testdata/gptoss_tiny.gguf"}, decoder.Options{}, false},
	{"sliding-window/windowed", []string{"mistral-tiny-window", "gemma2-tiny", "gemma3-vl-tiny"}, decoder.Options{}, true},
	{"pairwise-rope", []string{"cohere-tiny", "cohere2-tiny"}, decoder.Options{}, false},
	{"pairwise-rope/windowed", []string{"cohere2-tiny"}, decoder.Options{}, true},
	{"m-rope", []string{"qwen25vl-tiny", "qwen3vl-tiny"}, decoder.Options{}, false},
}

// kv4bPair loads a tiny fixture twice, at -kv f32 and -kv f16, with the group's options and one slot.
func kv4bPair(t *testing.T, fx string, opts decoder.Options, windowed bool) (m32, m16 *decoder.Model, r32, r16 *cudaResident) {
	t.Helper()
	path := filepath.Join("..", "testdata", fx)
	if strings.Contains(fx, "/") {
		path = filepath.Join("..", fx)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	if windowed {
		withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, windowedKVTestSlack-windowedKVTestStep)
		opts.ResidentWindowedKV = true
	}
	load := func(kv string) (*decoder.Model, *cudaResident) {
		o := opts
		o.Backend, o.Quant, o.KVPrecision, o.ResidentKVSlots = "cuda", "int4", kv, 1
		m, err := decoder.Load(path, o)
		if err != nil {
			t.Fatalf("load %s (-kv %s): %v", fx, kv, err)
		}
		t.Cleanup(func() { m.Close() })
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("%s not CUDA-resident: %s", fx, m.ResidentDecline())
		}
		if windowed && !r.kvWin {
			t.Fatalf("%s: windowed KV did not engage", fx)
		}
		return m, r
	}
	m32, r32 = load("f32")
	m16, r16 = load("f16")
	if r32.kvF16 || !r16.kvF16 {
		t.Fatalf("resident flags: f32 arm kvF16=%v, f16 arm kvF16=%v", r32.kvF16, r16.kvF16)
	}
	return m32, m16, r32, r16
}

// TestKV4b_deliveredBytes is G3 for every delivered group: at -kv f16 each layer that keeps K/V holds it as f16, its pools hold the
// same number of elements as the f32 arm's (so half the bytes), and decoder.ResidentKVBytes prices exactly what the build allocates.
func TestKV4b_deliveredBytes(t *testing.T) {
	requireCUDADevice(t)
	for _, g := range kv4bGroups {
		for _, fx := range g.fixtures {
			t.Run(g.group+"/"+filepath.Base(fx), func(t *testing.T) {
				m32, m16, r32, r16 := kv4bPair(t, fx, g.opts, g.windowed)
				if got := m16.ResidentKVPrecision(); got != "f16" {
					t.Fatalf("-kv f16 reports %q", got)
				}
				half := 0
				for l := range r16.layers {
					Ly := &r16.layers[l]
					if Ly.noKV() || Ly.isMLA() || Ly.kvShared {
						continue
					}
					if !Ly.kvHalf || r32.layers[l].kvHalf {
						t.Fatalf("layer %d: kvHalf f16 arm %v, f32 arm %v", l, Ly.kvHalf, r32.layers[l].kvHalf)
					}
					if r16.kc[l].Len() != r32.kc[l].Len() || r16.vc[l].Len() != r32.vc[l].Len() {
						t.Fatalf("layer %d: element counts differ (%d/%d vs %d/%d)", l, r16.kc[l].Len(), r16.vc[l].Len(), r32.kc[l].Len(), r32.vc[l].Len())
					}
					half++
				}
				if half == 0 {
					t.Fatal("no layer holds an f16 cache: the check is vacuous")
				}
				b16, n16 := kvBytesForCap(r16.ctxCap, r16.layers), kvPoolNullBytes(r16.layers)
				b32, n32 := kvBytesForCap(r32.ctxCap, r32.layers), kvPoolNullBytes(r32.layers)
				if want := m16.ResidentKVBytes("cuda", r16.ctxCap, true, false); want != b16+n16 {
					t.Errorf("decoder prices the f16 cache at %d bytes, a one-slot build allocates %d", want, b16+n16)
				}
				if want := m32.ResidentKVBytes("cuda", r32.ctxCap, false, false); want != b32+n32 {
					t.Errorf("decoder prices the f32 cache at %d bytes, a one-slot build allocates %d", want, b32+n32)
				}
				t.Logf("%d f16 layers; one slot at %d positions: f32 %d bytes, f16 %d", half, r16.ctxCap, b32+n32, b16+n16)
			})
		}
	}
}

// TestKV4b_layer0StoreIsTheRoundedF32Store is G1's store half for every delivered group: layer 0's K and V come from the same embedding by
// the same f32 arithmetic in both arms, so the f16 arm's layer-0 cache equals the f32 arm's rounded to nearest even, bit for bit, at every
// position written by the batched prefill and by decode. The group's attention readers are the _kv16 kernels TestKV16_exactInputIdentity_*
// hold to the same bar.
func TestKV4b_layer0StoreIsTheRoundedF32Store(t *testing.T) {
	requireCUDADevice(t)
	for _, g := range kv4bGroups {
		for _, fx := range g.fixtures {
			t.Run(g.group+"/"+filepath.Base(fx), func(t *testing.T) {
				_, m16, r32, r16 := kv4bPair(t, fx, g.opts, g.windowed)
				if r16.layers[0].noKV() || r16.layers[0].isMLA() {
					t.Skip("layer 0 keeps no f16 K/V")
				}
				_, _, _, _, _, _, vocab := m16.Dims()
				nPrompt, nDecode := min(70, r16.ctxCap/2), 6
				embs := make([][]float32, nPrompt)
				for i := range embs {
					embs[i] = m16.EmbedResidentForTest(3 + (37*i)%(vocab-3))
				}
				ctx := context.Background()
				for _, r := range []*cudaResident{r32, r16} {
					if _, err := r.PrefillLast(ctx, embs, 0); err != nil {
						t.Fatalf("prefill: %v", err)
					}
					for i := range nDecode {
						if _, err := r.Forward(m16.EmbedResidentForTest(3+(11*i)%(vocab-3)), nPrompt+i); err != nil {
							t.Fatalf("decode %d: %v", i, err)
						}
					}
				}
				nPos := nPrompt + nDecode
				n := nPos * r32.layers[0].kvDim
				rows := func(r *cudaResident) (k, v []float32) {
					if err := r.do(func() error {
						if e := r.stream.Sync(); e != nil {
							return e
						}
						var e error
						k, v, e = r.kvRowsForTest(r.kvSlot, 0, 0, nPos)
						return e
					}); err != nil {
						t.Fatal(err)
					}
					return k, v
				}
				k32, v32 := rows(r32)
				k16, v16 := rows(r16)
				for _, c := range []struct {
					name string
					f    []float32
					h    []uint16
				}{{"K", k32, kvWordsToHalves(k16)}, {"V", v32, kvWordsToHalves(v16)}} {
					bad, nonzero := 0, 0
					for i := range n {
						if want := f32ToF16RNE(c.f[i]); c.h[i] != want {
							if bad < 3 {
								t.Errorf("layer 0 %s[%d]: f16 cache %#04x, f32 %g rounds to %#04x", c.name, i, c.h[i], c.f[i], want)
							}
							bad++
						}
						if c.f[i] != 0 {
							nonzero++
						}
					}
					if bad > 0 {
						t.Errorf("layer 0 %s: %d of %d elements differ from the rounded f32 store", c.name, bad, n)
					}
					if nonzero < n/2 {
						t.Errorf("layer 0 %s: only %d of %d elements are nonzero: the comparison is vacuous", c.name, nonzero, n)
					}
				}
			})
		}
	}
}
