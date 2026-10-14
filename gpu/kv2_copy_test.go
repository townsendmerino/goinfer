//go:build gpu

package gpu

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// kv2Slot is every per-sequence buffer of slot s (slotKVBuffers, each once) as 32-bit words, with each buffer's words
// per position.
func kv2Slot(rd *residentDecoder, s int) (bufs [][]uint32, rowWords []int) {
	seen := map[any]bool{}
	for l := range rd.slots[s].rm.layers {
		for _, p := range slotKVBuffers(&rd.slots[s].rm.layers[l]) {
			b := *p
			if b == nil || seen[b] {
				continue
			}
			seen[b] = true
			words := int(b.GetSize() / 4)
			bufs = append(bufs, rd.c.readWords(b, words))
			rowWords = append(rowWords, words/rd.ctxCap)
		}
	}
	return bufs, rowWords
}

// kv2Diff is the first buffer and position where a and b differ in rows [lo, hi), or -1, -1.
func kv2Diff(a, b [][]uint32, rowWords []int, lo, hi int) (int, int) {
	for i := range a {
		w := rowWords[i]
		for p := lo; p < hi; p++ {
			if !slices.Equal(a[i][p*w:(p+1)*w], b[i][p*w:(p+1)*w]) {
				return i, p
			}
		}
	}
	return -1, -1
}

func kv2Embs(hidden, n int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, hidden)
		for j := range out[i] {
			out[i][j] = float32(rng.NormFloat64())
		}
	}
	return out
}

// kv2Prefill writes embs from start on the bound slot as residentPrefillSeed would: the batched pass, or token by token
// where it declines (int8 KV, MLA). It returns the last logits.
func kv2Prefill(t *testing.T, rd *residentDecoder, embs [][]float32, start int) []float32 {
	t.Helper()
	if lg, err := rd.PrefillLast(context.Background(), embs, start); err == nil {
		return slices.Clone(lg)
	}
	var lg []float32
	for i, e := range embs {
		var err error
		if lg, err = rd.Forward(e, start+i); err != nil {
			t.Fatal(err)
		}
	}
	return slices.Clone(lg)
}

var kv2Fixtures = []struct {
	name, dir string
	opts      decoder.Options
}{
	{"llama-tiny f32 KV", "llama-tiny", decoder.Options{Quant: "int4"}},
	{"llama-tiny f16 KV", "llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "f16"}},
	{"llama-tiny i8 KV", "llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "i8"}},
	{"deepseek-tiny (MLA latent)", "deepseek-tiny", decoder.Options{Quant: "int4"}},
}

// TestKV2_copyKVExact is G2 on WebGPU (docs/measurements/kv2-2026-10-14.md): after CopyKV(src, dst, start, end) dst's
// rows in the range equal src's, dst's other rows and every other slot are unchanged, and a range past the cache is
// declined with nothing written.
func TestKV2_copyKVExact(t *testing.T) {
	for _, fx := range kv2Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", fx.dir), fx.opts, 3)
			defer m.Close()
			if len(rd.slots) != 3 || rd.ctxCap < 64 {
				t.Skipf("%d slots, a %d-position cache; the test needs 3 and 64", len(rd.slots), rd.ctxCap)
			}
			hidden, _, _, _, _, _, _ := m.Dims()
			for s, n := range []int{min(120, rd.ctxCap), min(120, rd.ctxCap), 40} {
				if err := rd.UseKVSlot(s); err != nil {
					t.Fatal(err)
				}
				kv2Prefill(t, rd, kv2Embs(hidden, n, int64(10+s)), 0)
			}
			for _, tc := range []struct{ src, dst, start, end int }{{0, 1, 0, 64}, {0, 1, 0, 128}, {1, 2, 64, 128}} {
				if tc.end > rd.ctxCap {
					continue
				}
				var before, after [3][][]uint32
				var rw []int
				for s := range 3 {
					before[s], rw = kv2Slot(rd, s)
				}
				if err := rd.CopyKV(tc.src, tc.dst, tc.start, tc.end); err != nil {
					t.Fatalf("CopyKV%v: %v", tc, err)
				}
				for s := range 3 {
					after[s], _ = kv2Slot(rd, s)
				}
				if b, p := kv2Diff(after[tc.dst], before[tc.src], rw, tc.start, tc.end); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d is not src's", tc, b, p)
				}
				if b, p := kv2Diff(after[tc.dst], before[tc.dst], rw, 0, tc.start); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d below the range changed", tc, b, p)
				}
				if b, p := kv2Diff(after[tc.dst], before[tc.dst], rw, tc.end, rd.ctxCap); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d above the range changed", tc, b, p)
				}
				for s := range 3 {
					if s != tc.dst {
						if b, p := kv2Diff(after[s], before[s], rw, 0, rd.ctxCap); b >= 0 {
							t.Errorf("CopyKV%v: slot %d buffer %d row %d changed", tc, s, b, p)
						}
					}
				}
			}
			before, rw := kv2Slot(rd, 2)
			var d *decoder.KVCopyDecline
			if err := rd.CopyKV(0, 2, 0, rd.ctxCap+64); !errors.As(err, &d) {
				t.Errorf("a range past the cache: %v, want a KVCopyDecline", err)
			}
			after, _ := kv2Slot(rd, 2)
			if b, p := kv2Diff(after, before, rw, 0, rd.ctxCap); b >= 0 {
				t.Errorf("a declined copy wrote buffer %d row %d", b, p)
			}
		})
	}
}

// TestKV2_copyThenSuffixEqualsWhole is G0's split arms and G3's device check on WebGPU: 64 rows copied from a slot that
// prefilled a longer prompt sharing them, then the rest, gives the whole prefill's KV and last logits bit for bit. The
// rest runs as production would: a batched pass for 36 rows, and decode token by token for 5 (residentPrefillSeed sends
// a suffix under 8 down the sequential path on a resident without PrefillTailExact). A control with a different source
// row must differ from that row on.
func TestKV2_copyThenSuffixEqualsWhole(t *testing.T) {
	for _, fx := range kv2Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", fx.dir), fx.opts, 2)
			defer m.Close()
			if len(rd.slots) != 2 || rd.ctxCap < 128 {
				t.Skipf("%d slots, a %d-position cache", len(rd.slots), rd.ctxCap)
			}
			hidden, _, _, _, _, _, _ := m.Dims()
			shared := kv2Embs(hidden, 80, 1)
			src := append(slices.Clone(shared), kv2Embs(hidden, 30, 2)...)
			run := func(srcEmbs [][]float32, prompt [][]float32, sequential bool) ([]float32, [][]uint32, []int) {
				if err := rd.UseKVSlot(0); err != nil {
					t.Fatal(err)
				}
				kv2Prefill(t, rd, srcEmbs, 0)
				if err := rd.CopyKV(0, 1, 0, 64); err != nil {
					t.Fatal(err)
				}
				if err := rd.UseKVSlot(1); err != nil {
					t.Fatal(err)
				}
				var lg []float32
				var err error
				if sequential {
					for p := 64; p < len(prompt); p++ {
						if lg, err = rd.Forward(prompt[p], p); err != nil {
							t.Fatal(err)
						}
					}
				} else {
					lg = kv2Prefill(t, rd, prompt[64:], 64)
				}
				kv, rw := kv2Slot(rd, 1)
				return slices.Clone(lg), kv, rw
			}
			for _, tc := range []struct {
				n          int
				sequential bool
			}{{100, false}, {69, true}} {
				prompt := append(slices.Clone(shared[:64]), kv2Embs(hidden, tc.n-64, 3)...)
				if err := rd.UseKVSlot(1); err != nil {
					t.Fatal(err)
				}
				whole := kv2Prefill(t, rd, prompt, 0)
				want, _ := kv2Slot(rd, 1)
				got, kv, rw := run(src, prompt, tc.sequential)
				if b, p := kv2Diff(kv, want, rw, 0, tc.n); b >= 0 {
					t.Errorf("%d tokens (sequential suffix %v): copied + suffix KV differs from whole at buffer %d row %d", tc.n, tc.sequential, b, p)
				}
				for i := range whole {
					if math.Float32bits(whole[i]) != math.Float32bits(got[i]) {
						t.Errorf("%d tokens (sequential suffix %v): last logits differ at %d (%g vs %g)", tc.n, tc.sequential, i, got[i], whole[i])
						break
					}
				}
			}
			prompt := append(slices.Clone(shared[:64]), kv2Embs(hidden, 36, 3)...)
			if err := rd.UseKVSlot(1); err != nil {
				t.Fatal(err)
			}
			kv2Prefill(t, rd, prompt, 0)
			want, _ := kv2Slot(rd, 1)
			pert := slices.Clone(src)
			pert[40] = kv2Embs(hidden, 1, 99)[0]
			_, kv, rw := run(pert, prompt, false)
			if b, p := kv2Diff(kv, want, rw, 0, 100); b < 0 || p != 40 {
				t.Errorf("control: first difference at buffer %d row %d, want row 40", b, p)
			}
		})
	}
}
