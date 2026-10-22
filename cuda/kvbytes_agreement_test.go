//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentKVBytes_matchesCUDAAllocation pins the CUDA half of docs/tasks/task-memory-accounting-2026-09.md
// item 1: decoder.Model.ResidentKVBytes("cuda", …) — what Plan("cuda") and `fit` price — equals the K/V
// CUDA actually allocates, on real residents built from real fixtures, one per layout family:
//
//   - dense (llama-tiny, phi3-tiny, qwen2.5-0.5b), sliding window (mistral-tiny-window, gemma3-vl-tiny,
//     gemma3-1b: CUDA gives local layers the FULL context — the window is a mask, not a ring buffer),
//     per-layer geometry (gemma4-dense-twogeom, gemma4-moe), a DeltaNet hybrid (qwen35-tiny: no cache on
//     linear layers), and MLA (deepseek-tiny: ONE latent buffer per layer, not K and V);
//   - at every KV precision a caller can request — CUDA allocates f32 regardless, so the figure must not
//     move with kvF16 / kvI8.
//
// Without the "cuda" branch Plan's per-position formula prices MLA at twice the allocation and a
// requested f16 / i8 at a fraction of it (docs/measurements/memory-accounting-cuda-2026-09-25.md):
// deepseek-tiny and the f16 / i8 rows fail. Three links are checked, so a drift anywhere shows:
// the buffers' bytes == kvBytesForCap (the resident's own fit figure) == ResidentKVBytes("cuda"), and each
// buffer's driver allocation stays within one allocQuantumBytes (2 MiB) of its bytes.
//
// Every fixture is built with 2 resident KV slots (MC1, docs/tasks/task-concurrency-2026-09.md) over the paged pools
// (kvpool.go): the pools and tables hold exactly 2 × kvBytesForCap plus the null blocks, and ResidentKVBytes("cuda") is
// a one-slot load's figure, kvBytesForCap plus the null blocks. The DeltaNet hybrid must build one slot (recurrent state
// is not part of a slot); every other layout, 2.
func TestResidentKVBytes_matchesCUDAAllocation(t *testing.T) {
	home, _ := os.UserHomeDir()
	fixtures := []string{
		"../testdata/llama-tiny", "../testdata/phi3-tiny", "../testdata/mistral-tiny-window",
		"../testdata/gemma3-vl-tiny", "../testdata/gemma4-dense-twogeom-tiny", "../testdata/gemma4-moe-tiny", "../testdata/gemma4-emodel-tiny",
		"../testdata/qwen35-tiny", "../testdata/deepseek-tiny",
		filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		filepath.Join(home, "models", "gemma3-1b-q4_k_m.gguf"),
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	ran := 0
	for _, p := range fixtures {
		t.Run(filepath.Base(p), func(t *testing.T) {
			if _, err := os.Stat(p); err != nil {
				t.Skipf("no fixture at %s", p)
			}
			m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 2})
			if err != nil {
				t.Skipf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok || r == nil {
				t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
			}
			ran++
			wantSlots := 2
			if r.dnet != nil {
				wantSlots = 1
			}
			if r.KVSlots() != wantSlots {
				t.Fatalf("built %d KV slots with 2 requested, want %d", r.KVSlots(), wantSlots)
			}
			// The pools hold every slot's blocks and the null blocks, and the tables 4 B per block per group per slot (kvpool.go).
			forCap := kvBytesForCap(r.ctxCap, r.layers)
			null := kvPoolNullBytes(r.layers)
			bytes := int64(r.kvTabDev.Len()) * 4
			for l := range r.layers {
				if r.layers[l].kvShared {
					// An E-model's KV-shared layer holds its source's pools, not bytes of its own: counted once, at the owner. Assert the alias instead.
					if r.kc[l] != r.kc[r.layers[l].kvSrc] || r.vc[l] != r.vc[r.layers[l].kvSrc] {
						t.Errorf("layer %d: shared layer does not alias layer %d's pools", l, r.layers[l].kvSrc)
					}
					continue
				}
				for _, b := range []Buffer{r.kc[l], r.vc[l]} {
					n := int64(b.Len()) * 4 // Len is in elements: f32 (r.af), or f16 (r.akv) on a layer whose cache is half
					if r.layers[l].kvHalf {
						n = int64(b.Len()) * 2
					}
					bytes += n
					if alloc := (n + allocQuantumBytes - 1) / allocQuantumBytes * allocQuantumBytes; alloc-n >= allocQuantumBytes {
						t.Errorf("layer %d: a buffer of %d B rounds to %d B — more than one allocation quantum", l, n, alloc)
					}
				}
			}
			if want := int64(wantSlots)*forCap + null; bytes != want {
				t.Fatalf("the pools and tables hold %d B, %d slots of kvBytesForCap (%d B at ctx %d) and the null blocks (%d B) say %d B — the resident's own fit figure no longer matches its allocation",
					bytes, wantSlots, forCap, r.ctxCap, null, want)
			}
			// What the pricing says against what was allocated: ResidentKVBytes is a one-slot load, kvBytesForCap and the null blocks. The resident's own
			// precision prices exactly its allocation. Asking for the other one prices the other; f16 on a model the f16 kernels do not cover is still f32 on
			// the device. int8 is not a CUDA cache.
			asF32 := slices.Clone(r.layers)
			for i := range asF32 {
				asF32[i].kvHalf = false
			}
			asF16 := slices.Clone(r.layers)
			for i := range asF16 {
				asF16[i].kvHalf = !asF16[i].isMLA() && !asF16[i].noKV() && !asF16[i].kvShared
			}
			one := forCap + null
			f32Bytes := kvBytesForCap(r.ctxCap, asF32) + kvPoolNullBytes(asF32)
			f16Bytes := kvBytesForCap(r.ctxCap, asF16) + kvPoolNullBytes(asF16)
			for _, c := range []struct {
				name     string
				f16, i8b bool
				want     []int64
			}{{"f32", false, false, []int64{f32Bytes}}, {"f16 requested", true, false, []int64{one, f16Bytes, f32Bytes}}, {"i8 requested", false, true, []int64{f32Bytes}}} {
				got := m.ResidentKVBytes("cuda", r.ctxCap, c.f16, c.i8b)
				if !slices.Contains(c.want, got) {
					t.Errorf("%s: ResidentKVBytes(\"cuda\", %d) = %d B, want one of %v B (resident kvF16=%v, allocates %d B a slot)", c.name, r.ctxCap, got, c.want, r.kvF16, forCap)
				}
				if r.kvF16 && c.f16 && got != one {
					t.Errorf("an f16 resident prices %d B a one-slot load and ResidentKVBytes(f16) says %d B", one, got)
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no fixture built a CUDA resident here")
	}
}
