//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
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
// Before the "cuda" branch, Plan's per-position formula priced MLA at twice the allocation and a requested
// f16 / i8 at a half / ~0.28 of it (docs/measurements/memory-accounting-cuda-2026-09-25.md); deepseek-tiny
// and the f16 / i8 rows fail without the branch. Three links are checked, so a drift anywhere shows:
// the buffers' bytes == kvBytesForCap (the resident's own fit figure) == ResidentKVBytes("cuda"), and each
// buffer's driver allocation stays within one allocQuantumBytes (2 MiB) of its bytes.
//
// Every fixture is built with 2 resident KV slots (MC1, docs/tasks/task-concurrency-2026-09.md), and every slot is
// checked: each holds exactly kvBytesForCap, so the device holds slots × ResidentKVBytes("cuda"). The DeltaNet hybrid
// must build one slot (recurrent state is not part of a slot); every other layout, 2.
func TestResidentKVBytes_matchesCUDAAllocation(t *testing.T) {
	home, _ := os.UserHomeDir()
	fixtures := []string{
		"../testdata/llama-tiny", "../testdata/phi3-tiny", "../testdata/mistral-tiny-window",
		"../testdata/gemma3-vl-tiny", "../testdata/gemma4-dense-twogeom-tiny", "../testdata/gemma4-moe-tiny",
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
			slots := r.kvSlotBufs
			if slots == nil {
				slots = []cudaKVSlot{{r.kc, r.vc}}
			}
			if len(slots) != wantSlots || r.KVSlots() != wantSlots {
				t.Fatalf("built %d KV slots (KVSlots %d) with 2 requested, want %d", len(slots), r.KVSlots(), wantSlots)
			}
			forCap := kvBytesForCap(r.ctxCap, r.layers)
			for s, sl := range slots {
				var bytes int64
				for l := range sl.kc {
					for _, b := range []Buffer{sl.kc[l], sl.vc[l]} {
						n := int64(b.Len()) * 4 // Len is in f32 elements (r.af)
						bytes += n
						if alloc := (n + allocQuantumBytes - 1) / allocQuantumBytes * allocQuantumBytes; alloc-n >= allocQuantumBytes {
							t.Errorf("slot %d layer %d: a buffer of %d B rounds to %d B — more than one allocation quantum", s, l, n, alloc)
						}
					}
				}
				if bytes != forCap {
					t.Fatalf("slot %d: allocated K/V buffers hold %d B, kvBytesForCap says %d B at ctx %d — the resident's own fit figure no longer matches its allocation", s, bytes, forCap, r.ctxCap)
				}
			}
			for _, c := range []struct {
				name     string
				f16, i8b bool
			}{{"f32", false, false}, {"f16 requested", true, false}, {"i8 requested", false, true}} {
				if got := m.ResidentKVBytes("cuda", r.ctxCap, c.f16, c.i8b); got != forCap {
					t.Errorf("%s: ResidentKVBytes(\"cuda\", %d) = %d B, CUDA allocates %d B (%.3fx)", c.name, r.ctxCap, got, forCap, float64(got)/float64(forCap))
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no fixture built a CUDA resident here")
	}
}
