//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestKV2_G0splitInvariance is G0 on the Mac's WebGPU (docs/measurements/kv2-2026-10-14.md), inside the tiny fixture's
// 128 positions: a 120-token prompt prefilled in two calls split at 64 and in successive chunks of 32, against the prompt
// in one call, bit for bit, after KV0's control. The decode arm (token by token) is reported: WebGPU's class map is the
// one-class map, which is right only if decode rows equal the batched pass's.
func TestKV2_G0splitInvariance(t *testing.T) {
	for _, fx := range kv2Fixtures[:3] {
		t.Run(fx.name, func(t *testing.T) {
			m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", fx.dir), fx.opts, 2)
			defer m.Close()
			const N = 120
			if len(rd.slots) != 2 || rd.ctxCap < N {
				t.Skipf("%d slots, a %d-position cache", len(rd.slots), rd.ctxCap)
			}
			hidden, _, _, _, _, _, _ := m.Dims()
			embs := kv2Embs(hidden, N, 77)
			fill := func(slot int, cuts []int, e [][]float32) ([]float32, [][]uint32, []int) {
				if err := rd.UseKVSlot(slot); err != nil {
					t.Fatal(err)
				}
				var lg []float32
				prev := 0
				for _, c := range append(slices.Clone(cuts), N) {
					lg = kv2Prefill(t, rd, e[prev:c], prev)
					prev = c
				}
				kv, rw := kv2Slot(rd, slot)
				return lg, kv, rw
			}
			wholeLg, whole, rw := fill(0, nil, embs)
			pert := slices.Clone(embs)
			pert[70] = kv2Embs(hidden, 1, 5)[0]
			_, ctl, _ := fill(1, nil, pert)
			if b, p := kv2Diff(ctl, whole, rw, 0, N); b < 0 || p != 70 {
				t.Fatalf("control: first difference at buffer %d row %d, want row 70", b, p)
			}
			for _, arm := range []struct {
				name string
				cuts []int
			}{{"split@64", []int{64}}, {"chunk=32", []int{32, 64, 96}}} {
				lg, kv, _ := fill(1, arm.cuts, embs)
				b, p := kv2Diff(kv, whole, rw, 0, N)
				lgDiff := 0
				for i := range lg {
					if math.Float32bits(lg[i]) != math.Float32bits(wholeLg[i]) {
						lgDiff++
					}
				}
				fmt.Fprintf(os.Stderr, "[kv2-g0 webgpu %s] %s: first KV difference buffer %d row %d (-1: none); last logits %d of %d differ\n", fx.name, arm.name, b, p, lgDiff, len(lg))
				if b >= 0 || lgDiff != 0 {
					t.Errorf("%s is not invariant: KV differs at buffer %d row %d, %d logits differ", arm.name, b, p, lgDiff)
				}
			}
			if err := rd.UseKVSlot(1); err != nil {
				t.Fatal(err)
			}
			for p, e := range embs {
				if _, err := rd.Forward(e, p); err != nil {
					t.Fatal(err)
				}
			}
			dec, _ := kv2Slot(rd, 1)
			b, p := kv2Diff(dec, whole, rw, 0, N)
			fmt.Fprintf(os.Stderr, "[kv2-g0 webgpu %s] decode token by token against the batched pass: first KV difference buffer %d row %d (-1: none)\n", fx.name, b, p)
		})
	}
}
