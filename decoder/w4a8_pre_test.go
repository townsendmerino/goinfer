package decoder

import (
	"context"
	"math"
	"os"
	"testing"
)

// TestW4A8Pre_decodeBitIdenticalAndTaken is R-13's goinfer gate (docs/tasks/task-recompute-audit.md): decoding with
// each layer's normed row quantized once for the projections that share it (q/k/v, gate/up, every MoE expert) gives the
// logits bits of decoding with every matmul quantizing its own input (w4a8PreOff), on dense and MoE tiny fixtures at
// int4, with one activation scale per row and per 32. The shared path must have run: at least two matmuls a layer a
// token (a DeltaNet layer's in_proj_qkv and in_proj_z), and none with it off.
func TestW4A8Pre_decodeBitIdenticalAndTaken(t *testing.T) {
	defer func() { w4a8PreOff = false }()
	for _, fx := range []string{"../testdata/llama-attnfa-tiny", "../testdata/mixtral-tiny", "../testdata/llama4-tiny", "../testdata/deepseek-tiny",
		"../testdata/qwen35-tiny", "../testdata/qwen3next-tiny", "../testdata/kimi-tiny", "../testdata/gemma4-dense-twogeom-tiny", "../testdata/gemma4-moe-tiny"} {
		if _, err := os.Stat(fx + "/config.json"); err != nil {
			t.Logf("%s: no fixture", fx)
			continue
		}
		for _, group := range []int{0, 32} {
			const steps = 32
			run := func(off bool) ([][]float32, int64, int) {
				w4a8PreOff = off
				m, err := Load(fx, Options{Quant: "int4", ActQuantGroup: group})
				if err != nil {
					t.Fatalf("%s: load: %v", fx, err)
				}
				defer m.Close()
				cache := m.NewCache(steps + 4)
				c0 := w4a8PreCalls.Load()
				var out [][]float32
				for i := range steps {
					lg, err := m.forward((i*7919+3)%m.w.arch.VocabSize, cache)
					if err != nil {
						t.Fatalf("%s step %d: %v", fx, i, err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out, w4a8PreCalls.Load() - c0, m.w.arch.NumLayers
			}
			off, offCalls, _ := run(true)
			on, onCalls, layers := run(false)
			if offCalls != 0 || onCalls < int64(2*layers*steps) {
				t.Fatalf("%s group %d: shared-block matmuls %d with the path off, %d on (want 0 and at least %d)", fx, group, offCalls, onCalls, 2*layers*steps)
			}
			for s := range on {
				for j := range on[s] {
					if math.Float32bits(on[s][j]) != math.Float32bits(off[s][j]) {
						t.Fatalf("%s group %d step %d logit %d: shared %v, per-matmul %v", fx, group, s, j, on[s][j], off[s][j])
					}
				}
			}
			t.Logf("%s group %d: %d steps bit-identical; %d matmuls on a shared block (%.1f a layer a token)", fx, group, steps, onCalls, float64(onCalls)/float64(layers*steps))
		}
	}
}

// TestW4A8Pre_prefillBitIdenticalAndTaken extends R-13's gate to the batched prefill (forwardN): a 40-token prompt's
// logits, every row, are bit-identical with the K-row block quantized once per projection group and with a
// quantization per matmul, on a dense, a Qwen3.5 hybrid and a Gemma 4 fixture, and the shared path ran.
func TestW4A8Pre_prefillBitIdenticalAndTaken(t *testing.T) {
	defer func() { w4a8PreOff = false }()
	const n = 40
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*7919 + 3) % 200
	}
	for _, fx := range []string{"../testdata/llama-attnfa-tiny", "../testdata/qwen35-tiny", "../testdata/gemma4-dense-twogeom-tiny"} {
		if _, err := os.Stat(fx + "/config.json"); err != nil {
			t.Logf("no fixture %s", fx)
			continue
		}
		for _, group := range []int{0, 32} {
			run := func(off bool) ([][]float32, int64, int) {
				w4a8PreOff = off
				m, err := Load(fx, Options{Quant: "int4", ActQuantGroup: group})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				c0 := w4a8PreCalls.Load()
				out, err := m.forwardN(context.Background(), ids, m.NewCache(n+4))
				if err != nil {
					t.Fatal(err)
				}
				return out, w4a8PreCalls.Load() - c0, m.w.arch.NumLayers
			}
			off, offCalls, _ := run(true)
			on, onCalls, layers := run(false)
			if offCalls != 0 || onCalls < int64(2*layers) {
				t.Fatalf("%s group %d: shared-block matmuls %d off, %d on (want 0 and at least %d)", fx, group, offCalls, onCalls, 2*layers)
			}
			for r := range on {
				for j := range on[r] {
					if math.Float32bits(on[r][j]) != math.Float32bits(off[r][j]) {
						t.Fatalf("%s group %d row %d logit %d: shared %v, per-matmul %v", fx, group, r, j, on[r][j], off[r][j])
					}
				}
			}
			t.Logf("%s group %d: %d rows bit-identical; %d matmuls on a shared block", fx, group, len(on), onCalls)
		}
	}
}
