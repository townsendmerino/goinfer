package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// writeHalfFixture re-encodes the tiny fixture at src as a checkpoint of 16-bit tensors (dtype "BF16" or "F16": every tensor, norms
// included, rounded to that precision) in a new directory, and returns it.
func writeHalfFixture(t *testing.T, src, dtype string) string {
	t.Helper()
	st, err := openCheckpointMmap(src)
	if err != nil {
		t.Fatalf("open %s: %v", src, err)
	}
	defer st.Close()
	names := slices.Clone(st.Names())
	sort.Strings(names)
	hdr := map[string]any{}
	var data []byte
	for _, n := range names {
		tn, err := st.Tensor(n)
		if err != nil {
			t.Fatal(err)
		}
		vals, err := st.TensorF32(n)
		if err != nil {
			t.Fatal(err)
		}
		lo := len(data)
		for _, v := range vals {
			if dtype == "BF16" {
				data = binary.LittleEndian.AppendUint16(data, linalg.F32ToBF16(v))
			} else {
				data = binary.LittleEndian.AppendUint16(data, linalg.F32ToF16(v))
			}
		}
		hdr[n] = map[string]any{"dtype": dtype, "shape": tn.Shape, "data_offsets": []int{lo, len(data)}}
	}
	h, _ := json.Marshal(hdr)
	b := append(binary.LittleEndian.AppendUint64(nil, uint64(len(h))), h...)
	b = append(b, data...)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "generation_config.json"} {
		raw, err := os.ReadFile(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, f), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// Quant "bf16" / "f16" over a checkpoint stored in that dtype must give the f32 load's logits to the bit (the weights are the same
// numbers; the matmul widens a tile and runs the same f32 kernel), at roughly half the weight bytes. Dense and MoE fixtures, so the
// experts, the router (kept f32), the embedding and the head are all in play.
func TestHalf16Weights_bitIdenticalToF32(t *testing.T) {
	for _, fx := range []string{"llama-tiny", "mixtral-tiny", "qwen3moe-tiny"} {
		for _, tc := range []struct{ dtype, quant string }{{"BF16", "bf16"}, {"F16", "f16"}} {
			t.Run(fx+"/"+tc.quant, func(t *testing.T) {
				dir := writeHalfFixture(t, filepath.Join("..", "testdata", fx), tc.dtype)
				ref, err := Load(dir, Options{Backend: "cpu"})
				if err != nil {
					t.Fatalf("f32 load: %v", err)
				}
				defer ref.Close()
				half, err := Load(dir, Options{Backend: "cpu", Quant: tc.quant})
				if err != nil {
					t.Fatalf("%s load: %v", tc.quant, err)
				}
				defer half.Close()
				cr, ch := ref.NewCache(16), half.NewCache(16)
				for pos, id := range []int{3, 17, 5, 29, 11} {
					want, err := ref.forward(id, cr)
					if err != nil {
						t.Fatal(err)
					}
					got, err := half.forward(id, ch)
					if err != nil {
						t.Fatal(err)
					}
					for i := range want {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("position %d logit %d: %s %v, f32 %v", pos, i, tc.quant, got[i], want[i])
						}
					}
				}
				// The batched prefill (M > 1: the MoE batching, the grouped projections) reads the same kind through other call sites.
				ids := []int{3, 17, 5, 29, 11, 2, 7, 19, 23, 13, 31, 1}
				for _, on := range []string{"0", "1"} { // expert-major off and on: the second reads each expert's rows as one block
					setKnob(t, ref, knobMoEExpertMajor, on)
					setKnob(t, half, knobMoEExpertMajor, on)
					wantN, errR := ref.forwardLayersN(context.Background(), ids, ref.NewCache(len(ids)+8), false)
					gotN, errH := half.forwardLayersN(context.Background(), ids, half.NewCache(len(ids)+8), false)
					if errR != nil || errH != nil {
						t.Skipf("no batched prefill on this fixture (f32: %v, %s: %v)", errR, tc.quant, errH)
					}
					for i := range wantN {
						if math.Float32bits(gotN[i]) != math.Float32bits(wantN[i]) {
							t.Fatalf("batched prefill (expert-major=%s) element %d: %s %v, f32 %v", on, i, tc.quant, gotN[i], wantN[i])
						}
					}
				}
				rb, hb := ref.ResidentWeightBytes(), half.ResidentWeightBytes()
				// Routers, norms and biases stay f32, so the saving is a little under half on a tiny model.
				if hb <= 0 || float64(hb) > 0.75*float64(rb) {
					t.Fatalf("%s weights are %d bytes against f32's %d: not close to half", tc.quant, hb, rb)
				}
			})
		}
	}
}

func TestHalf16Weights_refusedWhereItCannotRun(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-tiny")
	if _, err := Load(dir, Options{Backend: "cuda", Quant: "bf16"}); err == nil {
		t.Fatal("bf16 with a non-cpu backend must be refused")
	}
	if err := checkHalf16Load(quantBF16, "x.gguf", "cpu"); err == nil {
		t.Fatal("bf16 over a .gguf must be refused")
	}
	if err := checkHalf16Load(quantInt8, "x.gguf", "cuda"); err != nil {
		t.Fatalf("other quants are not this check's business: %v", err)
	}
	if _, err := parseQuant("fp16"); err == nil {
		t.Fatal("unknown quant must be refused")
	}
}
