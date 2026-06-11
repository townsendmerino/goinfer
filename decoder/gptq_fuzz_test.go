package decoder

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
	"testing/fstest"

	"github.com/townsendmerino/aikit/embed"
)

// Track 2.4 (testing campaign): goinfer's interpretation of a pre-quantized
// GPTQ/AWQ safetensors checkpoint — parseQuantConfig (config.json) and the
// qweight/qzeros/scales/g_idx reconstruction. The safetensors CONTAINER parse is
// aikit's; here we build well-formed containers and fuzz goinfer's packed-int4
// dequant index math, which must never panic on a hostile/corrupt checkpoint.

// --- in-memory safetensors fixture builder (well-formed; aikit parses it) ---

type stTyped struct {
	dtype string
	shape []int
	data  []byte
}

func i32Bytes(v []int32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(x))
	}
	return b
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

// buildSafetensors encodes the safetensors frame (u64 header len + JSON header +
// blob) that aikit/embed parses.
func buildSafetensors(tensors map[string]stTyped) []byte {
	type meta struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	header := map[string]meta{}
	var blob []byte
	// deterministic order isn't required for correctness (offsets are explicit).
	for name, ten := range tensors {
		off := len(blob)
		header[name] = meta{ten.dtype, ten.shape, [2]int{off, off + len(ten.data)}}
		blob = append(blob, ten.data...)
	}
	hjson, _ := json.Marshal(header)
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(hjson)))
	out = append(out, hjson...)
	out = append(out, blob...)
	return out
}

func loadST(t *testing.T, raw []byte) *embed.SafetensorsFile {
	t.Helper()
	st, err := embed.OpenSafetensorsFromFS(fstest.MapFS{"m.safetensors": {Data: raw}}, "m.safetensors")
	if err != nil {
		t.Fatalf("OpenSafetensorsFromFS: %v", err)
	}
	return st
}

// gptqFixture builds a self-consistent (per the shape checks) GPTQ linear of the
// given dims and group count, contents arbitrary.
func gptqFixture(in, out, groups int) map[string]stTyped {
	outP := out / 8
	qw := make([]int32, (in/8)*out)
	gidx := make([]int32, in) // all group 0 — valid
	sc := make([]float32, groups*out)
	for i := range sc {
		sc[i] = 1
	}
	qz := make([]int32, groups*outP)
	return map[string]stTyped{
		"l.qweight": {"I32", []int{in / 8, out}, i32Bytes(qw)},
		"l.qzeros":  {"I32", []int{groups, outP}, i32Bytes(qz)},
		"l.g_idx":   {"I32", []int{in}, i32Bytes(gidx)},
		"l.scales":  {"F32", []int{groups, out}, f32Bytes(sc)},
	}
}

// TestGPTQReconstruct_nonMultipleOf8 is the regression: in/out that aren't a
// multiple of 8 made the integer-division shape check (in/8)*out pass with an
// undersized qweight, then the dequant loop indexed out of bounds and panicked.
// reconstruct must return a typed error instead.
func TestGPTQReconstruct_nonMultipleOf8(t *testing.T) {
	for _, d := range []struct{ in, out int }{{4, 8}, {8, 4}, {9, 8}, {16, 12}} {
		st := loadST(t, buildSafetensors(gptqFixture(d.in, d.out, 1)))
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("gptqReconstruct(in=%d,out=%d) panicked: %v", d.in, d.out, r)
				}
			}()
			if _, err := gptqReconstruct(st, "l", d.in, d.out); err == nil {
				t.Errorf("gptqReconstruct(in=%d,out=%d): want error for non-multiple-of-8 dims", d.in, d.out)
			}
		}()
	}
	// A clean multiple-of-8 case still reconstructs.
	st := loadST(t, buildSafetensors(gptqFixture(16, 8, 1)))
	if _, err := gptqReconstruct(st, "l", 16, 8); err != nil {
		t.Fatalf("valid gptq reconstruct rejected: %v", err)
	}
}

// fill writes n int32s drawn from content (cycled) — deterministic per input.
func fillI32(n int, content []byte) []int32 {
	out := make([]int32, n)
	if len(content) == 0 {
		return out
	}
	for i := range out {
		var w uint32
		for b := range 4 {
			w = w<<8 | uint32(content[(i*4+b)%len(content)])
		}
		out[i] = int32(w)
	}
	return out
}

func clampDim(v int) int { // 0..127, exercising both multiple-of-8 and not
	if v < 0 {
		v = -v
	}
	return v % 128
}

// FuzzParseQuantConfig fuzzes the config.json quantization_config parser: a typed
// error or a resolved config, never a panic.
func FuzzParseQuantConfig(f *testing.F) {
	f.Add([]byte(`{"quant_method":"gptq","bits":4,"group_size":128,"desc_act":true}`))
	f.Add([]byte(`{"quant_method":"awq","bits":4,"group_size":64}`))
	f.Add([]byte(`null`))
	f.Add([]byte(`{"quant_method":"gptq","bits":4,"group_size":-1}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = parseQuantConfig(data)
	})
}

// FuzzGPTQReconstruct builds a well-formed GPTQ safetensors at fuzz-chosen dims
// (including non-multiple-of-8) and arbitrary tensor contents, then reconstructs:
// never a panic, and a successful result is exactly out*in long.
func FuzzGPTQReconstruct(f *testing.F) {
	f.Add(16, 8, 1, []byte{1, 2, 3, 4})
	f.Add(4, 8, 1, []byte{0})
	f.Fuzz(func(t *testing.T, inRaw, outRaw, groupsRaw int, content []byte) {
		in, out := clampDim(inRaw), clampDim(outRaw)
		groups := 1 + clampDim(groupsRaw)%16
		if in == 0 || out == 0 {
			return
		}
		outP := out / 8
		ts := map[string]stTyped{
			"l.qweight": {"I32", []int{in / 8, out}, i32Bytes(fillI32((in/8)*out, content))},
			"l.qzeros":  {"I32", []int{groups, outP}, i32Bytes(fillI32(groups*outP, content))},
			"l.g_idx":   {"I32", []int{in}, i32Bytes(gidxValid(in, groups, content))},
			"l.scales":  {"F32", []int{groups, out}, f32Bytes(make([]float32, groups*out))},
		}
		st := loadST(t, buildSafetensors(ts))
		res, err := gptqReconstruct(st, "l", in, out)
		if err == nil && len(res) != out*in {
			t.Fatalf("gptq result len %d, want %d (in=%d out=%d)", len(res), out*in, in, out)
		}
	})
}

// FuzzAWQReconstruct is the AWQ analogue.
func FuzzAWQReconstruct(f *testing.F) {
	f.Add(16, 8, 1, []byte{1, 2, 3, 4})
	f.Add(8, 4, 1, []byte{0})
	f.Fuzz(func(t *testing.T, inRaw, outRaw, groupsRaw int, content []byte) {
		in, out := clampDim(inRaw), clampDim(outRaw)
		groups := 1 + clampDim(groupsRaw)%16
		if in == 0 || out == 0 {
			return
		}
		outP := out / 8
		ts := map[string]stTyped{
			"l.qweight": {"I32", []int{in, outP}, i32Bytes(fillI32(in*outP, content))},
			"l.qzeros":  {"I32", []int{groups, outP}, i32Bytes(fillI32(groups*outP, content))},
			"l.scales":  {"F32", []int{groups, out}, f32Bytes(make([]float32, groups*out))},
		}
		st := loadST(t, buildSafetensors(ts))
		res, err := awqReconstruct(st, "l", in, out)
		if err == nil && len(res) != out*in {
			t.Fatalf("awq result len %d, want %d (in=%d out=%d)", len(res), out*in, in, out)
		}
	})
}

// gidxValid returns in group indices in [0,groups) (so the dequant loop runs
// rather than failing the g_idx range check), derived from content.
func gidxValid(in, groups int, content []byte) []int32 {
	out := make([]int32, in)
	for i := range out {
		if len(content) > 0 {
			out[i] = int32(uint(content[i%len(content)]) % uint(groups))
		}
	}
	return out
}
