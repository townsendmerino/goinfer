package quant

import (
	"encoding/binary"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

func TestParseConfigAndReconstructors(t *testing.T) {
	cases := []struct {
		name       string
		raw        string
		wantMethod string
	}{
		{"gptq", `{"quant_method":"gptq","bits":4,"group_size":128}`, "gptq"},
		{"awq", `{"quant_method":"awq","bits":4,"group_size":128}`, "awq"},
		{"fp8", `{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]}`, "fp8"},
		{"nvfp4", `{"quant_method":"modelopt","quant_algo":"NVFP4"}`, "nvfp4"},
		{"mlx-affine", `{"bits":4,"group_size":64,"mode":"affine"}`, "mlx"},
		{"mlx-nvfp4", `{"mode":"nvfp4"}`, "mlx"},
		{"mlx-mxfp4", `{"mode":"mxfp4"}`, "mlx"},
	}
	for _, tc := range cases {
		cfg, err := ParseConfig(json.RawMessage(tc.raw))
		if err != nil {
			t.Fatalf("%s: ParseConfig failed: %v", tc.name, err)
		}
		if cfg == nil || cfg.Method != tc.wantMethod {
			t.Fatalf("%s: got %+v, want Method=%q", tc.name, cfg, tc.wantMethod)
		}
		r := cfg.Reconstructor()
		if r == nil || r.Method() != tc.wantMethod {
			t.Fatalf("%s: Reconstructor() = %v, want Method()=%q", tc.name, r, tc.wantMethod)
		}
	}

	var _ RowStreamReconstructor = NVFP4Reconstructor{}
	var _ RowStreamReconstructor = MLXReconstructor{}
}

func TestFP8AndNVFP4Tables(t *testing.T) {
	if got := FP8E4M3ToF32(0x7E); got != 448 {
		t.Fatalf("FP8E4M3ToF32(0x7E) = %v, want 448", got)
	}
	if !math.IsNaN(float64(FP8E4M3ToF32(0x7F))) {
		t.Fatal("FP8E4M3ToF32(0x7F) must be NaN")
	}
	if got := NVFP4E2M1ToF32(0x07); got != 6.0 {
		t.Fatalf("NVFP4E2M1ToF32(0x07) = %v, want 6.0", got)
	}
	if got := NVFP4E2M1ToF32(0x0F); got != -6.0 {
		t.Fatalf("NVFP4E2M1ToF32(0x0F) = %v, want -6.0", got)
	}
}

func TestParseConfig_accepts(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		want      Config
	}{
		{"gptq act-order, sym by default", `{"quant_method":"gptq","bits":4,"group_size":128,"desc_act":true}`,
			Config{Method: "gptq", Bits: 4, GroupSize: 128, DescAct: true, Sym: true}},
		{"gptq asymmetric", `{"quant_method":"gptq","bits":4,"group_size":32,"sym":false}`,
			Config{Method: "gptq", Bits: 4, GroupSize: 32}},
		{"gptq v1 named", `{"quant_method":"gptq","bits":4,"group_size":64,"checkpoint_format":"gptq"}`,
			Config{Method: "gptq", Bits: 4, GroupSize: 64, Sym: true}},
		{"gptq odd group size", `{"quant_method":"gptq","bits":4,"group_size":7}`,
			Config{Method: "gptq", Bits: 4, GroupSize: 7, Sym: true}},
		{"awq", `{"quant_method":"awq","bits":4,"group_size":128,"zero_point":true,"version":"gemm"}`,
			Config{Method: "awq", Bits: 4, GroupSize: 128, Sym: true}},
		{"fp8 without fmt", `{"quant_method":"fp8","weight_block_size":[128,64]}`,
			Config{Method: "fp8", BlockR: 128, BlockC: 64}},
		{"fp8 1x1 block", `{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[1,1]}`,
			Config{Method: "fp8", BlockR: 1, BlockC: 1}},
		{"nvfp4 method", `{"quant_method":"nvfp4"}`, Config{Method: "nvfp4", Bits: 4, GroupSize: 16}},
		{"modelopt, no algo", `{"quant_method":"modelopt"}`, Config{Method: "nvfp4", Bits: 4, GroupSize: 16}},
		{"modelopt, lower-case algo", `{"quant_method":"modelopt","quant_algo":"nvfp4"}`, Config{Method: "nvfp4", Bits: 4, GroupSize: 16}},
		{"compressed-tensors groups", `{"quant_method":"nvfp4","config_groups":{"group_0":{"weights":{"num_bits":4,"type":"float","group_size":16}},"g1":{"weights":{}}}}`,
			Config{Method: "nvfp4", Bits: 4, GroupSize: 16}},
		{"mlx default mode is affine", `{"bits":8,"group_size":32}`, Config{Method: "mlx", Bits: 8, GroupSize: 32, Mode: "affine"}},
		{"mlx named", `{"quant_method":"mlx","bits":4,"group_size":128,"mode":"affine"}`,
			Config{Method: "mlx", Bits: 4, GroupSize: 128, Mode: "affine"}},
		{"mlx nvfp4 defaults", `{"mode":"nvfp4"}`, Config{Method: "mlx", Bits: 4, GroupSize: 16, Mode: "nvfp4"}},
		{"mlx mxfp4 defaults", `{"mode":"mxfp4"}`, Config{Method: "mlx", Bits: 4, GroupSize: 32, Mode: "mxfp4"}},
		{"mlx mxfp4 explicit", `{"mode":"mxfp4","bits":4,"group_size":32}`, Config{Method: "mlx", Bits: 4, GroupSize: 32, Mode: "mxfp4"}},
	} {
		got, err := ParseConfig(json.RawMessage(tc.raw))
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if got == nil || *got != tc.want {
			t.Errorf("%s: %+v, want %+v", tc.name, got, tc.want)
		}
	}
	for _, raw := range []string{"", "null"} {
		if got, err := ParseConfig(json.RawMessage(raw)); got != nil || err != nil {
			t.Errorf("ParseConfig(%q) = %+v, %v; want a full-precision nil", raw, got, err)
		}
	}
}

func TestParseConfig_refuses(t *testing.T) {
	for _, tc := range []struct{ name, raw, wantErr string }{
		{"malformed JSON", `{"bits":4`, "quantization_config"},
		{"not an object", `[4]`, "quantization_config"},
		{"bits as a string", `{"quant_method":"gptq","bits":"4","group_size":128}`, "quantization_config"},
		{"unknown method", `{"quant_method":"bitsandbytes","bits":4}`, `method "bitsandbytes" unsupported`},
		{"fp8 e5m2", `{"quant_method":"fp8","fmt":"e5m2","weight_block_size":[128,128]}`, `fmt "e5m2"`},
		{"fp8 per-tensor", `{"quant_method":"fp8"}`, "weight_block_size"},
		{"fp8 1-D block", `{"quant_method":"fp8","weight_block_size":[128]}`, "weight_block_size"},
		{"fp8 3-D block", `{"quant_method":"fp8","weight_block_size":[1,128,128]}`, "weight_block_size"},
		{"fp8 zero block", `{"quant_method":"fp8","weight_block_size":[0,128]}`, "weight_block_size"},
		{"fp8 negative block", `{"quant_method":"fp8","weight_block_size":[128,-1]}`, "weight_block_size"},
		{"modelopt fp8 algo", `{"quant_method":"modelopt","quant_algo":"FP8"}`, `quant_algo "FP8"`},
		{"nvfp4 8-bit group", `{"quant_method":"nvfp4","config_groups":{"g":{"weights":{"num_bits":8}}}}`, "num_bits=8"},
		{"nvfp4 group of 32", `{"quant_method":"modelopt","config_groups":{"g":{"weights":{"num_bits":4,"group_size":32}}}}`, "group_size=32"},
		{"mlx affine 3-bit", `{"bits":3,"group_size":64}`, "3-bit"},
		{"mlx affine no bits", `{}`, "0-bit"},
		{"mlx affine 2-bit", `{"quant_method":"mlx","bits":2,"group_size":64}`, "2-bit"},
		{"mlx affine group 16", `{"bits":4,"group_size":16}`, "group_size 16"},
		{"mlx affine group 256", `{"bits":4,"group_size":256}`, "group_size 256"},
		{"mlx affine no group", `{"bits":4}`, "group_size 0"},
		{"mlx nvfp4 8-bit", `{"mode":"nvfp4","bits":8}`, "requires bits=4 group_size=16"},
		{"mlx nvfp4 group 32", `{"mode":"nvfp4","group_size":32}`, "requires bits=4 group_size=16"},
		{"mlx mxfp4 group 16", `{"mode":"mxfp4","group_size":16}`, "requires bits=4 group_size=32"},
		{"mlx unknown mode", `{"mode":"int4","bits":4,"group_size":64}`, `mode "int4"`},
		{"gptq 8-bit", `{"quant_method":"gptq","bits":8,"group_size":128}`, "8-bit"},
		{"gptq 3-bit", `{"quant_method":"gptq","bits":3,"group_size":128}`, "3-bit"},
		{"gptq no bits", `{"quant_method":"gptq","group_size":128}`, "0-bit"},
		{"gptq per-channel group -1", `{"quant_method":"gptq","bits":4,"group_size":-1}`, "group_size -1"},
		{"gptq no group", `{"quant_method":"gptq","bits":4}`, "group_size 0"},
		{"gptq v2 export", `{"quant_method":"gptq","bits":4,"group_size":128,"checkpoint_format":"gptq_v2"}`, `checkpoint_format "gptq_v2"`},
		{"awq 8-bit", `{"quant_method":"awq","bits":8,"group_size":128}`, "8-bit"},
		{"awq no group", `{"quant_method":"awq","bits":4}`, "group_size 0"},
	} {
		got, err := ParseConfig(json.RawMessage(tc.raw))
		if err == nil {
			t.Errorf("%s: accepted as %+v", tc.name, got)
			continue
		}
		if !strings.Contains(err.Error(), tc.wantErr) {
			t.Errorf("%s: error %q does not name %q", tc.name, err, tc.wantErr)
		}
	}
}

// AutoAWQ's GEMV export packs qweight along the input dimension ([out, in/8]) where GEMM packs along the output ([in, out/8]);
// both hold in*out/8 words, so AWQReconstruct's length checks cannot tell them apart and a GEMV checkpoint would load
// finite, wrong weights. Its config names the layout in "version".
func TestParseConfig_awqGEMVRefused(t *testing.T) {
	for _, v := range []string{"gemv", "GEMV", "gemv_fast"} {
		raw := `{"quant_method":"awq","bits":4,"group_size":128,"version":"` + v + `"}`
		if got, err := ParseConfig(json.RawMessage(raw)); err == nil {
			t.Errorf("ParseConfig accepts AWQ version %q as %+v; AWQReconstruct reads only the GEMM layout", v, got)
		}
	}
}

func TestParseCheckpointConfig(t *testing.T) {
	gptq := json.RawMessage(`{"quant_method":"gptq","bits":4,"group_size":128}`)
	mlx := json.RawMessage(`{"bits":4,"group_size":64}`)
	for _, tc := range []struct {
		name       string
		qc, q      json.RawMessage
		wantMethod string
	}{
		{"quantization_config wins", gptq, mlx, "gptq"},
		{"null quantization_config falls to quantization", json.RawMessage("null"), mlx, "mlx"},
		{"only quantization", nil, mlx, "mlx"},
		{"neither", nil, nil, ""},
		{"both null", json.RawMessage("null"), json.RawMessage("null"), ""},
	} {
		got, err := ParseCheckpointConfig(tc.qc, tc.q)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if (got == nil) != (tc.wantMethod == "") || (got != nil && got.Method != tc.wantMethod) {
			t.Errorf("%s: %+v, want method %q", tc.name, got, tc.wantMethod)
		}
	}
	if _, err := ParseCheckpointConfig(json.RawMessage(`{"quant_method":"fp8"}`), mlx); err == nil {
		t.Error("an invalid quantization_config was skipped for the valid quantization beside it")
	}
}

func TestConfig_Reconstructor(t *testing.T) {
	var nilCfg *Config
	if r := nilCfg.Reconstructor(); r != nil {
		t.Errorf("nil config: %v, want nil", r)
	}
	if r, ok := (&Config{Method: "fp8", BlockR: 64, BlockC: 32}).Reconstructor().(FP8Reconstructor); !ok || r.BlockR != 64 || r.BlockC != 32 {
		t.Errorf("fp8 reconstructor does not carry its block: %+v", r)
	}
	for m, want := range map[string]Reconstructor{
		"gptq": GPTQReconstructor{}, "awq": AWQReconstructor{}, "nvfp4": NVFP4Reconstructor{}, "mlx": MLXReconstructor{},
	} {
		if got := (&Config{Method: m}).Reconstructor(); got != want {
			t.Errorf("%s: %T, want %T", m, got, want)
		}
	}
}

// stTensor is one tensor of a synthetic safetensors file.
type stTensor struct {
	dtype string
	shape []int
	data  []byte
}

// openST writes tensors as a safetensors file and opens it.
func openST(t *testing.T, tensors map[string]stTensor) *embed.SafetensorsFile {
	t.Helper()
	header := map[string]any{}
	var payload []byte
	for _, n := range slices.Sorted(maps.Keys(tensors)) {
		tt := tensors[n]
		shape := tt.shape
		if shape == nil {
			shape = []int{}
		}
		header[n] = map[string]any{"dtype": tt.dtype, "shape": shape, "data_offsets": []int{len(payload), len(payload) + len(tt.data)}}
		payload = append(payload, tt.data...)
	}
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(h)%8 != 0 {
		h = append(h, ' ')
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	file = append(append(file, h...), payload...)
	p := filepath.Join(t.TempDir(), "m.safetensors")
	if err := os.WriteFile(p, file, 0o600); err != nil {
		t.Fatal(err)
	}
	st, err := embed.OpenSafetensors(p)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func f32s(v ...float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func i32s(v ...int32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, uint32(x))
	}
	return b
}

func u32s(v ...uint32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, x)
	}
	return b
}

func f32T(shape []int, v ...float32) stTensor { return stTensor{"F32", shape, f32s(v...)} }
func i32T(shape []int, v ...int32) stTensor   { return stTensor{"I32", shape, i32s(v...)} }
func u32T(shape []int, v ...uint32) stTensor  { return stTensor{"U32", shape, u32s(v...)} }

func wantErr(t *testing.T, name string, err error, sub string) {
	t.Helper()
	if err == nil {
		t.Errorf("%s: no error, want one naming %q", name, sub)
	} else if !strings.Contains(err.Error(), sub) {
		t.Errorf("%s: error %q does not name %q", name, err, sub)
	}
}

func TestFP8E4M3ToF32_boundaries(t *testing.T) {
	for _, tc := range []struct {
		b    byte
		want float64
	}{
		{0x00, 0}, {0x01, 0x1p-9}, {0x07, 7 * 0x1p-9}, {0x08, 0x1p-6}, {0x38, 1}, {0x39, 1.125}, {0x40, 2},
		{0x77, 240}, {0x7E, 448}, {0xB8, -1}, {0xFE, -448}, {0x81, -0x1p-9},
	} {
		if got := FP8E4M3ToF32(tc.b); float64(got) != tc.want {
			t.Errorf("FP8E4M3ToF32(%#02x) = %v, want %v", tc.b, got, tc.want)
		}
	}
	if z := FP8E4M3ToF32(0x80); z != 0 || !math.Signbit(float64(z)) {
		t.Errorf("FP8E4M3ToF32(0x80) = %v, want -0", z)
	}
	var nans []int
	for i, v := range FP8E4M3Table {
		if math.IsNaN(float64(v)) {
			nans = append(nans, i)
		} else if math.IsInf(float64(v), 0) || math.Abs(float64(v)) > 448 {
			t.Errorf("table[%#02x] = %v outside e4m3fn's finite range", i, v)
		}
	}
	if !slices.Equal(nans, []int{0x7F, 0xFF}) {
		t.Errorf("NaN bytes %#x, want only 0x7f and 0xff", nans)
	}
}

func TestFP8DequantBlocked_partialBlocks(t *testing.T) {
	// [3,5] in 2x2 blocks: a 2x3 scale grid whose last row and column cover partial blocks.
	w := slices.Repeat([]byte{0x38}, 15) // 1.0 everywhere
	scales := []float32{1, 2, 3, 4, 5, 6}
	got, err := FP8DequantBlocked(w, scales, 3, 5, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		1, 1, 2, 2, 3,
		1, 1, 2, 2, 3,
		4, 4, 5, 5, 6,
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %v\nwant %v", got, want)
	}
	// A block larger than the tensor: one scale.
	if got, err := FP8DequantBlocked([]byte{0x40, 0xC0}, []float32{0.5}, 1, 2, 128, 128); err != nil || !slices.Equal(got, []float32{1, -1}) {
		t.Errorf("oversized block: %v, %v", got, err)
	}
	if got, err := FP8DequantBlocked(nil, nil, 0, 0, 128, 128); err != nil || len(got) != 0 {
		t.Errorf("empty tensor: %v, %v", got, err)
	}
	_, err = FP8DequantBlocked(w, scales, 3, 5, 0, 2)
	wantErr(t, "zero block", err, "must be positive")
	_, err = FP8DequantBlocked(w[:14], scales, 3, 5, 2, 2)
	wantErr(t, "short payload", err, "14 payload bytes")
	_, err = FP8DequantBlocked(w, scales[:5], 3, 5, 2, 2)
	wantErr(t, "short scale grid", err, "5 scales")
	_, err = FP8DequantBlocked(w, append(slices.Clone(scales), 7), 3, 5, 2, 2)
	wantErr(t, "long scale grid", err, "7 scales")
}

func TestFP8Reconstruct(t *testing.T) {
	// [out=2, in=3] at 1x2 blocks: scale grid [2, 2].
	w := []byte{0x38, 0x40, 0xB8, 0x00, 0x38, 0x40} // 1 2 -1 / 0 1 2
	scaleInv := f32T([]int{2, 2}, 1, 10, 2, 20)
	st := openST(t, map[string]stTensor{
		"l.weight":                  {"F8_E4M3", []int{2, 3}, w},
		"l.weight_scale_inv":        scaleInv,
		"t.weight":                  {"F8_E4M3", []int{3, 2}, w},
		"t.weight_scale_inv":        f32T([]int{3, 1}, 1, 1, 1),
		"u8.weight":                 {"U8", []int{2, 3}, w},
		"noscale.weight":            {"F8_E4M3", []int{2, 3}, w},
		"badscale.weight":           {"F8_E4M3", []int{2, 3}, w},
		"badscale.weight_scale_inv": f32T([]int{1, 2}, 1, 1),
	})
	got, err := FP8Reconstruct(st, "l.weight", 3, 2, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{1, 2, -10, 0, 2, 40}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	got2, fresh, err := ReconstructMat2D(st, &Config{Method: "fp8", BlockR: 1, BlockC: 2}, "l.weight", 2, 3)
	if err != nil || !fresh || !slices.Equal(got2, got) {
		t.Errorf("ReconstructMat2D through the config: %v fresh=%v %v", got2, fresh, err)
	}

	_, err = FP8Reconstruct(st, "t.weight", 3, 2, 1, 2)
	wantErr(t, "transposed", err, "transposed")
	_, err = FP8Reconstruct(st, "u8.weight", 3, 2, 1, 2)
	wantErr(t, "U8 payload", err, "want F8_E4M3")
	_, err = FP8Reconstruct(st, "missing.weight", 3, 2, 1, 2)
	wantErr(t, "missing", err, "not found")
	_, err = FP8Reconstruct(st, "noscale.weight", 3, 2, 1, 2)
	wantErr(t, "no scale_inv", err, "noscale.weight_scale_inv")
	_, err = FP8Reconstruct(st, "badscale.weight", 3, 2, 1, 2)
	wantErr(t, "scale grid for another block", err, "badscale.weight_scale_inv")
	_, err = FP8Reconstruct(st, "l.weight", 3, 2, 2, 2) // a [1,2] grid is wanted, the file has [2,2]
	wantErr(t, "block size disagreeing with the grid", err, "scales")
}

// gptqFixture is a 4-bit GPTQ linear with in=16 rows of codes and out=8 columns in two groups of 8, codes
// code(i,j) = (i+3j) mod 16, stored zero-points 7 (group 0) and 2 (group 1) in v1 form (+1 on read), scale (g+1)*(j+1)/4.
func gptqFixture(gidx []int32) map[string]stTensor {
	const in, out = 16, 8
	qw := make([]int32, (in/8)*out)
	for i := range in {
		for j := range out {
			code := uint32((i + 3*j) % 16)
			qw[(i/8)*out+j] |= int32(code << (4 * (i % 8)))
		}
	}
	qz := []int32{0x77777777, 0x22222222}
	sc := make([]float32, 2*out)
	for g := range 2 {
		for j := range out {
			sc[g*out+j] = float32(g+1) * float32(j+1) / 4
		}
	}
	return map[string]stTensor{
		"p.qweight": i32T([]int{in / 8, out}, qw...),
		"p.qzeros":  i32T([]int{2, out / 8}, qz...),
		"p.scales":  f32T([]int{2, out}, sc...),
		"p.g_idx":   i32T([]int{in}, gidx...),
	}
}

func TestGPTQReconstruct(t *testing.T) {
	const in, out = 16, 8
	seq := make([]int32, in)
	for i := range in {
		seq[i] = int32(i / 8)
	}
	st := openST(t, gptqFixture(seq))
	got, err := GPTQReconstruct(st, "p", in, out)
	if err != nil {
		t.Fatal(err)
	}
	// Spot values by hand: W[j][i] = (code - (z+1)) * scale.
	for _, tc := range []struct {
		j, i int
		want float32
	}{
		{0, 0, (0 - 8) * 0.25}, // group 0, zero 8, scale 1/4
		{0, 7, (7 - 8) * 0.25},
		{1, 7, (10 - 8) * 0.5},       // code 10 in the top nibble: the packed int32 is negative
		{1, 5, 0},                    // code 5+3 = 8, the zero point
		{7, 15, (4 - 3) * 2 * 8 / 4}, // group 1: code (15+21)%16=4, zero 3, scale 2*8/4
		{2, 13, 0},                   // code 3, the zero point
	} {
		if g := got[tc.j*in+tc.i]; g != tc.want {
			t.Errorf("W[%d][%d] = %v, want %v", tc.j, tc.i, g, tc.want)
		}
	}
	// Act-order: g_idx sends row i to group 1-(i%2) whatever its position.
	perm := make([]int32, in)
	for i := range in {
		perm[i] = int32(1 - i%2)
	}
	gotP, err := GPTQReconstructor{}.ReconstructMat2D(openST(t, gptqFixture(perm)), "p.weight", out, in)
	if err != nil {
		t.Fatal(err)
	}
	if w := gotP[0*in+0]; w != (0-3)*2*1.0/4 { // row 0 is in group 1
		t.Errorf("act-order W[0][0] = %v, want %v", w, (0-3)*2*1.0/4)
	}
	if w := gotP[0*in+1]; w != (1-8)*0.25 { // row 1 is in group 0
		t.Errorf("act-order W[0][1] = %v, want %v", w, (1-8)*0.25)
	}

	for _, tc := range []struct {
		name    string
		in, out int
		sub     string
	}{
		{"in not a multiple of 8", 12, 8, "multiples of 8"},
		{"out not a multiple of 8", 16, 4, "multiples of 8"},
		{"zero in", 0, 8, "multiples of 8"},
		{"shape disagreeing with the tensors", 8, 16, "bad shapes"},
	} {
		_, err := GPTQReconstruct(st, "p", tc.in, tc.out)
		wantErr(t, tc.name, err, tc.sub)
	}
	bad := gptqFixture(seq)
	bad["p.g_idx"] = i32T([]int{in}, append(slices.Clone(seq[:in-1]), 2)...)
	_, err = GPTQReconstruct(openST(t, bad), "p", in, out)
	wantErr(t, "g_idx past the groups", err, "g_idx[15]=2")
	bad["p.g_idx"] = i32T([]int{in}, append([]int32{-1}, seq[1:]...)...)
	_, err = GPTQReconstruct(openST(t, bad), "p", in, out)
	wantErr(t, "negative g_idx", err, "g_idx[0]=-1")
	bad = gptqFixture(seq)
	bad["p.qzeros"] = i32T([]int{1, 1}, 0x77777777)
	_, err = GPTQReconstruct(openST(t, bad), "p", in, out)
	wantErr(t, "qzeros for one group of two", err, "qzeros len 1")
	bad = gptqFixture(seq)
	delete(bad, "p.g_idx")
	_, err = GPTQReconstruct(openST(t, bad), "p", in, out)
	wantErr(t, "no g_idx", err, "p.g_idx")
	bad = gptqFixture(seq)
	bad["p.scales"] = f32T([]int{12}, make([]float32, 12)...)
	_, err = GPTQReconstruct(openST(t, bad), "p", in, out)
	wantErr(t, "scales not a whole number of rows", err, "bad shapes")
}

// awqPackOrder is AutoAWQ's GEMM packing: nibble k of a word holds output column packOrder[k] of its group of eight.
var awqPackOrder = [8]int{0, 2, 4, 6, 1, 3, 5, 7}

// awqFixture is a 4-bit AWQ GEMM linear with in rows, out=8 columns and len(zeros) groups, code(i,j) = (2i+j) mod 16,
// one zero-point per group (no +1) and scale (g+1)*(j+1)/8.
func awqFixture(in int, zeros []uint32) map[string]stTensor {
	const out = 8
	nG := len(zeros)
	pack := func(val func(j int) uint32) int32 {
		var w uint32
		for k, j := range awqPackOrder {
			w |= val(j) << (4 * k)
		}
		return int32(w)
	}
	qw := make([]int32, in)
	for i := range in {
		qw[i] = pack(func(j int) uint32 { return uint32((2*i + j) % 16) })
	}
	qz := make([]int32, nG)
	sc := make([]float32, nG*out)
	for g := range nG {
		qz[g] = pack(func(int) uint32 { return zeros[g] })
		for j := range out {
			sc[g*out+j] = float32(g+1) * float32(j+1) / 8
		}
	}
	return map[string]stTensor{
		"a.qweight": i32T([]int{in, out / 8}, qw...),
		"a.qzeros":  i32T([]int{nG, out / 8}, qz...),
		"a.scales":  f32T([]int{nG, out}, sc...),
	}
}

func TestAWQReconstruct(t *testing.T) {
	const in, out = 6, 8 // two groups of 3: AWQ takes any group that divides in
	got, err := AWQReconstructor{}.ReconstructMat2D(openST(t, awqFixture(in, []uint32{8, 15})), "a.weight", out, in)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		j, i int
		want float32
	}{
		{0, 0, (0 - 8) * 1.0 / 8},
		{1, 0, (1 - 8) * 2.0 / 8}, // column 1 is nibble 4
		{7, 1, (9 - 8) * 8.0 / 8}, // code 9 in the top nibble: the packed int32 is negative
		{6, 2, (10 - 8) * 7.0 / 8},
		{7, 5, (1 - 15) * 2 * 8.0 / 8}, // group 1: code (10+7)%16=1, top nibble, zero 15
		{4, 3, (10 - 15) * 2 * 5.0 / 8},
	} {
		if g := got[tc.j*in+tc.i]; g != tc.want {
			t.Errorf("W[%d][%d] = %v, want %v", tc.j, tc.i, g, tc.want)
		}
	}
	st := openST(t, awqFixture(in, []uint32{8, 15}))
	for _, tc := range []struct {
		name    string
		in, out int
		sub     string
	}{
		{"out not a multiple of 8", in, 12, "multiple of 8"},
		{"zero in", 0, 8, "invalid"},
		{"rows disagreeing with qweight", 7, 8, "bad shapes"},
	} {
		_, err := AWQReconstruct(st, "a", tc.in, tc.out)
		wantErr(t, tc.name, err, tc.sub)
	}
	_, err = AWQReconstruct(openST(t, awqFixture(7, []uint32{8, 8})), "a", 7, out)
	wantErr(t, "groups not dividing in", err, "not divisible by groups=2")
	bad := awqFixture(in, []uint32{8, 15})
	bad["a.qzeros"] = i32T([]int{1, 1}, 0)
	_, err = AWQReconstruct(openST(t, bad), "a", in, out)
	wantErr(t, "qzeros for one group of two", err, "qzeros len 1")
	bad = awqFixture(in, []uint32{8, 15})
	bad["a.scales"] = f32T([]int{0, out})
	bad["a.qzeros"] = i32T([]int{0, 1})
	_, err = AWQReconstruct(openST(t, bad), "a", in, out)
	wantErr(t, "no groups", err, "groups=0")
	bad = awqFixture(in, []uint32{8, 15})
	bad["a.qweight"] = stTensor{"F32", []int{in, 1}, f32s(make([]float32, in)...)}
	_, err = AWQReconstruct(openST(t, bad), "a", in, out)
	wantErr(t, "qweight not I32", err, "I32")
}

// e2m1 is the FP4 E2M1 value of each nibble, in nibble order.
var e2m1 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

func TestNVFP4E2M1_values(t *testing.T) {
	for n, want := range e2m1 {
		got := NVFP4E2M1ToF32(byte(n))
		if got != want || NVFP4E2M1Table[n] != want {
			t.Errorf("nibble %#x: %v (table %v), want %v", n, got, NVFP4E2M1Table[n], want)
		}
	}
	if z := NVFP4E2M1ToF32(0x8); !math.Signbit(float64(z)) {
		t.Errorf("nibble 0x8 = %v, want -0", z)
	}
	// Only the low nibble is read.
	if NVFP4E2M1ToF32(0xF3) != 1.5 {
		t.Errorf("nibble 0xF3 = %v, want 1.5 (high bits ignored)", NVFP4E2M1ToF32(0xF3))
	}
}

func TestNVFP4DequantRow(t *testing.T) {
	// Two blocks; element 2j is byte j's low nibble, 2j+1 its high nibble.
	packed := make([]byte, 16)
	packed[0], packed[1] = 0x21, 0xF7 // 0.5 1 | 6 -6
	packed[8] = 0x0D                  // block 1: -3 0
	dst := make([]float32, 33)
	dst[32] = 99
	if err := NVFP4DequantRow(packed, []float32{2, 4}, 0.5, 32, dst); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 33)
	copy(want, []float32{0.5, 1, 6, -6})
	want[16] = -6 // -3 * 4 * 0.5
	want[32] = 99 // past in: untouched
	if !slices.Equal(dst, want) {
		t.Errorf("got  %v\nwant %v", dst, want)
	}
	for _, tc := range []struct {
		name   string
		packed []byte
		scales []float32
		in     int
		dst    int
		sub    string
	}{
		{"zero width", nil, nil, 0, 16, "not a positive multiple of 16"},
		{"width 24", make([]byte, 12), []float32{1, 1}, 24, 24, "not a positive multiple of 16"},
		{"short packed", make([]byte, 7), []float32{1}, 16, 16, "7 packed bytes"},
		{"scale per element pair", make([]byte, 8), []float32{1, 1}, 16, 16, "2 scales for 1 blocks"},
		{"short dst", make([]byte, 8), []float32{1}, 16, 15, "dst len 15"},
	} {
		err := NVFP4DequantRow(tc.packed, tc.scales, 1, tc.in, make([]float32, tc.dst))
		wantErr(t, tc.name, err, tc.sub)
	}
}

func TestNVFP4Reconstruct(t *testing.T) {
	const out, in = 2, 16
	w := append(slices.Repeat([]byte{0x22}, 8), slices.Repeat([]byte{0x77}, 8)...) // row 0 all 1.0, row 1 all 6.0
	base := map[string]stTensor{
		"n.weight":         {"U8", []int{out, in / 2}, w},
		"n.weight_scale":   {"F8_E4M3", []int{out, 1}, []byte{0x38, 0x40}}, // 1.0, 2.0
		"n.weight_scale_2": f32T([]int{1}, 0.5),
	}
	st := openST(t, base)
	got, fresh, err := ReconstructMat2D(st, nil, "n.weight", out, in)
	if err != nil || !fresh {
		t.Fatalf("ReconstructMat2D: fresh=%v %v", fresh, err)
	}
	want := append(slices.Repeat([]float32{0.5}, in), slices.Repeat([]float32{6}, in)...)
	if !slices.Equal(got, want) {
		t.Errorf("got  %v\nwant %v", got, want)
	}
	rowInto, err := NVFP4Reconstructor{}.RowReader2D(st, "n.weight", out, in)
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float32, in)
	if err := rowInto(1, row); err != nil || !slices.Equal(row, want[in:]) {
		t.Errorf("row 1: %v, %v", row, err)
	}

	// The per-tensor scale: absent is 1, a scalar [] or [1] is read, anything wider is refused.
	noGlobal := with(base, "n.weight_scale_2", stTensor{})
	if g, err := NVFP4GlobalScale(openST(t, noGlobal), "n"); g != 1 || err != nil {
		t.Errorf("absent weight_scale_2: %v, %v; want 1", g, err)
	}
	if g, err := NVFP4GlobalScale(openST(t, with(base, "n.weight_scale_2", f32T(nil, 0.25))), "n"); g != 0.25 || err != nil {
		t.Errorf("scalar weight_scale_2: %v, %v; want 0.25", g, err)
	}
	_, err = NVFP4Reconstruct(openST(t, with(base, "n.weight_scale_2", f32T([]int{2}, 1, 1))), "n.weight", in, out)
	wantErr(t, "two global scales", err, "want scalar")

	// Block scales may also be stored wide.
	wide := with(base, "n.weight_scale", f32T([]int{out, 1}, 1, 2))
	if got, err := NVFP4Reconstruct(openST(t, wide), "n.weight", in, out); err != nil || !slices.Equal(got, want) {
		t.Errorf("F32 block scales: %v, %v", got, err)
	}
	// An unquantized layer of an NVFP4 checkpoint reads as itself.
	plain := f32T([]int{1, 16}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	if got, err := NVFP4Reconstruct(openST(t, map[string]stTensor{"f.weight": plain}), "f.weight", 16, 1); err != nil || got[15] != 16 {
		t.Errorf("F32 passthrough: %v, %v", got, err)
	}

	for _, tc := range []struct {
		name    string
		tensors map[string]stTensor
		out, in int
		sub     string
	}{
		{"width not a multiple of 16", base, out, 24, "positive multiple of 16"},
		{"zero rows", base, 0, in, "positive multiple of 16"},
		{"unpacked shape", base, out, 2 * in, "want [2,16]"},
		{"I32 payload", with(base, "n.weight", i32T([]int{out, 2}, 0, 0, 0, 0)), out, in, "want U8"},
		{"scale grid of another block size", with(base, "n.weight_scale", stTensor{"F8_E4M3", []int{out, 2}, []byte{1, 1, 1, 1}}), out, in, "n.weight_scale"},
		{"no block scales", with(base, "n.weight_scale", stTensor{}), out, in, "n.weight_scale"},
		{"missing weight", map[string]stTensor{}, out, in, "not found"},
	} {
		_, err := NVFP4RowReader2D(openST(t, tc.tensors), "n.weight", tc.out, tc.in)
		wantErr(t, tc.name, err, tc.sub)
	}
}

// with is a copy of m with key set to v, or removed when v is the zero stTensor.
func with(m map[string]stTensor, key string, v stTensor) map[string]stTensor {
	c := maps.Clone(m)
	if v.dtype == "" {
		delete(c, key)
	} else {
		c[key] = v
	}
	return c
}

func TestTensorU32(t *testing.T) {
	st := openST(t, map[string]stTensor{
		"w":     u32T([]int{2, 2}, 0, 1, 0xFFFFFFFF, 0x80000000),
		"i":     i32T([]int{1}, 1),
		"empty": u32T([]int{0}),
	})
	if got, err := TensorU32(st, "w", 2, 2); err != nil || !slices.Equal(got, []uint32{0, 1, 0xFFFFFFFF, 0x80000000}) {
		t.Errorf("w: %v, %v", got, err)
	}
	if got, err := TensorU32(st, "w"); err != nil || len(got) != 4 {
		t.Errorf("w without a shape check: %v, %v", got, err)
	}
	if got, err := TensorU32(st, "empty"); err != nil || got != nil {
		t.Errorf("empty: %v, %v; want nil, nil", got, err)
	}
	_, err := TensorU32(st, "w", 4)
	wantErr(t, "rank mismatch", err, "!= want [4]")
	_, err = TensorU32(st, "w", 2, 3)
	wantErr(t, "dim mismatch", err, "!= want [2 3]")
	_, err = TensorU32(st, "i")
	wantErr(t, "I32 tensor", err, "expected U32")
	_, err = TensorU32(st, "nope")
	wantErr(t, "missing", err, "not found")
}

func TestMLXDequantAffineRow(t *testing.T) {
	dst := make([]float32, 16)
	if err := MLXDequantAffineRow([]uint32{0x76543210, 0xFEDCBA98}, []float32{1, 0.5}, []float32{-1, 10}, 16, 4, 8, dst); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		want := float32(i) - 1
		if i >= 8 {
			want = float32(i)*0.5 + 10
		}
		if got != want {
			t.Errorf("4-bit [%d] = %v, want %v", i, got, want)
		}
	}
	dst = make([]float32, 8)
	if err := MLXDequantAffineRow([]uint32{0x04030201, 0xFF000080}, []float32{2, 1}, []float32{0, -128}, 8, 8, 4, dst); err != nil {
		t.Fatal(err)
	}
	if want := []float32{2, 4, 6, 8, 0, -128, -128, 127}; !slices.Equal(dst, want) {
		t.Errorf("8-bit: %v, want %v", dst, want)
	}
	// A group that straddles words: in=16, groups of 4 across two 4-bit words.
	dst = make([]float32, 16)
	if err := MLXDequantAffineRow([]uint32{0x11111111, 0x11111111}, []float32{1, 2, 3, 4}, []float32{0, 0, 0, 0}, 16, 4, 4, dst); err != nil {
		t.Fatal(err)
	}
	if dst[3] != 1 || dst[4] != 2 || dst[11] != 3 || dst[12] != 4 {
		t.Errorf("groups of 4: %v", dst)
	}
	for _, tc := range []struct {
		name                string
		words               int
		nScale, in, bits, g int
		dst                 int
		sub                 string
	}{
		{"zero group", 2, 2, 16, 4, 0, 16, "invalid row geometry"},
		{"group not dividing width", 2, 2, 16, 4, 6, 16, "invalid row geometry"},
		{"zero width", 0, 0, 0, 4, 8, 0, "invalid row geometry"},
		{"one scale for two groups", 2, 1, 16, 4, 8, 16, "1 scales / 1 biases for 2 groups"},
		{"short dst", 2, 2, 16, 4, 8, 15, "dst len 15"},
		{"3-bit", 2, 2, 16, 3, 8, 16, "bit-width 3"},
		{"4-bit word count", 3, 2, 16, 4, 8, 16, "3 U32 words for 4-bit"},
		{"4-bit width not a multiple of 8", 1, 3, 12, 4, 4, 12, "for 4-bit row of width 12"},
		{"8-bit width not a multiple of 4", 2, 3, 6, 8, 2, 6, "for 8-bit row of width 6"},
	} {
		s := make([]float32, tc.nScale)
		err := MLXDequantAffineRow(make([]uint32, tc.words), s, s, tc.in, tc.bits, tc.g, make([]float32, tc.dst))
		wantErr(t, tc.name, err, tc.sub)
	}
}

func TestMLXDequantFP4Rows(t *testing.T) {
	// nvfp4: groups of 16, F8_E4M3 scale 2.0; nibble k of a word is element k.
	dst := make([]float32, 16)
	if err := MLXDequantNVFP4Row([]uint32{0x76543210, 0xFEDCBA98}, []byte{0x40}, 16, dst); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 2*e2m1[i] {
			t.Errorf("nvfp4 [%d] = %v, want %v", i, got, 2*e2m1[i])
		}
	}
	// mxfp4: groups of 32, E8M0 scales 2^(e-127): 127 is 1, 129 is 4.
	words := slices.Repeat([]uint32{0x76543210}, 8)
	dst = make([]float32, 64)
	if err := MLXDequantMXFP4Row(words, []byte{127, 129}, 64, dst); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		want := e2m1[i%8]
		if i >= 32 {
			want *= 4
		}
		if got != want {
			t.Errorf("mxfp4 [%d] = %v, want %v", i, got, want)
		}
	}
	err := MLXDequantNVFP4Row(make([]uint32, 1), []byte{1}, 8, make([]float32, 8))
	wantErr(t, "nvfp4 width 8", err, "not a positive multiple of 16")
	err = MLXDequantNVFP4Row(make([]uint32, 2), []byte{1, 1}, 16, make([]float32, 16))
	wantErr(t, "nvfp4 scale per 8", err, "2 scales")
	err = MLXDequantNVFP4Row(make([]uint32, 3), []byte{1}, 16, make([]float32, 16))
	wantErr(t, "nvfp4 word count", err, "3 U32 words")
	err = MLXDequantMXFP4Row(make([]uint32, 2), []byte{1}, 16, make([]float32, 16))
	wantErr(t, "mxfp4 width 16", err, "not a positive multiple of 32")
	err = MLXDequantMXFP4Row(make([]uint32, 4), []byte{1, 1}, 32, make([]float32, 32))
	wantErr(t, "mxfp4 scale per 16", err, "2 scales")
	err = MLXDequantMXFP4Row(make([]uint32, 4), nil, 0, nil)
	wantErr(t, "mxfp4 zero width", err, "not a positive multiple of 32")
}

func TestMLXRowReader2D(t *testing.T) {
	ramp := slices.Repeat([]uint32{0x76543210}, 4) // 32 4-bit elements, element i = i%8
	affine := map[string]stTensor{
		"m.weight": u32T([]int{2, 4}, slices.Concat(ramp, ramp)[:8]...),
		"m.scales": f32T([]int{2, 1}, 1, 2),
		"m.biases": f32T([]int{2, 1}, 0, -1),
	}
	got, err := MLXReconstructor{}.ReconstructMat2D(openST(t, affine), "m.weight", 2, 32)
	if err != nil {
		t.Fatal(err)
	}
	for i, g := range got {
		k := float32(i % 8)
		want := k
		if i >= 32 {
			want = 2*k - 1
		}
		if g != want {
			t.Errorf("affine 4-bit [%d] = %v, want %v", i, g, want)
		}
	}
	// 8-bit is inferred from the packed width: [1, 8] words for 32 elements.
	eight := map[string]stTensor{
		"e.weight": u32T([]int{1, 8}, slices.Repeat([]uint32{0xFF000102}, 8)...),
		"e.scales": f32T([]int{1, 2}, 1, 0.5),
		"e.biases": f32T([]int{1, 2}, 0, 0),
	}
	if got, err := MLXReconstruct2D(openST(t, eight), "e.weight", 1, 32); err != nil || got[0] != 2 || got[3] != 255 || got[16] != 1 || got[19] != 127.5 {
		t.Errorf("affine 8-bit: %v, %v", got, err)
	}
	// Bias-free modes: nvfp4 (groups of 16, fp8 scales) and mxfp4 (groups of 32, E8M0 scales).
	fp4 := map[string]stTensor{
		"nv.weight": u32T([]int{1, 2}, 0x76543210, 0xFEDCBA98),
		"nv.scales": {"F8_E4M3", []int{1, 1}, []byte{0x38}},
		"mx.weight": u32T([]int{1, 4}, ramp...),
		"mx.scales": {"U8", []int{1, 1}, []byte{128}},
	}
	st := openST(t, fp4)
	if got, err := MLXReconstruct2D(st, "nv.weight", 1, 16); err != nil || !slices.Equal(got, e2m1[:]) {
		t.Errorf("nvfp4: %v, %v", got, err)
	}
	if got, err := MLXReconstruct2D(st, "mx.weight", 1, 32); err != nil || got[7] != 12 || got[31] != 12 || got[1] != 1 {
		t.Errorf("mxfp4 (scale 2): %v, %v", got, err)
	}
	if got, err := MLXReconstruct2D(openST(t, map[string]stTensor{"f.weight": f32T([]int{1, 2}, 3, 4)}), "f.weight", 1, 2); err != nil || !slices.Equal(got, []float32{3, 4}) {
		t.Errorf("F32 passthrough: %v, %v", got, err)
	}

	for _, tc := range []struct {
		name    string
		tensors map[string]stTensor
		w       string
		out, in int
		sub     string
	}{
		{"F32 weight", map[string]stTensor{"f.weight": f32T([]int{1, 2}, 3, 4)}, "f.weight", 1, 2, "want U32"},
		{"zero rows", affine, "m.weight", 0, 32, "incompatible"},
		{"rows disagreeing", affine, "m.weight", 3, 32, "incompatible"},
		{"2-bit inferred", affine, "m.weight", 2, 64, "inferred bits=2"},
		{"16-bit inferred", affine, "m.weight", 2, 8, "inferred bits=16"},
		{"packed width not a whole element count", affine, "m.weight", 2, 48, "not divisible by in=48"},
		{"scales not dividing the row", with(affine, "m.scales", f32T([]int{2, 3}, 1, 1, 1, 1, 1, 1)), "m.weight", 2, 32, "m.scales"},
		{"scales for other rows", with(affine, "m.scales", f32T([]int{1, 1}, 1)), "m.weight", 2, 32, "m.scales"},
		{"no scales", with(affine, "m.scales", stTensor{}), "m.weight", 2, 32, "m.scales"},
		{"biases of another shape", with(affine, "m.biases", f32T([]int{2, 2}, 0, 0, 0, 0)), "m.weight", 2, 32, "m.biases"},
		{"8-bit without biases", with(eight, "e.biases", stTensor{}), "e.weight", 1, 32, `missing "e.biases" for 8-bit`},
		{"bias-free groups of 64", map[string]stTensor{"g.weight": u32T([]int{1, 8}, make([]uint32, 8)...),
			"g.scales": {"U8", []int{1, 1}, []byte{127}}}, "g.weight", 1, 64, "group_size=64"},
		{"bias-free groups of 32 with fp8 scales", map[string]stTensor{"g.weight": u32T([]int{1, 4}, ramp...),
			"g.scales": {"F8_E4M3", []int{1, 1}, []byte{0x38}}}, "g.weight", 1, 32, "scales dtype=F8_E4M3"},
		{"bias-free with F32 scales", map[string]stTensor{"g.weight": u32T([]int{1, 2}, 0, 0),
			"g.scales": f32T([]int{1, 1}, 1)}, "g.weight", 1, 16, "g.scales"},
	} {
		_, err := MLXRowReader2D(openST(t, tc.tensors), tc.w, tc.out, tc.in)
		wantErr(t, tc.name, err, tc.sub)
	}
}

func TestMLXExpertSlice3D(t *testing.T) {
	st := openST(t, map[string]stTensor{
		"x.weight": f32T([]int{2, 2, 2}, 1, 2, 3, 4, 5, 6, 7, 8),
		// Two experts of one 32-wide 4-bit row: expert e is k*scale[e] + bias[e].
		"s.weight": u32T([]int{2, 1, 4}, slices.Repeat([]uint32{0x76543210}, 8)...),
		"s.scales": f32T([]int{2, 1, 1}, 1, 3),
		"s.biases": f32T([]int{2, 1, 1}, 0, 1),
		// One bias-free nvfp4 expert, fp8 scale 2.0.
		"n.weight": u32T([]int{1, 1, 2}, 0x76543210, 0xFEDCBA98),
		"n.scales": {"F8_E4M3", []int{1, 1, 1}, []byte{0x40}},
		"i.weight": i32T([]int{1, 1, 1}, 0),
	})
	read, err := MLXExpertSlice3D(st, "x", 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	e1, err := read(1)
	if err != nil || !slices.Equal(e1, []float32{5, 6, 7, 8}) {
		t.Fatalf("F32 expert 1: %v, %v", e1, err)
	}
	e1[0] = -1 // the reader hands out a copy of an F32 mapping
	if again, _ := read(1); again[0] != 5 {
		t.Errorf("writing a returned expert changed the next read: %v", again)
	}

	read, err = MLXExpertSlice3D(st, "s", 2, 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if e1, err := read(1); err != nil || e1[0] != 1 || e1[7] != 22 || e1[31] != 22 {
		t.Errorf("affine expert 1: %v, %v", e1, err)
	}
	if e0, err := read(0); err != nil || e0[7] != 7 {
		t.Errorf("affine expert 0: %v, %v", e0, err)
	}

	read, err = MLXExpertSlice3D(st, "n", 1, 1, 16)
	if err != nil {
		t.Fatal(err)
	}
	if e0, err := read(0); err != nil || e0[7] != 12 || e0[15] != -12 {
		t.Errorf("nvfp4 expert 0: %v, %v", e0, err)
	}

	for _, tc := range []struct {
		name             string
		prefix           string
		nExpert, out, in int
		sub              string
	}{
		{"F32 of another expert count", "x", 3, 2, 2, "want [3 2 2]"},
		{"F32 of another width", "x", 2, 2, 3, "want [2 2 3]"},
		{"I32 weights", "i", 1, 1, 1, "want U32 or F32"},
		{"quantized of another expert count", "s", 3, 1, 32, "incompatible"},
		{"2-bit inferred", "s", 2, 1, 64, "inferred bits=2"},
		{"packed width not a whole element count", "s", 2, 1, 48, "not divisible by in=48"},
		{"missing", "nope", 1, 1, 1, "not found"},
	} {
		_, err := MLXExpertSlice3D(st, tc.prefix, tc.nExpert, tc.out, tc.in)
		wantErr(t, tc.name, err, tc.sub)
	}
}

// MLXRowReader2D refuses a bias-free tensor whose scale dtype does not match its group size (E8M0 U8 for groups of 32,
// fp8 for 16); the stacked-expert reader should refuse the same tensors rather than decode fp8 bytes as E8M0.
func TestMLXExpertSlice3D_biasFreeModeChecked(t *testing.T) {
	st := openST(t, map[string]stTensor{
		"g.weight": u32T([]int{1, 1, 4}, slices.Repeat([]uint32{0x76543210}, 4)...),
		"g.scales": {"F8_E4M3", []int{1, 1, 1}, []byte{0x38}},
	})
	if _, err := MLXRowReader2D(openST(t, map[string]stTensor{
		"g.weight": u32T([]int{1, 4}, slices.Repeat([]uint32{0x76543210}, 4)...),
		"g.scales": {"F8_E4M3", []int{1, 1}, []byte{0x38}},
	}), "g.weight", 1, 32); err == nil {
		t.Fatal("the 2-D reader accepts fp8 scales on groups of 32")
	}
	read, err := MLXExpertSlice3D(st, "g", 1, 1, 32)
	if err == nil {
		e0, rerr := read(0)
		t.Errorf("MLXExpertSlice3D accepts fp8 scales on groups of 32 and decodes them as E8M0: expert 0 = %v (err %v); "+
			"fp8 0x38 is 1.0, so element 7 should be 6", e0, rerr)
	}
}

func TestClassifyAndForTensor(t *testing.T) {
	st := openST(t, map[string]stTensor{
		"mlx.weight":      u32T([]int{1, 1}, 0),
		"mlx.scales":      f32T([]int{1, 1}, 1),
		"u32.weight":      u32T([]int{1, 1}, 0),
		"nv.weight":       {"U8", []int{1, 1}, []byte{0}},
		"nv.weight_scale": {"F8_E4M3", []int{1, 1}, []byte{0x38}},
		"u8.weight":       {"U8", []int{1, 1}, []byte{0}},
		"f.weight":        f32T([]int{1, 2}, 3, 4),
		"f.scales":        f32T([]int{1, 1}, 1),
		"bias":            u32T([]int{1, 1}, 0),
		"bias.scales":     f32T([]int{1, 1}, 1),
	})
	for _, tc := range []struct {
		name   string
		want   TensorFormat
		method string
	}{
		{"mlx.weight", FormatMLX, "mlx"},
		{"u32.weight", FormatNone, ""},
		{"nv.weight", FormatNVFP4, "nvfp4"},
		{"u8.weight", FormatNone, ""},
		{"f.weight", FormatNone, ""},
		{"bias", FormatNone, ""}, // not a .weight tensor
		{"missing.weight", FormatNone, ""},
	} {
		if got := ClassifyTensor(st, tc.name); got != tc.want {
			t.Errorf("ClassifyTensor(%s) = %d, want %d", tc.name, got, tc.want)
		}
		r := ForTensor(st, nil, tc.name)
		if (r == nil) != (tc.method == "") || (r != nil && r.Method() != tc.method) {
			t.Errorf("ForTensor(%s) = %v, want method %q", tc.name, r, tc.method)
		}
	}
	// A checkpoint-level config decides for every tensor, a plain one included.
	if r := ForTensor(st, &Config{Method: "awq"}, "f.weight"); r == nil || r.Method() != "awq" {
		t.Errorf("ForTensor with a config: %v, want awq", r)
	}
	data, fresh, err := ReconstructMat2D(st, nil, "f.weight", 1, 2)
	if err != nil || fresh || !slices.Equal(data, []float32{3, 4}) {
		t.Errorf("unquantized ReconstructMat2D: %v fresh=%v %v", data, fresh, err)
	}
	_, _, err = ReconstructMat2D(st, nil, "f.weight", 2, 1)
	wantErr(t, "unquantized of another shape", err, "!= want [2 1]")
}
