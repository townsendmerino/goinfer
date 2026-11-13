package quant

import (
	"encoding/json"
	"math"
	"testing"
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
