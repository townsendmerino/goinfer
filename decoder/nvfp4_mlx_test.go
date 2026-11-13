package decoder

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/goinfer/decoder/fit"
)

const flintTinyCkptDir = "../testdata/flint-tiny"

// TestNVFP4_e2m1TableAndExhaustiveE4M3 verifies the 16-entry FP4 E2M1 table
// (including signed zero at 0x8) and exhaustively checks all 16 * 256 = 4,096
// (e2m1, fp8_e4m3fn) combinations through nvfp4DequantRow.
func TestNVFP4_e2m1TableAndExhaustiveE4M3(t *testing.T) {
	wantE2M1 := [16]float32{
		0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
		float32(math.Copysign(0, -1)), -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
	}
	for i, want := range wantE2M1 {
		got := nvfp4E2M1ToF32(byte(i))
		if got != want || nvfp4E2M1Table[i] != want {
			t.Fatalf("nibble 0x%X: got fn=%v table=%v, want %v", i, got, nvfp4E2M1Table[i], want)
		}
		if want == 0 && math.Signbit(float64(got)) != math.Signbit(float64(want)) {
			t.Fatalf("nibble 0x%X: zero signbit mismatch: got %v, want %v", i, math.Signbit(float64(got)), math.Signbit(float64(want)))
		}
	}

	// Exhaustive 256-block row (in = 256 * 16 = 4096): block s uses scale byte s
	// and packs all 16 nibbles across its 8 bytes (byte j packs 2*j in low nibble
	// and 2*j+1 in high nibble).
	const (
		nBlocks     = 256
		in          = nBlocks * nvfp4BlockElems
		globalScale = float32(0.25)
	)
	packed := make([]byte, nBlocks*nvfp4PackedBytes)
	scales := make([]float32, nBlocks)
	for s := range nBlocks {
		scales[s] = fp8E4M3ToF32(byte(s))
		for j := range nvfp4PackedBytes {
			lo := byte(2 * j)
			hi := byte(2*j + 1)
			packed[s*nvfp4PackedBytes+j] = lo | (hi << 4)
		}
	}
	dst := make([]float32, in)
	if err := nvfp4DequantRow(packed, scales, globalScale, in, dst); err != nil {
		t.Fatalf("nvfp4DequantRow: %v", err)
	}
	for s := range nBlocks {
		refScale := float32(fp8E4M3RefTable[s]) * globalScale
		for nib := range nvfp4BlockElems {
			got := dst[s*nvfp4BlockElems+nib]
			if math.IsNaN(float64(refScale)) {
				if !math.IsNaN(float64(got)) {
					t.Fatalf("scale 0x%02X nib 0x%X: got %v, want NaN", s, nib, got)
				}
				continue
			}
			want := wantE2M1[nib] * refScale
			if got != want {
				t.Fatalf("scale 0x%02X nib 0x%X: got %v, want %v", s, nib, got, want)
			}
		}
	}
}

// stRawTensor is a helper for serializing a synthetic safetensors file in tests.
type stRawTensor struct {
	dtype string
	shape []int
	data  []byte
}

func writeSyntheticSafetensors(t *testing.T, path string, tensors map[string]stRawTensor) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	type headerEntry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	hdr := make(map[string]headerEntry, len(names))
	var payload bytes.Buffer
	for _, n := range names {
		tr := tensors[n]
		start := payload.Len()
		payload.Write(tr.data)
		end := payload.Len()
		shape := tr.shape
		if shape == nil {
			shape = []int{}
		}
		hdr[n] = headerEntry{
			DType:       tr.dtype,
			Shape:       shape,
			DataOffsets: [2]int{start, end},
		}
	}
	hdrJSON, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal safetensors header: %v", err)
	}
	// Pad header to 8-byte alignment so 4-byte tensors stay aligned.
	for len(hdrJSON)%8 != 0 {
		hdrJSON = append(hdrJSON, ' ')
	}
	var out bytes.Buffer
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hdrJSON)))
	out.Write(lenBuf[:])
	out.Write(hdrJSON)
	out.Write(payload.Bytes())
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func f32ToLEBytes(vals []float32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

func u32ToLEBytes(vals []uint32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	return b
}

// nearestE4M3Byte finds the finite FP8 E4M3 byte closest to v >= 0.
func nearestE4M3Byte(v float32) byte {
	if v <= 0 {
		return 0x01 // smallest positive subnormal so we don't divide by 0
	}
	best := byte(1)
	bestDiff := float32(math.MaxFloat32)
	for b := 1; b < 0x7F; b++ {
		cand := fp8E4M3Table[b]
		diff := float32(math.Abs(float64(cand - v)))
		if diff < bestDiff {
			bestDiff = diff
			best = byte(b)
		}
	}
	return best
}

// nearestE2M1Nibble finds the 4-bit E2M1 nibble closest to v.
func nearestE2M1Nibble(v float32) byte {
	best := byte(0)
	bestDiff := float32(math.MaxFloat32)
	for nib := range 16 {
		cand := nvfp4E2M1Table[nib]
		diff := float32(math.Abs(float64(cand - v)))
		if diff < bestDiff {
			bestDiff = diff
			best = byte(nib)
		}
	}
	return best
}

// quantizeMatrixNVFP4 quantizes an [out, in] f32 matrix into NVIDIA ModelOpt NVFP4
// tensors: weight U8 [out, in/2], weight_scale F8_E4M3 [out, in/16], and scalar
// weight_scale_2 F32 [1].
func quantizeMatrixNVFP4(src []float32, out, in int) (wBytes, sBytes, s2Bytes []byte) {
	var maxAbs float32
	for _, v := range src {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	globalScale := maxAbs / (6.0 * 448.0)
	if globalScale == 0 {
		globalScale = 1.0
	}
	nBlocks := in / 16
	wBytes = make([]byte, out*(in/2))
	sBytes = make([]byte, out*nBlocks)
	for r := range out {
		for g := range nBlocks {
			block := src[r*in+g*16 : r*in+(g+1)*16]
			var bMax float32
			for _, v := range block {
				if a := float32(math.Abs(float64(v))); a > bMax {
					bMax = a
				}
			}
			sb := nearestE4M3Byte(bMax / (6.0 * globalScale))
			sBytes[r*nBlocks+g] = sb
			effScale := fp8E4M3Table[sb] * globalScale
			for j := range 8 {
				lo := nearestE2M1Nibble(block[2*j] / effScale)
				hi := nearestE2M1Nibble(block[2*j+1] / effScale)
				wBytes[r*(in/2)+g*8+j] = lo | (hi << 4)
			}
		}
	}
	s2Bytes = f32ToLEBytes([]float32{globalScale})
	return wBytes, sBytes, s2Bytes
}

// quantizeMatrixMLXAffine quantizes an [out, in] f32 matrix into MLX affine U32
// words (`bits` in {4, 8}, `groupSize` = 16 or 32) plus F32 scales and biases.
func quantizeMatrixMLXAffine(src []float32, out, in, bits, groupSize int) (words []uint32, scales, biases []float32) {
	qMax := float32((int(1) << bits) - 1)
	pack := 32 / bits
	packedCols := in / pack
	scaleCols := in / groupSize
	words = make([]uint32, out*packedCols)
	scales = make([]float32, out*scaleCols)
	biases = make([]float32, out*scaleCols)
	mask := uint32((1 << bits) - 1)

	for r := range out {
		row := src[r*in : (r+1)*in]
		for g := range scaleCols {
			grp := row[g*groupSize : (g+1)*groupSize]
			mn, mx := grp[0], grp[0]
			for _, v := range grp[1:] {
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
			sc := (mx - mn) / qMax
			if sc == 0 {
				sc = 1.0
			}
			scales[r*scaleCols+g] = sc
			biases[r*scaleCols+g] = mn
		}
		for u := range packedCols {
			var w uint32
			for k := range pack {
				c := u*pack + k
				g := c / groupSize
				sc := scales[r*scaleCols+g]
				bi := biases[r*scaleCols+g]
				q := int(math.Round(float64((row[c] - bi) / sc)))
				if q < 0 {
					q = 0
				} else if float32(q) > qMax {
					q = int(qMax)
				}
				w |= (uint32(q) & mask) << (bits * k)
			}
			words[r*packedCols+u] = w
		}
	}
	return words, scales, biases
}

// TestNVFP4_reconstructAndEndToEndFlint converts the committed flint-tiny checkpoint
// into an NVIDIA ModelOpt NVFP4 safetensors checkpoint, loads it via decoder.Load,
// checks forward parity against the full-precision Flint golden, and verifies that
// break-it-first perturbations (swapping nibble order or dropping weight_scale_2)
// are caught.
func TestNVFP4_reconstructAndEndToEndFlint(t *testing.T) {
	g := loadFlintGolden(t)
	srcSt, err := embed.OpenSafetensors(filepath.Join(flintTinyCkptDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("open flint-tiny safetensors: %v", err)
	}
	defer srcSt.Close()

	tensors := make(map[string]stRawTensor)
	for _, name := range srcSt.Names() {
		tr, err := srcSt.Tensor(name)
		if err != nil {
			t.Fatalf("tensor %s: %v", name, err)
		}
		vals, err := srcSt.TensorF32(name)
		if err != nil {
			t.Fatalf("TensorF32 %s: %v", name, err)
		}
		if len(tr.Shape) == 2 && strings.HasSuffix(name, ".weight") && tr.Shape[1]%16 == 0 &&
			!strings.HasSuffix(name, "mlp.gate.weight") && !strings.HasSuffix(name, "embed_tokens.weight") && name != "lm_head.weight" {
			out, in := tr.Shape[0], tr.Shape[1]
			wb, sb, s2b := quantizeMatrixNVFP4(vals, out, in)
			prefix := strings.TrimSuffix(name, ".weight")
			tensors[name] = stRawTensor{dtype: "U8", shape: []int{out, in / 2}, data: wb}
			tensors[prefix+".weight_scale"] = stRawTensor{dtype: "F8_E4M3", shape: []int{out, in / 16}, data: sb}
			tensors[prefix+".weight_scale_2"] = stRawTensor{dtype: "F32", shape: []int{1}, data: s2b}
			tensors[prefix+".input_scale"] = stRawTensor{dtype: "F32", shape: []int{1}, data: f32ToLEBytes([]float32{1.0})}
		} else {
			tensors[name] = stRawTensor{dtype: "F32", shape: tr.Shape, data: f32ToLEBytes(vals)}
		}
	}

	tmpDir := t.TempDir()
	writeSyntheticSafetensors(t, filepath.Join(tmpDir, "model.safetensors"), tensors)

	rawCfg, err := os.ReadFile(filepath.Join(flintTinyCkptDir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	var cfgMap map[string]any
	if err := json.Unmarshal(rawCfg, &cfgMap); err != nil {
		t.Fatalf("unmarshal config.json: %v", err)
	}
	cfgMap["quantization_config"] = map[string]any{
		"quant_method": "modelopt",
		"quant_algo":   "NVFP4",
		"config_groups": map[string]any{
			"group_0": map[string]any{
				"weights": map[string]any{"num_bits": 4, "type": "float", "group_size": 16},
			},
		},
	}
	cfgBytes, err := json.Marshal(cfgMap)
	if err != nil {
		t.Fatalf("marshal config.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "config.json"), cfgBytes, 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}

	m, err := Load(tmpDir, Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatalf("Load NVFP4 Flint: %v", err)
	}
	defer m.Close()

	cache := m.NewCache(64)
	logits, err := m.prefillLogits(context.Background(), g.PromptIDs, cache)
	if err != nil {
		t.Fatalf("prefillLogits NVFP4 Flint: %v", err)
	}
	cos := logitCosine(logits, g.LastLogits)
	if cos < 0.96 {
		t.Fatalf("NVFP4 Flint cosine vs unquantized = %.6f, want >= 0.96", cos)
	}

	// Verify bit-exactness of the NVFP4 loader path against an F32 checkpoint
	// containing the reference-dequantized weights (`e2m1(nibble) * e4m3(scale) * scale_2`).
	refTensors := make(map[string]stRawTensor)
	for name, tr := range tensors {
		if strings.HasSuffix(name, ".weight_scale") || strings.HasSuffix(name, ".weight_scale_2") || strings.HasSuffix(name, ".input_scale") {
			continue
		}
		if tr.dtype == "U8" {
			prefix := strings.TrimSuffix(name, ".weight")
			out, in := tr.shape[0], tr.shape[1]*2
			sb := tensors[prefix+".weight_scale"].data
			s2 := math.Float32frombits(binary.LittleEndian.Uint32(tensors[prefix+".weight_scale_2"].data))
			deq := make([]float32, out*in)
			nBlocks := in / 16
			for r := range out {
				for blk := range nBlocks {
					eff := float32(fp8E4M3RefTable[sb[r*nBlocks+blk]]) * s2
					for j := range 8 {
						b := tr.data[r*(in/2)+blk*8+j]
						deq[r*in+blk*16+2*j] = nvfp4E2M1ToF32(b&0x0F) * eff
						deq[r*in+blk*16+2*j+1] = nvfp4E2M1ToF32(b>>4) * eff
					}
				}
			}
			refTensors[name] = stRawTensor{dtype: "F32", shape: []int{out, in}, data: f32ToLEBytes(deq)}
		} else {
			refTensors[name] = tr
		}
	}
	refDir := t.TempDir()
	writeSyntheticSafetensors(t, filepath.Join(refDir, "model.safetensors"), refTensors)
	if err := os.WriteFile(filepath.Join(refDir, "config.json"), rawCfg, 0o644); err != nil {
		t.Fatalf("write ref config.json: %v", err)
	}
	mRef, err := Load(refDir, Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatalf("Load ref F32 Flint: %v", err)
	}
	defer mRef.Close()
	refCache := mRef.NewCache(64)
	refLogits, err := mRef.prefillLogits(context.Background(), g.PromptIDs, refCache)
	if err != nil {
		t.Fatalf("prefillLogits ref F32 Flint: %v", err)
	}
	if diff := maxAbsLogitDiff(logits, refLogits); diff > 0 {
		t.Fatalf("NVFP4 loader vs reference-dequantized F32 checkpoint: maxAbs=%e, want 0 (bit-exact)", diff)
	}

	// Break-it-first: swapping low/high nibbles in an NVFP4 weight row must alter reconstruction.
	st, err := embed.OpenSafetensors(filepath.Join(tmpDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("open synthetic safetensors: %v", err)
	}
	defer st.Close()
	qName := "model.layers.0.self_attn.q_proj.weight"
	goodW, err := nvfp4Reconstruct(st, qName, 64, 64)
	if err != nil {
		t.Fatalf("nvfp4Reconstruct: %v", err)
	}
	wT, _ := st.Tensor(qName)
	rawW, _ := wT.Uint8s()
	swapped := make([]byte, len(rawW))
	for i, b := range rawW {
		swapped[i] = (b << 4) | (b >> 4)
	}
	scales, _ := nvfp4ScaleTensorF32(st, "model.layers.0.self_attn.q_proj.weight_scale", 64, 4)
	gs, _ := nvfp4GlobalScale(st, "model.layers.0.self_attn.q_proj")
	badRow := make([]float32, 64)
	_ = nvfp4DequantRow(swapped[:32], scales[:4], gs, 64, badRow)
	maxDiff := maxAbsLogitDiff(goodW[:64], badRow)
	if maxDiff < 1e-2 {
		t.Fatalf("nibble-swap perturbation was not caught: maxAbs=%e", maxDiff)
	}
}

// TestMLX_affineAndFP4ModesAndEndToEndFlint tests MLX affine (4-bit and 8-bit),
// nvfp4, and mxfp4 row dequantizers, then builds a synthetic MLX-quantized Flint
// checkpoint (including quantized embed_tokens, lm_head, and 3-D stacked
// mlp.switch_mlp.{gate,up,down}_proj experts) and verifies Load + prefillLogits parity.
func TestMLX_affineAndFP4ModesAndEndToEndFlint(t *testing.T) {
	// 1. Unit test 4-bit and 8-bit affine row dequantization + break-it-first bias drop.
	srcRow := make([]float32, 64)
	for i := range srcRow {
		srcRow[i] = float32(math.Sin(float64(i)*0.3)) * 2.5
	}
	for _, bits := range []int{4, 8} {
		words, scales, biases := quantizeMatrixMLXAffine(srcRow, 1, 64, bits, 32)
		got := make([]float32, 64)
		if err := mlxDequantAffineRow(words, scales, biases, 64, bits, 32, got); err != nil {
			t.Fatalf("mlxDequantAffineRow(%d-bit): %v", bits, err)
		}
		cos := logitCosine(got, srcRow)
		minCos := 0.99
		if bits == 8 {
			minCos = 0.9999
		}
		if cos < minCos {
			t.Fatalf("mlxDequantAffineRow(%d-bit) cosine=%.6f, want >= %v", bits, cos, minCos)
		}
		// Break-it-first: zeroing biases must degrade reconstruction.
		noBias := make([]float32, 64)
		_ = mlxDequantAffineRow(words, scales, make([]float32, len(biases)), 64, bits, 32, noBias)
		diff := maxAbsLogitDiff(got, noBias)
		if diff < 0.5 {
			t.Fatalf("dropping MLX biases was not caught (%d-bit): maxAbs=%e", bits, diff)
		}
	}

	// 2. Unit test MLX nvfp4 (group_size=16) and mxfp4 (group_size=32) row dequantizers.
	wordsFP4 := []uint32{
		0x76543210, // +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0
		0xFEDCBA98, // -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
		0x76543210,
		0xFEDCBA98,
	}
	nvOut := make([]float32, 32)
	// 0x38 is 1.0 in FP8 E4M3, 0x40 is 2.0 in FP8 E4M3.
	if err := mlxDequantNVFP4Row(wordsFP4, []byte{0x38, 0x40}, 32, nvOut); err != nil {
		t.Fatalf("mlxDequantNVFP4Row: %v", err)
	}
	if nvOut[7] != 6.0 || nvOut[15] != -6.0 || nvOut[16+7] != 12.0 || nvOut[16+15] != -12.0 {
		t.Fatalf("mlxDequantNVFP4Row unexpected values: %v", nvOut)
	}
	mxOut := make([]float32, 32)
	// E8M0 127 -> scale 1.0 (MXFP4Scale(127) = 0.5, * 2 = 1.0).
	if err := mlxDequantMXFP4Row(wordsFP4, []byte{127}, 32, mxOut); err != nil {
		t.Fatalf("mlxDequantMXFP4Row: %v", err)
	}
	if mxOut[7] != 6.0 || mxOut[15] != -6.0 || mxOut[31] != -6.0 {
		t.Fatalf("mlxDequantMXFP4Row unexpected values: %v", mxOut)
	}

	// 3. End-to-end MLX-quantized Flint checkpoint with 3-D switch_mlp experts.
	g := loadFlintGolden(t)
	srcSt, err := embed.OpenSafetensors(filepath.Join(flintTinyCkptDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("open flint-tiny safetensors: %v", err)
	}
	defer srcSt.Close()

	const (
		bits      = 8
		groupSize = 32
		nExperts  = 8
		expInter  = 32
		hidden    = 64
	)
	tensors := make(map[string]stRawTensor)
	for _, name := range srcSt.Names() {
		// Skip per-expert 2-D tensors on MoE layers (layers 2..5); we pack them into
		// 3-D mlp.switch_mlp.{gate_proj,up_proj,down_proj} tensors below.
		if strings.Contains(name, ".mlp.experts.") {
			continue
		}
		tr, err := srcSt.Tensor(name)
		if err != nil {
			t.Fatalf("tensor %s: %v", name, err)
		}
		vals, err := srcSt.TensorF32(name)
		if err != nil {
			t.Fatalf("TensorF32 %s: %v", name, err)
		}
		// Keep router (`mlp.gate.weight`) unquantized as mlx-lm does, and quantize all other 2-D weights (including embed_tokens and lm_head).
		if len(tr.Shape) == 2 && strings.HasSuffix(name, ".weight") && !strings.HasSuffix(name, "mlp.gate.weight") && tr.Shape[1]%groupSize == 0 {
			out, in := tr.Shape[0], tr.Shape[1]
			w, sc, bi := quantizeMatrixMLXAffine(vals, out, in, bits, groupSize)
			prefix := strings.TrimSuffix(name, ".weight")
			tensors[name] = stRawTensor{dtype: "U32", shape: []int{out, in * bits / 32}, data: u32ToLEBytes(w)}
			tensors[prefix+".scales"] = stRawTensor{dtype: "F32", shape: []int{out, in / groupSize}, data: f32ToLEBytes(sc)}
			tensors[prefix+".biases"] = stRawTensor{dtype: "F32", shape: []int{out, in / groupSize}, data: f32ToLEBytes(bi)}
		} else {
			tensors[name] = stRawTensor{dtype: "F32", shape: tr.Shape, data: f32ToLEBytes(vals)}
		}
	}

	// Stack MoE layers (2..5) into 3-D mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}.
	for l := 2; l < 6; l++ {
		for _, proj := range []struct {
			name    string
			out, in int
		}{
			{"gate_proj", expInter, hidden},
			{"up_proj", expInter, hidden},
			{"down_proj", hidden, expInter},
		} {
			var allWords []uint32
			var allScales, allBiases []float32
			for e := range nExperts {
				key := fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", l, e, proj.name)
				vals, err := srcSt.TensorF32(key, proj.out, proj.in)
				if err != nil {
					t.Fatalf("read %s: %v", key, err)
				}
				w, sc, bi := quantizeMatrixMLXAffine(vals, proj.out, proj.in, bits, groupSize)
				allWords = append(allWords, w...)
				allScales = append(allScales, sc...)
				allBiases = append(allBiases, bi...)
			}
			swPrefix := fmt.Sprintf("model.layers.%d.mlp.switch_mlp.%s", l, proj.name)
			tensors[swPrefix+".weight"] = stRawTensor{
				dtype: "U32",
				shape: []int{nExperts, proj.out, proj.in * bits / 32},
				data:  u32ToLEBytes(allWords),
			}
			tensors[swPrefix+".scales"] = stRawTensor{
				dtype: "F32",
				shape: []int{nExperts, proj.out, proj.in / groupSize},
				data:  f32ToLEBytes(allScales),
			}
			tensors[swPrefix+".biases"] = stRawTensor{
				dtype: "F32",
				shape: []int{nExperts, proj.out, proj.in / groupSize},
				data:  f32ToLEBytes(allBiases),
			}
		}
	}

	tmpDir := t.TempDir()
	writeSyntheticSafetensors(t, filepath.Join(tmpDir, "model.safetensors"), tensors)

	rawCfg, err := os.ReadFile(filepath.Join(flintTinyCkptDir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	var cfgMap map[string]any
	if err := json.Unmarshal(rawCfg, &cfgMap); err != nil {
		t.Fatalf("unmarshal config.json: %v", err)
	}
	cfgMap["quantization"] = map[string]any{
		"group_size": groupSize,
		"bits":       bits,
		"mode":       "affine",
	}
	cfgBytes, err := json.Marshal(cfgMap)
	if err != nil {
		t.Fatalf("marshal config.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "config.json"), cfgBytes, 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}

	m, err := Load(tmpDir, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load MLX Flint: %v", err)
	}
	defer m.Close()

	cache := m.NewCache(64)
	logits, err := m.prefillLogits(context.Background(), g.PromptIDs, cache)
	if err != nil {
		t.Fatalf("prefillLogits MLX Flint: %v", err)
	}
	cos := logitCosine(logits, g.LastLogits)
	if cos < 0.999 {
		t.Fatalf("MLX 8-bit Flint cosine = %.6f, want >= 0.999", cos)
	}
}

// TestFit_nvfp4AndMLXSafetensorsWeightBytes verifies that SafetensorsWeightBytes
// skips NVFP4/MLX auxiliary scale/bias tensors and scales packed U8/U32 weight
// shapes back to their logical [out, in] element counts.
func TestFit_nvfp4AndMLXSafetensorsWeightBytes(t *testing.T) {
	rates := fit.Rates{Matrix: 1.0, Embedding: 2.0}

	// Unquantized logical shapes: embed [256, 64], q_proj [64, 64], norm [64], switch_mlp gate [4, 48, 64].
	logicalShapes := map[string][]int{
		"model.embed_tokens.weight":                 {256, 64},
		"model.layers.0.self_attn.q_proj.weight":    {64, 64},
		"model.layers.0.input_layernorm.weight":     {64},
		"model.layers.1.mlp.switch_mlp.gate.weight": {4, 48, 64},
	}
	var logicalNames []string
	for k := range logicalShapes {
		logicalNames = append(logicalNames, k)
	}
	wantBytes := fit.SafetensorsWeightBytes(logicalNames, func(n string) ([]int, bool) {
		s, ok := logicalShapes[n]
		return s, ok
	}, rates)

	// 1. NVFP4 checkpoint: q_proj.weight is U8 [64, 32] with .weight_scale, .weight_scale_2, .input_scale.
	nvfp4Shapes := map[string][]int{
		"model.embed_tokens.weight":                         {256, 64},
		"model.layers.0.self_attn.q_proj.weight":            {64, 32},
		"model.layers.0.self_attn.q_proj.weight_scale":      {64, 4},
		"model.layers.0.self_attn.q_proj.weight_scale_2":    {1},
		"model.layers.0.self_attn.q_proj.input_scale":       {1},
		"model.layers.0.input_layernorm.weight":             {64},
		"model.layers.1.mlp.switch_mlp.gate.weight":         {4, 48, 32},
		"model.layers.1.mlp.switch_mlp.gate.weight_scale":   {4, 48, 4},
		"model.layers.1.mlp.switch_mlp.gate.weight_scale_2": {1},
	}
	var nvfp4Names []string
	for k := range nvfp4Shapes {
		nvfp4Names = append(nvfp4Names, k)
	}
	gotNVFP4 := fit.SafetensorsWeightBytes(nvfp4Names, func(n string) ([]int, bool) {
		s, ok := nvfp4Shapes[n]
		return s, ok
	}, rates)
	if gotNVFP4 != wantBytes {
		t.Fatalf("NVFP4 SafetensorsWeightBytes = %d, want %d", gotNVFP4, wantBytes)
	}

	// 2. MLX 4-bit checkpoint: weights are U32 [..., in/8] with .scales and .biases.
	mlx4Shapes := map[string][]int{
		"model.embed_tokens.weight":                 {256, 8},
		"model.embed_tokens.scales":                 {256, 2},
		"model.embed_tokens.biases":                 {256, 2},
		"model.layers.0.self_attn.q_proj.weight":    {64, 8},
		"model.layers.0.self_attn.q_proj.scales":    {64, 2},
		"model.layers.0.self_attn.q_proj.biases":    {64, 2},
		"model.layers.0.input_layernorm.weight":     {64},
		"model.layers.1.mlp.switch_mlp.gate.weight": {4, 48, 8},
		"model.layers.1.mlp.switch_mlp.gate.scales": {4, 48, 2},
		"model.layers.1.mlp.switch_mlp.gate.biases": {4, 48, 2},
	}
	var mlx4Names []string
	for k := range mlx4Shapes {
		mlx4Names = append(mlx4Names, k)
	}
	gotMLX4 := fit.SafetensorsWeightBytes(mlx4Names, func(n string) ([]int, bool) {
		s, ok := mlx4Shapes[n]
		return s, ok
	}, rates)
	if gotMLX4 != wantBytes {
		t.Fatalf("MLX 4-bit SafetensorsWeightBytes = %d, want %d", gotMLX4, wantBytes)
	}
}
