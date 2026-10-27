//go:build cuda

package cuda

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/audio"
)

// S14.6 lever B's kernels and encoder (docs/tasks/task-multimodal-support-2026-10.md, G-S14t4): the exact GELU against aikit's contract kernel BIT FOR BIT, the im2col
// layouts of both stems against a host loop, and the whole device encoder against the CPU encoder on the tiny fixture, with each planted defect red. The real checkpoints are
// audio_encoder_real_test.go. Needs a CUDA device; skips without one.

// TestAudioGeluErf_bitsMatchContract runs the kernel over a sweep that crosses both erf branches (|x/sqrt2| below and above 1), the exp edge cases (underflow, the clamp) and
// both signs, and compares every bit with linalg.GELUContractInto.
func TestAudioGeluErf_bitsMatchContract(t *testing.T) {
	ops := newTestTower(t, 16)
	var pm Pipeline
	ops.mustRun(t, func() {})
	if err := ops.do(func() error {
		lib, e := ops.dev.CompileLibrary(audioEncoderPTX)
		if e != nil {
			return e
		}
		pm, e = ops.dev.NewComputePipeline(lib, "audio_gelu_erf")
		return e
	}); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	src := make([]float32, 0, 1<<20)
	for x := float32(-40); x <= 40; x += 0.0007 {
		src = append(src, x)
	}
	for len(src) < 1<<20 {
		src = append(src, float32(rng.NormFloat64())*4)
	}
	src = append(src, 0, float32(math.Copysign(0, -1)), 1e-30, -1e-30, 1.4142135, -1.4142135, 5.7, -5.7, 6, -6, 9.5, -9.5, 100, -100, 1e4, -1e4)
	want := make([]float32, len(src))
	linalg.GELUContractInto(want, src)
	got := make([]float32, len(src))
	if err := ops.do(func() error {
		defer ops.releaseScratch()
		b := ops.af(len(src))
		if e := gpu.Upload(b, src); e != nil {
			return e
		}
		ops.launch(pm, gpu.Grid1D(len(src), 256), Arg(b), i64(int64(len(src))))
		if e := ops.finish(); e != nil {
			return e
		}
		return gpu.Download(b, got)
	}); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			if bad < 5 {
				t.Errorf("x=%v: device %v (%#x), contract %v (%#x)", src[i], got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d values differ from the contract kernel", bad, len(src))
	}
}

// TestAudioIm2col_bothStems checks the kernel's two uses against a host loop: stem one (channel-major [mels][T], stride 1) and stem two (time-major [T][D], stride 2), padding included.
func TestAudioIm2col_bothStems(t *testing.T) {
	ops := newTestTower(t, 16)
	var pm Pipeline
	if err := ops.do(func() error {
		lib, e := ops.dev.CompileLibrary(audioEncoderPTX)
		if e != nil {
			return e
		}
		pm, e = ops.dev.NewComputePipeline(lib, "audio_im2col_k3")
		return e
	}); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(8))
	for _, tc := range []struct {
		name                    string
		Tin, stride, C          int
		chStride, tStride, Tout int
	}{
		{"stem1 channel-major", 37, 1, 5, 37, 1, 37},
		{"stem2 time-major", 37, 2, 5, 1, 5, 19},
		{"stem2 even length", 40, 2, 4, 1, 4, 20},
	} {
		src := make([]float32, tc.Tin*tc.C)
		for i := range src {
			src[i] = float32(rng.NormFloat64())
		}
		n := tc.Tout * tc.C * 3
		want := make([]float32, n)
		for r := 0; r < tc.Tout; r++ {
			for ch := 0; ch < tc.C; ch++ {
				for k := 0; k < 3; k++ {
					if ti := r*tc.stride - 1 + k; ti >= 0 && ti < tc.Tin {
						want[r*tc.C*3+ch*3+k] = src[ch*tc.chStride+ti*tc.tStride]
					}
				}
			}
		}
		got := make([]float32, n)
		if err := ops.do(func() error {
			defer ops.releaseScratch()
			s, d := ops.af(len(src)), ops.af(n)
			if e := gpu.Upload(s, src); e != nil {
				return e
			}
			ops.launch(pm, gpu.Grid1D(n, 256), Arg(s), Arg(d), i32(int32(tc.Tin)), i32(int32(tc.stride)), i32(int32(tc.C)), i64(int64(tc.chStride)), i64(int64(tc.tStride)), i64(int64(n)))
			if e := ops.finish(); e != nil {
				return e
			}
			return gpu.Download(d, got)
		}); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: element %d is %v, want %v", tc.name, i, got[i], want[i])
			}
		}
	}
}

// audioMinCosine is the smallest per-position cosine between two [rows][d] matrices.
func audioMinCosine(a, b []float32, d int) float64 {
	worst := 1.0
	for r := 0; r+d <= len(a); r += d {
		var dot, na, nb float64
		for j := 0; j < d; j++ {
			x, y := float64(a[r+j]), float64(b[r+j])
			dot += x * y
			na += x * x
			nb += y * y
		}
		if c := dot / math.Sqrt(na*nb); c < worst || math.IsNaN(c) {
			worst = c
		}
	}
	return worst
}

func audioRandFeats(mels int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	f := make([]float32, mels*audio.WhisperFrames)
	for i := range f {
		f[i] = float32(rng.NormFloat64()*0.6 - 0.3)
	}
	return f
}

// TestAudioEncoder_tinyMatchesCPUAndDefectsAreRed is G-S14t4 on the tiny fixture: the clean device encoder holds every position at cosine >= 0.99999 against the CPU encoder, and
// each defect injectable on the device drops it below that.
func TestAudioEncoder_tinyMatchesCPUAndDefectsAreRed(t *testing.T) {
	enc, err := audio.LoadWhisperEncoder("../testdata/whisper-tiny-rand")
	if err != nil {
		t.Skipf("no tiny Whisper fixture: %v", err)
	}
	tw, err := newAudioTower(enc)
	if err != nil {
		t.Skipf("no CUDA device for the audio encoder: %v", err)
	}
	t.Cleanup(func() { _ = tw.Close() })
	feats := audioRandFeats(enc.Cfg.MelBins, 3)
	want, err := enc.Forward(feats)
	if err != nil {
		t.Fatal(err)
	}
	D := enc.Cfg.DModel
	got, err := tw.Forward(feats)
	if err != nil {
		t.Fatal(err)
	}
	if c := audioMinCosine(got, want, D); c < 0.99999 {
		t.Fatalf("clean device encoder: min per-position cosine %.8f against the CPU encoder, want >= 0.99999", c)
	}
	t.Logf("clean: min cosine %.9f", audioMinCosine(got, want, D))
	for d := audioDefectNoPos; d <= audioDefectPostNorm; d++ {
		tw.planted = d
		bad, err := tw.Forward(feats)
		if d == audioDefectWindowed {
			if err == nil {
				t.Errorf("defect %d (windowed) should be refused as not injectable", d)
			}
			continue
		}
		if err != nil {
			t.Fatalf("defect %d: %v", d, err)
		}
		c := audioMinCosine(bad, want, D)
		t.Logf("defect %d: min cosine %.6f", d, c)
		if c >= 0.99999 {
			t.Errorf("defect %d stayed green: min cosine %.8f against the clean CPU encoder", d, c)
		}
	}
	tw.planted = audioDefectNone
}

// The streamed tower (one layer slot refilled from the host before each layer) runs the same kernels on the same bytes as the resident one, so
// its output is the resident output bit for bit; and it still holds the defect list, which share the code.
func TestAudioEncoder_streamedIsBitIdenticalToResident(t *testing.T) {
	enc, err := audio.LoadWhisperEncoder("../testdata/whisper-tiny-rand")
	if err != nil {
		t.Skipf("no tiny Whisper fixture: %v", err)
	}
	res, err := newAudioTowerMode(enc, false)
	if err != nil {
		t.Skipf("no CUDA device for the audio encoder: %v", err)
	}
	t.Cleanup(func() { _ = res.Close() })
	str, err := newAudioTowerMode(enc, true)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = str.Close() })
	if len(str.layers) != 1 || !str.streamed || res.streamed {
		t.Fatalf("modes: streamed tower has %d layer slots (streamed=%v), resident streamed=%v", len(str.layers), str.streamed, res.streamed)
	}
	feats := audioRandFeats(enc.Cfg.MelBins, 9)
	want, err := res.Forward(feats)
	if err != nil {
		t.Fatal(err)
	}
	for pass := 0; pass < 2; pass++ { // the second pass proves the slot is refilled, not left holding the last layer
		got, err := str.Forward(feats)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("pass %d: streamed output differs from resident at %d: %v vs %v", pass, i, got[i], want[i])
			}
		}
	}
	// a streamed tower with its first layer's weights left in the slot for every layer would NOT match: the check has teeth
	str.host = append([]audio.WhisperLayerWeights(nil), str.host...)
	str.host[1] = str.host[0]
	bad, _ := str.Forward(feats)
	same := true
	for i := range bad {
		if math.Float32bits(bad[i]) != math.Float32bits(want[i]) {
			same = false
			break
		}
	}
	if same {
		t.Fatal("swapping layer 1's weights for layer 0's left the output unchanged: the streamed fill is not being read")
	}
}
