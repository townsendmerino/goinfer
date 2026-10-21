package decoder

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// q4kFitModels are the real q4_k_m GGUFs the q4k price is checked on: the lane's own 0.5B, the 1.5B, and Phi-3 mini, the
// model Metal's q4k lane is for.
var q4kFitModels = []struct {
	env, file string
	heavy     bool
}{
	{"GOINFER_QWEN05_GGUF", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", false},
	{"GOINFER_QWEN15_GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", false},
	{"GOINFER_PHI3_GGUF", "phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf", true},
}

func q4kFitPath(env, file string) string {
	if p := os.Getenv(env); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "models", file)
}

// q4kFitBand is the agreement the estimate must reach with a load's ResidentWeightBytes, as estimate/measured.
const q4kFitLo, q4kFitHi = 0.95, 1.05

// TestFitEstimate_q4kAgreesWithResidentWeightBytes: the pre-load estimate at q4k is within q4kFitBand of the weight bytes
// a q4k load holds, on each real q4_k_m file present, and the int4 price q4k used to take is outside it. A load the fit
// guard refuses is reported and fails: that model's row has not been measured. Phi-3's row is heavy
// (GOINFER_HEAVY_TESTS=1): this 16 GB Mac admits its load only with the desktop's memory free.
func TestFitEstimate_q4kAgreesWithResidentWeightBytes(t *testing.T) {
	ran := 0
	for _, c := range q4kFitModels {
		t.Run(filepath.Base(c.file), func(t *testing.T) {
			if c.heavy && os.Getenv("GOINFER_HEAVY_TESTS") == "" {
				t.Skip("heavy-checkpoint row: set GOINFER_HEAVY_TESTS=1")
			}
			path := q4kFitPath(c.env, c.file)
			if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
				t.Fatalf("%s is on the archive, not ~/models (CLAUDE.md)", path)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no model at %s", path)
			}
			est := estimateGGUFWeightBytes(path, quantQ4K)
			q4kPriceAsInt4ForTest = true
			old := estimateGGUFWeightBytes(path, quantQ4K)
			q4kPriceAsInt4ForTest = false
			m, err := Load(path, Options{Quant: "q4k", ResidentContext: 64})
			var fit *FitDeclineError
			if errors.As(err, &fit) {
				t.Fatalf("the fit guard refused the measuring load, so this row is unmeasured: %v", err)
			}
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			got := m.ResidentWeightBytes()
			r, rOld := float64(est)/float64(got), float64(old)/float64(got)
			t.Logf("measured %d bytes; estimate %d (ratio %.4f); the int4 price %d (ratio %.4f)", got, est, r, old, rOld)
			if r < q4kFitLo || r > q4kFitHi {
				t.Errorf("the q4k estimate is %.4f of the measured weight bytes, outside %.2f..%.2f", r, q4kFitLo, q4kFitHi)
			}
			if rOld >= q4kFitLo && rOld <= q4kFitHi {
				t.Errorf("the int4 price is %.4f of the measured weight bytes, inside the band: this test cannot see the defect", rOld)
			}
			ran++
		})
	}
	if ran == 0 {
		t.Skip("no q4_k_m GGUF in ~/models")
	}
}

// TestFitEstimate_otherQuantsUnchanged: every quant but q4k is still priced uniformly, each 2-D tensor at its mode's
// measured cost and every 1-D or f32-pinned tensor at f32, recomputed here from the file's metadata.
func TestFitEstimate_otherQuantsUnchanged(t *testing.T) {
	path := q4kFitPath(q4kFitModels[0].env, q4kFitModels[0].file)
	if _, err := os.Stat(path); err != nil {
		path = "testdata/gptoss_tiny.gguf"
	}
	g, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	for _, q := range []struct {
		name string
		mode quantMode
	}{{"f32", quantNone}, {"int8", quantInt8}, {"int8int8", quantInt8I8}, {"int4", quantInt4}, {"int4mix", quantInt4Mix}} {
		var want float64
		for _, name := range g.Names() {
			dims, _ := g.Dims(name)
			n := 1
			for _, d := range dims {
				n *= d
			}
			if len(dims) < 2 || f32PinnedTensorName(name) {
				want += 4 * float64(n)
			} else {
				want += quantBytesPerElem(q.mode) * float64(n)
			}
		}
		if got := estimateGGUFWeightBytes(path, q.mode); got != int64(want) {
			t.Errorf("%s: estimate %d, the uniform price %d", q.name, got, int64(want))
		}
	}
}
