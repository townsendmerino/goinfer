//go:build gpu && goinfer_testhooks

package gpu

import (
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// eModelImageRows is a prompt of an E-model's text rows with a run of image rows in the middle: random features
// through the decoder's own row builder, which gives them PAD's token-identity term.
func eModelImageRows(m *decoder.Model, rng *rand.Rand, n int) [][]float32 {
	hidden, _, _, _, _, _, vocab := m.Dims()
	rows := make([][]float32, n)
	for i := range rows {
		if i >= n/4 && i < n/4+n/2 {
			f := make([]float32, hidden)
			for j := range f {
				f[j] = float32(rng.NormFloat64()) * 0.5
			}
			rows[i] = m.Gemma4ResidentMediaRowForTest(f)
			continue
		}
		rows[i] = m.EmbedResidentForTest(rng.Intn(vocab - 1))
	}
	return rows
}

var prefillGemmaFixtures = []struct {
	name, dir string
	opts      decoder.Options
	rows      []int
	mkRows    func(m *decoder.Model, rng *rand.Rand, n int) [][]float32
}{
	{"window", "mistral-tiny-window", decoder.Options{}, []int{7, 20, 40}, nil},
	{"window-kv-f16", "mistral-tiny-window", decoder.Options{KVPrecision: "f16"}, []int{7, 40}, nil},
	{"gemma3", "gemma3-vl-tiny", decoder.Options{}, []int{7, 20, 40}, nil},
	{"gemma3-kv-f16", "gemma3-vl-tiny", decoder.Options{KVPrecision: "f16"}, []int{7, 40}, nil},
	{"gemma4-two-geometries", "gemma4-dense-twogeom-tiny", decoder.Options{}, []int{7, 20, 40}, nil},
	{"gemma4-e-model", "gemma4-emodel-tiny", decoder.Options{}, []int{7, 20, 40}, nil},
	{"gemma4-e-model-image-rows", "gemma4-emodel-tiny", decoder.Options{}, []int{20, 40}, eModelImageRows},
	{"gemma4-e-model-kv-f16", "gemma4-emodel-tiny", decoder.Options{KVPrecision: "f16"}, []int{40}, nil},
}

func prefillGemmaDir(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join("..", "testdata", dir)
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err != nil {
		t.Skipf("no fixture (%s)", path)
	}
	return path
}

// G-S12.4b-3a (docs/tasks/task-multimodal-support-2026-10.md): the batched pass on the Gemma shape against the
// row-by-row walk, the last row and the next 8 decode steps bit for bit, at row counts inside and past the sliding
// window and on both sides of a 16-row block.
func TestPrefillGemma_matchesWalk(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	for _, f := range prefillGemmaFixtures {
		for _, n := range f.rows {
			t.Run(fmt.Sprintf("%s/%d-rows", f.name, n), func(t *testing.T) {
				opts := f.opts
				opts.ResidentContext = 64
				want, got := prefillRowsRun(t, prefillGemmaDir(t, f.dir), opts, n, w4MultiAuto, f.mkRows)
				if why := prefillInt4Grade(want, got); why != "" {
					t.Error(why)
				}
			})
		}
	}
}

// G-S12.4b-3b: each planted defect, alone, must turn G-S12.4b-3a red on a fixture that has the part it breaks.
func TestPrefillGemma_plantedDefects(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	defer func() { kernels.PrefillDefectForTest = 0 }()
	defects := []struct {
		d      int
		name   string
		dir    string
		mkRows func(m *decoder.Model, rng *rand.Rand, n int) [][]float32
	}{
		{4, "each row's per-layer inputs bound to the next row's", "gemma4-emodel-tiny", eModelImageRows},
		{5, "the per-layer-input branch dropped", "gemma4-emodel-tiny", nil},
		{6, "a sliding-window layer attending its whole prefix", "gemma3-vl-tiny", nil},
		{7, "the post-attention norm skipped", "gemma3-vl-tiny", nil},
		{8, "the layer scalar skipped", "gemma4-dense-twogeom-tiny", nil},
		{9, "one feed-forward width for every layer", "gemma4-emodel-tiny", nil},
		{10, "the value norm skipped on a K=V layer", "gemma4-dense-twogeom-tiny", nil},
		{11, "a shared-K/V layer reading the first layer's cache", "gemma4-emodel-tiny", nil},
	}
	for _, d := range defects {
		t.Run(d.name, func(t *testing.T) {
			dir := prefillGemmaDir(t, d.dir)
			kernels.PrefillDefectForTest = d.d
			want, got := prefillRowsRun(t, dir, decoder.Options{ResidentContext: 64}, 40, w4MultiAuto, d.mkRows)
			kernels.PrefillDefectForTest = 0
			why := prefillInt4Grade(want, got)
			if why == "" {
				t.Errorf("the gate stayed green")
				return
			}
			t.Logf("red (%s)", why)
		})
	}
}
