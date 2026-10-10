package decoder

import "testing"

// TestFitEstimate_f32PinnedMixersAgreeWithResident is M-29's headline case (docs/audit-2026-09-10.md): a Mamba-2
// mixer's or MLA's f32-pinned tensors must be priced at f32, not at the requested quant's rate, which under-counts them
// at int4/int8. Same estimate-vs-accountant band as TestFitEstimate_safetensorsAgreesWithResidentWeightBytes, on real
// f32-pinned-mixer fixtures.
func TestFitEstimate_f32PinnedMixersAgreeWithResident(t *testing.T) {
	for _, tc := range []struct {
		name string
		dir  string
	}{
		{"deepseek-MLA", "../testdata/deepseek-tiny"},
		{"nemotron-mamba2", "../testdata/nemotron-tiny"},
		{"granite-mamba2", "../testdata/granite-tiny"},
		{"bailing-MLA+KDA", "../testdata/bailing_hybrid-tiny"}, // also confirms KDA is NOT over-priced
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, q := range []struct {
				name string
				mode quantMode
			}{{"int4", quantInt4}, {"int8int8", quantInt8I8}} {
				t.Run(q.name, func(t *testing.T) {
					est := estimateSafetensorsWeightBytes(tc.dir, q.mode)
					if est <= 0 {
						t.Skipf("no fixture at %s (or estimator returned 0)", tc.dir)
					}
					m, err := Load(tc.dir, Options{Quant: q.name})
					if err != nil {
						t.Skipf("cannot load fixture at %s: %v", tc.dir, err)
					}
					defer m.Close()
					actual := m.ResidentWeightBytes()
					if actual <= 0 {
						t.Skip("accountant reported 0 for this fixture")
					}
					ratio := float64(est) / float64(actual)
					// Same band as the dense estimate-vs-accountant tests. MLA's floor is a hair lower (0.80, not 0.85):
					// f32PinnedTensorName's doc comment documents a deliberate residual gap, MLA's output projection
					// (o_proj/dense.weight) is ALSO f32-pinned but shares its name with every ordinary family's quantizable o_proj, so
					// it is left unmatched rather than risk a much larger false positive elsewhere. 0.80 is that gap's size, not a
					// loosened bar to paper over a regression.
					floor := 0.85
					if tc.name == "deepseek-MLA" {
						floor = 0.80
					}
					if ratio < floor || ratio > 1.25 {
						t.Errorf("%s/%s: estimate %d vs accounted %d (ratio %.2f) — f32-pinned tensors "+
							"still mispriced", tc.name, q.name, est, actual, ratio)
					}
					t.Logf("%s/%s: estimate %d, accounted %d, ratio %.2f", tc.name, q.name, est, actual, ratio)
				})
			}
		})
	}
}
