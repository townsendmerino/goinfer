package vision

import (
	"os"
	"testing"
)

// BenchmarkSiglipTower is the arbiter for perf-campaign item 13's vision half.
//
// It runs the REAL-SIZED towers from scripts/oracle/gen_siglip_bench.py rather than the
// tiny parity fixture (hidden 32, 2 layers), because the share this item targets
// is a function of tower size: attention softmax is O(patches²) while the
// projections are O(patches), so the transcendental share only shows up at a
// realistic patch count. The weights are random, which is fine — throughput does
// not depend on weight values.
//
//	siglip-bench    hidden 512, 12 layers, 196 patches  (ViT-B/16-ish)
//	siglip-bench-l  hidden 768, 12 layers, 576 patches
func BenchmarkSiglipTower(b *testing.B) {
	for _, tc := range []struct{ name, dir string }{
		{"p196_h512", "testdata/siglip-bench"},
		{"p576_h768", "testdata/siglip-bench-l"},
	} {
		if _, err := os.Stat(tc.dir); err != nil {
			b.Skipf("%s not present; run scripts/oracle/gen_siglip_bench.py", tc.dir)
		}
		e, err := LoadEncoder(tc.dir, false)
		if err != nil {
			b.Fatalf("LoadEncoder(%s): %v", tc.dir, err)
		}
		c := e.Cfg
		pixels := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
		for i := range pixels {
			pixels[i] = float32(i%251)/251*2 - 1
		}
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				out, err := e.Forward(pixels)
				if err != nil {
					b.Fatal(err)
				}
				sinkVision = out
			}
		})
	}
}

var sinkVision []float32
