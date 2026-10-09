//go:build cuda && goinfer_testhooks

package cuda

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Since 2026-10-09 Options.EmbedInt4 reaches every family loader (decoder/embed_int4_loaders_test.go), so a family that used to get an int8 head by default
// now gets an int4 one. The resident must take that table without changing its decision: where a fixture builds resident with the int8 head it builds resident
// with the int4 head (the same decode path, the head reading int4), and where it declines it declines for the same reason either way. A resident that
// declined an int4 head would silently move a whole family to the CPU the day the default reached it.
func TestEmbedInt4_residentDecisionDoesNotChange(t *testing.T) {
	requireCUDADevice(t)
	residentSeen := 0
	for _, d := range []string{"llama-tiny", "granite-dense-tiny", "phi3-tiny", "glm-ocr-tiny", "granite-tiny", "nemotron-tiny", "nemotron3nano-tiny", "llama4-tiny", "gpt2"} {
		t.Run(d, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", d)
			var path, head [2]string
			for i, on := range []bool{false, true} {
				m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", EmbedInt4: on, ResidentContext: 64})
				if err != nil {
					t.Skipf("EmbedInt4=%v: %v", on, err)
				}
				path[i], head[i] = m.DecodePath(), m.HeadTable()
				m.Close()
			}
			if head[0] != "int8" || head[1] != "int4" {
				t.Errorf("head tables %q / %q, want int8 / int4", head[0], head[1])
			}
			if path[0] != path[1] {
				t.Errorf("the decode path changed with the head table:\n  int8 head: %s\n  int4 head: %s", path[0], path[1])
			}
			if strings.HasPrefix(path[1], "cuda-resident") {
				residentSeen++
			}
		})
	}
	if residentSeen < 2 {
		t.Errorf("only %d of the fixtures built resident: the test is not looking at a resident that takes an int4 head", residentSeen)
	}
}
