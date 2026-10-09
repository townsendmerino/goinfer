//go:build gpu && goinfer_testhooks

package gpu

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The WebGPU twin of cuda's TestEmbedInt4_residentDecisionDoesNotChange. Options.EmbedInt4 reaches every family loader since 2026-10-09, so the families that
// used to be handed an int8 head by default are now handed an int4 one. The resident must take it without changing its decision: the same decode path with
// either table, and the head reading int8 then int4.
func TestEmbedInt4_residentDecisionDoesNotChange(t *testing.T) {
	residentSeen := 0
	for _, d := range []string{"llama-tiny", "granite-dense-tiny", "nemotron-tiny", "phi3-tiny", "glm-ocr-tiny", "granite-tiny", "llama4-tiny", "gpt2"} {
		t.Run(d, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", d)
			var path, head [2]string
			for i, on := range []bool{false, true} {
				m, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4", EmbedInt4: on, ResidentContext: 64})
				if err != nil {
					requireGPU(t, err)
				}
				requireGPU(t, nil)
				path[i], head[i] = m.DecodePath(), m.HeadTable()
				m.Close()
			}
			if head[0] != "int8" || head[1] != "int4" {
				t.Errorf("head tables %q / %q, want int8 / int4", head[0], head[1])
			}
			if path[0] != path[1] {
				t.Errorf("the decode path changed with the head table:\n  int8 head: %s\n  int4 head: %s", path[0], path[1])
			}
			if strings.Contains(path[1], "-resident") {
				residentSeen++
			}
		})
	}
	if residentSeen < 2 {
		t.Errorf("only %d of the fixtures built resident: the test is not looking at a resident that takes an int4 head", residentSeen)
	}
}
