package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

// TestProgramShape_matchesTheAccessors is OP4cu-b1's B-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): on every tracked
// tiny checkpoint the decoder loads on the CPU, program.ShapeOf equals the nine Architecture accessors the residents read, and
// program.MixerOf equals ResidentLayerOps on every layer. Each plant must turn it red on at least one fixture.
func TestProgramShape_matchesTheAccessors(t *testing.T) {
	plants := map[program.ShapePlant]string{
		program.ShapePlantSandwichPost:    "sandwich read from a post norm alone",
		program.ShapePlantNoWhole:         "QK-norm Whole read as false",
		program.ShapePlantFFNOnlyAsAttend: "an FFN-only block read as attention",
	}
	red := map[program.ShapePlant]int{}
	compared, skipped := 0, 0
	for _, dir := range shapeTestFixtures(t) {
		name := filepath.Base(dir)
		if shapeTestTooBig(dir) {
			t.Logf("NOT RUN: %s: over 1 GB of weights, not a tiny checkpoint", name)
			skipped++
			continue
		}
		m, err := Load(dir, Options{})
		if err != nil {
			if shapeTestNotDecoders[name] {
				t.Logf("NOT RUN: %s: not a decoder (%v)", name, err)
				skipped++
				continue
			}
			t.Errorf("%s: unexpected load failure: %v", name, err)
			continue
		}
		if d := programShapeDiffs(m); len(d) > 0 {
			t.Errorf("%s: %v", name, d)
		}
		for plant := range plants {
			program.ShapePlantForTest = plant
			if len(programShapeDiffs(m)) > 0 {
				red[plant]++
			}
			program.ShapePlantForTest = program.ShapePlantNone
		}
		m.Close()
		compared++
	}
	if compared < 10 {
		t.Fatalf("only %d fixtures compared: a comparison over few fixtures proves little", compared)
	}
	t.Logf("%d fixtures compared, %d NOT RUN", compared, skipped)
	for plant, name := range plants {
		t.Logf("plant %q: red on %d of %d fixtures", name, red[plant], compared)
		if red[plant] == 0 {
			t.Errorf("plant %q: no fixture goes red", name)
		}
	}
}

// programShapeDiffs names each field where the program's shape or a layer's mixer differs from the accessors; an error from
// ShapeOf is one entry.
func programShapeDiffs(m *Model) []string {
	var d []string
	p := m.Program()
	s, err := program.ShapeOf(p)
	if err != nil {
		return []string{"ShapeOf: " + err.Error()}
	}
	for _, c := range []struct {
		name      string
		got, want bool
	}{
		{"sandwich", s.Sandwich, m.SandwichNormResident()},
		{"postOnly", s.PostOnly, m.PostOnlyNormResident()},
		{"parallel", s.Parallel, m.ParallelBlockResident()},
		{"nonGated", s.NonGated, m.NonGatedMLPResident()},
		{"layerNorm", s.LayerNorm, m.LayerNormResident()},
		{"qkNorm", s.QKNorm, m.HasQKNorm()},
		{"qkNormWhole", s.QKNormWhole, m.QKNormWholeResident()},
		{"outBias", s.OutBias, m.OutBiasResident()},
		{"learnedPos", s.LearnedPos, m.LearnedPosResident()},
	} {
		if c.got != c.want {
			d = append(d, fmt.Sprintf("%s: program %v, accessor %v", c.name, c.got, c.want))
		}
	}
	mixers := map[program.Mixer]ResidentMixer{program.MixerAttention: MixerAttention, program.MixerNone: MixerNone,
		program.MixerDeltaNet: MixerDeltaNet, program.MixerMamba: MixerMamba, program.MixerMLA: MixerMLA}
	for l := range p.Layers {
		gm, gf := program.MixerOf(p.Layers[l])
		wm, wf := m.ResidentLayerOps(l)
		if got, ok := mixers[gm]; !ok || got != wm || gf != wf {
			d = append(d, fmt.Sprintf("layer %d: MixerOf (%d, ffn %v), ResidentLayerOps (%v, ffn %v)", l, gm, gf, wm, wf))
			break
		}
	}
	return d
}

// shapeTestFixtures are the tracked tiny checkpoints B-1 loads: every directory under the two testdata roots with a config.json
// and safetensors weights.
func shapeTestFixtures(t *testing.T) []string {
	t.Helper()
	var dirs []string
	for _, root := range []string{"../testdata", "testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range ents {
			d := filepath.Join(root, e.Name())
			ws, _ := filepath.Glob(filepath.Join(d, "*.safetensors"))
			if _, err := os.Stat(filepath.Join(d, "config.json")); e.IsDir() && err == nil && len(ws) > 0 {
				dirs = append(dirs, d)
			}
		}
	}
	return dirs
}

// shapeTestTooBig reports a fixture whose weights pass 1 GB: a slice of a real model, not a tiny checkpoint.
func shapeTestTooBig(dir string) bool {
	ws, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	var n int64
	for _, w := range ws {
		if fi, err := os.Stat(w); err == nil {
			n += fi.Size()
		}
	}
	return n > 1<<30
}

// shapeTestNotDecoders are the fixture directories B-1 expects Load to refuse: encoders and vision or audio towers.
var shapeTestNotDecoders = map[string]bool{
	"embeddinggemma2-tiny": true, "gemma4-audio-tiny": true, "gemma4-vision-tiny": true, "glm-ocr-vision-tiny": true,
	"qwen3vl-vision-tiny": true, "siglip-tiny": true, "whisper-tiny-rand": true, "whisper-tiny-ts": true,
}
