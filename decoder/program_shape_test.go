//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

// TestProgramShape_matchesTheAccessors is OP4cu-b1's B-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): on every tracked
// tiny checkpoint the decoder loads on the CPU, program.ShapeOf equals the nine Architecture accessors the residents read, and
// program.MixerOf equals ResidentLayerOps on every layer: the mixer where a resident runs it, and the FFN everywhere. A layer
// holding KDA or LFM2's short convolution, which have no ResidentMixer, must map to its own mixer. Each plant must add a
// difference on at least one fixture.
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
		base := programShapeDiffs(m)
		if len(base) > 0 {
			t.Errorf("%s: %v", name, base)
		}
		seen := map[string]bool{}
		for _, d := range base {
			seen[d] = true
		}
		for plant, pname := range plants {
			t.Run(name+"/"+pname, func(t *testing.T) {
				program.SetShapePlantForTest(t, plant)
				// A plant is red on a fixture only when it adds a difference the unplanted run does not have.
				for _, d := range programShapeDiffs(m) {
					if !seen[d] {
						red[plant]++
						return
					}
				}
			})
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
	for l, ops := range p.Layers {
		gm, gf := program.MixerOf(ops)
		wm, wf := m.ResidentLayerOps(l)
		if gf != wf {
			d = append(d, fmt.Sprintf("layer %d: MixerOf's ffn %v, ResidentLayerOps' %v", l, gf, wf))
		}
		// KDA and the short convolution have no ResidentMixer, and every backend declines their families first: such a layer,
		// read from its operations, must map to its own mixer, and its ResidentMixer is not compared.
		if want, own := ownMixer(ops); own {
			if gm != want {
				d = append(d, fmt.Sprintf("layer %d: MixerOf %v, want %v from its operations", l, gm, want))
			}
			continue
		}
		if got, ok := mixers[gm]; !ok || got != wm {
			d = append(d, fmt.Sprintf("layer %d: MixerOf %v, ResidentLayerOps %v", l, gm, wm))
		}
	}
	return d
}

// ownMixer reports a layer whose operations hold a mixer no resident runs, and that mixer.
func ownMixer(ops []program.Op) (program.Mixer, bool) {
	for i := range ops {
		switch ops[i].Kind {
		case program.OpKDA:
			return program.MixerKDA, true
		case program.OpShortConv:
			return program.MixerShortConv, true
		}
	}
	return 0, false
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

// sublayersAccept reports whether program.SublayersOf and program.CheckSublayerClaims accept every layer of p, as Metal's walker
// derives its steps.
func sublayersAccept(p *program.Program) error {
	for l, ops := range p.Layers {
		spans, err := program.SublayersOf(ops, "Metal")
		if err == nil {
			err = program.CheckSublayerClaims(ops, spans)
		}
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
	}
	return nil
}

// tableHasSteps reports whether s has a step for every operation of p (program.Support.HasStep: the kinds and FFN forms, not
// the parameters), naming the first it lacks.
func tableHasSteps(s *program.Support, p *program.Program) error {
	for l, ops := range p.Layers {
		for i := range ops {
			if o := &ops[i]; !s.HasStep(o) && o.Kind == program.OpFFN {
				return fmt.Errorf("layer %d: operation %d (ffn, form %v)", l, i, o.FFN.Form)
			} else if !s.HasStep(o) {
				return fmt.Errorf("layer %d: operation %d (%v)", l, i, o.Kind)
			}
		}
	}
	return nil
}

// TestSublayers_consumeEveryOperationOnce is OP4m-b2's M2 and OP4m-c1's C1-1 over the fixtures, moved here from metal with the
// derivation (OP4cu-b2, docs/tasks/task-forward-architecture-2026-10.md §4.9 and §4.9a): on every tracked tiny checkpoint the
// decoder loads on the CPU, the sublayers of each layer consume each of its operations exactly once, in order, each only
// operations its block emits, exactly when Metal's table (program.Metal) has a step for every operation. Two plants must each
// turn it red on at least one fixture: the attention sublayer also consumes the FFN sublayer's input norm, and the table lists
// Kimi Delta Attention.
func TestSublayers_consumeEveryOperationOnce(t *testing.T) {
	plantedTable := *program.Metal
	plantedTable.Kinds = append(slices.Clone(program.Metal.Kinds), program.OpKDA)
	compared, red, tableRed, declined := 0, 0, 0, 0
	for _, dir := range shapeTestFixtures(t) {
		name := filepath.Base(dir)
		if shapeTestTooBig(dir) {
			continue
		}
		m, err := Load(dir, Options{})
		if err != nil {
			if !shapeTestNotDecoders[name] {
				t.Errorf("%s: unexpected load failure: %v", name, err)
			}
			continue
		}
		p := m.Program()
		walk, table := sublayersAccept(p), tableHasSteps(program.Metal, p)
		if (walk == nil) != (table == nil) {
			t.Errorf("%s: the walker gives %v and the table %v; want both to accept or both to decline", name, walk, table)
		}
		if (walk == nil) != (tableHasSteps(&plantedTable, p) == nil) {
			tableRed++
		}
		compared++
		if walk != nil {
			declined++
			t.Logf("%s declines: %v", name, walk)
			m.Close()
			continue
		}
		t.Run(name+"/claim one more", func(t *testing.T) {
			program.SetShapePlantForTest(t, program.ShapePlantClaimOneMore)
			if sublayersAccept(p) != nil {
				red++
			}
		})
		m.Close()
	}
	t.Logf("%d fixtures checked, %d declined for an operation with no step; plant red on %d, table plant red on %d", compared, declined, red, tableRed)
	if declined == 0 {
		t.Errorf("no fixture declines: the decline half of the agreement is untested")
	}
	if red == 0 {
		t.Errorf("plant (the attention sublayer consumes the FFN's input norm): no fixture goes red")
	}
	if tableRed == 0 {
		t.Errorf("plant (the table lists kda): no fixture goes red")
	}
}
