package decoder

import (
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestRow4GiwKind_sizeArithmetic is a throwaway diagnostic: before blaming the kind-4
// paged-decode regression on kernel/layout locality, check whether the on-disk growth
// (+94.9% measured on both real models) is explained by straightforward full duplication
// of expert int4 bytes (nibbles + f32 group scales, both stored at native size -- the
// f16-scale compaction in docs/task-giw-f16-scales.md was explicitly NOT folded into kind
// 4) or whether something else is inflating the file. Sums actual Int4()/Int4Row4() byte
// lengths across every MoE expert, the dominant byte consumer in both checkpoints.
func TestRow4GiwKind_sizeArithmetic(t *testing.T) {
	requireHeavyModel(t)
	path := envOr("GOINFER_ROW4_SIZECHECK_GIW", "")
	if path == "" {
		t.Skip("set GOINFER_ROW4_SIZECHECK_GIW to a real kind-4 .giw path")
	}
	path = expandHome(t, path)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no .giw at %s: %v", path, err)
	}

	m, err := Load(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	var canonNibbles, canonScales, row4Nibbles, row4Scales int64
	var eligible, ineligible int

	visit := func(wm *linalg.WeightMat) {
		q4, q4s, _, ok := wm.Int4()
		if !ok {
			return // not an int4 tensor at all
		}
		canonNibbles += int64(len(q4))
		canonScales += int64(len(q4s)) * 4
		p4, s4, rok := wm.Int4Row4()
		if rok {
			eligible++
			row4Nibbles += int64(len(p4))
			row4Scales += int64(len(s4)) * 4
		} else {
			ineligible++
		}
	}

	for i := range m.w.Layers {
		l := &m.w.Layers[i]
		for e := range l.Experts {
			visit(&l.Experts[e].Gate)
			visit(&l.Experts[e].Up)
			visit(&l.Experts[e].Down)
		}
		if l.gemma4moe != nil {
			for e := range l.gemma4moe.expertsGateUp {
				visit(&l.gemma4moe.expertsGateUp[e])
			}
			for e := range l.gemma4moe.expertsDown {
				visit(&l.gemma4moe.expertsDown[e])
			}
		}
	}

	total := canonNibbles + canonScales
	row4Total := row4Nibbles + row4Scales
	t.Logf("experts: %d row4-eligible, %d not", eligible, ineligible)
	t.Logf("canonical: nibbles=%.3f GB scales=%.3f GB total=%.3f GB", float64(canonNibbles)/1e9, float64(canonScales)/1e9, float64(total)/1e9)
	t.Logf("row4 added: nibbles=%.3f GB scales=%.3f GB total=%.3f GB", float64(row4Nibbles)/1e9, float64(row4Scales)/1e9, float64(row4Total)/1e9)
	if total > 0 {
		t.Logf("row4-added / canonical ratio (expert bytes only): %.4f (1.0 = exact full duplication)", float64(row4Total)/float64(total))
	}
	if row4Nibbles > 0 {
		t.Logf("row4 nibbles vs canonical nibbles: %.4f (1.0 = exact same byte count, just reordered)", float64(row4Nibbles)/float64(canonNibbles))
	}
	if row4Scales > 0 {
		t.Logf("row4 scales vs canonical scales: %.4f (1.0 = both f32, 0.5 = row4 compacted to f16)", float64(row4Scales)/float64(canonScales))
	}
}
