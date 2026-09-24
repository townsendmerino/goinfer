//go:build darwin

package metal

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// S6's first registered gate: logits byte-identical aliased vs copied — same bytes reach the same
// kernels, so a difference is a defect, not a tolerance. Both arms build a resident from the SAME .giw
// (GOINFER_METAL_ALIAS=0 then =1), run the same prefill + decode with the same embeddings, and compare
// every logit's bits. The alias arm must ALSO have really aliased something (r.alias.tensors > 0), or a
// passing comparison would be two runs of the copy path.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_TEST_GIW=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.giw \
//	  go test -tags goinfer_testhooks ./metal/ -run TestWeightAlias_logitsByteIdentical -v
func TestWeightAlias_logitsByteIdentical(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real .giw twice)")
	}
	path := os.Getenv("GOINFER_TEST_GIW")
	if path == "" {
		t.Skip("set GOINFER_TEST_GIW to a metal-target .giw (prequant -quant int4 -target metal)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}

	const prefillLen, nSteps = 24, 6
	type run struct {
		logits  [][]float32
		aliased int
		bytes   int64
		copied  int
	}
	runOne := func(alias bool) run {
		t.Helper()
		if alias {
			t.Setenv("GOINFER_METAL_ALIAS", "1")
		} else {
			t.Setenv("GOINFER_METAL_ALIAS", "0")
		}
		m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		b := &metalBackend{}
		rf, ok, err := b.BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("BuildResident: ok=%v err=%v", ok, err)
		}
		r := rf.(*metalResident).r
		var out run
		if r.alias != nil {
			out.aliased, out.bytes, out.copied = r.alias.tensors, r.alias.aliased, r.alias.copied
		}
		rng := rand.New(rand.NewSource(11))
		emb := func() []float32 {
			e := make([]float32, r.H)
			for j := range e {
				e[j] = float32(rng.NormFloat64()) * 0.05
			}
			return e
		}
		embs := make([][]float32, prefillLen)
		for i := range embs {
			embs[i] = emb()
		}
		if _, err := r.ForwardBatch(embs, 0); err != nil {
			t.Fatalf("ForwardBatch: %v", err)
		}
		for s := range nSteps {
			l := r.ForwardEmb(emb(), prefillLen+s)
			out.logits = append(out.logits, append([]float32(nil), l...))
		}
		b.Close() // resident first (releases every MTLBuffer, no-copy ones included) ...
		m.Close() // ... then the model unmaps the .giw the no-copy buffers pointed into
		return out
	}

	copyRun := runOne(false)
	aliasRun := runOne(true)

	if copyRun.aliased != 0 {
		t.Fatalf("the copy arm aliased %d tensors — the toggle is not off", copyRun.aliased)
	}
	if aliasRun.aliased == 0 {
		t.Fatalf("the alias arm aliased NOTHING (copied %d) — the comparison below would be copy-vs-copy", aliasRun.copied)
	}
	t.Logf("alias arm: %d tensors, %.1f MB of nibbles not copied", aliasRun.aliased, float64(aliasRun.bytes)/(1<<20))

	for s := range copyRun.logits {
		a, c := aliasRun.logits[s], copyRun.logits[s]
		if len(a) != len(c) {
			t.Fatalf("step %d: %d vs %d logits", s, len(a), len(c))
		}
		var ba, bc bytes.Buffer
		_ = binary.Write(&ba, binary.LittleEndian, a)
		_ = binary.Write(&bc, binary.LittleEndian, c)
		if !bytes.Equal(ba.Bytes(), bc.Bytes()) {
			diff, maxAbs := 0, 0.0
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(c[i]) {
					diff++
					maxAbs = math.Max(maxAbs, math.Abs(float64(a[i]-c[i])))
				}
			}
			t.Errorf("step %d: aliased logits differ from copied in %d/%d positions (max |diff| %g)", s, diff, len(a), maxAbs)
		}
	}
}

// The opt-in must decline itself on a mapping that is a large fraction of RAM — the one configuration where
// aliasing was measured to collapse the machine's memory (gemma4-26b, 3 of 3) — and must not decline the
// configurations it was measured safe on. "force" overrides for research, and an unreadable RAM size is
// "unknown", which does not decline.
func TestAliasDecision(t *testing.T) {
	const gb = int64(1) << 30
	for _, tc := range []struct {
		name  string
		mode  string
		file  int64
		ram   uint64
		want  bool
		wantW bool // a reason is given
	}{
		{"off by default", "", 1 * gb, uint64(16 * gb), false, false},
		{"explicitly off", "0", 1 * gb, uint64(16 * gb), false, false},
		{"1.5B on a 16 GB Mac", "1", 1 * gb, uint64(16 * gb), true, false},
		{"7B (5 GB) on a 16 GB Mac", "1", 5 * gb, uint64(16 * gb), true, false},
		{"exactly half of RAM is allowed", "1", 8 * gb, uint64(16 * gb), true, false},
		{"M26 (15 GB) on a 16 GB Mac is declined", "1", 15 * gb, uint64(16 * gb), false, true},
		{"M35 (22 GB) on a 16 GB Mac is declined", "1", 22 * gb, uint64(16 * gb), false, true},
		{"the same 15 GB on a 64 GB Mac is allowed", "1", 15 * gb, uint64(64 * gb), true, false},
		{"force overrides the size check", "force", 15 * gb, uint64(16 * gb), true, false},
		{"unreadable RAM is unknown, not a decline", "1", 15 * gb, 0, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			on, why := aliasDecision(tc.mode, tc.file, tc.ram)
			if on != tc.want {
				t.Errorf("aliasDecision(%q, %d, %d) on = %v, want %v", tc.mode, tc.file, tc.ram, on, tc.want)
			}
			if (why != "") != tc.wantW {
				t.Errorf("reason %q (want a reason: %v)", why, tc.wantW)
			}
		})
	}
}
