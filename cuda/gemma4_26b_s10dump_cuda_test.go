//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4_26B_s10DumpCUDA is the separating test the 26B's near-tie read proposed (docs/tasks/task-multimodal-support-2026-10.md, S1.0): the CUDA resident's full logits at every position of the
// sequences the Mac's Metal dump used (seqs.json, teacher-forced, the same .int4.metal.giw), written as logits-cuda.f32 beside them, so the CPU grade (decoder TestGemma4_26B_s10Grade) can add CUDA as a
// third arm. If CUDA agrees with the CPU at the early positions where Metal does not, the divergence is Metal-specific; if it diverges the same way, the CPU or a shared stage is the suspect.
// A correctness run, not a measurement: the 26B runs paged (C' streaming) on the 8 GB card. Heavy.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_S10_DUMP_DIR=<dir> GOINFER_S10_GIW=<giw> go test -tags 'cuda goinfer_testhooks' -run TestGemma4_26B_s10DumpCUDA -v -timeout 40m ./cuda/
func TestGemma4_26B_s10DumpCUDA(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	dir, giw := os.Getenv("GOINFER_S10_DUMP_DIR"), os.Getenv("GOINFER_S10_GIW")
	if dir == "" || giw == "" {
		t.Skip("set GOINFER_S10_DUMP_DIR and GOINFER_S10_GIW")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "seqs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var s struct {
		Seqs [][]int `json:"seqs"`
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1") // C' expert staging: the 26B does not fit the 8 GB card otherwise (as TestMoEPerLayerHitRate)
	t0 := time.Now()
	m, err := decoder.Load(giw, decoder.Options{Backend: "cuda"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("the 26B did not build a CUDA resident: %s", m.ResidentDecline())
	}
	t.Logf("loaded %s on CUDA in %s", filepath.Base(giw), time.Since(t0).Round(time.Second))
	out, err := os.Create(filepath.Join(dir, "logits-cuda.f32"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	n := 0
	last := time.Now()
	for pi, seq := range s.Seqs {
		r.Reset()
		for i := 0; i < len(seq)-1; i++ {
			l, err := r.Forward(m.EmbedResidentForTest(seq[i]), i)
			if err != nil {
				t.Fatalf("prompt %d position %d: %v", pi+1, i, err)
			}
			buf := make([]byte, 4*len(l))
			for j, v := range l {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(v))
			}
			if _, err := out.Write(buf); err != nil {
				t.Fatal(err)
			}
			n++
			if time.Since(last) > 30*time.Second {
				t.Logf("heartbeat: %d positions done (prompt %d, position %d), %s elapsed", n, pi+1, i, time.Since(t0).Round(time.Second))
				last = time.Now()
			}
		}
		t.Logf("prompt %d: %d positions", pi+1, len(seq)-1)
	}
	t.Logf("wrote %d positions of logits-cuda.f32 in %s", n, time.Since(t0).Round(time.Second))
}
