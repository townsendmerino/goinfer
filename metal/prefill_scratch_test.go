//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillScratch_zeroFilled: the pass's f16 scratch (prefillScratchU16) relies on Metal handing back a new buffer
// full of zeros. Each round dirties and releases a buffer of the same size first, so a recycled allocation that kept
// its old bytes would show here.
func TestPrefillScratch_zeroFilled(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, n := range []int{8, 4096, 1 << 20, 6 << 20} {
		for round := range 8 {
			dirty := make([]uint16, n)
			for i := range dirty {
				dirty[i] = 0x3c00 // 1.0h
			}
			d.ReleaseBuf(NewBufferU16s(d, dirty))
			b := gpu.NewBufferLenOf[uint16](d, n)
			for i, h := range b.U16s()[:n] {
				if h != 0 {
					d.ReleaseBuf(b)
					t.Fatalf("n=%d round %d: element %d of a new buffer is %#x, not zero", n, round, i, h)
				}
			}
			d.ReleaseBuf(b)
		}
	}
}

// TestPrefillScratch_identical (default-run, tiny dense fixture): the pass's last-row logits are bit-equal with its
// scratch allocated zero-filled by Metal and with it copied from a zeroed Go slice, as it was.
func TestPrefillScratch_identical(t *testing.T) {
	m, err := decoder.Load(mc3ChainFixture, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 512})
	if err != nil {
		t.Skipf("load %s: %v", mc3ChainFixture, err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	defer func() { prefillScratchCopy = false }()
	for _, n := range []int{20, 100, 300} {
		embs := getEmbs(a.r, mc3ChainPrompt(3, n))
		var out [2][]float32
		for i, copyArm := range []bool{true, false} {
			prefillScratchCopy = copyArm
			lg, err := a.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatalf("n=%d: %v", n, err)
			}
			out[i] = append([]float32(nil), lg...)
		}
		for j := range out[0] {
			if math.Float32bits(out[0][j]) != math.Float32bits(out[1][j]) {
				t.Fatalf("n=%d logit %d: zero-filled %v, copied %v", n, j, out[1][j], out[0][j])
			}
		}
	}
}

// TestPrefillScratch_AB: a fresh prompt's pass, wall time, with the scratch copied from Go (as it was) and zero-filled
// by Metal, arms alternated rep by rep, logits compared every rep. GOINFER_AUDIT_MODEL picks the checkpoint (default the
// 1.5B), GOINFER_PSCRATCH_MS the prompt lengths (default 512, 2048). By day, in-process.
//
//	GOINFER_PSCRATCH=1 go test -tags goinfer_testhooks -count=1 -run '^TestPrefillScratch_AB$' -v ./metal/
func TestPrefillScratch_AB(t *testing.T) {
	if os.Getenv("GOINFER_PSCRATCH") != "1" {
		t.Skip("set GOINFER_PSCRATCH=1 (loads a real checkpoint)")
	}
	Ms := []int{512, 2048}
	if v := os.Getenv("GOINFER_PSCRATCH_MS"); v != "" {
		Ms = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				Ms = append(Ms, n)
			}
		}
	}
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, slices.Max(Ms)+64, nil)
	defer func() { prefillScratchCopy = false }()
	t0 := time.Now()
	reps := auditReps(7)
	for _, M := range Ms {
		embs := auditEmbs(a.r, M, M)
		ms := map[bool][]float64{}
		var ref []float32
		for rep := range reps + 1 {
			order := []bool{true, false}
			if rep%2 == 1 {
				order = []bool{false, true}
			}
			for _, copyArm := range order {
				prefillScratchCopy = copyArm
				st := time.Now()
				lg, err := a.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("M=%d: %v", M, err)
				}
				dt := time.Since(st).Seconds() * 1e3
				if ref == nil {
					ref = append([]float32(nil), lg...)
				}
				for j := range ref {
					if math.Float32bits(ref[j]) != math.Float32bits(lg[j]) {
						t.Fatalf("M=%d rep %d: logit %d differs between the arms", M, rep, j)
					}
				}
				if rep > 0 { // rep 0 warms both arms
					ms[copyArm] = append(ms[copyArm], dt)
				}
			}
		}
		ratio := make([]float64, reps)
		for i := range ratio {
			ratio[i] = ms[true][i] / ms[false][i]
		}
		auditHB("p-scratch", t0, "%s M=%d: copied %.2f ms, zero-filled %.2f ms (medians of %d); RESULT copied/zero-filled median %.3f (per rep %s)",
			name, M, auditMedian(ms[true]), auditMedian(ms[false]), reps, auditMedian(ratio), auditFmt3(ratio))
	}
}
