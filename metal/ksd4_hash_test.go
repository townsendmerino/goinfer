//go:build darwin && goinfer_testhooks

package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestKSD4_towerHashes is D4-G2's instrument (docs/tasks/task-kernel-survey-2026-10.md, "KS-D4 gates"): every tower of
// mem2Towers over its committed tiny fixture and growing inputs, the SHA-256 of each output's bits. Run from a test binary at
// the change and one at its base; every line must match.
func TestKSD4_towerHashes(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, tw := range mem2Towers(t) {
		inst := tw.build(t)
		for i := range tw.n {
			out, err := inst.fwd(i)
			if err != nil {
				t.Fatalf("%s input %d: %v", tw.name, i, err)
			}
			h := sha256.New()
			for _, v := range out {
				_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
			}
			fmt.Fprintf(os.Stderr, "[D4-G2] %s input %d: %d values, sha256 %x\n", tw.name, i, len(out), h.Sum(nil))
		}
		if err := inst.close(); err != nil {
			t.Fatal(err)
		}
	}
}

// TestKSD4_g4vSpeed is D4-G3: the Gemma 4 E4B vision tower (~/models/gemma-4-E4B-qat-unq-towers) on Metal, one forward per
// image over S3's four images at serve's soft-token cap, tower_gemm_w32 against aikit's GEMM (vitGemmForTest) on one
// accelerator, A B B A over 10 pairs after 2 warm runs each; a run is the four forwards. The two arms' outputs must be
// equal bit for bit. A timed run: hold the timing lock (GOINFER_KSD4_TIMING=1).
func TestKSD4_g4vSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSD4_TIMING") == "" {
		t.Skip("set GOINFER_KSD4_TIMING=1 (a timed run under the timing lock)")
	}
	dir := os.ExpandEnv("$HOME/models/gemma-4-E4B-qat-unq-towers")
	enc, err := vision.LoadGemma4Encoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	a, err := newG4VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	pp, err := multimodal.LoadGemma4PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	type img struct {
		patches []float32
		pos     [][2]int
	}
	var imgs []img
	for _, name := range s3Images {
		b, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		p, pos, err := vision.Gemma4Preprocess(b, pp.MaxSoftTokens)
		if err != nil {
			t.Fatal(err)
		}
		imgs = append(imgs, img{p, pos})
	}
	run := func(old bool) (float64, [][]float32) {
		a.vitGemmForTest = old
		defer func() { a.vitGemmForTest = false }()
		var outs [][]float32
		t0 := time.Now()
		for _, im := range imgs {
			out, err := a.Hidden(im.patches, im.pos)
			if err != nil {
				t.Fatal(err)
			}
			outs = append(outs, slices.Clone(out))
		}
		return time.Since(t0).Seconds(), outs
	}
	for range 2 {
		run(true)
		run(false)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var o, n float64
		var oo, no [][]float32
		if i%2 == 0 {
			o, oo = run(true)
			n, no = run(false)
		} else {
			n, no = run(false)
			o, oo = run(true)
		}
		for k := range oo {
			for j := range oo[k] {
				if math.Float32bits(oo[k][j]) != math.Float32bits(no[k][j]) {
					t.Fatalf("pair %d image %d: the arms' outputs differ at %d", i, k, j)
				}
			}
		}
		ratios = append(ratios, n/o)
		if n < o {
			below++
		}
		fmt.Fprintf(os.Stderr, "[D4-G3] pair %d (load %s): aikit %.1f ms, tower_gemm_w32 %.1f ms (4 images), new/old %.4f\n", i, la[:len(la)-1], o*1e3, n*1e3, n/o)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[D4-G3] median new/old %.4f, %d of 10 pairs below 1\n", (ratios[4]+ratios[5])/2, below)
}
