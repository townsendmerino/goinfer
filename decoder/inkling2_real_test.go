//go:build realckpt

package decoder

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// TestInkling2Real_dump runs the real Inkling-2-Mini GGUF (GOINFER_INKLING2_GGUF) over a prompt of token ids
// (GOINFER_INKLING2_IDS, comma-separated) at int8, token by token, and writes every visit's output at every position to
// GOINFER_INKLING2_OUT/visits.bin ([position][visit][hidden] float32, after a little-endian int64 header of the three
// sizes) and the greedy continuation (GOINFER_INKLING2_GREEDY tokens) to greedy.txt. scripts/inkling2/compare_real.py
// differences it against llama.cpp's per-slot dump of the same prompt. Skips without the checkpoint.
func TestInkling2Real_dump(t *testing.T) {
	path, out := os.Getenv("GOINFER_INKLING2_GGUF"), os.Getenv("GOINFER_INKLING2_OUT")
	if path == "" || out == "" {
		t.Skip("GOINFER_INKLING2_GGUF and GOINFER_INKLING2_OUT unset")
	}
	var ids []int
	for _, s := range strings.Split(os.Getenv("GOINFER_INKLING2_IDS"), ",") {
		id, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			t.Fatalf("GOINFER_INKLING2_IDS: %v", err)
		}
		ids = append(ids, id)
	}
	nGreedy, _ := strconv.Atoi(os.Getenv("GOINFER_INKLING2_GREEDY"))
	quant := os.Getenv("GOINFER_INKLING2_QUANT")
	if quant == "" {
		quant = "int8"
	}
	t0 := time.Now()
	m, err := Load(path, Options{Quant: quant, DirectLoad: true})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	fmt.Fprintf(os.Stderr, "inkling2 real: loaded %s at %s in %v\n", filepath.Base(path), quant, time.Since(t0).Round(time.Second))
	a := m.w.Arch
	visits := make([]int, a.NumLayers)
	for i := range visits {
		visits[i] = i
	}
	f, err := os.Create(filepath.Join(out, "visits.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	binary.Write(f, binary.LittleEndian, [3]int64{int64(len(ids)), int64(a.NumLayers), int64(a.HiddenDim)})
	cache := m.NewCache(len(ids) + nGreedy + 1)
	var logits []float32
	for p, id := range ids {
		tp := time.Now()
		var hidden [][]float32
		logits, hidden, err = m.ForwardCapture(id, cache, visits)
		if err != nil {
			t.Fatalf("position %d: %v", p, err)
		}
		for _, h := range hidden {
			binary.Write(f, binary.LittleEndian, h)
		}
		fmt.Fprintf(os.Stderr, "inkling2 real: prompt position %d/%d in %v (elapsed %v)\n", p+1, len(ids), time.Since(tp).Round(time.Millisecond), time.Since(t0).Round(time.Second))
	}
	lf, err := os.Create(filepath.Join(out, "logits_last.bin"))
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(lf, binary.LittleEndian, logits)
	lf.Close()
	var greedy []string
	for s := range nGreedy {
		next := sampling.Argmax(logits)
		greedy = append(greedy, strconv.Itoa(next))
		if logits, err = m.forward(next, cache); err != nil {
			t.Fatalf("greedy step %d: %v", s, err)
		}
		if math.IsNaN(float64(logits[0])) {
			t.Fatalf("greedy step %d: NaN logits", s)
		}
	}
	if err := os.WriteFile(filepath.Join(out, "greedy.txt"), []byte(strings.Join(greedy, " ")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "inkling2 real: GREEDY %s (total %v)\n", strings.Join(greedy, " "), time.Since(t0).Round(time.Second))
}
