//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPLE_qualityE2BMetal is PLE-G3 (docs/tasks/task-memory-campaign-2026-10.md, MEM8(b) gates): E2B on Metal with its
// per-layer table at int8 (the sidecar written before the table left the head pin) against int4 (the .ple4 sidecar), one
// arm at a time. The int8 arm free-runs each G3 prompt 32 tokens greedy; the int4 arm is teacher-forced on that sequence.
// PASS: top-1 agreement over every position >= 95.0% and mean KL(int8 || int4) <= 0.05 nats; 90.0-95.0% or KL 0.05-0.10
// is ambiguous (parked for the owner); worse fails.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags goinfer_testhooks -run '^TestPLE_qualityE2BMetal$' -v ./metal/
func TestPLE_qualityE2BMetal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it")
	gguf, int8GIW := base+".gguf", base+".int4.metal.v17.giw"
	for _, p := range []string{gguf, int8GIW} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no %s: %v", p, err)
		}
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[PLE-G3 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	int4GIW, err := decoder.EnsureCachedGIW(context.Background(), gguf, "int4", "metal", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(filepath.Base(int4GIW), ".ple4.") {
		t.Fatalf("the int4 arm's sidecar %s has no .ple4 segment: this build does not key the table off the pin", int4GIW)
	}
	for _, p := range []string{int8GIW, int4GIW} {
		if fi, err := os.Stat(p); err == nil {
			logf("sidecar %s: %.1f MB", filepath.Base(p), float64(fi.Size())/(1<<20))
		}
	}
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	// arm loads one sidecar resident and calls fn with a step function: the logits after token tok at position pos.
	arm := func(giw, label string, fn func(step func(tok, pos int) []float32)) {
		m, err := decoder.Load(giw, decoder.Options{Quant: "int4", ResidentContext: 512})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if why := residentMemoryDecline(m); why != "" {
			t.Fatalf("%s: Metal's fit guard declines it: %s", label, why)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		logf("%s: resident", label)
		fn(func(tok, pos int) []float32 { return r.ForwardEmb(m.EmbedResidentForTest(tok), pos) })
	}
	logSoftmax := func(l []float32) []float64 {
		mx := math.Inf(-1)
		for _, v := range l {
			mx = math.Max(mx, float64(v))
		}
		s := 0.0
		for _, v := range l {
			s += math.Exp(float64(v) - mx)
		}
		ls := math.Log(s) + mx
		out := make([]float64, len(l))
		for i, v := range l {
			out[i] = float64(v) - ls
		}
		return out
	}
	type prompt struct {
		seq  []int
		lp8  [][]float64
		top8 []int
	}
	ps := make([]prompt, len(g3Prompts))
	arm(int8GIW, "int8 table", func(step func(tok, pos int) []float32) {
		for pi, text := range g3Prompts {
			ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
			if err != nil {
				t.Fatal(err)
			}
			seq := append([]int(nil), ids...)
			n := len(ids) + g3NewTokens
			for i := 0; i < n-1; i++ {
				l := step(seq[i], i)
				ps[pi].lp8 = append(ps[pi].lp8, logSoftmax(l))
				ps[pi].top8 = append(ps[pi].top8, argmaxF(l))
				if i >= len(ids)-1 {
					seq = append(seq, argmaxF(l))
				}
			}
			ps[pi].seq = seq
		}
	})
	agree, positions, klSum, klMax := 0, 0, 0.0, 0.0
	arm(int4GIW, "int4 table", func(step func(tok, pos int) []float32) {
		for pi := range ps {
			pa := 0
			for i, lp8 := range ps[pi].lp8 {
				l := step(ps[pi].seq[i], i)
				lp4 := logSoftmax(l)
				kl := 0.0
				for j, a := range lp8 {
					kl += math.Exp(a) * (a - lp4[j])
				}
				klSum += kl
				klMax = math.Max(klMax, kl)
				if argmaxF(l) == ps[pi].top8[i] {
					agree++
					pa++
				}
				positions++
			}
			logf("prompt %d: %d/%d positions agree", pi+1, pa, len(ps[pi].lp8))
		}
	})
	pct, kl := 100*float64(agree)/float64(positions), klSum/float64(positions)
	logf("PLE-G3: top-1 agreement %d/%d = %.2f%%, mean KL(int8 || int4) %.5f nats (max %.5f)", agree, positions, pct, kl, klMax)
	switch {
	case pct < 90.0 || kl > 0.10:
		t.Errorf("PLE-G3 FAIL: agreement %.2f%%, mean KL %.5f", pct, kl)
	case pct < 95.0 || kl > 0.05:
		t.Errorf("PLE-G3 AMBIGUOUS (parked for the owner): agreement %.2f%%, mean KL %.5f", pct, kl)
	default:
		t.Logf("PLE-G3 PASS: agreement %.2f%%, mean KL %.5f", pct, kl)
	}
}
