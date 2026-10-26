//go:build cuda && goinfer_testhooks

package cuda

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// ksh6Sampler streams nvidia-smi's used memory every 20 ms until stop; it returns the maximum used MiB it saw and the samples.
func ksh6Sampler(t *testing.T) (stop func() (maxUsed float64, samples int)) {
	t.Helper()
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "20")
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("nvidia-smi: %v", err)
	}
	done := make(chan struct{})
	var maxUsed float64
	var n int
	go func() {
		defer close(done)
		sc := bufio.NewScanner(out)
		for sc.Scan() {
			if v, err := strconv.ParseFloat(strings.TrimSpace(sc.Text()), 64); err == nil {
				maxUsed = max(maxUsed, v)
				n++
			}
		}
	}()
	return func() (float64, int) {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		<-done
		return maxUsed, n
	}
}

// ksh6UsedMiB is one nvidia-smi reading of the card's used memory.
func ksh6UsedMiB(t *testing.T) float64 {
	t.Helper()
	b, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		t.Fatalf("nvidia-smi: %v", err)
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// ksh6Model is one H6-R model: name, checkpoint, context (0: the build's own default), prompt (0: the context less 512) and serve's
// -moe-cache-experts.
type ksh6Model struct {
	name, path string
	ctx        int
	prompt     int
	moeCache   bool
}

// TestKSH6_marginPartsQwen3 and TestKSH6_marginParts26B are H6-R (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 reading"), a
// reading, not a gate, one model each (the heavy tier holds one large checkpoint a process): the model loaded as serve loads it
// (--kv auto, f16 where cleared), the free VRAM after the load, then a long prompt's prefill and 64 decode tokens with the card
// sampled every 20 ms. Each prints the parts the margins must cover; it asserts only that its run completed.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestKSH6_marginParts' -v ./cuda/
func TestKSH6_marginPartsQwen3(t *testing.T) {
	ksh6MarginParts(t, ksh6Model{"Qwen3-8B q4_k_m", modelPath("qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf"), 16384, 15000, false})
}

func TestKSH6_marginParts26B(t *testing.T) {
	ksh6MarginParts(t, ksh6Model{"Gemma 4 26B-A4B", modelPath("gemma-4-26b-a4b-it"), 0, 0, true})
}

func ksh6MarginParts(t *testing.T, c ksh6Model) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	kvF16Auto.Store(true) // serve's --kv auto; the suite defaults it off (kv16_default_test.go)
	defer kvF16Auto.Store(false)
	if _, err := os.Stat(c.path); err != nil {
		t.Skipf("no model %s: %v", c.path, err)
	}
	used0 := ksh6UsedMiB(t)
	m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: c.ctx, MoECacheExperts: c.moeCache})
	if err != nil {
		t.Fatalf("%s: load: %v", c.name, err)
	}
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		m.Close()
		t.Fatalf("%s: no CUDA resident: %s", c.name, m.ResidentDecline())
	}
	var free uint64
	if err := r.do(func() error { var e error; free, e = cudaFreeVRAM(r); return e }); err != nil {
		m.Close()
		t.Fatal(err)
	}
	usedLoad := ksh6UsedMiB(t)
	prompt := c.prompt
	if prompt == 0 {
		prompt = r.ctxCap - 512
	}
	pass512 := r.PrefillScratchBytes(512) + r.PrefillSpanScratchBytes(512, prompt)
	fmt.Fprintf(os.Stderr, "[H6] %s: context %d, KV f16 %v, expert cache %v (%d slots); free after the load %.1f MiB (cuMemGetInfo); nvidia-smi used %.0f → %.0f MiB over the load; a 512-row pass %.1f MiB of scratch at %d keys\n",
		c.name, r.ctxCap, r.kvF16, r.cacheExperts, r.cacheSlots, float64(free)/(1<<20), used0, usedLoad, float64(pass512)/(1<<20), prompt)
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, prompt)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
	}
	stop := ksh6Sampler(t)
	time.Sleep(200 * time.Millisecond)
	t0 := time.Now()
	lg, err := r.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		stop()
		m.Close()
		t.Fatalf("%s: prefill of %d: %v", c.name, prompt, err)
	}
	tPre := time.Since(t0)
	tok := 0
	for pos := prompt; pos < prompt+64 && pos < r.ctxCap; pos++ {
		tok = 0
		for i, v := range lg {
			if v > lg[tok] {
				tok = i
			}
		}
		if lg, err = r.Forward(m.EmbedResidentForTest(tok), pos); err != nil {
			stop()
			m.Close()
			t.Fatalf("%s: decode at %d: %v", c.name, pos, err)
		}
	}
	time.Sleep(200 * time.Millisecond)
	maxUsed, n := stop()
	var freeEnd uint64
	_ = r.do(func() error { var e error; freeEnd, e = cudaFreeVRAM(r); return e })
	fmt.Fprintf(os.Stderr, "[H6] %s: a %d-token prefill (%s, passes of %d rows, learned cap %d) and 64 decode tokens: nvidia-smi peak used %.0f MiB over %d samples, %.0f MiB above the post-load figure; the run's minimum free ≈ %.1f MiB (cuMemGetInfo after the load less that peak rise); free at the end %.1f MiB\n",
		c.name, prompt, tPre.Round(time.Millisecond), r.prefillPassRows(), r.prefillChunkCap.Load(), maxUsed, n, maxUsed-usedLoad,
		float64(free)/(1<<20)-(maxUsed-usedLoad), float64(freeEnd)/(1<<20))
	m.Close()
}
