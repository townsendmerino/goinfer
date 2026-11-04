//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD8_sequentialPrefillBaseline is KS-D8's baseline (docs/tasks/task-kernel-survey-2026-10.md, "KS-D8 baseline"):
// NVIDIA-Nemotron-Nano-9B-v2's int4 Metal .giw resident on Metal, whose Mamba-2 layers decline the batched prefill
// (recurrentPrefillOK), so a prompt runs one decode step a token. One 512-token greedy generation of one token warms it,
// then three timed ones, each on a prompt with its own first token so nothing is reused; the reading is 512 / time to
// the first token. Not gated. A fit refusal prints FIT-REFUSED and fails, which the night script turns into exit 75.
//
//	GOINFER_KSD8_BASELINE=1 go test -tags goinfer_testhooks -count=1 -timeout 30m -run '^TestKSD8_sequentialPrefillBaseline$' -v ./metal/
func TestKSD8_sequentialPrefillBaseline(t *testing.T) {
	if os.Getenv("GOINFER_KSD8_BASELINE") == "" {
		t.Skip("set GOINFER_KSD8_BASELINE=1 (a timed run of the 9B under the timing lock)")
	}
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/nemotron/nemotron-nano-9b-v2-int4-metal.v%d.giw", decoder.GIWFormatVersion))
	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[KS-D8] FIT-REFUSED: %v\n", err)
		}
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	fmt.Fprintf(os.Stderr, "[KS-D8] loaded %s in %.1f s; batched prefill admitted: %v\n", path, time.Since(t0).Seconds(), a.r.recurrentPrefillOK())
	_, _, _, _, _, _, vocab := m.Dims()
	const n = 512
	run := func(i int) float64 {
		prompt := make([]int, n)
		for j := range prompt {
			prompt[j] = (j*7919 + 13 + i) % vocab
		}
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		fmt.Fprintf(os.Stderr, "[KS-D8] run %d starts: load %s", i, la)
		start := time.Now()
		ch, g := m.Generate(context.Background(), prompt, 1, decoder.SamplingParams{Temperature: 0})
		for range ch {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generation: %v", err)
		}
		return time.Since(start).Seconds()
	}
	run(0)
	for i := 1; i <= 3; i++ {
		s := run(i)
		fmt.Fprintf(os.Stderr, "[KS-D8] run %d: %d tokens to the first token in %.2f s, %.1f tok/s\n", i, n, s, n/s)
	}
}
