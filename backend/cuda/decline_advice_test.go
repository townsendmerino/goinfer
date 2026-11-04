//go:build cuda

package cuda

import (
	"os"
	"strings"
	"testing"
)

// R20's gate: a resident decline's reason names a remedy, and a recovered executor panic's stack is not part of it. The two inputs
// are the errors the real 26B q4_0 produced on the 8 GB RTX 2070 SUPER (device OOM through runJob's panic boundary)
// and the cold-user run's (an empty weight), not strings written to suit the function.
func TestDeclineAdvice(t *testing.T) {
	oom := "cuda: executor job panicked: cuda: device allocation failed (typed-len, 253755392 bytes): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY\n" +
		"goroutine 789 [running, locked to thread]:\nruntime/debug.Stack()\n\t/x/stack.go:26 +0x5e\n"
	reason, detail := declineAdvice(oom, false)
	for _, want := range []string{"CUDA_ERROR_OUT_OF_MEMORY", "-moe-cache-experts", "-quant", "-ctx", "-backend cpu"} {
		if !strings.Contains(reason, want) {
			t.Errorf("OOM reason lacks %q: %s", want, reason)
		}
	}
	if strings.Contains(reason, "goroutine") || strings.Contains(reason, "\n") {
		t.Errorf("the stack leaked into the reason: %q", reason)
	}
	if !strings.HasPrefix(detail, "goroutine 789") {
		t.Errorf("the stack was dropped instead of kept for stderr: %q", detail)
	}

	// With the expert cache already on, offering it again is wrong advice; the next lever is its slot count.
	reason, _ = declineAdvice(oom, true)
	if strings.Contains(reason, "Try -moe-cache-experts") || !strings.Contains(reason, "-moe-cache-slots") {
		t.Errorf("with -moe-cache-experts already set the advice should be -moe-cache-slots, not the flag again: %s", reason)
	}

	empty := `cuda: unsupported projection kind ""`
	reason, detail = declineAdvice(empty, false)
	for _, want := range []string{`kind ""`, "-moe-cache-experts", "-quant int4", "-backend cpu"} {
		if !strings.Contains(reason, want) {
			t.Errorf("empty-weight reason lacks %q: %s", want, reason)
		}
	}
	if detail != "" {
		t.Errorf("no stack to keep, detail = %q", detail)
	}

	// A feature decline has its own explicit text; it is returned as it came, not decorated with memory advice.
	feat := "arch needs unimplemented feature(s) [softcap]"
	if r, d := declineAdvice(feat, false); r != feat || d != "" {
		t.Errorf("a feature decline was changed: %q / %q", r, d)
	}
}

// TestDeclineAdvice_wiredIntoBuildResident is the wiring half, as source — a BuildResident that reaches a real device OOM needs the
// card this test cannot assume: the one closure every decline in BuildResident goes through must route its error through
// declineAdvice, or the unit above vouches for a function nothing calls.
func TestDeclineAdvice_wiredIntoBuildResident(t *testing.T) {
	src, err := os.ReadFile("backend.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	i := strings.Index(body, "declined := func(e error)")
	if i < 0 {
		t.Fatal("the declined closure is not in backend.go — this guard is watching nothing")
	}
	j := strings.Index(body[i:], "\n\t}\n")
	if j < 0 || !strings.Contains(body[i:i+j], "declineAdvice(e.Error(), m.MoECacheExperts())") {
		t.Error("declined no longer routes its error through declineAdvice — a decline names no remedy again (R20)")
	}
}
