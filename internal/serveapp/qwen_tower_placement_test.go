package serveapp

import (
	"errors"
	"strings"
	"testing"
)

// qwenTowerPlacement (S4, docs/tasks/task-multimodal-support-2026-10.md): Qwen2.5-VL's tower goes to aikit's CUDA tower only under --backend cuda with a float32 tower; every
// other case is the CPU with its reason named, and -require-backend refuses the cuda fallbacks.
func TestQwenTowerPlacement(t *testing.T) {
	ok := func() error { return nil }
	declined := func() error { return errors.New("out of memory") }
	mustNotAttach := func() error { t.Fatal("attach must not be called"); return nil }
	cases := []struct {
		name         string
		backend      string
		int8, req    bool
		attach       func() error
		want         string // a prefix of the placement; "" with wantErr
		wantErr      string
		wantWarnPart string
	}{
		{"cuda attaches", "cuda", false, false, ok, "CUDA", "", ""},
		{"cuda attaches under require", "cuda", false, true, ok, "CUDA", "", ""},
		{"cuda declined falls to the CPU, named, with a warning", "cuda", false, false, declined, "CPU (cuda declined)", "", "cuda declined it: out of memory"},
		{"cuda declined is refused under require", "cuda", false, true, declined, "", "-require-backend: the Qwen2.5-VL tower could not start on cuda", ""},
		{"cuda int8 keeps the CPU, named", "cuda", true, false, mustNotAttach, "CPU (-vision-quant int8", "", ""},
		{"cuda int8 is refused under require", "cuda", true, true, mustNotAttach, "", "-require-backend: the Qwen2.5-VL CUDA tower is float32", ""},
		{"-vision-device cpu (backend cpu) never attaches", "cpu", false, true, mustNotAttach, "CPU", "", ""},
		{"metal has no Qwen2.5-VL device tower here", "metal", false, true, mustNotAttach, "CPU", "", ""},
		{"webgpu has none", "webgpu", false, false, mustNotAttach, "CPU", "", ""},
		{"auto/empty never attaches", "", false, false, mustNotAttach, "CPU", "", ""},
	}
	for _, c := range cases {
		var warn strings.Builder
		got, err := qwenTowerPlacement(c.backend, c.int8, c.req, c.attach, &warn)
		if c.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("%s: err = %v, want one containing %q", c.name, err, c.wantErr)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
			continue
		}
		if !strings.HasPrefix(got, c.want) {
			t.Errorf("%s: placement %q, want prefix %q", c.name, got, c.want)
		}
		if c.wantWarnPart != "" && !strings.Contains(warn.String(), c.wantWarnPart) {
			t.Errorf("%s: warning %q lacks %q", c.name, warn.String(), c.wantWarnPart)
		}
		if c.wantWarnPart == "" && warn.Len() != 0 {
			t.Errorf("%s: unexpected warning %q", c.name, warn.String())
		}
	}
}
