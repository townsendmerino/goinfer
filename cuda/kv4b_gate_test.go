//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// KV4b's G4 (docs/measurements/kv4-cuda-2026-10-21.md §2, G5): TestKV16GateVsReference's instrument on one checkpoint per f16 family group,
// one test per group so each process reaches one checkpoint family (cmd/gate's heavy partition). The depth and the prompt count come from
// GOINFER_KV4B_GATE_K (3900 or 8000) and GOINFER_KV4B_GATE_N (the first n prompts of set B; ten at 3900, four at 8000); run-g4.sh sets both.

func TestKV4bGate_QMOE(t *testing.T) {
	kv4bGate(t, "QMOE", "$HOME/models/qwen15-moe-a27b", decoder.Options{MoECacheExperts: true}) // group 1: MoE, paged experts
}

func TestKV4bGate_G34B(t *testing.T) {
	kv4bGate(t, "G34B", "$HOME/models/gemma-3-4b-it", decoder.Options{}) // group 2: sliding window
}

func TestKV4bGate_CR7B(t *testing.T) {
	kv4bGate(t, "CR7B", "$HOME/models/command-r7b", decoder.Options{}) // group 3: pairwise RoPE
}

func TestKV4bGate_QVL3(t *testing.T) {
	kv4bGate(t, "QVL3", "$HOME/models/qwen2.5-vl-3b", decoder.Options{}) // group 4: m-RoPE, text cells
}

func TestKV4bGate_Q359(t *testing.T) {
	kv4bGate(t, "Q359", "$HOME/models/qwen3.5-9b", decoder.Options{}) // group 5: the hybrids
}

func TestKV4bGate_DSV2L(t *testing.T) {
	kv4bGate(t, "DSV2L", "$HOME/models/deepseek-v2-lite", decoder.Options{MoECacheExperts: true}) // group 6: MLA's latent
}

// kv4bGate runs one decision cell of KV4b's G4 for the named checkpoint, skipping (with the reason) when the run is not configured or
// its CPU reference is missing.
func kv4bGate(t *testing.T, name, def string, opts decoder.Options) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints; needs the set-B reference files)")
	}
	if s, _ := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE")); s < 1 {
		t.Skip("set GOINFER_CUDA_FLASH_DECODE=16: both arms run the production decode lane")
	}
	K, _ := strconv.Atoi(os.Getenv("GOINFER_KV4B_GATE_K"))
	n, _ := strconv.Atoi(os.Getenv("GOINFER_KV4B_GATE_N"))
	if K < 1 {
		t.Skip("set GOINFER_KV4B_GATE_K (3900 or 8000) and GOINFER_KV4B_GATE_N (run-g4.sh)")
	}
	label, promptFiles := decoder.PrefillGatePromptSet()
	if label != "b" {
		t.Fatalf("the registered gate runs on held-out set B; GOINFER_PREFILL_GATE_PROMPTS selects set %q", label)
	}
	if n < 2 || n > len(promptFiles) {
		t.Fatalf("GOINFER_KV4B_GATE_N=%d: want 2 to %d", n, len(promptFiles))
	}
	files := promptFiles[:n]
	path := os.ExpandEnv(def)
	for _, bad := range []string{"/srv/models", "/Volumes/"} {
		if strings.HasPrefix(path, bad) {
			t.Fatalf("%s is the archive, not a bench surface (CLAUDE.md)", path)
		}
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	home, _ := os.UserHomeDir()
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref-b")
	for pi := range files {
		if _, err := os.Stat(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", name, K, pi))); err != nil {
			t.Fatalf("%s K=%d: reference missing for prompt %d: run run-g4.sh's reference phase on this tree first", name, K, pi)
		}
	}
	runKV16GateCell(t, path, name, K, files, refDir, true, opts)
}
