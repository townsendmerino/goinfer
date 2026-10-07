//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGemma4_26B_s10Dump is the Mac half of S1.0's 26B re-check (docs/tasks/task-multimodal-support-2026-10.md).
// Every 26B parity test here compares against a CPU 26B forward, which this 16 GB Mac must not run (the M26 rule:
// Metal only, paged, guards on). So the Mac dumps Metal's per-position logits and nobara supplies the CPU reference
// later, over the same token sequences.
//
// One arm per process: GOINFER_S10_DROP unset is the fixed arm (S1.0's layer scalar and v_norm in) and writes the
// sequences (each prompt plus 16 greedy tokens); GOINFER_S10_DROP=both is the before-fix arm and teacher-forces those
// same sequences. Output in GOINFER_S10_DUMP_DIR: seqs.json, and logits-<arm>.f32 (every position's full logits,
// little-endian float32, in sequence order).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_S10_DUMP_DIR=<dir> [GOINFER_S10_DROP=both] go test -count=1 -tags goinfer_testhooks \
//	    -run '^TestGemma4_26B_s10Dump$' -v ./metal/
func TestGemma4_26B_s10Dump(t *testing.T) {
	requireHeavyModel(t)
	out := os.Getenv("GOINFER_S10_DUMP_DIR")
	if out == "" {
		t.Skip("set GOINFER_S10_DUMP_DIR")
	}
	arm := "fixed"
	if d := os.Getenv("GOINFER_S10_DROP"); d != "" {
		arm = "drop-" + d
	}
	giw := modelPath("gemma4-26b-int4-v14st.metal.giw")
	tokPath := modelPath(filepath.Join("gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf")) // Gemma 4 shares one tokenizer
	for _, p := range []string{giw, tokPath} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is on the archive (CLAUDE.md)", p)
		}
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s: %v", p, err)
		}
	}
	rssMB := func() int {
		b, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
		if err != nil {
			return -1
		}
		kb, _ := strconv.Atoi(strings.TrimSpace(string(b)))
		return kb / 1024
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[26B %s %6.1fs] %s\n", arm, time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}

	// The M26 rule: Metal, paged, the guards on — and refuse anything that is not exactly that.
	m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4", MoECacheExperts: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if p := m.DecodePath(); !strings.HasPrefix(p, "metal-resident") {
		t.Fatalf("decode path %q is not metal-resident: refusing (the CPU path is forbidden for M26 on this Mac)", p)
	}
	mr, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || mr.r.g4moe == nil || !mr.r.g4moe.paged {
		t.Fatalf("the resident is not the paged Gemma 4 MoE one: refusing")
	}
	logf("loaded: %s, paged %d slots/layer, RSS %d MB", m.DecodePath(), mr.r.g4moe.slots, rssMB())

	tk, err := tokenizer.LoadGGUF(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	if v := m.Config().VocabSize; v != 262144 {
		t.Fatalf("26B vocab %d, expected Gemma 4's 262144 (the E2B tokenizer would not fit)", v)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	nPrompts, nNew := 3, 16
	if os.Getenv("GOINFER_S10_SMOKE") == "1" { // a load-and-run check before queueing: one prompt, two tokens
		nPrompts, nNew = 1, 2
	}
	seqPath := filepath.Join(out, "seqs.json")
	var seqs [][]int
	var promptLens []int
	if arm != "fixed" {
		raw, err := os.ReadFile(seqPath)
		if err != nil {
			t.Fatalf("the before-fix arm teacher-forces the fixed arm's sequences; run that arm first: %v", err)
		}
		var s struct {
			Seqs       [][]int `json:"seqs"`
			PromptLens []int   `json:"prompt_lens"`
		}
		if err := json.Unmarshal(raw, &s); err != nil {
			t.Fatal(err)
		}
		seqs, promptLens = s.Seqs, s.PromptLens
	}
	lf, err := os.Create(filepath.Join(out, "logits-"+arm+".f32"))
	if err != nil {
		t.Fatal(err)
	}
	defer lf.Close()
	buf := make([]byte, 4*262144)
	for pi := 0; pi < nPrompts; pi++ {
		var seq []int
		var plen int
		if arm == "fixed" {
			ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: g3Prompts[pi]}}), false)
			if err != nil {
				t.Fatal(err)
			}
			seq, plen = append([]int(nil), ids...), len(ids)
		} else {
			seq, plen = seqs[pi], promptLens[pi]
		}
		total := plen + nNew
		for i := 0; i < total-1; i++ {
			l, err := mr.Forward(m.EmbedResidentForTest(seq[i]), i)
			if err != nil {
				t.Fatalf("prompt %d pos %d: %v", pi+1, i, err)
			}
			for j, v := range l {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(v))
			}
			if _, err := lf.Write(buf[:4*len(l)]); err != nil {
				t.Fatal(err)
			}
			if arm == "fixed" && i >= plen-1 {
				seq = append(seq, argmaxF(l))
			}
			if rss := rssMB(); rss > 15000 {
				t.Fatalf("RSS %d MB over the 15000 MB safety ceiling at prompt %d pos %d: aborting", rss, pi+1, i)
			}
		}
		logf("prompt %d: %d positions dumped, RSS %d MB", pi+1, total-1, rssMB())
		if arm == "fixed" {
			seqs, promptLens = append(seqs, seq), append(promptLens, plen)
		}
	}
	if arm == "fixed" {
		b, _ := json.Marshal(map[string]any{"seqs": seqs, "prompt_lens": promptLens, "n_new": nNew,
			"model": filepath.Base(giw), "vocab": 262144, "note": "S1.0 26B re-check: Metal fixed arm's sequences; positions 0..len-2 of each are dumped"})
		if err := os.WriteFile(seqPath, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	logf("done")
}
