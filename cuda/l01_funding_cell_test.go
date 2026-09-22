//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestL01FundingCell is R11/L01's committed funding-cell driver (docs/tasks/task-l01-hybrid-moe-cpu-gpu.md
// §0's pre-registered decision rule, restated here, not re-derived: Qwen3.6-35B-A3B int4 on the 8 GB
// card, paired and interleaved against the shipped C′ path; fund at >=1.3x end-to-end, park below
// 1.15x, 1.15-1.3x ambiguous -> second mechanism). §9's remainder before this cell could run: async
// overlap (cuda/l01_cpu_offload.go, 2026-09-21 — goroutine-per-expert CPU compute alongside the
// GPU's own async hit-path launches).
//
// One process, one arm, one run: prints tok/s, C′ hit rate, and a SHA-256 of the output token ids
// (a precondition check — L01 is bit-identical end to end per TestL01_e2eDecode_matchesBaseline, so
// greedy decode must follow the IDENTICAL token trajectory regardless of the flag; a hash mismatch
// between arms voids the pairing, since it would mean the two arms are not decoding the same
// positions and their tok/s is not comparable).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_L01_CPU_OFFLOAD=1 GOINFER_L01_NEW_TOKENS=128 \
//	  go test -tags "cuda goinfer_testhooks" ./cuda/ -run TestL01FundingCell -v -timeout 30m
func TestL01FundingCell(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 35B decode")
	}
	path := os.Getenv("GOINFER_QWEN36_35B")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.6-35b-a3b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 35B checkpoint at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	if slots := os.Getenv("GOINFER_MOE_CACHE_SLOTS"); slots != "" {
		t.Setenv("GOINFER_MOE_CACHE_SLOTS", slots)
	}
	nNew := 128
	if v, err := strconv.Atoi(os.Getenv("GOINFER_L01_NEW_TOKENS")); err == nil && v > 0 {
		nNew = v
	}
	l01 := os.Getenv("GOINFER_CUDA_L01_CPU_OFFLOAD") != ""

	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(35B, cuda int4): %v", err)
	}
	defer m.Close()
	loadDur := time.Since(t0)
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("cuda resident DECLINED — decode path %q; decline: %s", m.DecodePath(), m.ResidentDecline())
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("C′ expert staging is OFF — this would measure a model that fit VRAM after all")
	}
	if l01 != r.l01Enabled {
		t.Fatalf("GOINFER_CUDA_L01_CPU_OFFLOAD=%v but r.l01Enabled=%v — the flag did not reach the resident", l01, r.l01Enabled)
	}
	t.Logf("loaded 35B in %s, l01=%v, cacheSlots=%d", loadDur.Round(time.Second), l01, r.cacheSlots)

	var tk *tokenizer.Tokenizer
	switch {
	case strings.HasSuffix(path, ".giw"):
		var tb []byte
		if tb, err = giw.ReadTokFile(path); err == nil {
			tk, err = tokenizer.LoadGGUFBytes(tb)
		}
	case strings.HasSuffix(path, ".gguf"):
		tk, err = tokenizer.LoadGGUF(path)
	default:
		tk, err = tokenizer.Load(path)
	}
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	turns := []chat.Turn{{Role: "user", Content: "Explain, in a few paragraphs, how a hash table resolves collisions and why the choice of load factor matters."}}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", turns), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	r.launchN = 0
	t1 := time.Now()
	ch, _ := m.Generate(context.Background(), ids, nNew, decoder.SamplingParams{Temperature: 0})
	var out []int
	for tok := range ch {
		out = append(out, tok)
	}
	genDur := time.Since(t1)
	if len(out) == 0 {
		t.Fatal("no tokens generated")
	}
	txt, _ := tk.Decode(out)
	rate := float64(len(out)) / genDur.Seconds()
	hits, misses := r.CacheStatsForTest()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}
	h := sha256.New()
	for _, id := range out {
		fmt.Fprintf(h, "%d,", id)
	}
	tokHash := hex.EncodeToString(h.Sum(nil))[:16]

	fmt.Printf("=== L01FUNDING l01=%v tokens=%d dur=%s rate=%.3f hitRate=%.4f hits=%d misses=%d tokHash=%s ===\n",
		l01, len(out), genDur.Round(time.Millisecond), rate, hitRate, hits, misses, tokHash)
	if ratio := distinctTrigramRatio(txt); ratio < 0.70 {
		t.Errorf("degenerate output: distinct-trigram ratio %.3f < 0.70", ratio)
	}
	t.Logf("continuation: %q", txt)
}
