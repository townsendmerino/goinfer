//go:build goinfer_testhooks

package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR9_decodeAttribution is docs/tasks/red-october.md R9, step 1, Mac half: the per-component
// ms/token table step 1 asks for, using the already-shipped GOINFER_DECODE_TIMING diagnostic
// (decoder/model.go's decodeTiming var — forward/sample/logitProc/embed, already split; this test
// adds nothing new there) at the brief's own decision-cell depth (128), greedy, on 0.5B/1.5B/7B.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_decodeAttribution -v -timeout 20m
//
// TestR9_w4a8BatchAt7B is a follow-up to R-06 (docs/tasks/task-recompute-audit.md), which measured
// GOINFER_W4A8_BATCH's fused q/k/v and gate/up matmuls on the 1.5B only (1.071x arm64/Metal,
// 1.066x amd64/CPU -- both ambiguous, parked). R9 step 1's own finding (MLP's share of the token
// grows with model size, largest at 7B) raises the natural follow-up: does R-06 behave differently
// at 7B? w4a8BatchEnabled (decoder/weightmat.go) is a package-level var read once from
// GOINFER_W4A8_BATCH at process start, so this test cannot toggle it mid-process -- it reports
// this run's own mean/stdev over repeated decode windows (fresh KV cache each, same loaded model,
// no reload between repeats) and is meant to be run TWICE, once per env setting, the results
// compared by hand (or by a small script) rather than paired within one process. That is a real
// limitation next to R-06's own interleaved design -- disclosed, not hidden -- see the record.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 GOINFER_NO_FIT_GUARD=1 GOINFER_W4A8_BATCH=0 go test -tags goinfer_testhooks ./decoder/ -run TestR9_w4a8BatchAt7B -v -timeout 20m
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 GOINFER_NO_FIT_GUARD=1 GOINFER_W4A8_BATCH=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_w4a8BatchAt7B -v -timeout 20m
func TestR9_w4a8BatchAt7B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real 7B checkpoint on CPU)")
	}
	if os.Getenv("GOINFER_DECODE_TIMING") == "" {
		t.Skip("set GOINFER_DECODE_TIMING=1 so decoder.decodeTiming prints the per-window forward ms/token")
	}
	path := os.Getenv("GOINFER_CPU_MODEL_D7")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s (set GOINFER_CPU_MODEL_D7)", path)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	text := "Continue this text. " + strings.Repeat(" the", 128)
	ids, err := tk.Encode(text, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) > 128 {
		ids = ids[:128]
	}
	const repeats = 6
	const decodeN = 16
	t.Logf("w4a8Batch env=%q, %d repeats x %d tokens, fresh cache each (see individual DECODE TIMING lines above)", os.Getenv("GOINFER_W4A8_BATCH"), repeats, decodeN)
	for i := range repeats {
		out, g := m.Generate(context.Background(), ids, decodeN, SamplingParams{Temperature: 0})
		n := 0
		for range out {
			n++
		}
		if g.err != nil {
			t.Fatalf("repeat %d: generate: %v", i, g.err)
		}
	}
}

func TestR9_decodeAttribution(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints on CPU)")
	}
	if os.Getenv("GOINFER_DECODE_TIMING") == "" {
		t.Skip("set GOINFER_DECODE_TIMING=1 so decoder.decodeTiming prints the per-component split")
	}

	models := []struct {
		name string
		env  string
		def  string
	}{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	const promptDepth = 128 // R9's own decision-cell depth
	const decodeN = 24      // enough tokens for decodeTiming's per-token average to settle

	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.env)
			}
			m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}
			// A calibrated-shape filler prompt (scripts/bench_prompts_calibrate.py's own
			// convention: " the" repeats at ~1 token/word) tokenized for real by this model's own
			// tokenizer -- content doesn't matter for a timing run, real tokenization does.
			text := "Continue this text. " + strings.Repeat(" the", promptDepth)
			ids, err := tk.Encode(text, true)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if len(ids) > promptDepth {
				ids = ids[:promptDepth]
			}
			t.Logf("%s: prompt %d tokens, decoding %d greedy", mc.name, len(ids), decodeN)
			out, g := m.Generate(context.Background(), ids, decodeN, SamplingParams{Temperature: 0})
			n := 0
			for range out {
				n++
			}
			if g.err != nil {
				t.Fatalf("generate: %v", g.err)
			}
			t.Logf("%s: %d tokens generated (DECODE TIMING line printed above by decoder.decodeTiming)", mc.name, n)
		})
	}
}

// TestR9_parWidthSweep is R9 step 1's fan-out-shape reading on the Linux box: the same decode
// window at matmul fan-out widths 1 / 2 / 4 / 8 / 16 (linalg.SetParallelWidth, numerically inert
// by contract — every output column is still computed whole by one worker), on one loaded model,
// widths interleaved so drift cannot pose as a curve. The serial (width 1) time is the compute
// floor; the gap between width 8 and 16 on an 8-core/16-thread part is the SMT contribution; the
// shortfall of width-8 against serial/8 is what the fork/join and imbalance cost. The attention
// head fan-out is NOT swept (decoder/scratch.go's maxAttnWorkers is a compile-time 6).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 go test ./decoder/ -run TestR9_parWidthSweep -v
func TestR9_parWidthSweep(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints on CPU)")
	}
	if os.Getenv("GOINFER_DECODE_TIMING") == "" {
		t.Skip("set GOINFER_DECODE_TIMING=1 so decoder.decodeTiming prints the per-window forward ms/token")
	}
	models := []struct{ name, env, def string }{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
	}
	widths := []int{16, 8, 4, 2, 1, 1, 2, 4, 8, 16} // ABBA-shaped: up then back down
	const promptDepth, decodeN = 128, 24
	defer linalg.SetParallelWidth(0)
	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.env)
			}
			m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}
			ids, err := tk.Encode("Continue this text. "+strings.Repeat(" the", promptDepth), true)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if len(ids) > promptDepth {
				ids = ids[:promptDepth]
			}
			// warm-up at the default width, discarded
			out, g := m.Generate(context.Background(), ids, 8, SamplingParams{Temperature: 0})
			for range out {
			}
			if g.err != nil {
				t.Fatalf("warm-up: %v", g.err)
			}
			for _, w := range widths {
				linalg.SetParallelWidth(w)
				fmt.Printf("R9 WIDTH %s width=%d GOMAXPROCS=%d\n", mc.name, w, runtime.GOMAXPROCS(0))
				out, g := m.Generate(context.Background(), ids, decodeN, SamplingParams{Temperature: 0})
				for range out {
				}
				if g.err != nil {
					t.Fatalf("width %d: %v", w, g.err)
				}
			}
		})
	}
}

// TestR9_mlpMatmulIsolated separates "the kernel" from "the token around it" on the real weights:
// the exact matmulInto call mlp() makes, on the loaded model's own MLP matrices, (a) HOT — one
// matrix repeated, cache-resident — and (b) COLD — every layer's gate/up/down swept once in
// order, DRAM-streamed exactly as a decode token streams them — at fan-out widths 1 and 16.
// The cold sweep's ms is directly comparable to the decode profile's MLP ms/token (same calls,
// minus the SwiGLU between them); the hot rate is the kernel's own ceiling at this shape.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_mlpMatmulIsolated -v
func TestR9_mlpMatmulIsolated(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints on CPU)")
	}
	models := []struct{ name, env, def string }{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
	}
	defer linalg.SetParallelWidth(0)
	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.env)
			}
			m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			L := m.w.Layers
			hidden, inter := L[0].GateProj.Cols(), L[0].GateProj.Rows()
			h := make([]float32, hidden)
			mid := make([]float32, inter)
			for i := range h {
				h[i] = float32(i%13)*0.01 - 0.06
			}
			for i := range mid {
				mid[i] = float32(i%7)*0.02 - 0.06
			}
			gate, up, out := make([]float32, inter), make([]float32, inter), make([]float32, hidden)
			ws := &linalg.Workspace{}
			ws.SetThreshold(DefaultDecodeParallelThreshold)
			macsPerSweep := 0.0
			for l := range L {
				macsPerSweep += float64(L[l].GateProj.Rows()*L[l].GateProj.Cols() + L[l].UpProj.Rows()*L[l].UpProj.Cols() + L[l].DownProj.Rows()*L[l].DownProj.Cols())
			}
			bytesPerSweep := macsPerSweep * 0.5 // int4 weights, scales ignored
			t.Logf("%s: %d layers, hidden %d, inter %d, %.0f MMAC / %.0f MB per MLP sweep", mc.name, len(L), hidden, inter, macsPerSweep/1e6, bytesPerSweep/1e6)
			for _, w := range []int{1, 16, 1, 16} {
				linalg.SetParallelWidth(w)
				// hot: layer 0 gate, repeated
				for i := 0; i < 20; i++ {
					matmulInto(ws, m.be, &L[0].GateProj, h, gate, 1)
				}
				const hotN = 200
				t0 := time.Now()
				for i := 0; i < hotN; i++ {
					matmulInto(ws, m.be, &L[0].GateProj, h, gate, 1)
				}
				hot := time.Since(t0) / hotN
				hotMACs := float64(L[0].GateProj.Rows() * L[0].GateProj.Cols())
				// cold: full sweep, gate/up/down per layer in order
				const sweeps = 8
				t1 := time.Now()
				for s := 0; s < sweeps; s++ {
					for l := range L {
						matmulInto(ws, m.be, &L[l].GateProj, h, gate, 1)
						matmulInto(ws, m.be, &L[l].UpProj, h, up, 1)
						matmulInto(ws, m.be, &L[l].DownProj, mid, out, 1)
					}
				}
				cold := time.Since(t1) / sweeps
				t.Logf("%s width=%2d: HOT gate[%dx%d] %.3f ms/call = %.1f GMAC/s | COLD full-MLP sweep %.1f ms = %.1f GMAC/s, %.1f GB/s weights",
					mc.name, w, L[0].GateProj.Rows(), L[0].GateProj.Cols(), float64(hot.Microseconds())/1000, hotMACs/hot.Seconds()/1e9,
					float64(cold.Microseconds())/1000, macsPerSweep/cold.Seconds()/1e9, bytesPerSweep/cold.Seconds()/1e9)
			}
		})
	}
}

// TestR9_groupedDepthSweep: R13's grouped attention kernels (GOINFER_ATTN_GROUPED, default on)
// against the per-head path, on the 1.5B (the 6-heads-per-KV geometry the Linux attribution
// found paying 11 ms/token at depth 128), across prompt depth, arms interleaved ABBA on one
// loaded model. attnGroupedEnabled reads the env per call, so the arm flips in-process. The
// R9 ATTN line's core term is the quantity; forward ms is the token.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 GOINFER_R9_DIAG=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_groupedDepthSweep -v
func TestR9_groupedDepthSweep(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint on CPU)")
	}
	if os.Getenv("GOINFER_DECODE_TIMING") == "" {
		t.Skip("set GOINFER_DECODE_TIMING=1")
	}
	path := os.Getenv("GOINFER_CPU_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	depths := []int{128, 512, 1024, 2048, 4096}
	if v := os.Getenv("GOINFER_R9_DEPTHS"); v != "" {
		depths = nil
		for _, f := range strings.Split(v, ",") {
			var d int
			fmt.Sscanf(f, "%d", &d)
			depths = append(depths, d)
		}
	}
	const decodeN = 24
	for _, depth := range depths {
		ids, err := tk.Encode("Continue this text. "+strings.Repeat(" the", depth+8), true)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		if len(ids) > depth {
			ids = ids[:depth]
		}
		for _, arm := range []string{"1", "0", "0", "1"} {
			os.Setenv("GOINFER_ATTN_GROUPED", arm)
			fmt.Printf("R9 GROUPED depth=%d grouped=%s\n", len(ids), arm)
			out, g := m.Generate(context.Background(), ids, decodeN, SamplingParams{Temperature: 0})
			for range out {
			}
			if g.err != nil {
				t.Fatalf("depth %d arm %s: %v", depth, arm, g.err)
			}
		}
	}
	os.Unsetenv("GOINFER_ATTN_GROUPED")
}

// TestR9_cpuTuningAB is the paired gate for the two Linux-attribution fixes
// (docs/measurements/cpu-decode-attribution-2026-09-22-linux.md): the activation fan-out off vs
// on, and the grouped-attention path off vs on, each flipped in-process on one loaded 1.5B,
// ABBA, reading DECODE SPLIT's own numbers. Both are bit-identical by construction (elementwise
// activation partition-independent; R13's grouped path already gated as bit-identical) — the
// speed is the question. Logged against the pre-registered 3% bar, not asserted.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_cpuTuningAB -v
func TestR9_cpuTuningAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint on CPU)")
	}
	path := os.Getenv("GOINFER_CPU_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	prevTiming := decodeTiming
	decodeTiming = true
	defer func() { decodeTiming = prevTiming }()
	m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	ids, err := tk.Encode("Continue this text. "+strings.Repeat(" the", 128), true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) > 128 {
		ids = ids[:128]
	}
	gen := func() float64 {
		out, g := m.Generate(context.Background(), ids, 24, SamplingParams{Temperature: 0})
		for range out {
		}
		if g.err != nil {
			t.Fatalf("generate: %v", g.err)
		}
		return lastDecodeSplit.fwd
	}
	gen() // warm-up
	type knob struct {
		name string
		set  func(on bool)
		part func() float64
	}
	knobs := []knob{
		{"activation fan-out", func(on bool) { activationFanoutEnabled = on }, func() float64 { return lastDecodeSplit.act }},
		{"grouped attention", func(on bool) { attnGroupedKernels = on }, func() float64 { return lastDecodeSplit.core }},
	}
	origAct, origGrp := activationFanoutEnabled, attnGroupedKernels
	defer func() { activationFanoutEnabled, attnGroupedKernels = origAct, origGrp }()
	for _, k := range knobs {
		const pairs = 3
		var rOn, rOff, pOn, pOff float64
		for p := 0; p < pairs; p++ {
			var on, off, partOn, partOff float64
			if p%2 == 0 {
				k.set(true)
				on, partOn = gen(), k.part()
				k.set(false)
				off, partOff = gen(), k.part()
			} else {
				k.set(false)
				off, partOff = gen(), k.part()
				k.set(true)
				on, partOn = gen(), k.part()
			}
			rOn, rOff, pOn, pOff = rOn+on, rOff+off, pOn+partOn, pOff+partOff
			t.Logf("%s pair %d: ON %.1f ms/tok (component %.2f) | OFF %.1f ms/tok (component %.2f) | OFF is %.3fx faster", k.name, p, on, partOn, off, partOff, on/off)
		}
		t.Logf("%s: pooled ON %.1f -> OFF %.1f ms/tok = %.3fx; component %.2f -> %.2f ms/tok", k.name, rOn/pairs, rOff/pairs, rOn/rOff, pOn/pairs, pOff/pairs)
	}
}
