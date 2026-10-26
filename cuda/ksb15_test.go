//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// KS-B15's gates (docs/tasks/task-kernel-survey-2026-10.md, "KS-B15 build: gates"): the exact prefill attention kernels opt in past
// the 48 KiB default, so a band of spans that took the row-batched kernels takes attn_batched; decode keeps the default bound, and a
// band prompt keeps the exact GEMM.

// ksb15Bound is the longest span the exact prefill attention holds in shared memory at r's limit.
func ksb15Bound(r *cudaResident) int { return r.prefillAttnShmemLimit()/4 - 128 }

// kv3PastBound is a prompt length a little past r's exact-attention bound, where the row-batched kernels serve.
func kv3PastBound(r *cudaResident) int { return ksb15Bound(r) + 144 }

// TestKSB15_limitArithmetic is B15-G1's struct-only half: with an opted-in limit the guard admits exactly the spans that fit it, a
// resident that set none keeps the 48 KiB bound, and decode's bound is the default one whatever the prefill's limit.
func TestKSB15_limitArithmetic(t *testing.T) {
	r := declineFixture(2, "int4")
	if got := r.prefillAttnShmemLimit(); got != singleBlockAttnShmemLimit {
		t.Fatalf("a resident that opted in to nothing reads a limit of %d B, want the default %d", got, singleBlockAttnShmemLimit)
	}
	r.prefillShmem = 64 << 10
	if b := ksb15Bound(r); b != 16256 {
		t.Fatalf("a 64 KiB limit holds %d keys, want 16,256", b)
	}
	if err := r.checkPrefillShmem(0, 16256, nil); err != nil {
		t.Errorf("16,256 keys under a 64 KiB limit must not decline: %v", err)
	}
	if err := r.checkPrefillShmem(0, 16257, nil); err == nil {
		t.Error("16,257 keys under a 64 KiB limit must decline")
	}
	if splitKVRequired(12160) || !splitKVRequired(12161) {
		t.Errorf("decode's bound moved: splitKVRequired(12160)=%v, (12161)=%v", splitKVRequired(12160), splitKVRequired(12161))
	}
}

// TestKSB15_optIn is B15-G1 on the card: each of the three exact prefill attention kernels takes more than 48 KiB, the prefill's
// limit is the smallest of them, a 16-row pass whose last row attends the bound runs on attn_batched and one attending a key more
// on the row kernels. Plant: a load with the opt-in skipped keeps 48 KiB, and a band span takes the row kernels there.
func TestKSB15_optIn(t *testing.T) {
	const base = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	g := kv0LoadCtx(t, dir, "f32", 17000, kv0RandomTokens)
	g.r.fastAttn, g.r.fastGemm = false, false
	lim := g.r.prefillAttnShmemLimit()
	if lim <= singleBlockAttnShmemLimit {
		t.Fatalf("the prefill limit is %d B: the opt-in did not take", lim)
	}
	least := 0
	for _, k := range []struct {
		name string
		p    Pipeline
	}{{"attn_batched", g.r.bAttn}, {"attn_img_batched", g.r.bAttnImg}, {"attn_span_batched", g.r.bAttnSpan}} {
		if k.p == (Pipeline{}) {
			t.Fatalf("%s did not load", k.name)
		}
		n := k.p.MaxDynamicSharedBytes(g.r.dev)
		fmt.Fprintf(os.Stderr, "[B15-G1] %s: %d B of dynamic shared memory\n", k.name, n)
		if n <= singleBlockAttnShmemLimit {
			t.Errorf("%s takes %d B, not past the default %d", k.name, n, singleBlockAttnShmemLimit)
		}
		if least == 0 || n < least {
			least = n
		}
	}
	if lim != least {
		t.Errorf("the prefill limit is %d B, the kernels' least %d", lim, least)
	}
	bound := ksb15Bound(g.r)
	fmt.Fprintf(os.Stderr, "[B15-G1] the prefill limit %d B holds %d keys (the default 48 KiB holds 12,160)\n", lim, bound)
	e := g.embs(g.tokens(bound+1, 11))
	lead := bound - 16
	g.prefill(e[:lead], 0)
	p0 := g.r.psLaunches
	g.prefill(e[lead:bound], lead) // the last row attends bound keys
	if n := g.r.psLaunches - p0; n != 0 {
		t.Errorf("a pass whose last row attends %d keys (the bound) ran %d row-batched groups, want attn_batched", bound, n)
	}
	p0 = g.r.psLaunches
	g.prefill(e[lead+1:bound+1], lead+1) // one key more
	if g.r.psLaunches == p0 {
		t.Errorf("a pass whose last row attends %d keys (one past the bound) ran no row-batched group", bound+1)
	}

	prefillShmemOptInOffForTest = true
	defer func() { prefillShmemOptInOffForTest = false }()
	h := kv0LoadCtx(t, dir, "f32", 17000, kv0RandomTokens)
	h.r.fastAttn, h.r.fastGemm = false, false
	if got := h.r.prefillAttnShmemLimit(); got != singleBlockAttnShmemLimit {
		t.Fatalf("plant: with the opt-in skipped the limit is %d B, want %d", got, singleBlockAttnShmemLimit)
	}
	f := h.embs(h.tokens(12400, 13))
	h.prefill(f[:12384], 0)
	p0 = h.r.psLaunches
	h.prefill(f[12384:], 12384)
	if h.r.psLaunches == p0 {
		t.Error("plant: a 12,400-key span with the opt-in skipped ran no row-batched group")
	}
}

// TestKSB15_bandBitsTiny is B15-G2's arm comparison: on the tiny fixture, fast levers off, f32 and f16 KV, a prompt in the band
// leaves the same logits and K/V under the opt-in (attn_batched, no row-batched group) as with the limit set back to 48 KiB on the
// same resident (the row kernels, today's route). Past the band both arms take the row kernels. Neither runs a tensor-core launch.
func TestKSB15_bandBitsTiny(t *testing.T) {
	const base = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	for _, kv := range []string{"f32", "f16"} {
		g := kv0LoadCtx(t, dir, kv, 17000, kv0RandomTokens)
		g.r.fastAttn, g.r.fastGemm = false, false
		opted := g.r.prefillShmem
		if opted == 0 {
			t.Fatal("the opt-in did not take")
		}
		for _, n := range []int{12400, 14000, 16000, 16384} {
			name := fmt.Sprintf("kv=%s n=%d", kv, n)
			e := g.embs(g.tokens(n, 7*n))
			arm := func(limit int) ([]float32, kv0KV, int, int) {
				g.r.prefillShmem = limit
				defer func() { g.r.prefillShmem = opted }()
				g.poison(n)
				p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
				lg := g.prefill(e, 0)
				return lg, g.snapshot(n), g.r.psLaunches - p0, g.r.fastGemmLaunches - m0
			}
			want, wantKV, psOld, mOld := arm(0)
			got, gotKV, psNew, mNew := arm(opted)
			kv3NoNaN(t, name+" today", want, wantKV)
			kv3NoNaN(t, name+" opt-in", got, gotKV)
			ld, worst := kv0LogitsDiff(want, got)
			kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
			band := g.r.prefillAttnFits(n)
			fmt.Fprintf(os.Stderr, "[B15-G2] %-16s in the band %v: row-batched groups %d today, %d opt-in; logits differing %d of %d (worst %.3g), KV differing %d of %d\n",
				name, band, psOld, psNew, ld, len(want), worst, kd, total)
			if psOld == 0 || (psNew == 0) != band || mOld != 0 || mNew != 0 {
				t.Errorf("%s: row-batched groups %d today (want some) and %d opt-in (want none: %v); tensor-core launches %d and %d (want 0)", name, psOld, psNew, band, mOld, mNew)
			}
			if ld != 0 || kd != 0 {
				t.Errorf("%s: the opt-in differs from today's route: %d logits, %d KV elements", name, ld, kd)
			}
		}
	}
}

// ksb15Gemma3 loads B15-G3/G4's model: Gemma-3-4B Q4_K_M, default levers, at a context that holds the 14,336-key pass. KV is f32;
// a load whose KV the card refuses at f32 is retried at f16 (the registered fallback), and the precision is printed.
func ksb15Gemma3(t *testing.T) *kv0Rig {
	t.Helper()
	path := modelPath("gemma-3-4b-it-Q4_K_M.gguf")
	requireDeviceAndFile(t, path)
	kv := "f32"
	probe, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ksb15Keys + 64, KVPrecision: kv})
	switch {
	case err != nil && strings.Contains(err.Error(), "GB of KV"):
		kv = "f16"
	case err != nil:
		t.Fatalf("fit check: %v", err)
	default:
		if _, ok := probe.ResidentForwardForTest().(*cudaResident); !ok {
			kv = "f16" // the default context declines rather than refuses
		}
		_ = probe.Close()
	}
	fmt.Fprintf(os.Stderr, "[B15] gemma-3-4b at %d positions: %s KV (fit check: %v)\n", ksb15Keys+64, kv, err)
	g := kv0LoadCtx(t, path, kv, ksb15Keys+64, kv0TextTokens(t, path))
	if g.r.prefillShmem == 0 {
		t.Fatal("the opt-in did not take")
	}
	if !g.r.prefillAttnFits(ksb15Keys) || !splitKVRequired(ksb15Keys) {
		t.Fatalf("%d keys is not in the band (bound %d)", ksb15Keys, ksb15Bound(g.r))
	}
	return g
}

// ksb15Lead and ksb15Keys are B15-G3/G4's pass: 512 rows after a 13,824-token lead, the last row attending 14,336 keys.
const ksb15Lead, ksb15Keys = 13824, 14336

// TestKSB15_bandBitsGemma3 is B15-G3: on Gemma-3-4B at default levers, the 512-row pass whose last row attends 14,336 keys leaves
// the same logits and K/V rows under the opt-in as at 48 KiB, the opt-in arm runs no row-batched group, and neither arm a
// tensor-core launch (a band prompt keeps the exact GEMM). KSB15_REAL=1.
func TestKSB15_bandBitsGemma3(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("KSB15_REAL") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and KSB15_REAL=1")
	}
	g := ksb15Gemma3(t)
	opted := g.r.prefillShmem
	e := g.embs(g.tokens(ksb15Keys, 77))
	t0 := time.Now()
	g.prefill(e[:ksb15Lead], 0)
	fmt.Fprintf(os.Stderr, "[B15-G3] gemma-3-4b: %d-token lead in %.1f s\n", ksb15Lead, time.Since(t0).Seconds())
	arm := func(limit int) ([]float32, kv0KV, int, int) {
		g.r.prefillShmem = limit
		defer func() { g.r.prefillShmem = opted }()
		p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
		lg := g.prefill(e[ksb15Lead:], ksb15Lead)
		return lg, g.snapshot(ksb15Keys), g.r.psLaunches - p0, g.r.fastGemmLaunches - m0
	}
	want, wantKV, psOld, mOld := arm(0)
	got, gotKV, psNew, mNew := arm(opted)
	kv3NoNaN(t, "today", want, wantKV)
	ld, worst := kv0LogitsDiff(want, got)
	kd, total := kv0CompareRows(wantKV, gotKV, ksb15Lead, ksb15Keys)
	fmt.Fprintf(os.Stderr, "[B15-G3] gemma-3-4b pass at %d keys: row-batched groups %d today, %d opt-in; tensor-core launches %d, %d; logits differing %d of %d (worst %.3g), KV differing %d of %d\n",
		ksb15Keys, psOld, psNew, mOld, mNew, ld, len(want), worst, kd, total)
	if psOld == 0 || psNew != 0 || mOld != 0 || mNew != 0 {
		t.Errorf("routes: row-batched groups %d today (want some), %d opt-in (want 0); tensor-core launches %d, %d (want 0)", psOld, psNew, mOld, mNew)
	}
	if ld != 0 || kd != 0 {
		t.Errorf("the opt-in differs from today's route: %d logits, %d KV elements", ld, kd)
	}
}

// ksb15Host is the 1-minute load average and MemAvailable (GB) at the call, for B15-G4's per-pair record.
func ksb15Host() (load float64, availGB float64) {
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		load, _ = strconv.ParseFloat(strings.Fields(string(b))[0], 64)
	}
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			if f := strings.Fields(ln); len(f) >= 2 && f[0] == "MemAvailable:" {
				kb, _ := strconv.ParseFloat(f[1], 64)
				availGB = kb / (1 << 20)
			}
		}
	}
	return load, availGB
}

// TestKSB15_speedGemma3 is B15-G4, a timed run under the timing lock (KSB15_TIMING=1): the B15-G3 pass, off (48 KiB) and on, A B B A
// over 10 pairs after 2 warm runs each; a pair started above load 8.0 is void and re-run. SHIP iff the median off/on is >= 1.00
// with at least 8 of 10 pairs faster on; KILL at a median < 0.98; otherwise parked. Then one whole 14,336-token PrefillLast, off
// then on, twice each, as a reading.
func TestKSB15_speedGemma3(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("KSB15_TIMING") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and KSB15_TIMING=1 (a timed run under the timing lock)")
	}
	g := ksb15Gemma3(t)
	opted := g.r.prefillShmem
	e := g.embs(g.tokens(ksb15Keys, 77))
	g.prefill(e[:ksb15Lead], 0)
	pass := func(on bool) time.Duration {
		g.r.prefillShmem = 0
		if on {
			g.r.prefillShmem = opted
		}
		defer func() { g.r.prefillShmem = opted }()
		t0 := time.Now()
		g.prefill(e[ksb15Lead:], ksb15Lead)
		return time.Since(t0)
	}
	for i := 0; i < 2; i++ {
		pass(false)
		pass(true)
	}
	var ratios []float64
	faster, voids := 0, 0
	for p := 0; len(ratios) < 10; p++ {
		if voids > 10 {
			t.Fatalf("%d void pairs: the box is too busy to time", voids)
		}
		load, avail := ksb15Host()
		var off, on time.Duration
		if p%2 == 0 {
			off, on = pass(false), pass(true)
		} else {
			on, off = pass(true), pass(false)
		}
		if load > 8.0 {
			voids++
			fmt.Fprintf(os.Stderr, "[B15-G4] pair %d VOID: load %.2f at its start (the line is 8.0), %.1f GB available\n", p, load, avail)
			continue
		}
		r := off.Seconds() / on.Seconds()
		ratios = append(ratios, r)
		if on < off {
			faster++
		}
		fmt.Fprintf(os.Stderr, "[B15-G4] pair %d: load %.2f, %.1f GB available; off %.1f ms, on %.1f ms, off/on %.4f\n",
			p, load, avail, float64(off.Microseconds())/1e3, float64(on.Microseconds())/1e3, r)
	}
	sort.Float64s(ratios)
	med := (ratios[4] + ratios[5]) / 2
	verdict := "PARKED (ambiguous)"
	switch {
	case med >= 1.00 && faster >= 8:
		verdict = "SHIP"
	case med < 0.98:
		verdict = "KILL"
	}
	fmt.Fprintf(os.Stderr, "[B15-G4] the %d-key pass: median off/on %.4f, %d of 10 pairs faster on, %d void: %s\n", ksb15Keys, med, faster, voids, verdict)
	for i := 0; i < 2; i++ {
		for _, on := range []bool{false, true} {
			g.r.prefillShmem = 0
			if on {
				g.r.prefillShmem = opted
			}
			t0 := time.Now()
			g.prefill(e, 0)
			fmt.Fprintf(os.Stderr, "[B15-G4] whole %d-token prompt, opt-in %v: %.2f s\n", ksb15Keys, on, time.Since(t0).Seconds())
		}
	}
	g.r.prefillShmem = opted
	if verdict == "KILL" {
		t.Errorf("B15-G4 KILL: median off/on %.4f", med)
	}
}
