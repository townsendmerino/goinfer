//go:build darwin && goinfer_testhooks

package metal

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR17KernelsCompile builds the prototype's pipelines — seconds, no checkpoint.
func TestR17KernelsCompile(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := d.CompileLibrary(r17Kernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	for _, k := range []string{"attention_fa_blk_g6", "attention_fa_blk_g7"} {
		if _, err := d.NewComputePipeline(lib, k); err != nil {
			t.Errorf("%s: %v", k, err)
		}
	}
}

// r17Install compiles the prototype for r's GQA group size and a no-op kernel, and grows r's attention_fa
// partial buffer when a split count above r.attnFAMaxSplit is wanted (maxSplit; 0 = no growth). It does not swap
// any pipeline in.
func r17Install(t *testing.T, r *resident, maxSplit int) (proto, noop Pipeline) {
	t.Helper()
	if r.attnFANKV == 0 || r.attnFAPartial == (Buffer{}) {
		t.Skip("attention_fa is not built for this model (head dim 128, dense GQA only)")
	}
	G := r.nH / r.attnFANKV
	if G != 6 && G != 7 {
		t.Skipf("prototype is instantiated for G=6 and G=7 only; this model has G=%d", G)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := r.d.CompileLibrary(r17Kernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prototype: %v", err)
	}
	if proto, err = r.d.NewComputePipeline(lib, fmt.Sprintf("attention_fa_blk_g%d", G)); err != nil {
		t.Fatalf("prototype pipeline: %v", err)
	}
	nlib, err := r.d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\nkernel void r17_noop() {}\n", MSL3_1)
	if err != nil {
		t.Fatalf("compile no-op: %v", err)
	}
	if noop, err = r.d.NewComputePipeline(nlib, "r17_noop"); err != nil {
		t.Fatalf("no-op pipeline: %v", err)
	}
	if maxSplit > r.attnFAMaxSplit {
		per := r.attnFAPartial.Len() / r.attnFAMaxSplit
		r.attnFAMaxSplit = maxSplit
		r.attnFAPartial = r.d.NewBufferLen(per * maxSplit)
	}
	return proto, noop
}

// TestR17_decodeFidelityGate is R17 precondition 1: R2's teacher-forced gate construction (runDecodeFidelityGate —
// S at K=3900, the S-K3900 CPU f32 reference, pooled §3.2 criteria) with the prototype as the candidate arm and the
// shipped exact attention kernel as the exact arm. GOINFER_METAL_R17_SPLIT fixes the prototype's split count
// (default 0 = production's attnFASplitFor rule).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run '^TestR17_decodeFidelityGate$' -v -timeout 60m
func TestR17_decodeFidelityGate(t *testing.T) {
	split := 0
	if v := os.Getenv("GOINFER_METAL_R17_SPLIT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatalf("GOINFER_METAL_R17_SPLIT: %v", err)
		}
		split = n
	}
	// GOINFER_METAL_R17_CAND picks the candidate arm: "proto" (default, the prototype as r.pAttnFA),
	// "attention_fa" (the shipped attention_fa at the given split — a reassociation-only perturbation of the shipped
	// kernel), or "exact-null" (the exact per-query-head kernel with each 4-term q·k partial sum reversed: a pure
	// rounding perturbation, identical in expected accuracy — the gate's purest null control).
	cand := os.Getenv("GOINFER_METAL_R17_CAND")
	if cand == "" {
		cand = "proto"
	}
	name := fmt.Sprintf("%s(S=%d)", cand, split)
	switch cand {
	case "exact-nudge":
		name = fmt.Sprintf("exact-nudge(k=%s)", os.Getenv("GOINFER_METAL_R17_NUDGE"))
	case "exact-vchunk":
		name = fmt.Sprintf("exact-vchunk(C=%s)", os.Getenv("GOINFER_METAL_R17_VCHUNK"))
	case "exact-vrev8":
		name = "exact-vrev8"
	}
	runDecodeFidelityGate(t, "r17-gate", name, func(t *testing.T, r *resident) {
		switch cand {
		case "proto":
			proto, _ := r17Install(t, r, split)
			r.pAttnFA = proto
			r.attnFASplitOverride = split
		case "attention_fa": // the PRODUCTION attention_fa path: since 2026-09-25 the block kernel on G = 6/7
			r.attnFASplitOverride = split
		case "attention_fa-legacy": // the R2 kernel, whatever production dispatches
			legacy, legacySplit := r17Legacy(t, r)
			r.pAttnFA = legacy
			r.attnFASplitOverride = legacySplit
			if split > 0 {
				r.attnFASplitOverride = split
			}
		case "exact-vchunk", "exact-vrev8":
			chunk := 0
			if cand == "exact-vchunk" {
				chunk, _ = strconv.Atoi(os.Getenv("GOINFER_METAL_R17_VCHUNK"))
				if chunk <= 0 {
					t.Fatalf("exact-vchunk needs GOINFER_METAL_R17_VCHUNK=<multiple of 8>")
				}
			}
			null := r17PatchKernel(t, r, "attention", "attention_vsum", r17VSumEdit(t, chunk))
			orig := r.pAttn
			r2GateArmHook = func(r *resident, c bool) {
				r.decodeAttnFA = false
				if c {
					r.pAttn = null
				} else {
					r.pAttn = orig
				}
			}
			t.Cleanup(func() { r2GateArmHook = nil })
		case "exact-null", "exact-nudge":
			var null Pipeline
			if cand == "exact-null" {
				null = r17ExactNull(t, r)
			} else {
				k, err := strconv.Atoi(os.Getenv("GOINFER_METAL_R17_NUDGE"))
				if err != nil || k == 0 {
					t.Fatalf("exact-nudge needs GOINFER_METAL_R17_NUDGE=<nonzero int k> (scale × (1 + k·2^-23))")
				}
				null = r17ExactNudge(t, r, k)
			}
			orig := r.pAttn
			r2GateArmHook = func(r *resident, c bool) {
				r.decodeAttnFA = false // both arms run the per-query-head kernel; only its pipeline differs
				if c {
					r.pAttn = null
				} else {
					r.pAttn = orig
				}
			}
			t.Cleanup(func() { r2GateArmHook = nil })
		default:
			t.Fatalf("GOINFER_METAL_R17_CAND=%q: want proto, attention_fa, attention_fa-legacy, exact-null, exact-nudge, exact-vchunk or exact-vrev8", cand)
		}
		fmt.Fprintf(os.Stderr, "[r17-gate] candidate %s installed (G=%d)\n", name, r.nH/r.attnFANKV)
	})
}

// r17ExactNudge builds "attention_nudge": the production `attention` kernel with its softmax scale multiplied by
// (1 + k·2^-23) — a relative change of k·1.19e-7, the same order as the kernels' own measured error vs float64
// (TestR17KernelAccuracy: ~2–6e-7 median). A family of same-magnitude, accuracy-neutral perturbations of the exact
// arm, one draw per k, for the gate's null distribution.
func r17ExactNudge(t *testing.T, r *resident, k int) Pipeline {
	t.Helper()
	const line = "sc[s - winStart]=a*scale;"
	return r17PatchedAttention(t, r, "attention_nudge", line,
		fmt.Sprintf("sc[s - winStart]=a*(scale*%.9ef);", 1+float64(k)*math.Exp2(-23)))
}

// r17PatchedAttention compiles allKernels plus a renamed copy of the production `attention` kernel with one exact
// text substitution (which must match exactly once inside that kernel).
func r17PatchedAttention(t *testing.T, r *resident, name, from, to string) Pipeline {
	t.Helper()
	i := strings.Index(allKernels, "kernel void attention(")
	if i < 0 {
		t.Fatalf("attention kernel not found in allKernels")
	}
	j := strings.Index(allKernels[i+1:], "\nkernel void ")
	if j < 0 {
		t.Fatalf("end of attention kernel not found")
	}
	src := allKernels[i : i+1+j]
	if strings.Count(src, from) != 1 {
		t.Fatalf("patch target %q found %d times in the attention kernel, want 1", from, strings.Count(src, from))
	}
	src = strings.Replace(strings.Replace(src, from, to, 1), "kernel void attention(", "kernel void "+name+"(", 1)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := r.d.CompileLibrary(allKernels+"\n"+src, MSL3_1)
	if err != nil {
		t.Fatalf("compile %s: %v", name, err)
	}
	p, err := r.d.NewComputePipeline(lib, name)
	if err != nil {
		t.Fatalf("%s pipeline: %v", name, err)
	}
	return p
}

// r17PatchKernel compiles allKernels plus a copy of production kernel `kernel` renamed to `name`, its text passed
// through edit (the copy runs from "kernel void <kernel>(" to its first column-0 closing brace).
func r17PatchKernel(t *testing.T, r *resident, kernel, name string, edit func(string) string) Pipeline {
	t.Helper()
	i := strings.Index(allKernels, "kernel void "+kernel+"(")
	if i < 0 {
		t.Fatalf("kernel %s not found in allKernels", kernel)
	}
	j := strings.Index(allKernels[i:], "\n}\n")
	if j < 0 {
		t.Fatalf("end of kernel %s not found", kernel)
	}
	src := allKernels[i : i+j+3]
	src = strings.Replace(edit(src), "kernel void "+kernel+"(", "kernel void "+name+"(", 1)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := r.d.CompileLibrary(allKernels+"\n"+src, MSL3_1)
	if err != nil {
		t.Fatalf("compile %s: %v", name, err)
	}
	p, err := r.d.NewComputePipeline(lib, name)
	if err != nil {
		t.Fatalf("%s pipeline: %v", name, err)
	}
	return p
}

// r17PreciseExp rewrites every exp( in a kernel's text to precise::exp( (Metal's fast-math exp is the default).
func r17PreciseExp(src string) string { return strings.ReplaceAll(src, "exp(", "precise::exp(") }

// r17Legacy returns the legacy attention_fa first pass (the R2 kernel; production dispatches the R17 block kernel
// instead for G = 6 and 7 since 2026-09-25) compiled under its own name, and the split count its core-count rule
// gives on r — so an arm that means "attention_fa" keeps meaning that kernel whatever r.pAttnFA is.
func r17Legacy(t *testing.T, r *resident) (Pipeline, int) {
	t.Helper()
	return r17PatchKernel(t, r, "attention_fa", "attention_fa_legacy", func(s string) string { return s }),
		(2*attnFACoreCount + r.attnFANKV - 1) / r.attnFANKV
}

// r17ProtoPrecise compiles the prototype with precise::exp in place of the fast-math exp.
func r17ProtoPrecise(t *testing.T, r *resident) Pipeline {
	t.Helper()
	src := strings.ReplaceAll(r17PreciseExp(r17Kernels), `host_name("attention_fa_blk_g`, `host_name("attention_fa_blkpx_g`)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := r.d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile precise prototype: %v", err)
	}
	p, err := r.d.NewComputePipeline(lib, fmt.Sprintf("attention_fa_blkpx_g%d", r.nH/r.attnFANKV))
	if err != nil {
		t.Fatalf("precise prototype pipeline: %v", err)
	}
	return p
}

// r17VSumEdit rewrites the exact kernel's single-tile V accumulation. chunk > 0: sum each chunk of `chunk` keys
// (a multiple of 8) into its own accumulator and add the chunk sums serially — a MORE accurate V sum, like the split
// kernels' (it removes most of the serial-f32 error). chunk == 0 ("vrev8"): keep the one serial accumulator but add each
// unrolled 8-term group in reverse — a rounding change of the same accuracy class as the shipped kernel.
func r17VSumEdit(t *testing.T, chunk int) func(string) string {
	return func(src string) string {
		// Patch the single-tile path only (nWin <= 4096 — every key count the gate reaches): the first occurrence,
		// which must lie before that path's `return;` (the multi-tile path after it repeats the same lines).
		must := func(s, from, to string) string {
			t.Helper()
			i, ret := strings.Index(s, from), strings.Index(s, "        return;\n    }")
			if i < 0 || ret < 0 || i > ret {
				t.Fatalf("V-sum patch target %q not found in the single-tile path (at %d, return at %d)", from, i, ret)
			}
			return s[:i] + to + s[i+len(from):]
		}
		const grp = "a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;\n" +
			"                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;"
		if chunk == 0 {
			return must(src, grp, "a += sc[sc_idx+7u]*v7; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+4u]*v4;\n"+
				"                a += sc[sc_idx+3u]*v3; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+0u]*v0;")
		}
		if chunk%8 != 0 {
			t.Fatalf("V chunk %d must be a multiple of 8", chunk)
		}
		src = must(src, "float a=0; uint s=winStart; uint nMain", "float a=0; float blk=0; uint s=winStart; uint nMain")
		src = must(src, grp, strings.ReplaceAll(grp, "a += ", "blk += ")+
			fmt.Sprintf("\n                if (((sc_idx + 8u) %% %du) == 0u) { a += blk; blk = 0.0f; }", chunk))
		src = must(src, "for (; s<nKeys; s++) a += sc[s - winStart]*float(vb[s*kvDim+d]);\n            out[qh*hd+d]=a/sum;",
			"for (; s<nKeys; s++) blk += sc[s - winStart]*float(vb[s*kvDim+d]);\n            a += blk;\n            out[qh*hd+d]=a/sum;")
		return src
	}
}

// r17ExactNull builds "attention_null": the production per-query-head `attention` kernel with each 4-term q·k
// partial sum accumulated in reverse order (d+3, d+2, d+1, d instead of d..d+3) — a rounding-only change with the
// same expected accuracy, compiled alongside allKernels.
func r17ExactNull(t *testing.T, r *resident) Pipeline {
	t.Helper()
	i := strings.Index(allKernels, "kernel void attention(")
	if i < 0 {
		t.Fatalf("attention kernel not found in allKernels")
	}
	j := strings.Index(allKernels[i+1:], "\nkernel void ")
	if j < 0 {
		t.Fatalf("end of attention kernel not found")
	}
	src := allKernels[i : i+1+j]
	const fwd = "a+=qr[d]*float(k4.x); a+=qr[d+1u]*float(k4.y); a+=qr[d+2u]*float(k4.z); a+=qr[d+3u]*float(k4.w);"
	const rev = "a+=qr[d+3u]*float(k4.w); a+=qr[d+2u]*float(k4.z); a+=qr[d+1u]*float(k4.y); a+=qr[d]*float(k4.x);"
	if strings.Count(src, fwd) != 1 {
		t.Fatalf("attention kernel's half4 q·k line not found exactly once (%d)", strings.Count(src, fwd))
	}
	src = strings.Replace(strings.Replace(src, fwd, rev, 1), "kernel void attention(", "kernel void attention_null(", 1)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := r.d.CompileLibrary(allKernels+"\n"+src, MSL3_1)
	if err != nil {
		t.Fatalf("compile attention_null: %v", err)
	}
	p, err := r.d.NewComputePipeline(lib, "attention_null")
	if err != nil {
		t.Fatalf("attention_null pipeline: %v", err)
	}
	return p
}

// TestR17AttentionProto measures R17's metric for the prototype: in-sequence attention work at a depth (default
// 3900) on the production decode token — the same no-op method as TestMetalDecodeDecomp — for the current kernel
// (attention_fa) and the prototype, at one or more split counts, arms interleaved rep by rep. It also reports how
// far the prototype's logits move from attention_fa's on the same decode step (a sanity check; the grading gate is
// R2's teacher-forced construction) and the prototype's attention time right after 2 s of idle.
//
//	GOINFER_METAL_R17=1 go test -tags goinfer_testhooks -run '^TestR17AttentionProto$' -v ./metal/
//
// Env: GOINFER_METAL_R17_MODEL, GOINFER_METAL_R17_DEPTHS (default 3900), GOINFER_METAL_R17_SPLITS (split counts for
// the prototype, default "0,32"; 0 = production's rule), GOINFER_METAL_R17_REPS (5), GOINFER_METAL_R17_TOKENS (20).
func TestR17AttentionProto(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R17") != "1" {
		t.Skip("set GOINFER_METAL_R17=1 (loads a real checkpoint; minutes of GPU time)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_R17_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	ints := func(env string, def []int) []int {
		v := os.Getenv(env)
		if v == "" {
			return def
		}
		var out []int
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				t.Fatalf("%s: %v", env, err)
			}
			out = append(out, n)
		}
		return out
	}
	depths := ints("GOINFER_METAL_R17_DEPTHS", []int{3900})
	splits := ints("GOINFER_METAL_R17_SPLITS", []int{0, 32})
	reps := ints("GOINFER_METAL_R17_REPS", []int{5})[0]
	steps := ints("GOINFER_METAL_R17_TOKENS", []int{20})[0]
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[r17 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}

	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: metalCtxCapDefault})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	defer r.Close()
	if !r.decodeAttnFA || r.attnFAPartial == (Buffer{}) || r.attnFANKV == 0 {
		t.Skip("attention_fa is not active for this model (head dim 128, dense GQA only)")
	}
	G := r.nH / r.attnFANKV
	if G != 6 && G != 7 {
		t.Skipf("prototype is instantiated for G=6 and G=7 only; this model has G=%d", G)
	}

	maxS := 0
	for _, sv := range splits {
		maxS = max(maxS, sv)
	}
	proto, noop := r17Install(t, r, maxS)
	current := r.pAttnFA
	hb("loaded %s: nH=%d nKV=%d G=%d; prototype attention_fa_blk_g%d; prototype split counts %v", filepath.Base(path), r.nH, r.attnFANKV, G, G, splits)

	vocab := r.V
	for _, D := range depths {
		if D+1 < attnFADepthFloor {
			t.Fatalf("depth %d is below attnFADepthFloor %d — attention_fa (and so the prototype) does not engage", D, attnFADepthFloor)
		}
		embs := make([][]float32, D)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
		}
		r.PrefillLast(embs, 0)
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("prefill: %v", err)
		}
		tok := (D*131 + 7) % vocab
		step := func() float64 {
			r.Forward(tok, D)
			if err := r.takeExecErr(); err != nil {
				t.Fatalf("decode: %v", err)
			}
			return (r.gpuEnd - r.gpuStart) * 1e3
		}
		type armSpec struct {
			name  string
			first Pipeline // what runs as r.pAttnFA
			split int      // attnFASplitOverride (0 = production rule)
			noAtt bool     // all attention pipelines no-op'd
		}
		arms := []armSpec{{name: "current (production)", first: current}}
		for _, sv := range splits {
			arms = append(arms, armSpec{name: fmt.Sprintf("prototype S=%d", sv), first: proto, split: sv})
		}
		arms = append(arms, armSpec{name: "no-op attention", first: noop, noAtt: true})
		setArm := func(a armSpec) (restore func()) {
			sa, sb, sc := r.pAttnFA, r.pAttnFACombine, r.pAttn
			r.pAttnFA = a.first
			r.attnFASplitOverride = a.split
			if a.noAtt {
				r.pAttnFACombine, r.pAttn = noop, noop
			}
			r.stopExec() // drop the command buffer pre-encoded under the previous arm (see runR2GateCell)
			return func() {
				r.pAttnFA, r.pAttnFACombine, r.pAttn = sa, sb, sc
				r.attnFASplitOverride = 0
				r.stopExec()
			}
		}
		runArm := func(a armSpec) float64 {
			restore := setArm(a)
			defer restore()
			xs := make([]float64, steps)
			for i := range xs {
				xs[i] = step()
			}
			return median(xs)
		}
		// warm, and the logits of each arm on the same step
		logitsOf := func(a armSpec) []float32 {
			restore := setArm(a)
			defer restore()
			for i := 0; i < 3; i++ {
				step()
			}
			return append([]float32(nil), r.logitsHost...)
		}
		// the exact kernel (r.decodeAttnFA off → the shipped per-query-head attention) as a third reference, and the
		// current kernel twice, so a disagreement can be placed on the arm that moved (or on nondeterminism)
		exactOf := func() []float32 {
			r.decodeAttnFA = false
			defer func() { r.decodeAttnFA = true }()
			return logitsOf(armSpec{first: current})
		}
		cosArg := func(got, ref []float32) (float64, int, int) {
			var dot, na, nb2 float64
			am, bm := 0, 0
			for i := range got {
				dot += float64(got[i]) * float64(ref[i])
				na += float64(got[i]) * float64(got[i])
				nb2 += float64(ref[i]) * float64(ref[i])
				if got[i] > got[am] {
					am = i
				}
				if ref[i] > ref[bm] {
					bm = i
				}
			}
			return dot / math.Sqrt(na*nb2), am, bm
		}
		exact := exactOf()
		ref := logitsOf(arms[0])
		fid := []struct {
			name string
			lg   []float32
		}{{"exact kernel (again)", exactOf()}, {arms[0].name, ref}, {arms[0].name + " (again)", logitsOf(arms[0])}}
		fidArms := append([]armSpec(nil), arms[1:len(arms)-1]...)
		for _, sv := range splits { // the control: the CURRENT kernel at the same split counts
			if sv != 0 {
				fidArms = append(fidArms, armSpec{name: fmt.Sprintf("production S=%d", sv), first: current, split: sv})
			}
		}
		for _, a := range fidArms {
			fid = append(fid, struct {
				name string
				lg   []float32
			}{a.name, logitsOf(a)})
		}
		maxDiff := func(a, b []float32) float64 {
			var d float64
			for i := range a {
				d = max(d, math.Abs(float64(a[i])-float64(b[i])))
			}
			return d
		}
		for _, f := range fid {
			ce, am, em := cosArg(f.lg, exact)
			cf, _, fm := cosArg(f.lg, ref)
			hb("depth %d %-26s logits vs exact kernel: cosine %.9f max|diff| %.4g (argmax %d vs %d); vs attention_fa: cosine %.9f max|diff| %.4g (argmax %d)",
				D, f.name, ce, maxDiff(f.lg, exact), am, em, cf, maxDiff(f.lg, ref), fm)
		}

		res := make([][]float64, len(arms))
		idleRes := make([][]float64, len(arms)) // one token right after 2 s idle, per arm per rep (R17 precondition 3)
		for rep := 0; rep < reps; rep++ {
			for ai, a := range arms {
				res[ai] = append(res[ai], runArm(a))
			}
			for ai, a := range arms {
				restore := setArm(a)
				step() // this arm's pipelines warm, then idle
				time.Sleep(2 * time.Second)
				idleRes[ai] = append(idleRes[ai], step())
				restore()
			}
			hb("depth %d rep %d/%d: %s", D, rep+1, reps, func() string {
				var b strings.Builder
				for ai, a := range arms {
					fmt.Fprintf(&b, "%s %.3f (idle %.3f)  ", a.name, res[ai][rep], idleRes[ai][rep])
				}
				return b.String()
			}())
		}

		noopAi := len(arms) - 1
		workOf := func(rs [][]float64, ai int) []float64 {
			var d []float64
			for i := range rs[ai] {
				d = append(d, rs[ai][i]-rs[noopAi][i])
			}
			return d
		}
		work := func(ai int) []float64 { return workOf(res, ai) }
		noopMs := res[noopAi]
		cur := work(0)
		curIdle := workOf(idleRes, 0)
		fmt.Fprintf(os.Stderr, "\n=== R17 step 2, depth %d: in-sequence attention work, current vs prototype (medians of %d reps × %d tokens) ===\n", D, reps, steps)
		fmt.Fprintf(os.Stderr, "  %-26s attention %7.3f ms  full token %7.3f ms  reps %s | after 2 s idle: attention %7.3f full %7.3f\n",
			arms[0].name, median(cur), median(res[0]), fmtMs(cur), median(curIdle), median(idleRes[0]))
		for ai := 1; ai < len(arms)-1; ai++ {
			w := work(ai)
			wi := workOf(idleRes, ai)
			var idleRatios []float64
			for i := range wi {
				idleRatios = append(idleRatios, curIdle[i]/wi[i])
			}
			var ratios []float64
			for i := range w {
				ratios = append(ratios, cur[i]/w[i])
			}
			rm := median(ratios)
			band := "KILL (< 1.5x)"
			switch {
			case rm >= 2.5:
				band = "SHIP band (>= 2.5x) — exploratory; a confirmation run grades"
			case rm >= 1.5:
				band = "PARK band (1.5-2.5x)"
			}
			fmt.Fprintf(os.Stderr, "  %-26s attention %7.3f ms  full token %7.3f ms  speedup %.2fx (reps %s) → %s | after 2 s idle: attention %7.3f full %7.3f speedup %.2fx (reps %s)\n",
				arms[ai].name, median(w), median(res[ai]), rm, fmtMs(ratios), band, median(wi), median(idleRes[ai]), median(idleRatios), fmtMs(idleRatios))
		}
		fmt.Fprintf(os.Stderr, "  %-26s full token %7.3f ms, after 2 s idle %7.3f (the attention-free baseline)\n\n", arms[noopAi].name, median(noopMs), median(idleRes[noopAi]))
	}
}

// TestR17StateLeak asks whether running a candidate attention kernel leaves state behind that changes a LATER run
// of the exact kernel: exact → candidate → exact on the same K-token input (prefill + a fixed teacher-forced
// continuation), the two exact runs compared bit for bit, for the prototype and — as the control — attention_fa.
// Built the way the gate builds its resident (decoder.Load, Backend metal, ResidentContext K+72).
//
//	GOINFER_METAL_R17=1 go test -tags goinfer_testhooks -run '^TestR17StateLeak$' -v ./metal/
func TestR17StateLeak(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R17") != "1" {
		t.Skip("set GOINFER_METAL_R17=1 (loads a real checkpoint)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	const K, contN = 3900, 16
	split := 16
	if v := os.Getenv("GOINFER_METAL_R17_SPLIT"); v != "" {
		split, _ = strconv.Atoi(v)
	}
	t.Setenv("GOINFER_METAL_ATTN_FA", "0")
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: K + contN + 8})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built")
	}
	r := rf.r
	proto, _ := r17Install(t, r, split)
	current := r.pAttnFA
	vocab := r.V
	embs := make([][]float32, K)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
	}
	ctx := context.Background()
	// flush: stop the pipelined executor after every toggle, dropping the command buffer it pre-encoded under the
	// previous state (execLoop encodes token t+1 right after committing t; ensureExec restarts it on the next job)
	flush := os.Getenv("GOINFER_METAL_R17_FLUSH") == "1"
	run := func(fa bool) [][]float32 {
		r.decodeAttnFA = false
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if flush {
			r.decodeAttnFA = fa
			r.stopExec()
		}
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		out := [][]float32{cloneF32(seed)}
		r.decodeAttnFA = fa
		defer func() { r.decodeAttnFA = false }()
		for i := 1; i < contN; i++ {
			lg, err := rf.Forward(m.EmbedResidentForTest((i*977+3)%vocab), K-1+i)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			out = append(out, cloneF32(lg))
		}
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("exec: %v", err)
		}
		return out
	}
	diffRows := func(a, b [][]float32) (rows, elems int) {
		for i := range a {
			n := 0
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					n++
				}
			}
			if n > 0 {
				rows++
				elems += n
			}
		}
		return
	}
	for _, c := range []struct {
		name  string
		p     Pipeline
		split int
		fa    bool
	}{{"exact (no candidate — the null control)", current, 0, false}, {"production attention_fa path (control)", current, 0, true},
		{fmt.Sprintf("prototype S=%d", split), proto, split, true}, {"exact (null control, again)", current, 0, false}} {
		a := run(false)
		r.pAttnFA, r.attnFASplitOverride = c.p, c.split
		run(c.fa)
		r.pAttnFA, r.attnFASplitOverride = current, 0
		b := run(false)
		b2 := run(false) // stale-KV over-read predicts b2 == a (b rewrote the continuation slots); a persistent change predicts b2 == b
		rows, elems := diffRows(a, b)
		r2a, _ := diffRows(a, b2)
		r2b, _ := diffRows(b, b2)
		fmt.Fprintf(os.Stderr, "[r17-leak flush=%v] exact → %s → exact: %d of %d logit rows differ (%d elements); a second exact run after it differs from the first exact run in %d rows, from the one before it in %d\n",
			flush, c.name, rows, len(a), elems, r2a, r2b)
		if rows > 0 {
			t.Errorf("%s changed a later exact-kernel run: %d rows differ", c.name, rows)
		}
	}
}

// TestR17StateLeakBisect localises what TestR17StateLeak found: after an attention_fa-path run, which piece of
// persistent state, restored to its value after the preceding exact run, makes the next exact run bit-identical
// again — the KV cache (every layer, every slot), or r.ctx plus the attention_fa partial buffer.
//
//	GOINFER_METAL_R17=1 go test -tags goinfer_testhooks -run '^TestR17StateLeakBisect$' -v ./metal/
func TestR17StateLeakBisect(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R17") != "1" {
		t.Skip("set GOINFER_METAL_R17=1 (loads a real checkpoint)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	const K, contN = 3900, 16
	t.Setenv("GOINFER_METAL_ATTN_FA", "0")
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: K + contN + 8})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built")
	}
	r := rf.r
	vocab := r.V
	embs := make([][]float32, K)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
	}
	ctx := context.Background()
	run := func(fa bool) [][]float32 {
		r.decodeAttnFA = false
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		out := [][]float32{cloneF32(seed)}
		r.decodeAttnFA = fa
		defer func() { r.decodeAttnFA = false }()
		for i := 1; i < contN; i++ {
			lg, err := rf.Forward(m.EmbedResidentForTest((i*977+3)%vocab), K-1+i)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			out = append(out, cloneF32(lg))
		}
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("exec: %v", err)
		}
		return out
	}
	rowsDiffer := func(a, b [][]float32) int {
		n := 0
		for i := range a {
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					n++
					break
				}
			}
		}
		return n
	}
	type snap struct{ kv [][]uint16 }
	halves := func(b Buffer) []uint16 { return b.U16s()[:b.Len()/2] } // a byte buffer's Len is bytes
	take := func() snap {
		var s snap
		for l := range r.kc {
			s.kv = append(s.kv, append([]uint16(nil), halves(r.kc[l])...), append([]uint16(nil), halves(r.vc[l])...))
		}
		return s
	}
	restore := func(s snap) {
		for l := range r.kc {
			copy(halves(r.kc[l]), s.kv[2*l])
			copy(halves(r.vc[l]), s.kv[2*l+1])
		}
	}
	kvDim := r.layers[0].geom.kvDim
	slotsDiffer := func(a, b snap) (lo, hi, n int) {
		lo, hi = -1, -1
		for l := range a.kv {
			for i := range a.kv[l] {
				if a.kv[l][i] != b.kv[l][i] {
					s := i / kvDim
					if lo < 0 || s < lo {
						lo = s
					}
					hi = max(hi, s)
					n++
				}
			}
		}
		return
	}
	for _, mode := range []string{"no restore", "restore KV cache", "restore r.ctx + partial"} {
		a := run(false)
		sa := take()
		ctxA := append([]float32(nil), r.ctx.Floats()...)
		partA := append([]float32(nil), r.attnFAPartial.Floats()...)
		run(true) // shipped attention_fa
		sf := take()
		lo, hi, n := slotsDiffer(sa, sf)
		switch mode {
		case "restore KV cache":
			restore(sa)
		case "restore r.ctx + partial":
			copy(r.ctx.Floats(), ctxA)
			copy(r.attnFAPartial.Floats(), partA)
		}
		b := run(false)
		fmt.Fprintf(os.Stderr, "[r17-bisect] %-24s: KV after the attention_fa run differs from after the exact run in %d halves, slots %d..%d (K=%d, continuation K..K+%d); next exact run differs from the first in %d of %d rows\n",
			mode, n, lo, hi, K, contN-2, rowsDiffer(a, b), len(a))
		run(false) // back to the exact-written state for the next mode
	}
}

// r17BufferBytes reads a Metal buffer's byte capacity from Download's overrun error (the capacity accessor is
// unexported in aikit) — test-only.
func r17BufferBytes(b Buffer) int {
	err := gpu.Download(b.At(1<<40), make([]byte, 1))
	if err == nil {
		return 0
	}
	mm := regexp.MustCompile(`overruns a (\d+)-byte buffer`).FindStringSubmatch(err.Error())
	if mm == nil {
		return 0
	}
	n, _ := strconv.Atoi(mm[1])
	return n
}

// r17Buffers walks r (struct fields, slices, arrays, pointers to structs — unexported ones included) and returns
// every distinct Buffer it can reach, by field path.
func r17Buffers(r *resident) map[string]Buffer {
	out := map[string]Buffer{}
	seenPtr := map[uintptr]bool{}
	bufT := reflect.TypeOf(Buffer{})
	var walk func(v reflect.Value, path string, depth int)
	walk = func(v reflect.Value, path string, depth int) {
		if depth > 6 || !v.IsValid() {
			return
		}
		if v.CanAddr() {
			v = reflect.NewAt(v.Type(), unsafe.Pointer(v.UnsafeAddr())).Elem()
		}
		if v.Type() == bufT {
			b := v.Interface().(Buffer)
			if b != (Buffer{}) {
				out[path] = b
			}
			return
		}
		switch v.Kind() {
		case reflect.Ptr:
			if v.IsNil() || v.Elem().Kind() != reflect.Struct || seenPtr[v.Pointer()] {
				return
			}
			seenPtr[v.Pointer()] = true
			walk(v.Elem(), path, depth+1)
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				walk(v.Field(i), path+"."+v.Type().Field(i).Name, depth+1)
			}
		case reflect.Slice, reflect.Array:
			if v.Len() > 0 && !r17MayHoldBuffer(v.Type().Elem(), 0) {
				return
			}
			for i := 0; i < v.Len(); i++ {
				walk(v.Index(i), fmt.Sprintf("%s[%d]", path, i), depth+1)
			}
		}
	}
	walk(reflect.ValueOf(r), "r", 0)
	return out
}

func r17MayHoldBuffer(t reflect.Type, depth int) bool {
	if depth > 4 {
		return false
	}
	if t == reflect.TypeOf(Buffer{}) {
		return true
	}
	switch t.Kind() {
	case reflect.Ptr, reflect.Slice, reflect.Array:
		return r17MayHoldBuffer(t.Elem(), depth+1)
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if r17MayHoldBuffer(t.Field(i).Type, depth+1) {
				return true
			}
		}
	}
	return false
}

// TestR17StateLeakAllBuffers: TestR17StateLeakBisect's restore, widened to every buffer the resident holds.
// After exact run a, every reachable buffer is snapshotted; after the attention_fa run, the ones whose bytes
// changed are listed, and restored in halves (bisection over the changed set) until the next exact run matching
// a pins the buffer(s) it depends on.
//
//	GOINFER_METAL_R17=1 go test -tags goinfer_testhooks -run '^TestR17StateLeakAllBuffers$' -v ./metal/
func TestR17StateLeakAllBuffers(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R17") != "1" {
		t.Skip("set GOINFER_METAL_R17=1 (loads a real checkpoint)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	const K, contN = 3900, 16
	t.Setenv("GOINFER_METAL_ATTN_FA", "0")
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: K + contN + 8})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built")
	}
	r := rf.r
	vocab := r.V
	embs := make([][]float32, K)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
	}
	ctx := context.Background()
	run := func(fa bool) [][]float32 {
		r.decodeAttnFA = false
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		out := [][]float32{cloneF32(seed)}
		r.decodeAttnFA = fa
		defer func() { r.decodeAttnFA = false }()
		for i := 1; i < contN; i++ {
			lg, err := rf.Forward(m.EmbedResidentForTest((i*977+3)%vocab), K-1+i)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			out = append(out, cloneF32(lg))
		}
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("exec: %v", err)
		}
		return out
	}
	same := func(a, b [][]float32) bool {
		for i := range a {
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					return false
				}
			}
		}
		return true
	}
	bufs := r17Buffers(r)
	type ent struct {
		b Buffer
		n int
	}
	ids := map[Buffer]string{}
	var names []string
	ents := map[string]ent{}
	total := 0
	for name, b := range bufs {
		b0 := b.At(0)
		if _, dup := ids[b0]; dup {
			continue
		}
		ids[b0] = name
		n := r17BufferBytes(b0)
		if n == 0 || n > 512<<20 {
			continue
		}
		ents[name] = ent{b0, n}
		names = append(names, name)
		total += n
	}
	sort.Strings(names)
	fmt.Fprintf(os.Stderr, "[r17-all] %d distinct buffers reachable, %d snapshotted (%.1f MB)\n", len(ids), len(names), float64(total)/1e6)
	snapshot := func() map[string][]byte {
		sn := map[string][]byte{}
		for _, nm := range names {
			e := ents[nm]
			buf := make([]byte, e.n)
			if err := gpu.Download(e.b, buf); err != nil {
				t.Fatalf("download %s: %v", nm, err)
			}
			sn[nm] = buf
		}
		return sn
	}
	a := run(false)
	sa := snapshot()
	run(true)
	sf := snapshot()
	var changed []string
	for _, nm := range names {
		if !bytes.Equal(sa[nm], sf[nm]) {
			changed = append(changed, nm)
		}
	}
	fmt.Fprintf(os.Stderr, "[r17-all] buffers changed by the attention_fa run: %d: %v\n", len(changed), changed)
	restore := func(set []string) {
		for _, nm := range set {
			if err := gpu.Upload(ents[nm].b, sa[nm]); err != nil {
				t.Fatalf("upload %s: %v", nm, err)
			}
		}
	}
	// does restoring ALL changed buffers make the next exact run match a? (then bisect)
	try := func(set []string) bool {
		run(true) // re-create the post-attention_fa state
		restore(set)
		ok := same(a, run(false))
		fmt.Fprintf(os.Stderr, "[r17-all] restore %d buffer(s) %v → next exact run matches the first: %v\n", len(set), set, ok)
		return ok
	}
	if !try(changed) {
		fmt.Fprintf(os.Stderr, "[r17-all] restoring every changed buffer does NOT remove the difference — the state is outside these buffers\n")
		return
	}
	set := changed
	for len(set) > 1 {
		h := set[:len(set)/2]
		if try(h) {
			set = h
		} else if try(set[len(set)/2:]) {
			set = set[len(set)/2:]
		} else {
			fmt.Fprintf(os.Stderr, "[r17-all] neither half alone suffices — the dependency spans both halves of %v\n", set)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "[r17-all] PINNED: the next exact run depends on %v\n", set)
}

// TestR17KernelAccuracy measures each attention kernel's own error against a float64 reference on REAL inputs,
// independent of the end-to-end gate's amplification through the W4A8 trunk: a real prompt is prefilled at K, then one
// decode step at pos K is encoded LAYER BY LAYER (exact kernel in the trunk); after each layer its post-RoPE q (r.qkv)
// and the cache (r.kc/r.vc, f16) are the inputs, and every kernel under test is dispatched standalone on exactly those
// buffers into its own output. The reference is attention computed on the host in float64 from the same f32 q and f16
// K/V, so KV quantisation (shared by every kernel) is excluded and only each kernel's arithmetic is measured.
// Sanity: the standalone exact dispatch must be bit-identical to what the layer itself wrote into r.ctx.
//
//	GOINFER_METAL_R17=1 go test -tags goinfer_testhooks -run '^TestR17KernelAccuracy$' -v ./metal/
//
// Env: GOINFER_METAL_R17_MODEL (default the 1.5B), GOINFER_METAL_R17_PROMPTS (default 3, of the gate's set "a").
func TestR17KernelAccuracy(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R17") != "1" {
		t.Skip("set GOINFER_METAL_R17=1 (loads a real checkpoint)")
	}
	path := os.Getenv("GOINFER_METAL_R17_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	nPrompts := 3
	if v := os.Getenv("GOINFER_METAL_R17_PROMPTS"); v != "" {
		nPrompts, _ = strconv.Atoi(v)
	}
	depths := []int{3900}
	if v := os.Getenv("GOINFER_METAL_R17_DEPTHS"); v != "" {
		depths = depths[:0]
		for _, f := range strings.Split(v, ",") {
			d, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil || d < 2 {
				t.Fatalf("GOINFER_METAL_R17_DEPTHS: bad depth %q", f)
			}
			depths = append(depths, d)
		}
	}
	maxD := 0
	for _, d := range depths {
		maxD = max(maxD, d)
	}
	t.Setenv("GOINFER_METAL_ATTN_FA", "0")
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: maxD + 72})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built")
	}
	r := rf.r
	proto, _ := r17Install(t, r, 32)
	current := r.pAttnFA
	tkPath := path
	if strings.HasSuffix(path, ".giw") {
		tkPath = strings.TrimSuffix(path, ".int4.metal.giw") + ".gguf"
	}
	tk, err := tokenizer.LoadGGUF(tkPath)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	setLabel, files := decoder.PrefillGatePromptSet()
	if nPrompts > len(files) {
		nPrompts = len(files)
	}
	type arm struct {
		name  string
		fa    bool
		p     Pipeline // first pass (fa) or the per-query-head kernel (!fa; zero = r.pAttn)
		comb  Pipeline // fa only; zero = r.pAttnFACombine
		split int
	}
	exactPx := r17PatchKernel(t, r, "attention", "attention_px", r17PreciseExp)
	faPx := r17PatchKernel(t, r, "attention_fa", "attention_fa_px", r17PreciseExp)
	combPx := r17PatchKernel(t, r, "attention_fa_combine", "attention_fa_combine_px", r17PreciseExp)
	protoPx := r17ProtoPrecise(t, r)
	// "attention_fa" arms are the LEGACY kernel at its own split rule (r17Legacy); "production" is whatever
	// r.pAttnFA is at the production split (since 2026-09-25 the block kernel on G = 6/7 — identical to
	// "prototype S=16" there).
	legacy, legacySplit := r17Legacy(t, r)
	arms := []arm{
		{name: "exact (shipped attention)"},
		{name: "attention_fa S=prod", fa: true, p: legacy, split: legacySplit},
		{name: "attention_fa S=16", fa: true, p: legacy, split: 16},
		{name: "attention_fa S=32", fa: true, p: legacy, split: 32},
		{name: "production (r.pAttnFA)", fa: true, p: current},
		{name: "prototype S=prod", fa: true, p: proto},
		{name: "prototype S=16", fa: true, p: proto, split: 16},
		{name: "prototype S=32", fa: true, p: proto, split: 32},
		{name: "exact, precise exp", p: exactPx},
		{name: "attention_fa S=prod, precise", fa: true, p: faPx, comb: combPx, split: legacySplit},
		{name: "prototype S=16, precise exp", fa: true, p: protoPx, comb: combPx, split: 16},
		{name: "exact-null (reversed q·k)", p: r17ExactNull(t, r)},
		{name: "exact-nudge k=+1", p: r17ExactNudge(t, r, 1)},
		{name: "exact-nudge k=-3", p: r17ExactNudge(t, r, -3)},
		{name: "exact-nudge k=+16", p: r17ExactNudge(t, r, 16)},
		{name: "exact-nudge k=+64", p: r17ExactNudge(t, r, 64)},
		{name: "exact-vrev8 (same class)", p: r17PatchKernel(t, r, "attention", "attention_vrev8", r17VSumEdit(t, 0))},
		{name: "exact-vchunk C=8", p: r17PatchKernel(t, r, "attention", "attention_vc8", r17VSumEdit(t, 8))},
		{name: "exact-vchunk C=32", p: r17PatchKernel(t, r, "attention", "attention_vc32", r17VSumEdit(t, 32))},
		{name: "exact-vchunk C=64", p: r17PatchKernel(t, r, "attention", "attention_vc64", r17VSumEdit(t, 64))},
		{name: "exact-vchunk C=256", p: r17PatchKernel(t, r, "attention", "attention_vc256", r17VSumEdit(t, 256))},
	} // GOINFER_METAL_R17_ACC_ARMS=decision keeps only the arms the pre-registered decision grades
	// (docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md): the exact kernel and the two candidates.
	if os.Getenv("GOINFER_METAL_R17_ACC_ARMS") == "decision" {
		keep := map[string]bool{"exact (shipped attention)": true, "attention_fa S=prod": true, "prototype S=16": true}
		var kept []arm
		for _, a := range arms {
			if keep[a.name] {
				kept = append(kept, a)
			}
		}
		arms = kept
	}
	if arms[0].name != "exact (shipped attention)" {
		t.Fatalf("arm 0 must be the exact kernel (every comparison is against it)")
	}

	nHhdMax := r.nH * 256
	outBuf := r.d.NewBufferLen(nHhdMax)
	rel := make([][]float64, len(arms))    // per (prompt, layer, head): ||out-ref|| / ||ref||
	relL := make([][][]float64, len(arms)) // the same, by layer
	for ai := range relL {
		relL[ai] = make([][]float64, r.nL)
	}
	// Shared-bias check: the gate's CPU reference computes attention as decoder/attention.go attendQuery does (f64
	// scores and softmax, then a SERIAL f32 V accumulation, keys in order — the exact kernel's own V order). Emulate
	// it on the same inputs, fused (arm64 gc fuses x*y+z) and unfused (amd64), and ask, per head, whether each kernel's
	// error vector vs float64 points the SAME WAY as the emulated reference's (cosine), and which kernel sits closer to
	// the emulated reference than the exact kernel does.
	vsExact := make([][]float64, len(arms)) // ||out_k - out_exact|| / ||ref|| per head: how far each arm moves the output
	var exactOut []float32
	cosF := make([][]float64, len(arms)) // cos(out_k - ref64, cpuFused - ref64) per head
	cosU := make([][]float64, len(arms))
	nearF := make([]int, len(arms)) // heads where arm is strictly closer to cpuFused than the exact kernel is
	var cpuRelF, cpuRelU []float64
	worst := make([]string, len(arms)) // where each arm's largest relative error sits
	worstV := make([]float64, len(arms))
	maxAbs := make([]float64, len(arms))
	win := make([]int, len(arms)) // heads where this arm is strictly closer to the reference than the exact kernel
	tie := make([]int, len(arms))
	sanityOK, sanityN := 0, 0
	ctx := context.Background()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	scale := float64(r.uScale.Floats()[0])
	for pi := 0; pi < nPrompts; pi++ {
		for _, K := range depths {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, files[pi], K)[:K]
			embs := make([][]float32, K)
			for i, id := range ids {
				embs[i] = m.EmbedResidentForTest(id)
			}
			r.decodeAttnFA = false
			seed, err := rf.PrefillLast(ctx, embs, 0)
			if err != nil {
				t.Fatalf("PrefillLast: %v", err)
			}
			tok := argmaxF(seed)
			r.stopExec()
			copy(r.x.Floats(), m.EmbedResidentForTest(tok))
			r.addLearnedPos(K)
			r.setPos(K)
			nKeys := K + 1
			for l := 0; l < r.nL; l++ {
				e := r.q.Begin()
				r.encodeLayer(e, l)
				e.End()
				if err := e.Err(); err != nil {
					t.Fatalf("layer %d: %v", l, err)
				}
				L := &r.layers[l]
				g := L.geom
				if g == nil || g.hd != 128 || L.window != 0 {
					continue
				}
				nHhd := r.nH * g.hd
				G := r.nH / g.nKV
				q := append([]float32(nil), r.qkv.Floats()[:nHhd]...)
				inLayer := append([]float32(nil), r.ctx.Floats()[:nHhd]...)
				kh := r.kc[l].U16s()[:r.kc[l].Len()/2]
				vh := r.vc[l].U16s()[:r.vc[l].Len()/2]
				// float64 reference
				ref := make([]float64, nHhd)
				sc := make([]float64, nKeys)
				for h := 0; h < r.nH; h++ {
					kvh := h / G
					mx := math.Inf(-1)
					for j := 0; j < nKeys; j++ {
						var a float64
						base := j*g.kvDim + kvh*g.hd
						for d := 0; d < g.hd; d++ {
							a += float64(q[h*g.hd+d]) * float64(f16ToF32(kh[base+d]))
						}
						sc[j] = a * scale
						mx = math.Max(mx, sc[j])
					}
					var sum float64
					for j := range sc {
						sc[j] = math.Exp(sc[j] - mx)
						sum += sc[j]
					}
					for d := 0; d < g.hd; d++ {
						var a float64
						for j := 0; j < nKeys; j++ {
							a += sc[j] * float64(f16ToF32(vh[j*g.kvDim+kvh*g.hd+d]))
						}
						ref[h*g.hd+d] = a / sum
					}
				}
				cpuF := make([]float64, nHhd) // stored as the f32 values the reference would hold
				cpuU := make([]float64, nHhd)
				{
					sc32 := make([]float32, nKeys)
					for h := 0; h < r.nH; h++ {
						kvh := h / G
						maxS := math.Inf(-1)
						for j := 0; j < nKeys; j++ {
							var dot float64
							base := j*g.kvDim + kvh*g.hd
							for d := 0; d < g.hd; d++ {
								dot += float64(q[h*g.hd+d]) * float64(f16ToF32(kh[base+d]))
							}
							x := dot * scale
							sc32[j] = float32(x)
							maxS = math.Max(maxS, x)
						}
						var sum float64
						for j := range sc32 {
							e := math.Exp(float64(sc32[j]) - maxS)
							sc32[j] = float32(e)
							sum += e
						}
						inv := 1.0 / sum
						accF := make([]float32, g.hd)
						accU := make([]float32, g.hd)
						for j := 0; j < nKeys; j++ {
							w := float32(float64(sc32[j]) * inv)
							base := j*g.kvDim + kvh*g.hd
							for d := 0; d < g.hd; d++ {
								v := f16ToF32(vh[base+d])
								accF[d] = float32(math.FMA(float64(w), float64(v), float64(accF[d])))
								accU[d] += float32(w * v) // the explicit conversion forbids fusion (Go spec)
							}
						}
						var nf, nu, dn float64
						for d := 0; d < g.hd; d++ {
							cpuF[h*g.hd+d], cpuU[h*g.hd+d] = float64(accF[d]), float64(accU[d])
							x, y, rr := float64(accF[d])-ref[h*g.hd+d], float64(accU[d])-ref[h*g.hd+d], ref[h*g.hd+d]
							nf, nu, dn = nf+x*x, nu+y*y, dn+rr*rr
						}
						cpuRelF, cpuRelU = append(cpuRelF, math.Sqrt(nf/dn)), append(cpuRelU, math.Sqrt(nu/dn))
					}
				}
				cosine := func(a, b []float64) float64 {
					var ab, aa, bb float64
					for i := range a {
						ab, aa, bb = ab+a[i]*b[i], aa+a[i]*a[i], bb+b[i]*b[i]
					}
					if aa == 0 || bb == 0 {
						return 0
					}
					return ab / math.Sqrt(aa*bb)
				}
				var exactDistF []float64
				var exactErr []float64
				for ai, a := range arms {
					e := r.q.Begin()
					if !a.fa {
						pa := r.pAttn
						if a.p != (Pipeline{}) {
							pa = a.p
						}
						e.Dispatch(pa, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l], r.vc[l], outBuf, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
					} else {
						r.attnFASplitOverride = a.split
						nSplit := r.attnFASplitFor(nKeys, g.nKV)
						r.attnFASplitOverride = 0
						r.uAttnFAG.SetU32(uint32(G))
						r.uAttnFANSplit.SetU32(uint32(nSplit))
						e.DispatchTG(a.p, g.nKV*nSplit*128, 128, 128*6*G*4, r.qkv, r.kc[l], r.vc[l], r.attnFAPartial,
							g.uNKV, r.uAttnFAG, r.uNKeys, r.uScale, L.uWindow, r.uAttnFANSplit)
						pc := r.pAttnFACombine
						if a.comb != (Pipeline{}) {
							pc = a.comb
						}
						e.Dispatch(pc, r.nH*g.hd, g.hd, r.attnFAPartial, outBuf, r.uAttnFAG, g.uHd, r.uAttnFANSplit)
					}
					e.End()
					if err := e.Err(); err != nil {
						t.Fatalf("layer %d arm %s: %v", l, a.name, err)
					}
					out := outBuf.Floats()[:nHhd]
					if ai == 0 {
						exactOut = append(exactOut[:0], out...)
					}
					for h := 0; h < r.nH; h++ {
						var num, den float64
						for d := 0; d < g.hd; d++ {
							x := float64(out[h*g.hd+d]) - float64(exactOut[h*g.hd+d])
							num, den = num+x*x, den+ref[h*g.hd+d]*ref[h*g.hd+d]
						}
						vsExact[ai] = append(vsExact[ai], math.Sqrt(num/den))
					}
					if ai == 0 {
						sanityN++
						same := true
						for i := range out {
							if math.Float32bits(out[i]) != math.Float32bits(inLayer[i]) {
								same = false
								break
							}
						}
						if same {
							sanityOK++
						}
					}
					for h := 0; h < r.nH; h++ {
						var num, den float64
						for d := 0; d < g.hd; d++ {
							x := float64(out[h*g.hd+d]) - ref[h*g.hd+d]
							num += x * x
							den += ref[h*g.hd+d] * ref[h*g.hd+d]
							maxAbs[ai] = math.Max(maxAbs[ai], math.Abs(x))
						}
						ek, ef, eu := make([]float64, g.hd), make([]float64, g.hd), make([]float64, g.hd)
						var distF float64
						for d := 0; d < g.hd; d++ {
							o := float64(out[h*g.hd+d])
							ek[d], ef[d], eu[d] = o-ref[h*g.hd+d], cpuF[h*g.hd+d]-ref[h*g.hd+d], cpuU[h*g.hd+d]-ref[h*g.hd+d]
							distF += (o - cpuF[h*g.hd+d]) * (o - cpuF[h*g.hd+d])
						}
						cosF[ai] = append(cosF[ai], cosine(ek, ef))
						cosU[ai] = append(cosU[ai], cosine(ek, eu))
						if ai == 0 {
							exactDistF = append(exactDistF, distF)
						} else if distF < exactDistF[h] {
							nearF[ai]++
						}
						e := math.Sqrt(num / den)
						rel[ai] = append(rel[ai], e)
						relL[ai][l] = append(relL[ai][l], e)
						if e > worstV[ai] {
							worstV[ai] = e
							worst[ai] = fmt.Sprintf("prompt %d depth %d layer %d head %d (||ref|| %.3g)", pi+1, K, l, h, math.Sqrt(den))
						}
						if ai == 0 {
							exactErr = append(exactErr, e)
						} else {
							switch {
							case e < exactErr[h]:
								win[ai]++
							case e == exactErr[h]:
								tie[ai]++
							}
						}
					}
				}
				r.setPos(K) // restore the production split uniform for the next layer's encode
			}
			fmt.Fprintf(os.Stderr, "[r17-acc] prompt %d/%d depth %d done (decode token %d at pos %d)\n", pi+1, nPrompts, K, tok, K)
		}
	}
	if sanityOK != sanityN {
		t.Errorf("capture sanity: the standalone exact dispatch matched the in-layer output in only %d of %d layers", sanityOK, sanityN)
	}
	pct := func(xs []float64, p float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[min(len(s)-1, int(p*float64(len(s))))]
	}
	fmt.Fprintf(os.Stderr, "\n=== R17 kernel accuracy vs float64, %s, prompt set %q, depths %v, %d prompts × %d layers × %d heads; capture sanity %d/%d layers bit-identical ===\n",
		filepath.Base(path), setLabel, depths, nPrompts, r.nL, r.nH, sanityOK, sanityN)
	fmt.Fprintf(os.Stderr, "  %-26s %12s %12s %12s %12s %12s  %s\n", "kernel", "relL2 median", "mean", "p99", "max", "max|abs|", "closer than exact / ties (of heads)   | moved from exact: median")
	for ai, a := range arms {
		var mean float64
		for _, x := range rel[ai] {
			mean += x
		}
		mean /= float64(len(rel[ai]))
		wt := "—"
		if ai > 0 {
			wt = fmt.Sprintf("%d / %d of %d", win[ai], tie[ai], len(rel[ai]))
		}
		fmt.Fprintf(os.Stderr, "  %-26s %12.3e %12.3e %12.3e %12.3e %12.3e  %-36s | %.3e\n", a.name, pct(rel[ai], 0.5), mean, pct(rel[ai], 0.99), pct(rel[ai], 1), maxAbs[ai], wt, pct(vsExact[ai], 0.5))
	}
	fmt.Fprintf(os.Stderr, "\n  shared-bias check vs an emulation of the CPU reference's attention (f64 softmax, serial f32 V sum): the emulation's own relL2 vs f64 median %.3e fused / %.3e unfused\n",
		pct(cpuRelF, 0.5), pct(cpuRelU, 0.5))
	fmt.Fprintf(os.Stderr, "  %-30s %22s %22s %28s\n", "kernel", "cos(err, cpuFused err)", "cos(err, cpuUnfused err)", "closer to cpuFused than exact")
	for ai, a := range arms {
		pos := 0
		for _, c := range cosF[ai] {
			if c > 0 {
				pos++
			}
		}
		nf := "—"
		if ai > 0 {
			nf = fmt.Sprintf("%d of %d", nearF[ai], len(cosF[ai]))
		}
		fmt.Fprintf(os.Stderr, "  %-30s  median %+.3f (%4.0f%% >0)  median %+.3f          %s\n", a.name, pct(cosF[ai], 0.5), 100*float64(pos)/float64(len(cosF[ai])), pct(cosU[ai], 0.5), nf)
	}
	fmt.Fprintf(os.Stderr, "\n  worst head per kernel:\n")
	for ai, a := range arms {
		fmt.Fprintf(os.Stderr, "  %-26s relL2 %.3e at %s\n", a.name, worstV[ai], worst[ai])
	}
	var show []int // exact, attention_fa at the production split, prototype S=16, and precise-exp exact / prototype
	for _, want := range []string{"exact (shipped attention)", "attention_fa S=prod", "prototype S=16", "exact, precise exp", "prototype S=16, precise exp"} {
		for ai, a := range arms {
			if a.name == want {
				show = append(show, ai)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "\n  per layer, relL2 median / max over prompts × heads:\n  %5s", "layer")
	for _, ai := range show {
		fmt.Fprintf(os.Stderr, "  %-29s", arms[ai].name)
	}
	fmt.Fprintln(os.Stderr)
	for l := 0; l < r.nL; l++ {
		if len(relL[0][l]) == 0 {
			continue
		}
		fmt.Fprintf(os.Stderr, "  %5d", l)
		for _, ai := range show {
			fmt.Fprintf(os.Stderr, "  %12.3e / %12.3e    ", pct(relL[ai][l], 0.5), pct(relL[ai][l], 1))
		}
		fmt.Fprintln(os.Stderr)
	}
	// P1 of docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md: a candidate passes if its median
	// AND its p99 per-head relative L2 error vs float64 are each <= the exact kernel's. The max is reported, not gated.
	exMed, exP99, exMax := pct(rel[0], 0.5), pct(rel[0], 0.99), pct(rel[0], 1)
	for ai, a := range arms {
		if a.name != "attention_fa S=prod" && a.name != "prototype S=16" {
			continue
		}
		med, p99 := pct(rel[ai], 0.5), pct(rel[ai], 0.99)
		v := map[bool]string{true: "PASS", false: "FAIL"}[med <= exMed && p99 <= exP99 && sanityOK == sanityN]
		fmt.Fprintf(os.Stderr, "=== P1 kernel accuracy (pre-registered), %s, set %q: %s median %.3e vs exact %.3e, p99 %.3e vs exact %.3e (max %.3e vs exact %.3e, reported) — %s ===\n",
			filepath.Base(path), setLabel, a.name, med, exMed, p99, exP99, pct(rel[ai], 1), exMax, v)
	}
}
