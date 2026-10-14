//go:build darwin

package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
)

// The attention kernels' raw test outputs, hashed, so two builds can be compared byte for byte (OP1's gate G1-K,
// docs/tasks/task-forward-architecture-2026-10.md §4.6). The kernel tests call recordAttnDigest where they read a
// kernel's output back; without -attn-digest it does nothing.
//
//	go test -tags goinfer_testhooks ./metal -run "$(attnDigestRun)" -count=1 -attn-digest=/tmp/base.tsv
//	go test -tags goinfer_testhooks ./metal -run '^TestAttnDigest_coversEveryKernel$' -attn-digest-check=/tmp/base.tsv
//
// Two files from two builds on one box are equal exactly when every recorded output is.
var (
	attnDigestOut   = flag.String("attn-digest", "", "append a sha256 of each attention kernel's raw test output to this file")
	attnDigestCheck = flag.String("attn-digest-check", "", "a file written by -attn-digest: fail unless every attention kernel in the shipped sources has a line in it")

	attnDigestMu  sync.Mutex
	attnDigestSeq = map[string]int{}
)

// attnDigestRun is the -run pattern of the tests that record digests.
const attnDigestRun = "^(TestAttention_ShippedKernelShapes|TestAttentionKernelsPastTileBound|TestAttentionFA_vsReference|TestAttnFABlk_anyGMatchesG7|" +
	"TestAttnFABlkMatchesFloat64|TestAttnFABlk64Digest|TestAttentionPrefillFused|TestAttentionPrefillSteelMatchesFloat64|" +
	"TestAttentionPrefillImg_matchesReference|TestGptOssAttnSink_metal|TestMetalKVI8_KVStoreAndAttentionParity|TestMetalKVI8_DeepContext|" +
	"TestMC3Step_rowsPathBitIdentical|TestMC3Step_faRowsBitIdenticalDeep|TestAttnDigest_mc3RowsOtherGeometries)$"

// recordAttnDigest appends one line for out, the raw output a test read back after dispatching kernels (comma-separated
// when a dispatch chains two). A line is kernels, test, the call's ordinal within that pair, element count, sha256.
func recordAttnDigest(t testing.TB, kernels string, out any) {
	if *attnDigestOut == "" {
		return
	}
	t.Helper()
	h := sha256.New()
	n := 0
	switch v := out.(type) {
	case []float32:
		n = len(v)
		_ = binary.Write(h, binary.LittleEndian, v)
	case []uint16:
		n = len(v)
		_ = binary.Write(h, binary.LittleEndian, v)
	default:
		t.Fatalf("recordAttnDigest: unsupported output type %T", out)
	}
	attnDigestMu.Lock()
	defer attnDigestMu.Unlock()
	key := kernels + "\t" + t.Name()
	seq := attnDigestSeq[key]
	attnDigestSeq[key]++
	f, err := os.OpenFile(*attnDigestOut, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("recordAttnDigest: %v", err)
	}
	defer f.Close()
	if _, err := fmt.Fprintf(f, "%s\t%d\t%d\t%x\n", key, seq, n, h.Sum(nil)); err != nil {
		t.Fatalf("recordAttnDigest: %v", err)
	}
}

// attnKernelNames is every attention kernel entry point in src: plain kernels and template instantiations.
func attnKernelNames(src string) []string {
	var names []string
	for _, re := range []*regexp.Regexp{
		regexp.MustCompile(`(?m)^kernel void ((?:mc3_)?attention\w*)\(`),
		regexp.MustCompile(`host_name\("((?:mc3_)?attention\w*)"\)`),
	} {
		for _, m := range re.FindAllStringSubmatch(src, -1) {
			if !slices.Contains(names, m[1]) {
				names = append(names, m[1])
			}
		}
	}
	slices.Sort(names)
	return names
}

// attnDigestRowsLabel names the kernels behind the batched step's logits in mc3IdentityWith, for a test that forces a
// geometry; "" lets mc3IdentityWith label them from the default fixture's.
var attnDigestRowsLabel string

// TestAttnDigest_coversEveryKernel reads a -attn-digest file and fails unless each attention kernel in the shipped
// sources has a line: the digest comparison says nothing about a kernel that was never run.
func TestAttnDigest_coversEveryKernel(t *testing.T) {
	names := attnKernelNames(allKernels + mc3RowsKernels + prefillKernels)
	if len(names) < 10 {
		t.Fatalf("found only %d attention kernels in the sources (%v): the patterns no longer match them", len(names), names)
	}
	if *attnDigestCheck == "" {
		t.Skipf("no -attn-digest-check file. Record one with -run '%s' -attn-digest=FILE; the sources have %d attention kernels: %s",
			attnDigestRun, len(names), strings.Join(names, " "))
	}
	raw, err := os.ReadFile(*attnDigestCheck)
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]int{}
	lines := 0
	for ln := range strings.SplitSeq(strings.TrimSpace(string(raw)), "\n") {
		f := strings.Split(ln, "\t")
		if len(f) != 5 {
			t.Fatalf("malformed digest line %q", ln)
		}
		lines++
		for k := range strings.SplitSeq(f[0], ",") {
			have[k]++
		}
	}
	// A template's own name is not an entry point; its instantiations are. The block kernels' rows forms are
	// instantiated for every group size the block kernels are, and are run at the two a fixture here has (6 and, at
	// head dimension 64, 7): the rest are the same template text, as TestAttnFABlk_anyGMatchesG7 shows for the parent.
	ran := 0
	for _, n := range names {
		if slices.ContainsFunc(names, func(o string) bool { return strings.HasPrefix(o, n+"_g") }) {
			continue
		}
		if have[n] > 0 {
			ran++
			continue
		}
		if strings.HasPrefix(n, "mc3_attention_fa_blk_rows_g") && have["mc3_attention_fa_blk_rows_g6"] > 0 {
			continue
		}
		t.Errorf("no digest recorded for %s", n)
	}
	t.Logf("%d digest lines; %d of the sources' %d attention kernel names ran directly", lines, ran, len(names))
}

// TestAttnFABlk64Digest drives attention_fa_blk64_g7 (head dimension 64, the 0.5B's shape) and attention_fa_combine the
// way the resident does, which no other kernel-level test does, and records its output. The reference check is loose on
// purpose: it shows the dispatch computed attention at all, and the digest carries the identity.
func TestAttnFABlk64Digest(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention_fa_blk64_g7")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	comb, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("combine: %v", err)
	}
	const G, nKV, hd = 7, 2, 64
	const nH = G * nKV
	cq := d.NewCommandQueue()
	scale := float32(1 / math.Sqrt(hd))
	for _, nKeys := range []int{attnFADepthFloor, attnFADepthFloor + 1, 3900, 4100} {
		for _, peaked := range []bool{false, true} {
			rng := rand.New(rand.NewSource(int64(nKeys*10 + 64)))
			q := make([]float32, nH*hd)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			kh, vh := make([]uint16, nKeys*nKV*hd), make([]uint16, nKeys*nKV*hd)
			for i := range kh {
				kh[i] = f32ToF16(float32(rng.NormFloat64() * 0.5))
				vh[i] = f32ToF16(float32(rng.NormFloat64()))
			}
			if peaked {
				for _, s := range []int{0, nKeys / 3, nKeys - 1} {
					for kvh := range nKV {
						for dd := range hd {
							kh[(s*nKV+kvh)*hd+dd] = f32ToF16(q[(kvh*G)*hd+dd] * 0.35)
						}
					}
				}
			}
			kb, vb := d.NewBufferBytes(len(kh)*2), d.NewBufferBytes(len(vh)*2)
			copy(kb.U16s()[:len(kh)], kh)
			copy(vb.U16s()[:len(vh)], vh)
			partial := d.NewBufferLen(nKV * attnFABlkSplit * G * (hd + 2))
			out := d.NewBufferLen(nH * hd)
			uG, uSplit := NewBufferU32(d, G), NewBufferU32(d, attnFABlkSplit)
			e := cq.Begin()
			e.DispatchTG(pipe, nKV*attnFABlkSplit*128, 128, 128*6*G*4, NewBufferFloats(d, q), kb, vb, partial,
				NewBufferU32(d, nKV), uG, NewBufferU32(d, uint32(nKeys)), NewBufferFloats(d, []float32{scale}), NewBufferU32(d, 0), uSplit)
			e.Dispatch(comb, nH*hd, hd, partial, out, uG, NewBufferU32(d, hd), uSplit)
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			got := append([]float32(nil), out.Floats()[:nH*hd]...)
			recordAttnDigest(t, "attention_fa_blk64_g7,attention_fa_combine", got)

			kf, vf := make([]float32, len(kh)), make([]float32, len(vh))
			for i := range kh {
				kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
			}
			ref := cpuAttention(q, kf, vf, nH, nKV, hd, nKeys, 0, scale)
			var dot, na, nb float64
			for i := range ref {
				dot += float64(got[i]) * float64(ref[i])
				na += float64(got[i]) * float64(got[i])
				nb += float64(ref[i]) * float64(ref[i])
			}
			if cos := dot / math.Sqrt(na*nb); !(cos >= 0.9999) {
				t.Errorf("nKeys=%d peaked=%v: cosine %.6f against the CPU reference, want >= 0.9999", nKeys, peaked, cos)
			}
		}
	}
}

// TestAttnDigest_mc3RowsOtherGeometries runs the batched step's identity check on two generated models whose geometry
// selects the split-KV kernels the default fixture (G = 6, head dimension 128) does not: head dimension 64 with G = 7,
// which takes attention_fa_blk64 and its rows form, and G = 4 with the any-G block kernel off, which takes
// attention_fa and its rows form. Which kernel a geometry takes is read from buildResident's selection, and each case
// checks the conditions that selection keys on.
func TestAttnDigest_mc3RowsOtherGeometries(t *testing.T) {
	for _, c := range []struct {
		name               string
		heads, kvHeads, hd int
		anyG               bool
		label              string
	}{
		{"hd64-G7", 14, 2, 64, true, "mc3_attention_rows,mc3_attention_fa_blk64_rows_g7,mc3_attention_fa_combine_rows"},
		{"hd128-G4-legacy", 8, 2, 128, false, "mc3_attention_rows,mc3_attention_fa_rows,mc3_attention_fa_combine_rows"},
	} {
		t.Run(c.name, func(t *testing.T) {
			prevW, prevG, prevFA, prevL := mc3FixtureWriter, attnFABlkAnyG, mc3FARowsOn, attnDigestRowsLabel
			defer func() {
				mc3FixtureWriter, attnFABlkAnyG, mc3FARowsOn, attnDigestRowsLabel = prevW, prevG, prevFA, prevL
			}()
			mc3FixtureWriter = func(t *testing.T, maxPositions int) string {
				return writeMC3FixtureGeom(t, maxPositions, c.heads, c.kvHeads, c.hd)
			}
			attnFABlkAnyG, mc3FARowsOn, attnDigestRowsLabel = c.anyG, true, c.label
			mc3IdentityWith(t, []int{attnFADepthFloor + 400, attnFADepthFloor - 6, 100, attnFADepthFloor + 64}, 2048, func(r *resident) {
				if r.attnFANKV == 0 {
					t.Fatalf("the split-KV kernels do not engage for this geometry, so this case would record the exact kernel only")
				}
				if g, hd := r.nH/r.attnFANKV, r.layers[r.attnFALayer].geom.hd; g != c.heads/c.kvHeads || hd != c.hd {
					t.Fatalf("resident geometry G=%d hd=%d, want G=%d hd=%d", g, hd, c.heads/c.kvHeads, c.hd)
				}
			})
		})
	}
}

// writeMC3FixtureGeom is writeMC3Fixture's model with another attention geometry.
func writeMC3FixtureGeom(t *testing.T, maxPositions, heads, kvHeads, headDim int) string {
	t.Helper()
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(20261013))
	rnd := func(n int, s float32) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = (rng.Float32()*2 - 1) * s
		}
		return d
	}
	ones := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = 1 + (rng.Float32()*2-1)*0.05
		}
		return d
	}
	qDim, kvDim := heads*headDim, kvHeads*headDim
	writeConfig(t, dir, fmt.Sprintf(`{"model_type":"qwen2","vocab_size":%d,"hidden_size":%d,
		"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,
		"intermediate_size":%d,"max_position_embeddings":%d,"rms_norm_eps":1e-6,"rope_theta":1000000,"tie_word_embeddings":false}`,
		mfVocab, mfHidden, mfLayers, heads, kvHeads, headDim, mfInter, maxPositions))
	ts := map[string]stf32{
		"model.embed_tokens.weight": {[]int{mfVocab, mfHidden}, rnd(mfVocab*mfHidden, 0.4)},
		"model.norm.weight":         {[]int{mfHidden}, ones(mfHidden)},
		"lm_head.weight":            {[]int{mfVocab, mfHidden}, rnd(mfVocab*mfHidden, 0.4)},
	}
	for l := range mfLayers {
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"self_attn.q_proj.weight"] = stf32{[]int{qDim, mfHidden}, rnd(qDim*mfHidden, 0.3)}
		ts[p+"self_attn.q_proj.bias"] = stf32{[]int{qDim}, rnd(qDim, 0.1)}
		ts[p+"self_attn.k_proj.weight"] = stf32{[]int{kvDim, mfHidden}, rnd(kvDim*mfHidden, 0.3)}
		ts[p+"self_attn.k_proj.bias"] = stf32{[]int{kvDim}, rnd(kvDim, 0.1)}
		ts[p+"self_attn.v_proj.weight"] = stf32{[]int{kvDim, mfHidden}, rnd(kvDim*mfHidden, 0.3)}
		ts[p+"self_attn.v_proj.bias"] = stf32{[]int{kvDim}, rnd(kvDim, 0.1)}
		ts[p+"self_attn.o_proj.weight"] = stf32{[]int{mfHidden, qDim}, rnd(mfHidden*qDim, 0.1)}
		ts[p+"input_layernorm.weight"] = stf32{[]int{mfHidden}, ones(mfHidden)}
		ts[p+"post_attention_layernorm.weight"] = stf32{[]int{mfHidden}, ones(mfHidden)}
		ts[p+"mlp.gate_proj.weight"] = stf32{[]int{mfInter, mfHidden}, rnd(mfInter*mfHidden, 0.3)}
		ts[p+"mlp.up_proj.weight"] = stf32{[]int{mfInter, mfHidden}, rnd(mfInter*mfHidden, 0.3)}
		ts[p+"mlp.down_proj.weight"] = stf32{[]int{mfHidden, mfInter}, rnd(mfHidden*mfInter, 0.3)}
	}
	writeSTF32(t, filepath.Join(dir, "model.safetensors"), ts)
	return dir
}
