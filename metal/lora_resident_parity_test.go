//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestLoRAResidentParityMetal is the G3 numeric gate (docs/tasks/task-gpu-paths-2026-09.md): compute-time LoRA on Metal's
// resident decode path must match the CPU reference (the same adapter through decoder's generic gatedMLP/causalAttention
// forward) within the resident-vs-CPU floor. 0.95 is the floor gpt2_resident_parity_test.go set for whole-model resident-vs-CPU
// logits (hiddenlast_resident_parity_test.go explains why 0.9999 does not apply once the LM head amplifies quantization
// noise), here with a LoRA delta on both sides.
//
// It drives decoder.ResidentAdapter directly (ResidentForwardForTest/ResidentAdapterLayersForTest) to isolate kernel
// correctness; the wiring (does an adapter session reach ResidentAdapter, does a backend without it decline) is gated by
// decoder/resident_adapter_seam_test.go's fake-backend tests.
//
// Fixture: testdata/llama-tiny, a committed resident-eligible safetensors checkpoint with GQA (4 heads/2 kv heads), so the
// o-proj and q/k/v-of-different-widths sites are genuinely exercised. Loaded at int4: int8int8 becomes int4 numerics on Metal
// anyway (G10).
func TestLoRAResidentParityMetal(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	adapterDir := buildLlamaTinyLoRAFixture(t)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go resident (BuildResident refused) — decode path %q; decline: %s",
			mRes.DecodePath(), mRes.ResidentDecline())
	}
	ra, ok := rf.(decoder.ResidentAdapter)
	if !ok {
		t.Fatalf("metal resident runner does not implement decoder.ResidentAdapter")
	}
	if err := mRes.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter (metal side): %v", err)
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	if err := mCPU.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter (cpu side): %v", err)
	}

	_, _, _, _, _, _, vocab := mCPU.Dims()
	prompt := make([]int, 8)
	for i := range prompt {
		prompt[i] = (i*47 + 5) % vocab
	}

	want, err := mCPU.PrefillLogitsWithAdapterForTest(context.Background(), prompt, "a")
	if err != nil {
		t.Fatalf("cpu compute-time prefill: %v", err)
	}

	// runResident drives ntok sequential Forward calls from a clean KV (Reset) and returns a copy of the last logits: Forward's
	// slice aliases resident-owned storage and is reused across calls.
	runResident := func() []float32 {
		t.Helper()
		rf.Reset()
		var last []float32
		for i, tok := range prompt {
			lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("resident forward[%d]: %v", i, err)
			}
			last = lr
		}
		return append([]float32(nil), last...)
	}

	if err := ra.SetAdapter(mRes.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatalf("resident SetAdapter: %v", err)
	}
	got := runResident()
	cos, maxAbs := cosF32(want, got)
	t.Logf("resident-with-adapter vs CPU-with-adapter: cosine=%.6f maxAbs=%.4g argmax_match=%v",
		cos, maxAbs, argmaxF32(want) == argmaxF32(got))
	// N-14/N-52 (docs/audit-metal-2026-09-12.md, docs/audit-2026-09-10.md): 0.95 is far looser than a correct bind measures, so a
	// bug dropping one of the seven per-layer projections could still clear it when that projection's share of the variance is
	// small; the vacuousness check below proves only that some effect survives, not that every projection fired. Deliberately not
	// tightened (parked, not rejected): two single-machine runs are too few to set a measured-floor-minus-noise bar without
	// risking CI flakes. Figures: docs/code-notes/metal.md#TestLoRAResidentParityMetal.floor
	if cos < 0.95 {
		t.Errorf("resident LoRA cosine %.6f < 0.95 — below the established resident-vs-CPU floor", cos)
	}

	// Vacuousness check (as in decoder/lora_compute_test.go): clearing the adapter and re-running the same prompt from a clean
	// KV must give different logits. Otherwise every dispatch this row added could be a silent no-op (the class of bug G6 found
	// twice) and the cosine check above would still pass, comparing the CPU adapter's effect against identical unadapted output.
	if err := ra.SetAdapter(nil); err != nil {
		t.Fatalf("resident SetAdapter(nil): %v", err)
	}
	gotNoAdapter := runResident()
	if cosNoAdapter, _ := cosF32(got, gotNoAdapter); cosNoAdapter > 0.999999 {
		t.Error("resident logits WITH and WITHOUT the adapter are indistinguishable — the adapter " +
			"has no effect on Metal, the dispatches are silently no-op'ing")
	}
}

// TestLoRAResidentParityMetal_armedExecutorThenBind pins C-07 (docs/audit-2026-09-10.md) and closes G-06's Metal half. The gate
// above binds SetAdapter on a fresh executor (r.execReq == nil, nothing pre-encoded) and compares only the last token, so it
// can never observe a stale pre-encoded buffer. This test arms the pipelined executor with a plain (no-adapter) Forward first,
// which pre-encodes the next command buffer under r.loraLayers == nil, then binds the adapter and re-runs the same position.
// The KV cache is positional, so the second call overwrites the first and what survives is decided by which encode gets
// committed.
func TestLoRAResidentParityMetal_armedExecutorThenBind(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	adapterDir := buildLlamaTinyLoRAFixture(t)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go resident (BuildResident refused) — decode path %q; decline: %s",
			mRes.DecodePath(), mRes.ResidentDecline())
	}
	ra, ok := rf.(decoder.ResidentAdapter)
	if !ok {
		t.Fatalf("metal resident runner does not implement decoder.ResidentAdapter")
	}
	if err := mRes.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter (metal side): %v", err)
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	if err := mCPU.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter (cpu side): %v", err)
	}

	_, _, _, _, _, _, vocab := mCPU.Dims()
	tok := 5 % vocab

	// ARM: a plain Forward with no adapter bound. Through ForwardEmbPipe, this both commits an
	// encode for position 0 AND pre-encodes the NEXT command buffer — still under the no-adapter
	// state, since SetAdapter has not run yet.
	if _, err := rf.Forward(mRes.EmbedResidentForTest(tok), 0); err != nil {
		t.Fatalf("arm: %v", err)
	}

	if err := ra.SetAdapter(mRes.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatalf("resident SetAdapter after arming: %v", err)
	}
	got, err := rf.Forward(mRes.EmbedResidentForTest(tok), 0)
	if err != nil {
		t.Fatalf("resident forward after bind: %v", err)
	}
	got = append([]float32(nil), got...)

	want, err := mCPU.PrefillLogitsWithAdapterForTest(context.Background(), []int{tok}, "a")
	if err != nil {
		t.Fatalf("cpu compute-time prefill: %v", err)
	}

	cos, maxAbs := cosF32(want, got)
	t.Logf("armed-then-bound resident vs CPU-with-adapter: cosine=%.6f maxAbs=%.4g argmax_match=%v",
		cos, maxAbs, argmaxF32(want) == argmaxF32(got))
	// N-14/N-52: the same 0.95 floor as TestLoRAResidentParityMetal's, parked there (see its floor comment).
	if cos < 0.95 {
		t.Errorf("cosine %.6f < 0.95 after binding on an already-armed executor — the pre-encoded "+
			"buffer from the arming call was committed instead of a fresh encode under the bound "+
			"adapter (C-07)", cos)
	}

	if err := ra.SetAdapter(nil); err != nil {
		t.Fatalf("resident SetAdapter(nil): %v", err)
	}
}

// TestSetAdapter_cachesDeviceBuffersAcrossRebind pins P-10's Metal half (docs/audit-2026-09-10.md): SetAdapter must not
// release and re-upload every projection's device buffers on each bind (one chat session binds and clears the same adapter every
// turn). Pointer-level check: a rebind of the same adapter reuses the exact *residLoRAProj (no NewBufferFloats calls); a bind of
// a different adapter evicts the cache and uploads fresh. The identity check is pointer-based, not content-based: the test binds
// two adapters built from identical data (buildLlamaTinyLoRAFixture is deterministic) loaded as separate loraRuntime
// instances.
func TestSetAdapter_cachesDeviceBuffersAcrossRebind(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	adapterDirA := buildLlamaTinyLoRAFixture(t)
	adapterDirB := buildLlamaTinyLoRAFixture(t) // same content, different files/instance

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	mr, ok := rf.(*metalResident)
	if !ok {
		t.Fatalf("ResidentForwardForTest did not return *metalResident (%T)", rf)
	}
	if err := mRes.LoadAdapter("a", adapterDirA); err != nil {
		t.Fatalf("LoadAdapter a: %v", err)
	}
	if err := mRes.LoadAdapter("b", adapterDirB); err != nil {
		t.Fatalf("LoadAdapter b: %v", err)
	}

	if err := mr.SetAdapter(mRes.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatalf("bind a (1st): %v", err)
	}
	if mr.r.loraLayers == nil || mr.r.loraLayers[0].q == nil {
		t.Fatal("bind a (1st): loraLayers[0].q is nil — fixture targets q_proj on every layer")
	}
	firstQ := mr.r.loraLayers[0].q

	if err := mr.SetAdapter(nil); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if mr.r.loraLayers != nil {
		t.Error("loraLayers not nil after SetAdapter(nil) — dispatch sites would still apply a delta")
	}
	if mr.r.loraCached == nil {
		t.Fatal("P-10: loraCached was released on clear instead of kept for a same-adapter rebind")
	}

	if err := mr.SetAdapter(mRes.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatalf("bind a (2nd, rebind): %v", err)
	}
	if mr.r.loraLayers[0].q != firstQ {
		t.Error("P-10: rebinding the SAME adapter got a NEW *residLoRAProj — the device buffers " +
			"were re-uploaded instead of reused from the cache")
	}

	if err := mr.SetAdapter(mRes.ResidentAdapterLayersForTest("b")); err != nil {
		t.Fatalf("bind b (different adapter): %v", err)
	}
	if mr.r.loraLayers[0].q == firstQ {
		t.Error("binding a DIFFERENT adapter (identical content, separate loraRuntime instance) " +
			"reused adapter a's cached buffers — the identity check is matching by content, not " +
			"by pointer, which would silently apply the wrong adapter's delta for two distinct binds")
	}
}

// buildLlamaTinyLoRAFixture writes a PEFT adapter directory targeting ALL SEVEN projections
// (q,k,v,o,gate,up,down) across every layer of testdata/llama-tiny (4 layers, hidden=64,
// qDim=64, kvDim=32, inter=128 — testdata/llama-tiny/config.json), rank 4 / alpha 8 (scale 2).
// Deliberately exercises every projection G3 wires, not just the subset
// decoder/resident_adapter_seam_test.go's synthetic fixture targets.
func buildLlamaTinyLoRAFixture(t *testing.T) string {
	t.Helper()
	const hidden, qDim, kvDim, inter, layers, r = 64, 64, 32, 128, 4, 4
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			// Same magnitude decoder/lora_compute_test.go's TestLoRACompute_forwardParity already
			// proved gives a visible (non-vacuous) effect at this rank — not "as large as possible",
			// just large enough that cosine's precision can actually see the delta.
			d[i] = float32((i*5+seed)%11)*0.07 - 0.35
		}
		return d
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"),
		[]byte(`{"r":4,"lora_alpha":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	at := map[string]loraStTensor{}
	pfx := "base_model.model.model.layers."
	itoa := func(i int) string {
		return string(rune('0' + i))
	}
	for l := 0; l < layers; l++ {
		add := func(mod string, inDim, outDim, seed int) {
			at[pfx+itoa(l)+mod+".lora_A.weight"] = loraStTensor{[]int{r, inDim}, fill(r*inDim, seed)}
			at[pfx+itoa(l)+mod+".lora_B.weight"] = loraStTensor{[]int{outDim, r}, fill(outDim*r, seed+1)}
		}
		add(".self_attn.q_proj", hidden, qDim, 100+l*10)
		add(".self_attn.k_proj", hidden, kvDim, 200+l*10)
		add(".self_attn.v_proj", hidden, kvDim, 300+l*10)
		add(".self_attn.o_proj", qDim, hidden, 400+l*10)
		add(".mlp.gate_proj", hidden, inter, 500+l*10)
		add(".mlp.up_proj", hidden, inter, 600+l*10)
		add(".mlp.down_proj", inter, hidden, 700+l*10)
	}
	writeAdapterSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), at)
	return dir
}

// loraStTensor is one tensor's shape+data, for writeAdapterSafetensors below.
type loraStTensor struct {
	shape []int
	data  []float32
}

// writeAdapterSafetensors is a minimal F32 .safetensors writer — the same shape
// decoder/lora_test.go's writeSafetensors uses, duplicated here since it is unexported in
// package decoder and this test lives in package metal.
func writeAdapterSafetensors(t *testing.T, path string, tensors map[string]loraStTensor) {
	t.Helper()
	type meta struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	header := map[string]meta{}
	var blob []byte
	off := 0
	for _, n := range names {
		d := tensors[n]
		b := make([]byte, len(d.data)*4)
		for i, f := range d.data {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
		}
		header[n] = meta{"F32", d.shape, [2]int{off, off + len(b)}}
		blob = append(blob, b...)
		off += len(b)
	}
	hjson, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var buf []byte
	lenHdr := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenHdr, uint64(len(hjson)))
	buf = append(buf, lenHdr...)
	buf = append(buf, hjson...)
	buf = append(buf, blob...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}
