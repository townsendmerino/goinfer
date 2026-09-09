//go:build gpu && goinfer_testhooks

package gpu

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

// TestLoRAResidentParityWebGPU is the G3 numeric gate (docs/task-gpu-paths-2026-09.md): compute-
// time LoRA applied on this backend's resident decode path must match the CPU reference (the
// same adapter applied via decoder's generic gatedMLP/causalAttention forward). Mirrors
// metal/lora_resident_parity_test.go — same fixture (testdata/llama-tiny, GQA 4-heads/2-kv-heads
// so o-proj and the differently-widthed q/k/v sites are all genuinely exercised), same
// vacuousness check, same reasoning for the 0.95 floor (gpt2_resident_parity_test.go's
// established resident-vs-CPU decode-logit bar on the Metal side; this backend has no
// resident-vs-CPU whole-model floor of its own narrower than that, so it is the right anchor
// here too) — but drives decoder.ResidentAdapter directly via ResidentForwardForTest/
// ResidentAdapterLayersForTest to isolate the KERNEL correctness question, same as every other
// resident parity test in this file's package.
//
// Architecturally this backend's SetAdapter (gpu/lora_resident.go) is a genuinely different
// mechanism from Metal's — a flat, Go-side dispatch-step list rebuilt from a saved pristine plan
// plus spliced-in LoRA steps, not a per-token re-encode with a plain `if` — so this test is the
// gate for THAT mechanism specifically, not just a kernel port's numerics.
func TestLoRAResidentParityWebGPU(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	adapterDir := buildLlamaTinyLoRAFixtureGPU(t)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go resident (BuildResident refused) — decode path %q; decline: %s",
			mRes.DecodePath(), mRes.ResidentDecline())
	}
	ra, ok := rf.(decoder.ResidentAdapter)
	if !ok {
		t.Fatalf("webgpu resident runner does not implement decoder.ResidentAdapter")
	}
	if err := mRes.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter (webgpu side): %v", err)
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

	// runResident drives len(prompt) sequential Forward calls from a clean KV (Reset) and
	// returns a COPY of the last one's logits — Forward's own doc says its returned slice is
	// reused across calls, so a caller comparing two runs' results must copy before the second
	// run's calls overwrite the first's (the exact bug this same test's Metal twin hit first).
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
	cos, maxAbs := cosine(want, got)
	t.Logf("resident-with-adapter vs CPU-with-adapter: cosine=%.6f maxAbs=%.4g argmax_match=%v",
		cos, maxAbs, argmaxF(want) == argmaxF(got))
	if cos < 0.95 {
		t.Errorf("resident LoRA cosine %.6f < 0.95 — below the established resident-vs-CPU floor", cos)
	}

	// Vacuousness check (mirrors decoder/lora_compute_test.go's own "the adapter is not vacuous"
	// assertion): clearing the adapter and re-running the SAME prompt from a clean KV must
	// produce DIFFERENT logits. If it didn't, r.rebuildSteps could be silently leaving r.steps
	// unchanged (the WebGPU analogue of Metal's zero-value dispatch bug) and the cosine check
	// above would still pass by accident.
	if err := ra.SetAdapter(nil); err != nil {
		t.Fatalf("resident SetAdapter(nil): %v", err)
	}
	gotNoAdapter := runResident()
	if cosNoAdapter, _ := cosine(got, gotNoAdapter); cosNoAdapter > 0.999999 {
		t.Error("resident logits WITH and WITHOUT the adapter are indistinguishable — the adapter " +
			"has no effect on this backend, the dispatches are silently no-op'ing")
	}
}

// buildLlamaTinyLoRAFixtureGPU is metal/lora_resident_parity_test.go's buildLlamaTinyLoRAFixture,
// duplicated here since it is package-private there and this test lives in package gpu: a PEFT
// adapter directory targeting ALL SEVEN projections (q,k,v,o,gate,up,down) across every layer of
// testdata/llama-tiny (4 layers, hidden=64, qDim=64, kvDim=32, inter=128 —
// testdata/llama-tiny/config.json), rank 4 / alpha 8 (scale 2).
func buildLlamaTinyLoRAFixtureGPU(t *testing.T) string {
	t.Helper()
	const hidden, qDim, kvDim, inter, layers, r = 64, 64, 32, 128, 4, 4
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			// Same magnitude decoder/lora_compute_test.go's TestLoRACompute_forwardParity already
			// proved gives a visible (non-vacuous) effect at this rank.
			d[i] = float32((i*5+seed)%11)*0.07 - 0.35
		}
		return d
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"),
		[]byte(`{"r":4,"lora_alpha":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	at := map[string]loraGPUStTensor{}
	pfx := "base_model.model.model.layers."
	itoa := func(i int) string { return string(rune('0' + i)) }
	for l := 0; l < layers; l++ {
		add := func(mod string, inDim, outDim, seed int) {
			at[pfx+itoa(l)+mod+".lora_A.weight"] = loraGPUStTensor{[]int{r, inDim}, fill(r*inDim, seed)}
			at[pfx+itoa(l)+mod+".lora_B.weight"] = loraGPUStTensor{[]int{outDim, r}, fill(outDim*r, seed+1)}
		}
		add(".self_attn.q_proj", hidden, qDim, 100+l*10)
		add(".self_attn.k_proj", hidden, kvDim, 200+l*10)
		add(".self_attn.v_proj", hidden, kvDim, 300+l*10)
		add(".self_attn.o_proj", qDim, hidden, 400+l*10)
		add(".mlp.gate_proj", hidden, inter, 500+l*10)
		add(".mlp.up_proj", hidden, inter, 600+l*10)
		add(".mlp.down_proj", inter, hidden, 700+l*10)
	}
	writeLoraGPUSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), at)
	return dir
}

type loraGPUStTensor struct {
	shape []int
	data  []float32
}

// writeLoraGPUSafetensors is a minimal F32 .safetensors writer — the same shape
// decoder/lora_test.go's writeSafetensors (and metal/lora_resident_parity_test.go's own copy)
// use, duplicated here since both are unexported/package-private elsewhere.
func writeLoraGPUSafetensors(t *testing.T, path string, tensors map[string]loraGPUStTensor) {
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
