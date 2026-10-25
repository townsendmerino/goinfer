//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/decoder"
)

// kso1cModels are O1C-G3's two files: Qwen2.5-Coder-0.5B q4_k_m (Q6_K in half its layers' v and down) and Phi-3 mini q4
// (Q5_K qkv, Q6_K down in half its layers and the head).
var kso1cModels = []string{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"}

// kso1cWeights is the resident's weights in plan order, each with its label and source: every layer's projections, then the head.
func kso1cWeights(r *cudaResident, m *decoder.Model) (labels []string, ws []cudaWQ, srcs []*linalg.WeightMat) {
	w := m.Weights()
	for l, Ly := range r.layers {
		lw := &w.Layers[l]
		for _, p := range []struct {
			name string
			wq   cudaWQ
			src  *linalg.WeightMat
		}{{"q", Ly.q, &lw.QProj}, {"k", Ly.k, &lw.KProj}, {"v", Ly.v, &lw.VProj}, {"o", Ly.o, &lw.OProj}, {"gate", Ly.g, &lw.GateProj}, {"up", Ly.u, &lw.UpProj}, {"down", Ly.d, &lw.DownProj}} {
			if p.wq.N > 0 {
				labels, ws, srcs = append(labels, fmt.Sprintf("layer %d %s", l, p.name)), append(ws, p.wq), append(srcs, p.src)
			}
		}
	}
	head := &w.LMHead
	if head.Rows() == 0 {
		head = &w.Embed
	}
	return append(labels, "head"), append(ws, r.lmW), append(srcs, head)
}

// kso1cDeviceBytes is a weight's device bytes: its packed buffer and its scales.
func kso1cDeviceBytes(w cudaWQ) int64 {
	n := int64(w.W.Len()) * 4
	if w.kind == "q5k" || w.kind == "q6k" {
		n = int64(w.W.Len()) // one byte buffer
	}
	return n + int64(w.ws.Len())*4 + int64(w.ws16.Len())*2
}

// TestKSO1C_laneBytes is O1C-G3 (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on CUDA: gates"): each model loaded on CUDA at
// --quant q4k twice, the int8 arm (decoder.SetKQuantNativeOffForTest) and the native arm, after one discarded warm-up load. The
// native arm's device weight bytes must fall by exactly the fit pricing's saving, each native buffer must hold exactly the file's
// bytes for its weight, layer 0's native buffers and the head must read back as the file's blocks (Q6_K through its planes), and
// the load's Go heap allocation must not exceed the int8 arm's plus 1 MiB.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKSO1C_laneBytes$' -v ./cuda/
func TestKSO1C_laneBytes(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	const heapSlack = 1 << 20
	for _, name := range kso1cModels {
		path := modelPath(name)
		if _, err := os.Stat(path); err != nil {
			t.Skipf("no model %s: %v", path, err) // run-o1c-day.sh counts a skip as a failure
		}
		type arm struct {
			total, pricing, heap int64
			byKind               map[string]int64
		}
		load := func(native bool) (arm, *decoder.Model) {
			decoder.SetKQuantNativeOffForTest(!native)
			defer decoder.SetKQuantNativeOffForTest(false)
			runtime.GC()
			var m0, m1 runtime.MemStats
			runtime.ReadMemStats(&m0)
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "q4k", ResidentContext: 512})
			if err != nil {
				t.Fatalf("%s: load: %v", name, err)
			}
			runtime.ReadMemStats(&m1)
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				m.Close()
				t.Fatalf("%s: no CUDA resident: %s", name, m.ResidentDecline())
			}
			a := arm{heap: int64(m1.TotalAlloc - m0.TotalAlloc), pricing: m.ResidentDenseWeightBytesFor("cuda"), byKind: map[string]int64{}}
			_, ws, _ := kso1cWeights(r, m)
			for _, w := range ws {
				b := kso1cDeviceBytes(w)
				a.total += b
				a.byKind[w.kind] += b
			}
			return a, m
		}
		warm, wm := load(true)
		wm.Close()
		_ = warm
		old, om := load(false)
		om.Close()
		nat, nm := load(true)
		saving := nm.KQuantSavingsForTest("cuda")
		fmt.Fprintf(os.Stderr, "O1C-G3 %s: int8 arm %d device bytes %v, heap %d; native arm %d %v, heap %d; pricing saves %d (dense pricing %d → %d)\n",
			name, old.total, old.byKind, old.heap, nat.total, nat.byKind, nat.heap, saving, old.pricing, nat.pricing)
		if old.total-nat.total != saving || old.pricing-nat.pricing != saving || saving <= 0 {
			t.Errorf("%s: device bytes fell by %d, the pricing by %d, the saving is %d: all three must be equal and positive",
				name, old.total-nat.total, old.pricing-nat.pricing, saving)
		}
		if nat.heap > old.heap+heapSlack {
			t.Errorf("%s: native load allocated %d bytes of heap, the int8 arm %d (+%d allowed)", name, nat.heap, old.heap, heapSlack)
		}
		// Every native buffer holds exactly the file's bytes; layer 0's and the head are compared byte for byte.
		r := nm.ResidentForwardForTest().(*cudaResident)
		kq, err := nm.OpenKQuant()
		if err != nil || kq == nil {
			nm.Close()
			t.Fatalf("%s: OpenKQuant: %v", name, err)
		}
		labels, ws, srcs := kso1cWeights(r, nm)
		checked := 0
		for i, w := range ws {
			kind := nm.KQuantNative("cuda", srcs[i])
			if kind != w.kind && (kind != "" || w.kind == "q5k" || w.kind == "q6k") {
				t.Errorf("%s %s: uploaded as %q, the lane says %q", name, labels[i], w.kind, kind)
				continue
			}
			if kind == "" {
				continue
			}
			bb := map[string]int{"q5k": 176, "q6k": 210}[kind]
			if want := w.N * (w.K / 256) * bb; w.W.Len() != want {
				t.Errorf("%s %s: %d device bytes, the file's are %d", name, labels[i], w.W.Len(), want)
			}
			if !(labels[i] == "head" || len(labels[i]) > 8 && labels[i][:8] == "layer 0 ") {
				continue
			}
			var file []byte
			if err := kq.Rows(srcs[i], func(_ int, raw []byte) error { file = append(file, raw...); return nil }); err != nil {
				t.Fatalf("%s %s: %v", name, labels[i], err)
			}
			if kind == "q6k" {
				file = kso1cPlanes(file)
			}
			dev := make([]uint8, w.W.Len())
			if err := r.do(func() error { return gpu.Download(w.W, dev) }); err != nil {
				t.Fatalf("%s %s: download: %v", name, labels[i], err)
			}
			if !bytes.Equal(dev, file) {
				t.Errorf("%s %s: the device bytes are not the file's %s blocks", name, labels[i], kind)
			}
			checked++
		}
		_ = kq.Close()
		nm.Close()
		if checked == 0 {
			t.Errorf("%s: no native buffer was compared with the file", name)
		}
		fmt.Fprintf(os.Stderr, "O1C-G3 %s: %d native buffers (layer 0 and the head) equal the file's blocks\n", name, checked)
	}
}
