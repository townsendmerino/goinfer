//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"errors"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"

	"github.com/townsendmerino/goinfer/decoder"
)

// The streaming pack's gates (docs/measurements/cuda-stream-pack-2026-10-18/README.md): G-S1 device bytes identical to packWeight+upW, G-S2a host bytes
// allocated per tensor are the staging block, G-S2b nothing is packed while planning, G-S3 a decline allocates nothing.
//
//	go test -count=1 -timeout 10m -tags 'cuda goinfer_testhooks' -run '^TestPackStream_' -v ./cuda/

func streamRig(t *testing.T) *cudaResident {
	t.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no cuda device: %v", err)
	}
	t.Cleanup(dev.ReleaseObjects)
	return &cudaResident{dev: dev}
}

func withStage(t *testing.T, words int) {
	t.Helper()
	old := stagePackWords
	stagePackWords = words
	t.Cleanup(func() { stagePackWords = old })
}

// streamWeights returns one weight of each kind packWeight takes, with shapes that are not a whole number of rows per staging block.
func streamWeights() map[string]linalg.WeightMat {
	fill := func(n int, seed uint32) []float32 {
		f := make([]float32, n)
		s := seed
		for i := range f {
			s = s*1664525 + 1013904223
			f[i] = float32(int32(s>>8)%2000-1000) / 700
		}
		return f
	}
	const N, K = 70, 512
	q4k := make([]byte, N*K/256*144)
	s := uint32(99)
	for i := range q4k {
		s = s*1664525 + 1013904223
		q4k[i] = byte(s >> 13)
	}
	k4, err := linalg.WrapQ4K(q4k, N, K)
	if err != nil {
		panic(err)
	}
	return map[string]linalg.WeightMat{
		"int4":      linalg.QuantizeInt4(fill(N*K, 1), N, K, 32),
		"int8":      linalg.QuantizeInt8(fill(N*K, 2), N, K, true),
		"q4k":       k4,
		"f32->int8": linalg.WrapF32(fill(N*K, 3), N, K),
		// a bf16 weight, a block drafter's (AttachDrafter streams it): widened row by row and quantized to int8
		"bf16->int8": linalg.QuantizeBF16(fill(N*K, 4), N, K),
	}
}

// deviceBytes reads a device buffer back as bytes.
func deviceBytes(t *testing.T, b gpu.Buffer, bytesLen int) []byte {
	t.Helper()
	out := make([]byte, bytesLen)
	if err := gpu.Download(b, out); err != nil {
		t.Fatalf("download: %v", err)
	}
	return out
}

func devBytes(t *testing.T, w cudaWQ, h hostW) map[string][]byte {
	t.Helper()
	m := map[string][]byte{"W": deviceBytes(t, w.W, h.nWpk*4)}
	switch h.kind {
	case "int4":
		m["ws16"] = deviceBytes(t, w.ws16, h.nWs16*2)
	case "int8":
		m["ws"] = deviceBytes(t, w.ws, h.nWs*4)
	}
	return m
}

// G-S1: the streamed upload's device bytes are packWeight+upW's, at one block and at several.
func TestPackStream_deviceBytesIdenticalToPackWeight(t *testing.T) {
	r := streamRig(t)
	for name, wm := range streamWeights() {
		wm := wm
		plan, err := planWeight(&wm)
		if err != nil {
			t.Fatalf("%s: planWeight: %v", name, err)
		}
		eager, err := packWeight(&wm)
		if err != nil {
			t.Fatalf("%s: packWeight: %v", name, err)
		}
		if plan.kind != eager.kind || plan.nWpk != len(eager.wpk) || plan.nWs != len(eager.ws) || plan.nWs16 != len(eager.ws16) {
			t.Fatalf("%s: the plan names kind %s buffers %d/%d/%d, the packed weight is kind %s buffers %d/%d/%d", name,
				plan.kind, plan.nWpk, plan.nWs, plan.nWs16, eager.kind, len(eager.wpk), len(eager.ws), len(eager.ws16))
		}
		ref := r.upW(eager)
		want := devBytes(t, ref, plan)
		for _, words := range []int{1 << 20, 700, 64, 1} { // one block, then several; 1 forces one row per block
			withStage(t, words)
			r.pk = nil
			got := devBytes(t, r.upW(plan), plan)
			for k, b := range want {
				if !bytes.Equal(got[k], b) {
					i := 0
					for i < len(b) && i < len(got[k]) && b[i] == got[k][i] {
						i++
					}
					t.Errorf("%s, staging %d words: device buffer %s differs from packWeight's (first byte %d of %d)", name, words, k, i, len(b))
				}
			}
		}
	}
	if r.setupErr != nil {
		t.Fatalf("upload error: %v", r.setupErr)
	}
}

// G-S2a: the host bytes a streamed upload allocates are the staging block and a few headers, however large the tensors; the eager path allocates the packed copy.
func TestPackStream_hostBytesAreTheStagingBlockNotTheTensor(t *testing.T) {
	r := streamRig(t)
	const N, K, tensors = 2048, 2048, 6 // each int4 tensor packs to 2 MiB of words + 0.25 MiB of scales
	f := make([]float32, N*K)
	for i := range f {
		f[i] = float32(i%97-48) / 50
	}
	wm := linalg.QuantizeInt4(f, N, K, 32)
	plan, _ := planWeight(&wm)
	withStage(t, 1<<17) // 512 KiB: smaller than a tensor, so a per-tensor allocation shows
	alloc := func(fn func()) uint64 {
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		fn()
		runtime.ReadMemStats(&b)
		return b.TotalAlloc - a.TotalAlloc
	}
	r.pk = nil
	streamed := alloc(func() {
		for range tensors {
			_ = r.upW(plan)
		}
	})
	eagerBytes := alloc(func() {
		for range tensors {
			h, _ := packWeight(&wm)
			_ = r.upW(h)
		}
	})
	packed := uint64(tensors) * uint64(N*K/2+N*K/32*2)
	bound := uint64(stageCount())*uint64(stagePackWords)*4*2 + uint64(tensors)*64<<10 // the staging blocks (words, scales and their widening), plus headers
	if eagerBytes < packed {
		t.Fatalf("VOID: the eager path allocated %d host bytes for %d packed: the fixture is too small to see the difference", eagerBytes, packed)
	}
	if streamed > bound {
		t.Errorf("streaming %d tensors of %d packed bytes allocated %d host bytes; the bound is the staging block, %d (the eager path: %d)", tensors, packed/tensors, streamed, bound, eagerBytes)
	}
}

// G-S2b: planning packs nothing, and the allocation slack priced from planned layers is the slack of the same layers packed.
func TestPackStream_planningPacksNothing(t *testing.T) {
	for name, wm := range streamWeights() {
		wm := wm
		plan, _ := planWeight(&wm)
		if len(plan.wpk) != 0 || len(plan.ws) != 0 || len(plan.ws16) != 0 || plan.src == nil {
			t.Errorf("%s: a planned weight holds packed data (%d/%d/%d words) or no source", name, len(plan.wpk), len(plan.ws), len(plan.ws16))
		}
		eager, _ := packWeight(&wm)
		if a, b := packedAllocSlack([]any{[]hostW{plan, plan}}), packedAllocSlack([]any{[]hostW{eager, eager}}); a != b {
			t.Errorf("%s: the allocation slack priced from planned weights is %d, from packed ones %d", name, a, b)
		}
	}
}

// G-S3: a plan declines exactly as packWeight does, before anything is allocated; a model whose last layer declines uploads nothing from the earlier ones.
func TestPackStream_aDeclineAllocatesNothing(t *testing.T) {
	r := streamRig(t)
	good := streamWeights()
	badK := linalg.QuantizeInt4(make([]float32, 8*36), 8, 36, 12)    // K = 36: a multiple of 4, not of 32
	badQ4K, _ := linalg.WrapQ4K(make([]byte, 3*512/256*144), 4, 512) // 3 rows of bytes for 4 rows
	odd := linalg.QuantizeInt8(make([]float32, 8*30), 8, 30, true)   // K = 30: not a multiple of 4
	cases := map[string]linalg.WeightMat{"int4 K%32": badK, "q4k size": badQ4K, "K%4": odd}
	allocs := 0
	allocSeenForTest = func(int) { allocs++ }
	defer func() { allocSeenForTest = nil }()
	for name, wm := range cases {
		wm := wm
		_, perr := planWeight(&wm)
		_, kerr := packWeight(&wm)
		if perr == nil || kerr == nil || perr.Error() != kerr.Error() {
			t.Errorf("%s: planWeight %v, packWeight %v: they must decline with the same error", name, perr, kerr)
		}
	}
	// A model of three layers, the last of which declines: the build plans all three before it uploads any.
	layers := []linalg.WeightMat{good["int4"], good["int8"], badK}
	var plans []hostW
	var declined error
	for i := range layers {
		p, err := planWeight(&layers[i])
		if err != nil {
			declined = err
			break
		}
		plans = append(plans, p)
	}
	if declined == nil || len(plans) != 2 {
		t.Fatalf("expected the third layer to decline after two plans, got %v after %d", declined, len(plans))
	}
	if allocs != 0 || r.setupErr != nil {
		t.Errorf("planning and declining allocated %d device buffers (error %v)", allocs, r.setupErr)
	}
}

// G-S3, through the real build: BuildResident accepts every layer before it takes a byte of device memory, so a model that declines at its last layer has
// uploaded nothing from the earlier ones. Read for every build the load makes.
func TestPackStream_buildAllocatesNothingBeforeEveryLayerIsAccepted(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	var allocs, started, builds int
	allocSeenForTest = func(int) { allocs++ }
	planStartForTest = func() { started = allocs }
	planCompleteForTest = func() {
		builds++
		if d := allocs - started; d != 0 {
			t.Errorf("build %d allocated %d device buffers while it was still accepting layers", builds, d)
		}
	}
	defer func() { allocSeenForTest, planStartForTest, planCompleteForTest = nil, nil, nil }()
	g := kv2Load(t, dir, decoder.Options{ResidentKVSlots: 1, ResidentContext: 256})
	_ = g
	if builds == 0 {
		t.Fatal("VOID: the build never reached the end of planning (the resident was not built)")
	}
	if allocs == 0 {
		t.Error("VOID: the build allocated no device buffers at all")
	}
}

// A row source (planRows, a streamed drafter's matrices) uploads packWeight's device bytes for the same values, at one staging
// block and at several, and a reader that fails is an upload error rather than a weight of garbage.
func TestPackStream_rowSourceBytesIdenticalToPackWeight(t *testing.T) {
	r := streamRig(t)
	kinds := map[string]bool{}
	for name, wm := range streamWeights() {
		if k := wm.Kind(); k != "f32" && k != "bf16" {
			continue
		}
		kinds[wm.Kind()] = true
		wm := wm
		N, K := wm.Rows(), wm.Cols()
		eager, err := packWeight(&wm)
		if err != nil {
			t.Fatalf("%s: packWeight: %v", name, err)
		}
		plan, err := planRows(decoder.DrafterRows{Rows: N, Cols: K, Read: func(r0, n int, dst []float32) error {
			for i := range n {
				wm.Row(r0+i, dst[i*K:(i+1)*K])
			}
			return nil
		}})
		if err != nil {
			t.Fatalf("%s: planRows: %v", name, err)
		}
		want := devBytes(t, r.upW(eager), plan)
		for _, words := range []int{1 << 20, 700, 1} {
			withStage(t, words)
			r.pk = nil
			got := devBytes(t, r.upW(plan), plan)
			for k, b := range want {
				if !bytes.Equal(got[k], b) {
					t.Errorf("%s, staging %d words: device buffer %s differs from packWeight's", name, words, k)
				}
			}
		}
	}
	if !kinds["f32"] || !kinds["bf16"] {
		t.Fatalf("streamWeights gave the row-source check kinds %v, want both f32 and bf16", kinds)
	}
	if r.setupErr != nil {
		t.Fatalf("upload error: %v", r.setupErr)
	}
	// a reader that fails only on a later block: one block of all 8 rows, then blocks of 2 rows (an 8-word stage, 4 words a row)
	failing, err := planRows(decoder.DrafterRows{Rows: 8, Cols: 16, Read: func(r0, n int, _ []float32) error {
		if r0+n > 4 {
			return errors.New("read failed")
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, words := range []int{1 << 20, 8} {
		withStage(t, words)
		r.pk = nil
		r.upW(failing)
		if r.setupErr == nil || !strings.Contains(r.setupErr.Error(), "read failed") {
			t.Errorf("staging %d words, a reader failing from row 4: setupErr %v, want the reader's error", words, r.setupErr)
		}
		r.setupErr = nil
	}
}
