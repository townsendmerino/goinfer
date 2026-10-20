//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"errors"
	"runtime"
	"testing"

	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/decoder"
)

// B36's gates (docs/queue-engineering.md B36): under the C' expert cache the routed experts are packed straight into their pinned host copy.
//
//	go test -count=1 -timeout 10m -tags 'cuda goinfer_testhooks' -run '^TestExpertStackPack_' -v ./cuda/

// expertStackWeights is nW int4 weights of rows x k, the members of one expert stack, each with its own values.
func expertStackWeights(nW, rows, k int) []*linalg.WeightMat {
	out := make([]*linalg.WeightMat, nW)
	for i := range out {
		f := make([]float32, rows*k)
		s := uint32(7 + 31*i)
		for j := range f {
			s = s*1664525 + 1013904223
			f[j] = float32(int32(s>>8)%2000-1000) / 900
		}
		w := linalg.QuantizeInt4(f, rows, k, 32)
		out[i] = &w
	}
	return out
}

// hostAlloc is the bytes the Go heap is asked for while fn runs (cumulative, so a buffer freed again still counts).
func hostAlloc(fn func()) uint64 {
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	fn()
	runtime.ReadMemStats(&b)
	return b.TotalAlloc - a.TotalAlloc
}

// stagedPinned is the build B36 replaced: the stack is packed into Go memory, then copied into the pinned buffers. Only the plant runs it.
func stagedPinned(r *cudaResident, ws []*linalg.WeightMat) cudaWQ {
	h, err := packWeightStack(ws...)
	if err != nil {
		panic(err)
	}
	w := cudaWQ{kind: h.kind, N: h.N, K: h.K}
	w.srcW, w.srcS = r.mapBytes(append([]byte(nil), u32bytes(h.wpk)...)), r.mapBytes(append([]byte(nil), u16bytes(h.ws16)...))
	return w
}

func expertRig(t *testing.T, nE int) *cudaResident {
	r := streamRig(t)
	r.nE, r.cacheExperts = nE, true
	t.Setenv("GOINFER_MOE_PIN_REGISTER", "0")
	return r
}

// The pinned copy's bytes are packWeightStack's, and the strides cacheWQ derives from the stack are the same.
func TestExpertStackPack_pinnedBytesIdenticalToPackWeightStack(t *testing.T) {
	const nE, rows, k = 8, 48, 256
	ws := expertStackWeights(2*nE, rows, k)
	eager, err := packWeightStack(ws...)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := planWeightStack(ws...)
	if err != nil {
		t.Fatal(err)
	}
	if plan.kind != eager.kind || plan.N != eager.N || plan.K != eager.K || plan.nWpk != len(eager.wpk) || plan.nWs16 != len(eager.ws16) || len(plan.wpk) != 0 {
		t.Fatalf("the plan names kind %s N %d K %d buffers %d/%d (packed words held: %d); the packed stack is kind %s N %d K %d buffers %d/%d",
			plan.kind, plan.N, plan.K, plan.nWpk, plan.nWs16, len(plan.wpk), eager.kind, eager.N, eager.K, len(eager.wpk), len(eager.ws16))
	}
	r := expertRig(t, nE)
	want := r.cacheWQ(eager)
	got := r.cacheWQ(plan)
	if r.setupErr != nil {
		t.Fatalf("upload error: %v", r.setupErr)
	}
	if got.perExpertW != want.perExpertW || got.perExpertS != want.perExpertS || got.N != want.N || got.K != want.K {
		t.Errorf("strides %d/%d N %d K %d, want %d/%d N %d K %d", got.perExpertW, got.perExpertS, got.N, got.K, want.perExpertW, want.perExpertS, want.N, want.K)
	}
	if !bytes.Equal(got.srcW.Bytes(), want.srcW.Bytes()) || len(got.srcW.Bytes()) != len(u32bytes(eager.wpk)) {
		t.Errorf("the pinned weight bytes differ from packWeightStack's")
	}
	if !bytes.Equal(got.srcS.Bytes(), want.srcS.Bytes()) || len(got.srcS.Bytes()) != len(u16bytes(eager.ws16)) {
		t.Errorf("the pinned scale bytes differ from packWeightStack's")
	}
	if a, b := packedAllocSlack([]any{[]hostW{plan}}), packedAllocSlack([]any{[]hostW{eager}}); a != b {
		t.Errorf("allocation slack priced from the plan is %d, from the packed stack %d", a, b)
	}
}

// Heap bytes allocated while the stack is built are at most a tenth of the pinned bytes (the eager build's are more than all of them); the plant, the
// staged build, must exceed that bound, so a return to staging turns the gate red.
func TestExpertStackPack_heapAllocatedIsATenthOfPinnedAtMost(t *testing.T) {
	const nE, rows, k = 16, 256, 1024
	ws := expertStackWeights(2*nE, rows, k)
	plan, err := planWeightStack(ws...)
	if err != nil {
		t.Fatal(err)
	}
	pinned := uint64(plan.nWpk*4 + plan.nWs16*2)
	bound := pinned / 10
	r := expertRig(t, nE)
	r.cacheWQ(plan) // warm: the first call may start worker goroutines and fault in code
	planned := hostAlloc(func() { r.cacheWQ(plan) })
	staged := hostAlloc(func() { stagedPinned(r, ws) })
	t.Logf("pinned %d bytes: planned build allocated %d heap bytes, staged build %d", pinned, planned, staged)
	if planned > bound {
		t.Errorf("the planned build allocated %d heap bytes, over the bound %d (a tenth of the %d pinned)", planned, bound, pinned)
	}
	if staged <= bound {
		t.Errorf("VOID: the staged build (the plant) allocated %d heap bytes, within the bound %d, so this gate cannot go red", staged, bound)
	}
	if r.setupErr != nil {
		t.Fatalf("upload error: %v", r.setupErr)
	}
}

// A plan declines with packWeightStack's error, before anything is allocated: no host bytes beyond the error's own text.
func TestExpertStackPack_aDeclineAllocatesNothing(t *testing.T) {
	good := expertStackWeights(2, 32, 256)
	badK := linalg.QuantizeInt4(make([]float32, 8*36), 8, 36, 12)
	other := expertStackWeights(1, 32, 128)
	int8w := linalg.QuantizeInt8(make([]float32, 32*256), 32, 256, true)
	cases := map[string][]*linalg.WeightMat{
		"K%32":       {good[0], &badK},
		"ragged K":   {good[0], other[0]},
		"mixed kind": {good[0], &int8w},
		"empty":      nil,
	}
	for name, ws := range cases {
		_, kerr := packWeightStack(ws...)
		var perr error
		a := hostAlloc(func() { _, perr = planWeightStack(ws...) })
		if perr == nil || kerr == nil || perr.Error() != kerr.Error() {
			t.Errorf("%s: planWeightStack %v, packWeightStack %v: they must decline with the same error", name, perr, kerr)
		}
		if a > 4096 {
			t.Errorf("%s: declining allocated %d host bytes", name, a)
		}
	}
	if _, err := planWeightStack(good...); err != nil {
		t.Errorf("VOID: the control stack was declined: %v", err)
	}
}

// The build's abort poll reaches the stack packer, and a planned stack never reaches an upload that is not the cache.
func TestExpertStackPack_abortAndMisuse(t *testing.T) {
	ws := expertStackWeights(4, 32, 256)
	plan, _ := planWeightStack(ws...)
	r := expertRig(t, 2)
	ch := make(chan struct{})
	close(ch)
	r.loadAbort = ch
	r.cacheWQ(plan)
	if !errors.Is(r.setupErr, decoder.ErrLoadAborted) {
		t.Errorf("a closed abort channel recorded %v, want ErrLoadAborted", r.setupErr)
	}
	r2 := expertRig(t, 2)
	r2.cacheExperts = false
	r2.upExperts(plan)
	if r2.setupErr == nil {
		t.Errorf("a planned stack reached the uncached upload without an error")
	}
}
