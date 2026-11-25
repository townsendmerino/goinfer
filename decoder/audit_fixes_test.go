package decoder

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

func testResidentBatcher(lo, hi int) *residentBatcher {
	b := &residentBatcher{maxHolders: hi, slotBusy: make([]bool, hi)}
	b.cond = sync.NewCond(&b.mu)
	b.lo, b.hi, b.wait = lo, hi, time.Millisecond
	return b
}

func TestBatcher_claimExclusiveBlockedWhileBusy(t *testing.T) {
	b := testResidentBatcher(2, 2)
	var resBusy int32
	entered := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		b.exclusive(func() {
			close(entered)
			<-release
		})
	}()
	<-entered
	if b.claimExclusive(&resBusy) {
		t.Fatal("claimExclusive succeeded while batcher busy flag was held")
	}
	close(release)
	<-done
	if !b.claimExclusive(&resBusy) {
		t.Fatal("claimExclusive failed after batcher became idle")
	}
}

func TestBatcher_serveLockedRecoversStepPanic(t *testing.T) {
	b := testResidentBatcher(2, 2)
	b.step = func([]*batchReq) { panic("boom in StepBatch") }
	b.enterDecode()
	b.enterDecode()
	defer b.exitDecode()
	defer b.exitDecode()
	q1 := &batchReq{solo: func() error { return nil }}
	q2 := &batchReq{solo: func() error { return nil }}
	var wg sync.WaitGroup
	var err1, err2 error
	wg.Go(func() { err1 = b.forward(q1) })
	wg.Go(func() { err2 = b.forward(q2) })
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("serveLocked deadlocked after step panic")
	}
	if err1 == nil || !strings.Contains(err1.Error(), "boom in StepBatch") {
		t.Fatalf("q1 err = %v, want panic error", err1)
	}
	if err2 == nil || !strings.Contains(err2.Error(), "boom in StepBatch") {
		t.Fatalf("q2 err = %v, want panic error", err2)
	}
	// Batcher must still be usable afterwards (b.busy cleared).
	ran := false
	b.exclusive(func() { ran = true })
	if !ran {
		t.Fatal("exclusive did not run after recovered step panic")
	}
}

func TestExpertBufferPool_refillErrorPreservesFreeSlot(t *testing.T) {
	f := filepath.Join(t.TempDir(), "short.giw")
	if err := os.WriteFile(f, []byte{1, 2, 3, 4}, 0o600); err != nil {
		t.Fatal(err)
	}
	w1 := linalg.WrapInt8(make([]int8, 4), []float32{1}, 1, 4, false)
	w2 := linalg.WrapInt8(make([]int8, 4), []float32{1}, 1, 4, false)
	k1 := unsafe.Pointer(&w1)
	k2 := unsafe.Pointer(&w2)
	p, err := newExpertBufferPool(f, []poolMember{
		{key: k1, fields: []expertField{{wm: &w1, kind: fieldInt8, off: 0, n: 4, scales: []float32{1}, rows: 1, cols: 4}}},
		{key: k2, fields: []expertField{{wm: &w2, kind: fieldInt8, off: 100, n: 4, scales: []float32{1}, rows: 1, cols: 4}}}, // past EOF
	}, 4, 1) // 1 slot
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()
	if err := p.ensure(k1); err != nil {
		t.Fatalf("first ensure: %v", err)
	}
	if err := p.ensure(k2); err == nil {
		t.Fatal("expected error reading past EOF")
	}
	// Slot 0 must remain free (not leaked), so re-ensuring k1 succeeds without panicking on empty lru.
	if err := p.ensure(k1); err != nil {
		t.Fatalf("ensure after failed refill: %v", err)
	}
}

type fakeSnapHandle struct {
	bytes    int64
	released int
}

func (h *fakeSnapHandle) Rows() int       { return 4 }
func (h *fakeSnapHandle) Bytes() int64    { return h.bytes }
func (h *fakeSnapHandle) Serves(int) bool { return true }
func (h *fakeSnapHandle) Release()        { h.released++ }

func TestKVSnapStore_removeLockedIdempotent(t *testing.T) {
	st := newKVSnapStore(1024)
	h := &fakeSnapHandle{bytes: 128}
	x := &kvSnap{h: h, ids: []int{1, 2, 3, 4}}
	st.list = append(st.list, x)
	st.used = 128

	st.drop(x)
	st.drop(x) // second drop after prior eviction must be a no-op
	if st.used != 0 {
		t.Fatalf("used = %d, want 0", st.used)
	}
	if h.released != 1 {
		t.Fatalf("released = %d, want 1", h.released)
	}
}

func TestCheckGlobalPair_rejectsAsymmetricKV(t *testing.T) {
	if err := checkGlobalPair(0, "keys", "vals", 0, 0, 16); err != nil {
		t.Fatalf("both zero should pass: %v", err)
	}
	if err := checkGlobalPair(0, "keys", "vals", 16, 16, 16); err != nil {
		t.Fatalf("both matching should pass: %v", err)
	}
	if err := checkGlobalPair(0, "keys", "vals", 16, 0, 16); err == nil {
		t.Fatal("expected mismatch error when keys present and vals empty")
	}
}

func TestRunGen_recoversPanicIntoGenerationErr(t *testing.T) {
	m := &Model{}
	out := make(chan int)
	g := &Generation{}
	m.runGen(context.Background(), out, g, nil, func(context.Context) {
		panic("corrupt weights panic")
	})
	for range out {
	}
	if g.Err() == nil || !strings.Contains(g.Err().Error(), "corrupt weights panic") {
		t.Fatalf("g.Err() = %v, want recovered panic error", g.Err())
	}
}

func TestValidateConfigBounds_catchesGPT2Oversized(t *testing.T) {
	cfg := &Config{ModelType: "gpt2", NLayer: maxGGUFLayers + 1}
	if err := validateConfigBounds(cfg); err == nil {
		t.Fatal("expected oversized GPT-2 n_layer to be rejected")
	}
	if err := validateConfigBounds(&Config{ModelType: "gpt2", NEmbd: maxGGUFHidden + 1}); err == nil {
		t.Fatal("expected oversized GPT-2 n_embd to be rejected")
	}
	if err := validateConfigBounds(&Config{ModelType: "gpt2", NLayer: 12, NEmbd: 768, NHead: 12}); err != nil {
		t.Fatalf("valid GPT-2 rejected: %v", err)
	}
}

func TestMLARope_interleavedMatchesReference(t *testing.T) {
	const heads, ropeDim = 2, 16
	invFreq := make([]float64, ropeDim/2)
	for i := range invFreq {
		invFreq[i] = 1.0 / float64(i+1)
	}
	vec := make([]float32, heads*ropeDim)
	for i := range vec {
		vec[i] = float32(i+1) * 0.25
	}
	got := slices.Clone(vec)
	mlaRope(got, heads, ropeDim, 7, invFreq, 1.0, true)

	// Reference using explicit heap de-interleave + applyRoPE.
	want := slices.Clone(vec)
	half := ropeDim / 2
	for hh := range heads {
		src := want[hh*ropeDim : (hh+1)*ropeDim]
		tmp := make([]float32, ropeDim)
		for i := range half {
			tmp[i] = src[2*i]
			tmp[half+i] = src[2*i+1]
		}
		copy(src, tmp)
	}
	applyRoPE(want, heads, ropeDim, 7, invFreq, 1.0)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("idx %d: got %g, want %g", i, got[i], want[i])
		}
	}
}
