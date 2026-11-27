package decoder

import (
	"context"
	"encoding/binary"
	"errors"
	"github.com/townsendmerino/goinfer/internal/runtime/paging"
	"hash/crc32"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/families"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
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

func TestBatcher_claimExclusiveAndAdaptiveMutualExclusion(t *testing.T) {
	b := testResidentBatcher(2, 2)
	var resBusy int32

	// Whole-generation exclusive claim blocks adaptive entry.
	if !b.claimExclusive(&resBusy) {
		t.Fatal("initial claimExclusive failed")
	}
	if b.claimAdaptive(&resBusy) {
		t.Fatal("claimAdaptive succeeded while whole-generation resBusy was held")
	}
	atomic.StoreInt32(&resBusy, 0)

	// Active adaptive generation blocks whole-generation exclusive claim, and can take round claims when alone.
	if !b.claimAdaptive(&resBusy) {
		t.Fatal("claimAdaptive failed when idle")
	}
	if b.claimExclusive(&resBusy) {
		t.Fatal("claimExclusive succeeded while an adaptive generation was active")
	}
	if !b.claimAdaptiveRound(&resBusy) {
		t.Fatal("claimAdaptiveRound failed for a lone adaptive generation")
	}
	// A second adaptive generation may still register while the first holds its round claim,
	// after which neither can claim a solo round so both join the batcher cleanly.
	if !b.claimAdaptive(&resBusy) {
		t.Fatal("second claimAdaptive failed while first held only a round claim")
	}
	atomic.StoreInt32(&resBusy, 0)
	if b.claimAdaptiveRound(&resBusy) {
		t.Fatal("claimAdaptiveRound succeeded with two concurrent adaptive generations")
	}
	b.releaseAdaptive()
	if !b.claimAdaptiveRound(&resBusy) {
		t.Fatal("claimAdaptiveRound failed after second adaptive generation released")
	}
	atomic.StoreInt32(&resBusy, 0)
	b.releaseAdaptive()

	if !b.claimExclusive(&resBusy) {
		t.Fatal("claimExclusive failed after all adaptive generations released")
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
	p, err := paging.NewExpertBufferPool(f, []poolMember{
		{Key: k1, Fields: []expertField{paging.NewInt8ExpertField(&w1, 0, 4, []float32{1}, 1, 4)}},
		{Key: k2, Fields: []expertField{paging.NewInt8ExpertField(&w2, 100, 4, []float32{1}, 1, 4)}}, // past EOF
	}, 4, 1) // 1 slot
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Ensure(k1); err != nil {
		t.Fatalf("first ensure: %v", err)
	}
	if err := p.Ensure(k2); err == nil {
		t.Fatal("expected error reading past EOF")
	}
	// Slot 0 must remain free (not leaked), so re-ensuring k1 succeeds without panicking on empty lru.
	if err := p.Ensure(k1); err != nil {
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

func TestLoadSession_rejectsAsymmetricKVAndMissingScales(t *testing.T) {
	m := &Model{
		w: &darch.Weights{
			Arch: &darch.Architecture{
				NumLayers:  1,
				NumKVHeads: 2,
				HeadDim:    4,
			},
		},
		kvI8: true,
	}
	ref := m.NewCache(0)
	if ref.c.Quant != kvI8 {
		t.Fatalf("synthetic cache quant = %d, want kvI8 (%d)", ref.c.Quant, kvI8)
	}
	const pos = 2
	kvDim := ref.c.KVDim // 8
	nKV := kvDim / ref.c.HeadDim

	buildBlob := func(kLen, vLen, ksLen, vsLen int) []byte {
		wr := &giwWriter{}
		wr.Raw([]byte(kvSnapMagic))
		wr.U32(kvSnapVersion)
		wr.Str("m")
		wr.Str("")
		wr.U32(uint32(ref.c.NumLayers))
		wr.U32(uint32(kvDim))
		wr.U32(uint32(ref.c.Window))
		wr.U32(uint32(ref.c.HeadDim))
		wr.Raw([]byte{0, byte(kvI8)})
		wr.U32(pos)
		wr.ints([]int{1, 2})
		wr.I8(make([]int8, kLen))
		wr.I8(make([]int8, vLen))
		wr.F32Body(make([]float32, ksLen))
		wr.F32Body(make([]float32, vsLen))
		crc := crc32.ChecksumIEEE(wr.Buf)
		var tail [4]byte
		binary.LittleEndian.PutUint32(tail[:], crc)
		return append(wr.Buf, tail[:]...)
	}

	// Valid complete kvI8 snapshot loads cleanly.
	if _, err := m.LoadSession(buildBlob(pos*kvDim, pos*kvDim, pos*nKV, pos*nKV), "m"); err != nil {
		t.Fatalf("valid kvI8 snapshot failed: %v", err)
	}
	// keysQ present, valsQ empty -> rejected with SnapshotError.
	if _, err := m.LoadSession(buildBlob(pos*kvDim, 0, pos*nKV, pos*nKV), "m"); err == nil {
		t.Fatal("LoadSession accepted asymmetric keysQ/valsQ")
	} else if _, ok := errors.AsType[*SnapshotError](err); !ok || !strings.Contains(err.Error(), "presence mismatch") {
		t.Fatalf("asymmetric keysQ/valsQ err = %v, want SnapshotError presence mismatch", err)
	}
	// keysQ and valsQ present, keyScale and valScale empty -> rejected with SnapshotError.
	if _, err := m.LoadSession(buildBlob(pos*kvDim, pos*kvDim, 0, 0), "m"); err == nil {
		t.Fatal("LoadSession accepted keysQ with missing keyScale")
	} else if _, ok := errors.AsType[*SnapshotError](err); !ok || !strings.Contains(err.Error(), "presence mismatch") {
		t.Fatalf("missing keyScale err = %v, want SnapshotError presence mismatch", err)
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
	if err := families.ValidateConfigBounds(cfg); err == nil {
		t.Fatal("expected oversized GPT-2 n_layer to be rejected")
	}
	if err := families.ValidateConfigBounds(&Config{ModelType: "gpt2", NEmbd: maxGGUFHidden + 1}); err == nil {
		t.Fatal("expected oversized GPT-2 n_embd to be rejected")
	}
	if err := families.ValidateConfigBounds(&Config{ModelType: "gpt2", NLayer: 12, NEmbd: 768, NHead: 12}); err != nil {
		t.Fatalf("valid GPT-2 rejected: %v", err)
	}
}

func TestFlint_ggufLeadingDenseBounds(t *testing.T) {
	g := &embed.GGUFFile{
		Metadata: map[string]any{
			"general.architecture":                   "flint",
			"flint.block_count":                      uint32(6),
			"flint.embedding_length":                 uint32(64),
			"flint.feed_forward_length":              uint32(128),
			"flint.attention.head_count":             uint32(4),
			"flint.attention.head_count_kv":          uint32(2),
			"flint.attention.key_length":             uint32(16),
			"flint.attention.layer_norm_rms_epsilon": float32(1e-6),
			"flint.rope.freq_base":                   float32(10000.0),
			"flint.context_length":                   uint32(2048),
			"flint.leading_dense_block_count":        uint32(7), // exceeds block_count=6
			"tokenizer.ggml.tokens":                  make([]any, 256),
		},
	}
	if _, err := ggufConfig(g); err == nil || !strings.Contains(err.Error(), "leading_dense_block_count") {
		t.Fatalf("ggufConfig oversized flint.leading_dense_block_count err = %v, want leading_dense_block_count error", err)
	}
}

func TestMLARope_interleavedMatchesReferenceAndZeroAllocs(t *testing.T) {
	for _, ropeDim := range []int{16, 64, 320} {
		const heads = 2
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
		rope.Apply(want, heads, ropeDim, 7, invFreq, 1.0)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("ropeDim=%d idx %d: got %g, want %g", ropeDim, i, got[i], want[i])
			}
		}
		if ropeDim <= 256 {
			allocs := testing.AllocsPerRun(20, func() {
				mlaRope(got, heads, ropeDim, 7, invFreq, 1.0, true)
			})
			if allocs != 0 {
				t.Fatalf("mlaRope(ropeDim=%d) allocs = %v, want 0", ropeDim, allocs)
			}
		}
	}
}
