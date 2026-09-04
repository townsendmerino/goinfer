package decoder

import (
	"errors"
	"sync/atomic"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// blockspecStubHost is a minimal ResidentForward + ResidentDrafterHost for testing the R-00
// resBusy/resIDs bookkeeping in BlockSpec.generate without a real backend. setBatchedCaptureErr,
// when non-nil, makes SetBatchedCapture fail immediately — generate() returns before ever
// reaching PrefillLastNArgmax or the drafter, which is exactly what these tests need: they pin
// bookkeeping done BEFORE that point, not the verify loop itself (that needs a real device, per
// R-00's own gate note on CUDA being the only ResidentDrafterHost).
type blockspecStubHost struct {
	setBatchedCaptureErr   error
	setBatchedCaptureCalls int32
}

func (h *blockspecStubHost) Forward([]float32, int) ([]float32, error)      { return nil, nil }
func (h *blockspecStubHost) ForwardN([][]float32, int) ([][]float32, error) { return nil, nil }
func (h *blockspecStubHost) UploadKV(int, []float32, []float32) error       { return nil }
func (h *blockspecStubHost) TruncateTo(int)                                 {}
func (h *blockspecStubHost) Reset()                                         {}
func (h *blockspecStubHost) Close() error                                   { return nil }

func (h *blockspecStubHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return nil, errors.New("blockspecStubHost: AttachBlockDrafter should not be reached in this test")
}

func (h *blockspecStubHost) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return nil, errors.New("blockspecStubHost: PrefillLastNArgmax should not be reached in this test")
}

func (h *blockspecStubHost) SetBatchedCapture(taps []int) error {
	atomic.AddInt32(&h.setBatchedCaptureCalls, 1)
	return h.setBatchedCaptureErr
}

func (h *blockspecStubHost) BatchedCapture() [][]float32 { return nil }

var (
	_ ResidentForward     = (*blockspecStubHost)(nil)
	_ ResidentDrafterHost = (*blockspecStubHost)(nil)
)

// blockspecStubWeights is a minimal BlockDrafterWeights: generate() only reads BlockSize()
// before either test's controlled stopping point, so every other method panics if reached.
type blockspecStubWeights struct{ blockSize int }

func (blockspecStubWeights) DrafterGeometry() DrafterGeometry     { panic("not reached") }
func (blockspecStubWeights) DrafterFC() *linalg.WeightMat         { panic("not reached") }
func (blockspecStubWeights) DrafterHiddenNorm() []float32         { panic("not reached") }
func (blockspecStubWeights) DrafterFinalNorm() []float32          { panic("not reached") }
func (blockspecStubWeights) DrafterLayer(int) DrafterLayerWeights { panic("not reached") }
func (blockspecStubWeights) MaskTokenID() int                     { panic("not reached") }
func (w blockspecStubWeights) BlockSize() int                     { return w.blockSize }

var _ BlockDrafterWeights = blockspecStubWeights{}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestBlockSpecGenerate_residentBusyDeclines is audit R-00's concurrent-claim half: a generation
// already holding m.resBusy must make BlockSpec.generate return before touching the host or
// resIDs — the gap that let a plain-greedy commit and a concurrent block-spec generation
// interleave writes into the same positional resident KV with no error anywhere.
func TestBlockSpecGenerate_residentBusyDeclines(t *testing.T) {
	host := &blockspecStubHost{}
	m := &Model{resident: host, resIDs: []int{1, 2, 3}}
	atomic.StoreInt32(&m.resBusy, 1) // simulate another in-flight generation
	s := &BlockSpec{m: m, host: host, dw: blockspecStubWeights{blockSize: 8}}

	_, _, err := s.generate([]int{1, 2, 3}, BlockSpecOptions{VerifyWidth: 4}, nil)
	if !ErrBlockSpecResidentBusy(err) {
		t.Fatalf("generate() with resBusy held: err = %v, want ErrBlockSpecResidentBusy", err)
	}
	if got := atomic.LoadInt32(&host.setBatchedCaptureCalls); got != 0 {
		t.Errorf("SetBatchedCapture called %d times while resBusy was held — the busy path must "+
			"return before any device write", got)
	}
	if !equalIntSlices(m.resIDs, []int{1, 2, 3}) {
		t.Errorf("resIDs = %v after a declined claim, want unchanged [1 2 3] — a losing claim must "+
			"not forget a commit it never touched", m.resIDs)
	}
	if atomic.LoadInt32(&m.resBusy) != 1 {
		t.Error("resBusy released by a call that never held the claim")
	}
}

// TestBlockSpecGenerate_forgetsResIDsBeforeAnyWrite is R-00's other half: on a successful claim,
// resIDs must already be nil before the first device write (SetBatchedCapture), and resBusy must
// be released again on ANY exit — including this test's controlled early failure — so an early
// return never leaves the next turn trusting a half-written cache.
func TestBlockSpecGenerate_forgetsResIDsBeforeAnyWrite(t *testing.T) {
	wantErr := errors.New("stub: stop here")
	host := &blockspecStubHost{setBatchedCaptureErr: wantErr}
	m := &Model{resident: host, resIDs: []int{9, 9, 9}}
	s := &BlockSpec{m: m, host: host, dw: blockspecStubWeights{blockSize: 8}}

	_, _, err := s.generate([]int{1, 2, 3}, BlockSpecOptions{VerifyWidth: 4}, nil)
	if !errors.Is(err, wantErr) {
		t.Fatalf("generate() error = %v, want the stub's SetBatchedCapture error", err)
	}
	if got := atomic.LoadInt32(&host.setBatchedCaptureCalls); got != 1 {
		t.Fatalf("SetBatchedCapture called %d times, want exactly 1 — test setup assumption broke", got)
	}
	if m.resIDs != nil {
		t.Errorf("resIDs = %v after a claimed generation, want nil — the forget must happen before "+
			"the first device write, not only on a clean return", m.resIDs)
	}
	if atomic.LoadInt32(&m.resBusy) != 0 {
		t.Error("resBusy still held after generate() returned — an early error must still release the claim")
	}
}
