//go:build cuda

package cuda

import (
	"fmt"
	"time"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// headArgProf times the three phases of a batched head-to-argmax tail (stream drain, M×vocab D2H, host argmax loop)
// at its two sites, the drafter's DraftTokens and the verify's batchedHeadArgmax. Nil by default.
type headArgProf struct {
	sync, dl, host time.Duration
	calls, rows    int
}

// SetHeadArgProfForTest arms (or disarms) the head-argmax phase profiler.
func (r *cudaResident) SetHeadArgProfForTest(on bool) {
	if on {
		r.headProf = &headArgProf{}
	} else {
		r.headProf = nil
	}
}

// HeadArgProfForTest reports the accumulated phases; zero when disarmed.
func (r *cudaResident) HeadArgProfForTest() (sync, dl, host time.Duration, calls, rows int) {
	if r.headProf == nil {
		return 0, 0, 0, 0, 0
	}
	p := r.headProf
	return p.sync, p.dl, p.host, p.calls, p.rows
}

// hostArgmaxRows is the serial host argmax both tails share: strict > from element 0, so ties go
// to the lowest index — the same tie-break argmax_reduce implements on the device.
func (r *cudaResident) hostArgmaxRows(host []float32, M int, ids []int) {
	for m := range M {
		row := host[m*r.vocab : (m+1)*r.vocab]
		bi, bv := 0, row[0]
		for i, v := range row {
			if v > bv {
				bi, bv = i, v
			}
		}
		ids[m] = bi
	}
}

// argmaxRows is the batched head tail's reduction: argmax_rows over the M rows of logits on the device, one drain,
// an M-int readback, in place of downloading M×vocab floats and looping on the host. Same strided scan and
// tie-break as argmax_reduce, so row m gets the token the host loop (strict >, lowest index) gets. hostArgmaxPath
// keeps the host tail for the A/B; argCheck runs both and compares.
func (r *cudaResident) argmaxRows(logits, idxBuf Buffer, M int, ids []int) error {
	var t0 time.Time
	if r.hostArgmaxPath {
		if r.headProf != nil {
			t0 = time.Now()
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if r.headProf != nil {
			r.headProf.sync += time.Since(t0)
			t0 = time.Now()
		}
		host := make([]float32, M*r.vocab)
		if e := gpu.Download(logits, host); e != nil {
			return e
		}
		if r.headProf != nil {
			r.headProf.dl += time.Since(t0)
			t0 = time.Now()
		}
		r.hostArgmaxRows(host, M, ids)
		if r.headProf != nil {
			r.headProf.host += time.Since(t0)
			r.headProf.calls++
			r.headProf.rows += M
		}
		return nil
	}
	if e := r.launch(r.fArgRows, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256*4 + 256*4},
		Arg(logits), gpu.ArgValue(int32(r.vocab)), gpu.ArgValue(int32(M)), Arg(idxBuf)); e != nil {
		return e
	}
	if r.headProf != nil {
		t0 = time.Now()
	}
	if e := r.stream.Sync(); e != nil {
		return e
	}
	if r.headProf != nil {
		r.headProf.sync += time.Since(t0)
		t0 = time.Now()
	}
	out := make([]int32, M)
	if e := gpu.Download(idxBuf, out); e != nil {
		return e
	}
	if e := argmaxRowIDs(out, ids, func(m int) ([]float32, error) {
		row := make([]float32, r.vocab)
		return row, gpu.Download(logits.At(m*r.vocab*4), row)
	}); e != nil {
		return e
	}
	if r.headProf != nil {
		r.headProf.dl += time.Since(t0)
		r.headProf.calls++
		r.headProf.rows += M
	}
	if r.argCheck {
		host := make([]float32, M*r.vocab)
		if e := gpu.Download(logits, host); e != nil {
			return e
		}
		ref := make([]int, M)
		r.hostArgmaxRows(host, M, ref)
		for m := range ref {
			if ref[m] != ids[m] {
				return fmt.Errorf("cuda: argmax_rows row %d = %d, host loop = %d (logit %g vs %g)",
					m, ids[m], ref[m], host[m*r.vocab+ids[m]], host[m*r.vocab+ref[m]])
			}
		}
		r.argChecks += M
	}
	return nil
}

// SetHostArgmaxForTest forces (true) or releases (false) the host argmax tail, the A/B's do-nothing arm.
func (r *cudaResident) SetHostArgmaxForTest(host bool) { r.hostArgmaxPath = host }

// SetArgmaxCheckForTest arms the row-for-row device-vs-host comparison inside argmaxRows.
func (r *cudaResident) SetArgmaxCheckForTest(on bool) { r.argCheck = on; r.argChecks = 0 }

// ArgmaxChecksForTest reports how many rows the check mode compared (all agreed, or the call errored).
func (r *cudaResident) ArgmaxChecksForTest() int { return r.argChecks }

// DecodeClassProfForTest reports the per-class g4moe decode split (see profMark). Zero unless
// GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) DecodeClassProfForTest() (attn, dense, router, clear, rt, segC, head time.Duration) {
	return r.profAttn, r.profDense, r.profRouter, r.profClear, r.profRT, r.profSegC, r.profHead
}

// UploadProfForTest reports the expert-DMA split: the big weight copies vs the tiny scale copies, by bytes moved
// and COPY count. Zero unless GOINFER_MOE_CACHE_PROF is set. It reports no per-kind elapsed times: the copies are
// queued by appendExpertSlot and issued together by one UploadBatch, so there is no per-kind upload to time; the
// transfer+sync time is BatchProfForTest's.
func (r *cudaResident) UploadProfForTest() (wB, sB, wC, sC uint64) {
	return r.profWBytes, r.profSBytes, r.profWCalls, r.profSCalls
}

// BatchProfForTest reports the batched-upload cost: total time inside gpu.UploadBatch and the
// number of SYNCHRONIZES it performed (one per layer with at least one miss). This is the quantity
// the batching change moves — copy count is unchanged, sync count is not. Zero unless
// GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) BatchProfForTest() (batchTime time.Duration, syncCalls uint64) {
	return r.profBatchTime, r.profSyncCalls
}

// argmaxPick is argmax_reduce over r.logits[:v] and its readback (executor thread only), the one greedy pick ForwardArgmax,
// the batched step and gumbelPick's fallback share. The kernel starts at -1e30 and reports -1 when no entry beats it (every
// logit -inf, NaN, or at most -1e30); argmaxID then reads the row back and takes the host's argmax. The id is the host's on
// every row except one with a NaN at index 0 and a logit above -1e30, where sampling.Argmax keeps 0 (no comparison with NaN
// is true) and the kernel returns the largest logit.
func (r *cudaResident) argmaxPick(v int) (int, error) {
	if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), Arg(r.logits), gpu.ArgValue(int32(v)), Arg(r.argIdx), Arg(r.argVal)); e != nil {
		return 0, e
	}
	if e := r.stream.Sync(); e != nil {
		return 0, e
	}
	out := make([]int32, 1)
	if e := gpu.Download(r.argIdx, out); e != nil {
		return 0, e
	}
	return argmaxID(out[0], func() ([]float32, error) {
		row := make([]float32, v)
		return row, gpu.Download(r.logits, row)
	})
}

// argmaxRowIDs maps argmax_rows' M indices into ids through argmaxID: read(m) is called only for a row that came back -1,
// and that row gets the host's argmax.
func argmaxRowIDs(out []int32, ids []int, read func(m int) ([]float32, error)) error {
	for m, idx := range out {
		id, err := argmaxID(idx, func() ([]float32, error) { return read(m) })
		if err != nil {
			return err
		}
		ids[m] = id
	}
	return nil
}

// argmaxID is argmax_reduce's index when it found one, and otherwise (idx -1) sampling.Argmax over the row read returns,
// which is called only then.
func argmaxID(idx int32, read func() ([]float32, error)) (int, error) {
	if idx >= 0 {
		return int(idx), nil
	}
	row, err := read()
	if err != nil {
		return 0, err
	}
	return sampling.Argmax(row), nil
}

// argmaxState is the batched head tails' argmax: the device kernel, the host-loop arms that check it, the verify tail's logits
// scratch and the head profiler.
type argmaxState struct {
	headProf *headArgProf // the head-argmax phase profiler; nil is off
	// The batched head tails' argmax runs on the device (fArgRows, M ints back) unless hostArgmaxPath forces the
	// M×vocab download + host loop (the A/B's do-nothing arm). argCheck additionally runs the host loop on the same
	// logits and counts agreement.
	fArgRows       Pipeline
	hostArgmaxPath bool
	argCheck       bool
	argChecks      int
	// Batched-head verify scratch (PrefillLastNArgmax). Allocated lazily INSIDE the executor
	// job — af/ai require r.dev's context current — and reused across rounds. logitsBCap is the
	// row count the current allocation covers; a wider block reallocates once and then holds.
	logitsB    Buffer
	logitsBIdx Buffer // [logitsBCap] int32: argmax_rows' output for the verify tail
	logitsBCap int
}
