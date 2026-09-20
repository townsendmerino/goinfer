//go:build cuda

package cuda

import (
	"fmt"
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// topkMaxK is the largest K topk_select returns (TK_KMAX in topk.cu).
const topkMaxK = 1024

// TopKAvailable reports whether this resident's logits are the RAW LM-head output. The top-K path
// selects on the device's row, but step() applies a final-logit softcap and a logit scale on the HOST
// after readback (Gemma, Cohere): selecting before that would rank by a different function than the
// sampler sees, so those families keep the full-row path.
func (r *cudaResident) TopKAvailable() bool {
	return r.finalSoftcap == 0 && (r.logitScale == 0 || r.logitScale == 1) // applyLogitScale treats 0 and 1 as "none"
}

// ForwardTopK runs one token's forward and reduces the logits row on-device to its k best entries
// (decoder.ResidentTopK). Same kernel chain as Forward; only the readback differs — ~2 KB instead of
// vocab*4 bytes. The returned Full closure reads the whole row for the token where k is not enough and
// is valid only until the next forward.
func (r *cudaResident) ForwardTopK(embedding []float32, pos, k int, temperature float64, wantZ bool) (decoder.TopKRow, error) {
	if k < 1 || k > topkMaxK {
		return decoder.TopKRow{}, fmt.Errorf("cuda: ForwardTopK k=%d outside [1,%d]", k, topkMaxK)
	}
	if k > r.vocab {
		k = r.vocab
	}
	if e := r.checkCap(pos, 1); e != nil {
		return decoder.TopKRow{}, e
	}
	var row decoder.TopKRow
	err := r.do(func() error {
		if e := r.launchToken(embedding, pos, pos, true); e != nil {
			return e
		}
		var err error
		row, err = r.topkReadback(r.vocab, k, temperature, wantZ)
		return err
	})
	if err != nil {
		return decoder.TopKRow{}, err
	}
	row.Full = func() ([]float32, error) {
		var out []float32
		err := r.do(func() error {
			if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
				return e
			}
			out = r.logitsHost
			return nil
		})
		return out, err
	}
	return row, nil
}

// topkReadback launches topk_select over r.logits, syncs, and decodes the packed output. Executor
// thread only. Shared by ForwardTopK and the test hook so the tests exercise the decode path's kernel
// launch and unpacking exactly.
func (r *cudaResident) topkReadback(v, k int, temperature float64, wantZ bool) (decoder.TopKRow, error) {
	invT := float32(1)
	if temperature > 0 {
		invT = float32(1 / temperature)
	}
	wz := int32(0)
	if wantZ {
		wz = 1
	}
	if e := r.launch(r.fTopK, onecfg(1024, 0), Arg(r.logits), gpu.ArgValue(int32(v)), gpu.ArgValue(int32(k)),
		gpu.ArgValue(invT), gpu.ArgValue(wz), Arg(r.topkOut)); e != nil {
		return decoder.TopKRow{}, e
	}
	if e := r.stream.Sync(); e != nil {
		return decoder.TopKRow{}, e
	}
	buf := make([]int32, 2*k+2)
	if e := gpu.Download(r.topkOut, buf); e != nil {
		return decoder.TopKRow{}, e
	}
	row := decoder.TopKRow{IDs: buf[:k], Logits: make([]float32, k)}
	for i := 0; i < k; i++ {
		row.Logits[i] = math.Float32frombits(uint32(buf[k+i]))
	}
	if wantZ {
		hi := float64(math.Float32frombits(uint32(buf[2*k])))
		lo := float64(math.Float32frombits(uint32(buf[2*k+1])))
		row.Z = hi + lo
	}
	return row, nil
}

var _ decoder.ResidentTopK = (*cudaResident)(nil)
