//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/wpack"
)

// UploadKV writes a layer's post-RoPE K and raw V into the bound slot's caches at absolute positions base..base+n-1 (prefill
// bridge, rows in the layout the kernels read: [pos*kvDim + head*hd + d], each position's row placed through the slot's table). base
// is normally 0; a wrapped CPU-side sliding-window ring's live K/V can start later (KVCache.LayerKV's base return).
func (r *cudaResident) UploadKV(layer, base int, keys, vals []float32) error {
	if r.kvI8 {
		return fmt.Errorf("cuda: UploadKV into an int8 KV cache: the host rows would need quantizing with their scales, which no path does yet")
	}
	if r.layers[layer].kvShared {
		// Its r.kc/r.vc ARE its source's buffers: an upload here would overwrite the source layer's cache with another layer's rows.
		return fmt.Errorf("cuda: UploadKV to layer %d, which shares layer %d's KV cache (Gemma 4 E-model): upload the owning layer", layer, r.layers[layer].kvSrc)
	}
	kvDim := r.layers[layer].kvDim
	if kvDim == 0 {
		return nil
	}
	if e := r.checkCap(base, len(keys)/kvDim); e != nil {
		return e
	}
	return r.do(func() error {
		lo, k, v := base, keys, vals
		pass := r.kvUpload
		var w kvSlotWrite
		if pass != nil {
			w = pass.w // one write with the other layers: EndKVUpload ends it
			// A sliding layer's ring can have wrapped, so it may start late, as long as its rows cover its window. A full-attention layer's run from position 0.
			end, window := base+len(keys)/kvDim, int(r.layers[layer].window)
			if err := pass.pass.CheckLayer("cuda", pass.pass.Slot(), layer, base, end, window, false); err != nil {
				return err
			}
		} else {
			if e := r.kvWriteRefused(r.kvSlot, base); e != nil {
				return e
			}
			w = r.kvWriteBegin(r.kvSlot, base)
			r.kvWriteForget(w)
		}
		failed := func(e error) error {
			if pass == nil {
				r.kvWriteFailed(w, base)
			}
			return e
		}
		if r.layers[layer].kvWin {
			// a windowed layer keeps only the last window-1 positions (kvwindow.go)
			k, v = r.uploadKVWindowed(layer, base, keys, vals)
			lo = r.kvBases[r.kvSlot]
		}
		hi := base + len(keys)/kvDim
		s, g := r.kvSlot, r.kvGroupOf(layer)
		e := r.kvMap(s, g, lo, hi)
		if e == nil {
			e = r.kvFlushTabs(s)
		}
		if e != nil {
			return failed(e)
		}
		elem := r.kvElem(layer)
		var kh, vh []uint16
		if r.layers[layer].kvHalf {
			// The cache holds the nearest f16 of each value, ties to even: the rounding the _kv16 store kernels apply to what they write.
			kh, vh = make([]uint16, len(k)), make([]uint16, len(v))
			wpack.F32SliceToF16RNE(kh, k)
			wpack.F32SliceToF16RNE(vh, v)
		}
		if kvUploadFailForTest != nil {
			if e := kvUploadFailForTest(layer); e != nil {
				return failed(e)
			}
		}
		e = r.kvRowRuns(s, g, lo, hi, func(p, n, row int) error {
			src, dst := (p-lo)*kvDim, row*kvDim*elem
			if kh != nil {
				if e := gpu.Upload(r.kc[layer].At(dst), kh[src:src+n*kvDim]); e != nil {
					return e
				}
				if r.vc[layer].Len() == 0 {
					return nil
				}
				return gpu.Upload(r.vc[layer].At(dst), vh[src:src+n*kvDim])
			}
			if e := gpu.Upload(r.kc[layer].At(dst), k[src:src+n*kvDim]); e != nil {
				return e
			}
			if r.vc[layer].Len() == 0 {
				return nil // an MLA latent layer has no V
			}
			return gpu.Upload(r.vc[layer].At(dst), v[src:src+n*kvDim])
		})
		if e != nil {
			return failed(e)
		}
		if pass != nil {
			pass.pass.RecordLayer(layer, base, hi, int(r.layers[layer].window))
		} else {
			r.kvWriteDone(w, hi)
		}
		return nil
	})
}

// TruncateTo is a no-op: KV is positional and Forward sets nKeys=pos+1, so entries past pos
// are never read and get overwritten (matches the WebGPU path).
func (r *cudaResident) TruncateTo(pos int) {}

// KVSlots / UseKVSlot implement decoder.ResidentKVSubsystem: how many
// resident KV slots the build allocated, and binding one.
//
// Binding selects the slot's tables, done on the executor so it is ordered after every job already posted. That is
// the whole switch because nothing holds a slot's tables across calls: every launch that reads or writes KV (rope_kv,
// the attention kernels, the batched prefill, the MLA latent store) takes kvT(l) when it is issued, and the CUDA
// graphs (GOINFER_CUDA_GRAPHS) capture only segA/B/C, which touch no KV: rope_kv and attention run live in the gap
// between them. A recurrent layer's window and state are the slot's own (allocRecurrentSlots): a launch takes them when it is
// issued, so binding points the layer at them, and marked rows fold into the slot that marked them first.
func (r *cudaResident) KVSlots() int { return max(1, len(r.kvTabs)) }

// UseKVSlot binds slot i for the calls that follow. The binding shields nothing: kvEvictIdle releases an idle bound slot like any other,
// and only a running generation (SetKVSlotBusy) or the slot being written is skipped.
func (r *cudaResident) UseKVSlot(i int) error {
	if i < 0 || i >= r.KVSlots() {
		return fmt.Errorf("cuda: KV slot %d out of range (%d allocated)", i, r.KVSlots())
	}
	if i == r.kvSlot {
		return nil
	}
	if e := r.FlushFold(); e != nil {
		return e
	}
	return r.do(func() error {
		r.kvSlot, r.kvTabSlot = i, i
		for l := range r.layers {
			if L := &r.layers[l]; i < len(L.recSlots) {
				L.bindRecurrent(L.recSlots[i])
			}
		}
		if a := r.dnSpec; a != nil {
			a.avail = 0 // the last verify's rows are the slot's being left
		}
		return nil
	})
}
