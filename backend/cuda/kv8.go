//go:build cuda

package cuda

import (
	"fmt"
	"slices"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
)

// The int8 KV cache (docs/measurements/cuda-kv-i8-2026-11/scope.md). Each owning layer holds K and V as int8 in its pools (kc, vc) and one f32
// scale per (pool row, KV head) for each in ksc and vsc, laid out like the cache so the block table addresses both. The readers are the
// _kv8 module twins, which take the two scale pools as trailing arguments (kvtype.cuh's KV_SCALE_PARAMS); the rope kernels' int8 builds
// rotate K in place and store nothing, and kv8_store quantizes the rotated K and the V into the pools after them (kvStore8).

// kv8Unsupported is why this resident cannot run on an int8 cache, "" when it can: a layer kind whose kernels have no int8 twin.
// decoder.CUDAKVI8Decline refuses every family that has one before a build starts; this keeps a build that reached here anyway from
// binding an f32 kernel to int8 pools.
func (r *cudaResident) kv8Unsupported() string {
	switch {
	case r.isMLA:
		return "the latent (MLA) cache"
	case r.pairwiseRoPE:
		return "pairwise RoPE"
	case r.attnCap != 0:
		return "an attention softcap"
	case r.eModel:
		return "a Gemma 4 E-model"
	}
	for l := range r.layers {
		Ly := &r.layers[l]
		switch {
		case Ly.noKV():
		case Ly.kvShared:
			return fmt.Sprintf("layer %d's shared KV", l)
		case Ly.kEqV:
			return fmt.Sprintf("layer %d's K=V cache", l)
		case Ly.window > 0:
			return fmt.Sprintf("layer %d's sliding window", l)
		case Ly.vNorm:
			return fmt.Sprintf("layer %d's v_norm", l)
		}
	}
	return ""
}

// kvScaled appends layer l's K and V scale pools to a reader's arguments on an int8 cache (KV_SCALE_PARAMS, after the kernel's own trailing
// parameters, a softcap included); on any other cache it returns args unchanged.
func (r *cudaResident) kvScaled(l int, args ...gpu.KernelArg) []gpu.KernelArg {
	if !r.kvI8 {
		return args
	}
	return append(slices.Clone(args), Arg(r.ksc[l]), Arg(r.vsc[l]))
}

// kvStore8 quantizes rows rows of rotated K (k) and V (v), [rows][nKV*hd] f32, into layer l's int8 pools at absolute positions startPos..,
// through the slot table the kernels are addressing (kvT), with one scale per (row, head) each: what an f32 build's rope kernel stores
// itself. A no-op on any other cache, or for a layer that stores nothing (nKV 0).
func (r *cudaResident) kvStore8(l int, k, v Buffer, nKV, hd, startPos, rows int) error {
	if !r.kvI8 || nKV == 0 || rows == 0 {
		return nil
	}
	return r.launch(r.kv8Store, LaunchConfig{GridX: uint32(nKV), GridY: uint32(rows), GridZ: 1, BlockX: 32, BlockY: 1, BlockZ: 1},
		Arg(k), Arg(v), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.ksc[l]), Arg(r.vsc[l]), Arg(r.kvT(l)),
		gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rows)))
}

// loadKV8 binds the int8 cache's store; called by the build once the layers are known, only on an int8 cache.
func (r *cudaResident) loadKV8() error {
	if why := r.kv8Unsupported(); why != "" {
		return fmt.Errorf("cuda: --kv i8: %s has no int8 KV kernel", why)
	}
	mod, e := r.dev.CompileLibrary(kv8StorePTX)
	if e != nil {
		return fmt.Errorf("cuda: JIT kv8_store.ptx: %w", e)
	}
	if r.kv8Store, e = r.dev.NewComputePipeline(mod, "kv8_store"); e != nil {
		return e
	}
	return nil
}

// compileKV loads a KV-touching module picked by kvPTX, kvPTX8 or attnPTX; nil (no twin at this cache's precision) is an error, never a
// load of empty PTX.
func (r *cudaResident) compileKV(ptx []byte) (gpu.Library, error) {
	if len(ptx) == 0 {
		return gpu.Library{}, fmt.Errorf("cuda: this module has no build for the resident's KV precision (int8 %v, f16 %v)", r.kvI8, r.kvF16)
	}
	return r.dev.CompileLibrary(ptx)
}

// errKV8OneSlot is why an int8 cache declines what moves rows out of their slot (a copy to another slot, a host snapshot, a demotion): the
// scale pools would have to travel with them, which only the copy-on-write block copy does so far. The decoder recomputes instead.
const errKV8OneSlot = "an int8 KV cache stays in its one slot: copies and host snapshots of it are not built yet"
