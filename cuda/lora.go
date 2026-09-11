//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"unsafe"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// Compute-time LoRA on the resident path (G3, docs/task-gpu-paths-2026-09.md).
//
// Architecturally this backend is like Metal, not WebGPU: launchToken re-issues live kernel
// launches every token (segA/segB/segBFFN), so binding/clearing an adapter is just a Go-side
// `if r.loraLayers != nil` at each of the 7 projection sites — no fixed dispatch-plan surgery
// needed (contrast gpu/lora_resident.go's rebuildSteps, forced by WebGPU's fixed-plan-replay
// architecture). The one thing THIS backend has that neither of the others does is CUDA graphs
// (r.graphs, opt-in via GOINFER_CUDA_GRAPHS, off by default): a captured graph replays the exact
// launches recorded at capture time, which happened with r.loraLayers == nil, so a graph replay
// would silently skip every LoRA dispatch — SetAdapter refuses rather than bind under graphs (see
// resident.go's r.loraLayers field comment).
//
// The delta itself, and its two-kernel down/up split, mirror metal/lora.go and gpu/lora.go
// exactly: y[o] += scale·Σ_r B[o,r]·(A·x)[r], applied AFTER the base projection's matmul on the
// SAME quantized activation the base projection consumed.

// loraRMax bounds the LoRA rank this backend accepts — generous against real adapters
// (typically rank 4-64, rarely above 128). SetAdapter declines (does not silently truncate) a
// projection whose rank exceeds it.
const loraRMax = 256

// cudaLoraProj is one projection's bound delta: the device-resident A[R,In]/B[Out,R] buffers
// (row-major f32, PEFT's own layout) plus the Go-side scalars applyLora needs to size its two
// launches. Built fresh per SetAdapter call.
type cudaLoraProj struct {
	a, b  Buffer
	rank  int
	outn  int
	scale float32
}

// cudaLoraLayer is one transformer layer's per-projection bound deltas; a nil field is a
// projection this adapter does not target, matching decoder.ResidentAdapterLayer's own
// nil-means-untargeted convention exactly.
type cudaLoraLayer struct {
	q, k, v, o, gate, up, down *cudaLoraProj
}

func projForCUDA(l *decoder.ResidentAdapterLayer) [7]*decoder.ResidentAdapterProj {
	return [7]*decoder.ResidentAdapterProj{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down}
}

// releaseLoraLayers frees an explicit set of adapter buffers. This backend's Device ledger
// normally frees every allocation only at Close (ReleaseObjects); r.dev.ReleaseBuf exists for a
// dynamic allocation like an adapter's, which a different adapter's bind must return rather than
// accumulate. Used for the evicted cache entry and for a partially-built set on a failed bind.
func (r *cudaResident) releaseLoraLayers(layers []cudaLoraLayer) {
	for _, l := range layers {
		for _, p := range []*cudaLoraProj{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if p == nil {
				continue
			}
			r.dev.ReleaseBuf(p.a)
			r.dev.ReleaseBuf(p.b)
		}
	}
}

// loraCacheDisabled is the escape hatch / A-B switch for the adapter device cache (audit P-10),
// same convention as GOINFER_NO_RESIDENT_REUSE: with it set, every bind uploads and every clear
// frees, which is exactly the pre-cache behaviour.
func loraCacheDisabled() bool { return os.Getenv("GOINFER_NO_LORA_CACHE") != "" }

// loraKeyOf records what a bind would upload, projection by projection.
func loraKeyOf(layers []decoder.ResidentAdapterLayer) [][7]loraProjKey {
	key := make([][7]loraProjKey, len(layers))
	for i := range layers {
		for j, p := range projForCUDA(&layers[i]) {
			if p != nil {
				key[i][j] = loraProjKey{a: p.A, b: p.B, r: p.R, in: p.In, out: p.Out, scale: p.Scale}
			}
		}
	}
	return key
}

// sameLoraKey reports whether two keys name the same adapter: every projection's A and B are the
// SAME backing arrays (not merely equal contents — identity, which is what makes the cached
// device copy valid) with the same shape and scale.
func sameLoraKey(x, y [][7]loraProjKey) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		for j := range x[i] {
			p, q := &x[i][j], &y[i][j]
			if unsafe.SliceData(p.a) != unsafe.SliceData(q.a) || len(p.a) != len(q.a) ||
				unsafe.SliceData(p.b) != unsafe.SliceData(q.b) || len(p.b) != len(q.b) ||
				p.r != q.r || p.in != q.in || p.out != q.out || p.scale != q.scale {
				return false
			}
		}
	}
	return true
}

// SetAdapter implements decoder.ResidentAdapter: binds (layers != nil) or clears (layers == nil)
// the compute-time LoRA delta for every subsequent Forward call, until the next SetAdapter. Runs on
// the executor thread (r.do) like every other device-touching call in this file — CUDA contexts
// are thread-affine, and BuildResident's own setup follows the same rule.
//
// Clearing UNBINDS but keeps the uploaded buffers (audit P-10), so the next bind of the same
// adapter is a pointer swap rather than a full re-upload; a different adapter evicts them. The
// cache holds one adapter — the most recently uploaded — so it adds no VRAM beyond what a bound
// adapter already costs, and adapters that alternate still upload each time, as before.
func (r *cudaResident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return r.do(func() error {
		r.loraLayers = nil // unbind first: every error below leaves NO adapter bound
		if layers == nil {
			if loraCacheDisabled() {
				r.releaseLoraLayers(r.lora.cache)
				r.lora.cache, r.lora.key = nil, nil
			}
			return nil
		}
		if r.graphs {
			return fmt.Errorf("cuda: SetAdapter declined — this resident was built with CUDA " +
				"graphs enabled (GOINFER_CUDA_GRAPHS); a captured graph replays the exact launches " +
				"recorded without an adapter bound, so binding one here would silently apply no " +
				"delta. Disable GOINFER_CUDA_GRAPHS to use compute-time LoRA on this backend")
		}
		if len(layers) != len(r.layers) {
			return fmt.Errorf("cuda: SetAdapter got %d layers, model has %d", len(layers), len(r.layers))
		}
		key := loraKeyOf(layers)
		if r.lora.cache != nil && !loraCacheDisabled() && sameLoraKey(key, r.lora.key) {
			r.lora.hits++
			r.loraLayers = r.lora.cache
			return nil
		}
		r.releaseLoraLayers(r.lora.cache) // a different adapter: return the cached one first
		r.lora.cache, r.lora.key = nil, nil
		built := make([]cudaLoraLayer, len(layers))
		for i := range layers {
			projs := projForCUDA(&layers[i])
			dst := []**cudaLoraProj{&built[i].q, &built[i].k, &built[i].v, &built[i].o, &built[i].gate, &built[i].up, &built[i].down}
			for j, proj := range projs {
				if proj == nil {
					continue
				}
				if proj.R <= 0 || proj.R > loraRMax {
					r.releaseLoraLayers(built[:i+1])
					return fmt.Errorf("cuda: SetAdapter: LoRA rank %d out of range (want 1..%d)", proj.R, loraRMax)
				}
				*dst[j] = &cudaLoraProj{a: r.up32(proj.A), b: r.up32(proj.B), rank: proj.R, outn: proj.Out, scale: proj.Scale}
				if r.setupErr != nil {
					err := r.setupErr
					r.setupErr = nil
					r.releaseLoraLayers(built[:i+1])
					return fmt.Errorf("cuda: SetAdapter: uploading layer %d projection %d: %w", i, j, err)
				}
			}
		}
		r.lora.uploads++
		r.lora.cache, r.lora.key = built, key
		r.loraLayers = built
		return nil
	})
}

// applyLora dispatches one projection's compute-time LoRA delta into dst, ADDITIVELY — no-op if
// p is nil (an untargeted projection, or no adapter bound at all). aq/ascale must be the SAME
// quantized activation the base projection this delta rides alongside already consumed; dst
// must already hold the base projection's result — the delta is added on top, matching
// applyLoRA's CPU "matmul, then add" order exactly.
func (r *cudaResident) applyLora(p *cudaLoraProj, aq, ascale Buffer, k int, dst Buffer) error {
	if p == nil {
		return nil
	}
	if err := r.launch(r.fLoraDown, loraDownCfg(p.rank),
		Arg(aq), Arg(ascale), Arg(p.a), Arg(r.loraT),
		gpu.ArgValue(int32(k)), gpu.ArgValue(int32(p.rank))); err != nil {
		return err
	}
	return r.launch(r.fLoraUp, g1cfg(p.outn, 256),
		Arg(p.b), Arg(r.loraT), Arg(dst),
		gpu.ArgValue(int32(p.rank)), gpu.ArgValue(int32(p.outn)), gpu.ArgValue(p.scale))
}

// loraCacheState is the adapter device cache (audit P-10). generateInto binds an adapter before
// every adapter generation and clears it after, and SetAdapter used to upload every A/B matrix
// and free them all again each time — ~90 MB and ~400 allocations per request on a 7B at rank 16,
// before the first token. The cache keeps the most recently bound adapter's buffers across a
// clear, so repeated requests to one fine-tune (an agent loop) upload it once.
type loraCacheState struct {
	cache   []cudaLoraLayer  // device buffers of the most recently uploaded adapter
	key     [][7]loraProjKey // what those buffers were uploaded FROM
	uploads uint64           // binds that transferred an adapter to the device
	hits    uint64           // binds that reused the cached one
}

// loraProjKey is one projection's identity. It holds the backing slices themselves — not just
// their addresses — so the GC cannot free and reuse that memory while it is cached: the slices
// alias the adapter's own loraDelta storage (decoder/residency.go residentAdapterProj), which a
// fresh LoadAdapter never shares, even under the same name. The scalars are compared too.
type loraProjKey struct {
	a, b       []float32
	r, in, out int
	scale      float32
}

// loraDownCfg is lora_delta_down's launch shape (audit P-11): one 256-thread block per rank, so the
// R reductions run side by side instead of one after another in a single block. loraDownSerial
// restores the old single-block shape. It exists so a test can prove the two shapes bit-identical,
// and so the measurement can A/B them on one resident.
func loraDownCfg(rank int) LaunchConfig {
	cfg := onecfg(256, 256*4)
	if !loraDownSerial {
		cfg.GridX = uint32(rank)
	}
	return cfg
}

// loraDownSerial is test-only. Tests set it between Forward calls, which reach the executor
// through r.do's channel, so the write happens-before the launch that reads it.
var loraDownSerial bool
