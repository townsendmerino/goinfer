package memprice

import (
	"fmt"
	"slices"
	"strings"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

const (
	// WindowedKVSlack is the headroom, in positions, a windowed-KV layer holds beyond its window.
	WindowedKVSlack = 512

	// CUDAPackStageBytes is the host staging memory the CUDA build streams packed weights through (four 20 MiB blocks).
	CUDAPackStageBytes = 80 << 20
)

// CUDAChargedBytes returns what CUDA's allocator charges for a buffer of n bytes (2 MiB quantum at or above 2 MiB).
func CUDAChargedBytes(n int64) int64 {
	const quantum = 2 << 20
	if n < quantum {
		return n
	}
	return (n + quantum - 1) / quantum * quantum
}

// CUDAF16Delivered reports whether every KV-touching CUDA path has an f16 kernel for f's family.
func CUDAF16Delivered(f darch.KVFacts) bool {
	switch f.Family {
	case "llama", "mistral", "qwen2", "qwen3", "phi3", "qwen2_moe", "qwen3_moe", "mixtral", "gemma2", "gemma3", "gpt-oss", "cohere",
		"cohere2", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen3_next", "gemma4", "deepseek_v2", "deepseek_v3", "kimi_k2",
		"nemotron_h", "flint":
		return true
	}
	return false
}

// CUDAF16AutoCleared reports whether --kv auto puts a delivered model on f16 KV on CUDA.
func CUDAF16AutoCleared(f darch.KVFacts) bool {
	if f.SlidingWindow || f.PairwiseRoPE || f.Hybrid || f.MLA {
		return false
	}
	switch f.Family {
	case "llama", "mistral", "qwen2", "qwen3", "phi3", "qwen2_5_vl", "qwen3_vl", "qwen2_moe", "qwen3_moe", "mixtral", "qwen3_vl_moe":
		return true
	}
	return false
}

// CUDAF16Decline returns why a CUDA resident cannot hold f's KV cache as f16, or "" when it can.
func CUDAF16Decline(f darch.KVFacts) string {
	if !CUDAF16Delivered(f) {
		return "f16 KV on CUDA covers llama, mistral, qwen2, qwen3, phi3, qwen2_moe, qwen3_moe, mixtral, gemma2, gemma3, gpt-oss, cohere, cohere2, " +
			"qwen2_5_vl, qwen3_vl, qwen3_vl_moe, qwen3_5, qwen3_5_moe, qwen3_next, gemma4, deepseek_v2, deepseek_v3, kimi_k2, nemotron_h and flint so far; " +
			f.Family + " keeps f32"
	}
	return ""
}

// CUDAF16AutoDecline returns why --kv auto keeps a model CUDAF16Decline admits at f32 on CUDA, or "" when auto picks f16.
func CUDAF16AutoDecline(f darch.KVFacts) string {
	switch {
	case CUDAF16AutoCleared(f):
		return ""
	case f.Hybrid:
		return "f16 KV is delivered on request (--kv f16); auto keeps f32: the hybrids' quality gate did not pass (mean KL 1.21x f32's at 3,900 keys)"
	case f.Family == "gpt-oss":
		return "f16 KV is delivered on request (--kv f16); auto keeps f32: gpt-oss's attention sinks have no quality gate yet (a sinks G4 cell would clear it)"
	case f.PairwiseRoPE:
		return "f16 KV is delivered on request (--kv f16); auto keeps f32: the pairwise-RoPE quality gate did not pass at 8,000 keys (a passing K=8,000 cell would clear it)"
	case f.SlidingWindow:
		return "f16 KV is delivered on request (--kv f16); auto keeps f32: the sliding-window quality gate did not pass at 8,000 keys (mean KL 1.16x f32's over ten prompts, against a 1.10x bar; a passing K=8,000 cell would clear it)"
	case f.Family == "nemotron_h":
		return "f16 KV is delivered on request (--kv f16); auto keeps f32: no quality gate has read a Mamba-2 hybrid yet (a passing G4 cell would clear it)"
	}
	return "f16 KV is delivered on request (--kv f16) for this build; auto keeps f32 until its quality gate passes"
}

var cudaKVI8Families = []string{"llama", "mistral", "qwen2", "qwen3", "phi3"}

// CUDAI8Decline returns why a CUDA resident cannot hold f's KV cache as int8, or "" when it can.
func CUDAI8Decline(f darch.KVFacts) string {
	if !slices.Contains(cudaKVI8Families, f.Family) {
		return "int8 KV on CUDA covers the dense families (" + strings.Join(cudaKVI8Families, ", ") + "); " + f.Family + " is not among them (widening to other families is parked)"
	}
	switch {
	case f.SlidingWindow:
		return "int8 KV on CUDA does not cover sliding-window layers yet (the windowed ring and its rollback carry no scales)"
	case f.MoE || f.MLA || f.MRoPE || f.PairwiseRoPE || f.Hybrid:
		return "int8 KV on CUDA covers dense attention only so far"
	case f.KVRowNotWords > 0:
		return fmt.Sprintf("int8 KV on CUDA moves rows as whole 4-byte words, and this model's %d-element K/V row is not", f.KVRowNotWords)
	}
	return ""
}

// CUDAKVRowBytes returns one position's row of b in a CUDA pool at prec.
func CUDAKVRowBytes(b program.Buf, prec string) int64 {
	switch {
	case prec == mem.KVI8 && b.Prec == program.PrecKV:
		return int64(b.Elems) + 4*int64(b.Heads)
	case prec == mem.KVF16:
		return 2 * int64(b.Elems)
	}
	return 4 * int64(b.Elems)
}

// CUDAKVParts returns one slot's KV on CUDA at precision prec and its pools' null blocks over bufsOf.
func CUDAKVParts(a *darch.Architecture, ctx int, prec string, windowedKV bool, bufsOf func(l int) []program.Buf) (slot, null int64) {
	blocks := int64((ctx + kvmgr.BlockRows - 1) / kvmgr.BlockRows)
	var perBlock, windowed int64
	for l := range a.NumLayers {
		for _, b := range bufsOf(l) {
			if b.Unit != program.PerPosition {
				continue
			}
			block := kvmgr.BlockRows * CUDAKVRowBytes(b, prec)
			null += block
			if held := b.Window + WindowedKVSlack; b.Window > 0 && windowedKV && ctx > held {
				windowed += block * int64((held+kvmgr.BlockRows-1)/kvmgr.BlockRows+1)
				continue
			}
			perBlock += block
		}
	}
	return perBlock*blocks + windowed + 2*4*blocks, null
}

// CUDADerivedTableBytes returns the RoPE inverse-frequency and zero router-bias tables CUDA derives and uploads on the dense line.
func CUDADerivedTableBytes(w *darch.Weights, prog *program.Program) int64 {
	if w == nil || w.Arch == nil || prog == nil {
		return 0
	}
	layers := prog.Layers
	hybrid := false
	for _, ops := range layers {
		if x, _ := program.MixerOf(ops); x == program.MixerMamba {
			hybrid = true
		}
	}
	isMoE := w.Arch.MoE != nil
	nE := 0
	if isMoE {
		nE = w.Arch.MoE.NumExperts
	}
	var n int64
	for l, ops := range layers {
		if x, _ := program.MixerOf(ops); x == program.MixerAttention || x == program.MixerMLA {
			if hybrid {
				n += 4 * int64(LayerRotaryDim(w.Arch, l)/2)
			} else {
				n += 4 * int64(len(LayerRopeInvFreq(w.Arch, l)))
			}
		}
		if lw := &w.Layers[l]; isMoE && (len(lw.Experts) > 0 || lw.Gemma4MoE != nil) && len(lw.RouterBias) == 0 {
			n += 4 * int64(nE)
		}
	}
	return n
}

// CUDABuildQuery carries the neutral inputs to CUDABuildDemand.
type CUDABuildQuery struct {
	BackendCUDA, LinkedCUDA, StreamWeights, GGUF, FeaturesCovered bool
	ExpertCache, MoEExpertsUnpinned                               bool
	Layers                                                        int
	WeightBytes, ExpertBytes                                      int64
	MoELayers, NumExperts, ExpertIntermediate, Hidden             int
	StateBuffers, CheckpointBuffers                               []int64
}

// CUDABuildDemandResult is CUDA's pre-build host and device state demand.
type CUDABuildDemandResult struct {
	Resident, KVOnDevice, CheckpointPerSlot         bool
	HostPackBytes, HostBuildPeak, PinnedExpertBytes int64
	StateBytes, CheckpointBytes                     int64
}

// CUDABuildDemand computes CUDA's build-time host peak, pinned expert copy, and 2 MiB-rounded state/checkpoint bytes.
func CUDABuildDemand(q CUDABuildQuery) CUDABuildDemandResult {
	var d CUDABuildDemandResult
	d.Resident = q.BackendCUDA && !q.StreamWeights && q.LinkedCUDA && q.FeaturesCovered
	d.KVOnDevice = d.Resident
	if d.Resident {
		d.HostPackBytes = q.WeightBytes/int64(max(q.Layers, 1)) + CUDAPackStageBytes
	}
	if q.BackendCUDA && q.GGUF && !q.StreamWeights && q.ExpertBytes > 0 && q.ExpertCache {
		d.HostBuildPeak = q.WeightBytes + CUDAPackStageBytes
		if q.Layers > 0 {
			d.HostBuildPeak += q.WeightBytes / int64(q.Layers)
		}
		if !q.MoEExpertsUnpinned {
			d.HostBuildPeak += q.ExpertBytes
		}
	}
	if q.ExpertCache && d.Resident && q.NumExperts > 0 && q.ExpertIntermediate > 0 && !q.MoEExpertsUnpinned {
		d.PinnedExpertBytes = int64(q.MoELayers) * int64(q.NumExperts) * 3 * int64(q.ExpertIntermediate) * int64(q.Hidden) * 9 / 16
	}
	for _, b := range q.StateBuffers {
		d.StateBytes += CUDAChargedBytes(b)
	}
	for _, b := range q.CheckpointBuffers {
		d.CheckpointBytes += CUDAChargedBytes(b)
	}
	d.CheckpointPerSlot = d.CheckpointBytes > 0
	return d
}
