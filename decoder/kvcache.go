package decoder

import (
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
)

type (
	kvQuant        = cpuengine.KVQuant
	ring           = cpuengine.Ring
	deltaState     = cpuengine.DeltaState
	shortConvState = cpuengine.ShortConvState
	mamba2State    = cpuengine.Mamba2State
	kdaState       = cpuengine.KDAState
	deepstackRows  = cpuengine.DeepstackRows
	decodeScratch  = cpuengine.DecodeScratch
	attnScoreStats = cpuengine.AttnScoreStats
	seamPlant      = cpuengine.SeamPlant
	walkBatchPlant = cpuengine.WalkBatchPlant
)

const (
	kvqF32 = cpuengine.KVQF32
	kvI8   = cpuengine.KVI8
	kvqF16 = cpuengine.KVQF16

	kvRowPadAlignDefault = archparts.KVRowPadAlignDefault

	plantNone            = cpuengine.PlantNone
	plantHiddenNoNorm    = cpuengine.PlantHiddenNoNorm
	plantCaptureEarly    = cpuengine.PlantCaptureEarly
	plantSubPrePostNorm  = cpuengine.PlantSubPrePostNorm
	plantCostDoubleCount = cpuengine.PlantCostDoubleCount
	plantSkipPLECapture  = cpuengine.PlantSkipPLECapture
	plantCtxBeforeAttend = cpuengine.PlantCtxBeforeAttend
	plantCtxNarrow       = cpuengine.PlantCtxNarrow
	plantMLPPreAfterNorm = cpuengine.PlantMLPPreAfterNorm

	plantBatchNone            = cpuengine.PlantBatchNone
	plantBatchDropConvWindow  = cpuengine.PlantBatchDropConvWindow
	plantBatchNoAttnGate      = cpuengine.PlantBatchNoAttnGate
	plantBatchImagePLEToken   = cpuengine.PlantBatchImagePLEToken
	plantBatchNoBidirectional = cpuengine.PlantBatchNoBidirectional
)

var (
	kvRoundF16ForTest, kvF16NewRowsF32ForTest, kvF16RingOffsetForTest bool
	attnScoreStatsForTest                                             atomic.Pointer[attnScoreStats]
	seamPlantForTest                                                  seamPlant
	walkBatchPlantForTest                                             walkBatchPlant
	g4traceHidden                                                     func(layer int, h []float32)
	g4traceQKV                                                        func(layer int, q, k, v []float32)
)

// KVCache holds the per-sequence CPU attention KV cache and recurrent mixer states.
// It is a Hard-tier handle over internal/runtime/cpuengine.Cache; not goroutine-safe.
type KVCache struct {
	c    cpuengine.Cache
	lora *loraRuntime // active compute-time LoRA adapter (for Session/KV-key identity)
}

func unwrapCache(c *KVCache) *cpuengine.Cache {
	if c == nil {
		return nil
	}
	if c.lora != nil {
		c.c.LoRALayers = c.lora.layers
	} else if c.c.LoRALayers != nil {
		c.c.LoRALayers = nil
	}
	return &c.c
}

// NewKVCache allocates an empty cache for a model with the given geometry.
func NewKVCache(numLayers, numKVHeads, headDim, window, capHint int, skipLayer func(l int) bool) *KVCache {
	syncCPUHooks()
	return &KVCache{c: *cpuengine.NewCache(numLayers, numKVHeads, headDim, window, capHint, skipLayer)}
}

func (c *KVCache) Append(layer int, k, v []float32) int {
	syncCPUHooks()
	return c.c.Append(layer, k, v)
}
func (c *KVCache) Advance()                 { c.c.Advance() }
func (c *KVCache) Keys(layer int) []float32 { return c.c.Keys(layer) }
func (c *KVCache) Vals(layer int) []float32 { return c.c.Vals(layer) }
func (c *KVCache) LayerKV(layer int) (k, v []float32, base int) {
	return c.c.LayerKV(layer)
}
func (c *KVCache) AppendLatent(layer int, latent []float32) { c.c.AppendLatent(layer, latent) }
func (c *KVCache) Latent(layer int) []float32               { return c.c.Latent(layer) }
func (c *KVCache) Pos() int                                 { return c.c.Pos() }
func (c *KVCache) TruncateTo(pos int) (exact bool)          { syncCPUHooks(); return c.c.TruncateTo(pos) }
func (c *KVCache) WindowStart(pos int, global bool) int     { return c.c.WindowStart(pos, global) }
func (c *KVCache) SetImageBlocks(blocks [][2]int)           { c.c.SetImageBlocks(blocks) }

func (c *KVCache) setLora(rt *loraRuntime) { c.lora = rt }

func (c *KVCache) hasRecurrentState() bool { return c.c.HasRecurrentState() }
