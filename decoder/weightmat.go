package decoder

import (
	"os"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

type quantMode = wbuild.QuantMode

const (
	quantNone    = wbuild.QuantNone
	quantInt8    = wbuild.QuantInt8
	quantInt4    = wbuild.QuantInt4
	quantInt8I8  = wbuild.QuantInt8I8
	quantInt4Mix = wbuild.QuantInt4Mix
	quantQ4K     = wbuild.QuantQ4K
	quantBF16    = wbuild.QuantBF16
	quantF16     = wbuild.QuantF16

	int4GroupSize = wbuild.Int4GroupSize
)

// int4ParThreshold sets the fan-out threshold for the int4 (W4A8 and Q4_K) matmul below aikit's default (parThreshold =
// 1<<24 MACs) so the small decode matmuls parallelize. aikit runs a call serial when M*N*K (M*totalN*K for a batch) is
// under it. 1<<19 lets the smallest projections of the sub-1B models fan out (qwen2.5 0.5B's q/k/v batch, 1,032,192 MACs,
// and its o projection, 802,816), while ops under 512K MACs stay serial; threshold 0, which fans out everything,
// over-parallelizes on the 8-core Ryzen. TestInt4ParThreshold_qwen05bShapesFanOut pins the 0.5B shapes.
//
// Output is byte-identical (aikit partitions output columns in 8-wide groups, the width-invariant contract), and it only
// widens fan-out, so prefill's already-parallel large-M matmuls are unaffected. Re-run BenchmarkInt4ParThresholdSweep if
// the core topology or aikit's kernel changes. Measurements on both rigs: docs/code-notes/decoder.md#int4ParThreshold.
const int4ParThreshold = cpuengine.Int4ParThreshold

// w4a8SplitHalfRepacked, w4a8SplitHalfSkipped and w4a8SplitHalfBytes count what the repack did, at load only (one atomic
// add per weight tensor). They exist because the repack is otherwise silent: it returns a bool nobody reads, and a wiring
// that repacked nothing (wrong quant, wrong load path, a shape rule rejecting every tensor) would yield a benchmark that
// measures no difference and gets recorded as flat. An A/B against this repack must read these first and confirm the
// repacked count is non-zero.
var w4a8SplitHalfRepacked, w4a8SplitHalfSkipped, w4a8SplitHalfBytes atomic.Int64

// w4a8SplitHalfRepackEnabled is default-off by decision, not caution; GOINFER_W4A8_SPLITHALF=1 opts in. The repack is a
// second copy of every eligible tensor's nibbles (+0.5 byte per weight on top of an int4 tensor's 0.625, about +80% int4
// weight bytes), and canonical is never dropped (M>1 prefill and every non-AVX2 path read it). Its decode gain landed in
// the band the pre-registration named in advance as ambiguous, so it is parked with the code and wiring intact.
//
// Turning it on is defensible where decode latency outranks resident memory, on amd64 with AVX2 and no AVX-512 VNNI
// (aikit declines on VNNI hosts). It is not defensible as a default. Re-open the decision if the kernel gets faster or
// canonical can be dropped for a decode-only build. The A/B and the figures:
// docs/measurements/w4a8-splithalf-decode-ab-PREREGISTERED.md and docs/code-notes/decoder.md#w4a8SplitHalfRepackEnabled.
var w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""

// w4a8BatchEnabled runs a layer's q/k/v (and gate/up, where the fused gate+up does not take them) as one W4A8 fork/join
// instead of one per projection (aikit MatmulBTBatchInto). Bit-identical: every output column is the same dot product
// either way. The default is per architecture (cpuengine.W4A8BatchDefault, cpu_tuning_{arm64,other}.go); GOINFER_W4A8_BATCH=0
// opts out and =1 forces it on. Why the defaults differ: docs/code-notes/decoder.md#w4a8BatchEnabled.
var w4a8BatchEnabled = envBoolDefault("GOINFER_W4A8_BATCH", cpuengine.W4A8BatchDefault)

// row4RepackCallsForTest, when set, counts the tensors offered to the in-RAM row4 repack.
var row4RepackCallsForTest *atomic.Int64

// w4a8Row4RepackEnabled is a load-time-measurement toggle only: production never sets it, so it stays true in a real
// build. A test measuring the repack's load-time and resident-memory delta flips it off to get a baseline from the same
// load path.
var w4a8Row4RepackEnabled = true

// wantsCanonicalInt4 reports whether this load might need canonical int4 bytes (packed nibbles + scales) for a tensor,
// that is, whether the repacked-only layout is unsafe. Two consumers read them directly rather than through WeightMat's
// layout-agnostic methods:
//
//   - The staged per-token GPU consult (QuantBackend4.MatmulW4A8, QuantBatchBackend4.MatmulW4A8Batch in matmul,
//     matmulInto, matmulW4A8Batch) hands q4/q4s to the backend on every call. A repacked-only tensor has nothing to hand
//     it, and rebuilding canonical per token is the wrong shape for a per-call path.
//   - A resident GPU build (ResidencyBackend.BuildResident) reads w.Int4() directly to upload once at build time.
//
// backendName is Options.Backend as the caller wrote it, not be's resolved capabilities. Repacked-only is a promise that
// no GPU backend will ever touch this *Model, resident or staged, and a promise must be stated, never inferred from an
// omission: only the literal "cpu" states it. An empty name promises nothing, because a generic load may later be handed
// to any backend's resident-build machinery outside decoder.Load's dispatch (metal's tests do exactly this), and keying
// on be's interfaces alone sees the plain CPU backend, which implements none of them, and calls it safe.
//
// The interface assertions are a second guard for the "cpu" case itself: a "cpu" backend that implemented one of these
// would still decline repacked-only. Every other backendName returns true, so canonical(+row4) stays the default.
func wantsCanonicalInt4(backendName string, be Backend) bool {
	if backendName != "cpu" {
		return true
	}
	if _, ok := be.(QuantBackend4); ok {
		return true
	}
	if _, ok := be.(QuantBatchBackend4); ok {
		return true
	}
	_, ok := be.(ResidencyBackend)
	return ok
}

// GIWTarget names the single consumer a .giw bundle is built for.
type GIWTarget = giwcodec.Target

const (
	GIWTargetNone     = giwcodec.TargetNone
	GIWTargetCPUArm64 = giwcodec.TargetCPUArm64
	GIWTargetCPUAmd64 = giwcodec.TargetCPUAmd64
	GIWTargetMetal    = giwcodec.TargetMetal
	GIWTargetCUDA     = giwcodec.TargetCUDA
	GIWTargetWebGPU   = giwcodec.TargetWebGPU
)

// GIWTargetForBackend derives the GIWTarget for a caller naming backendName.
func GIWTargetForBackend(backendName string) GIWTarget {
	return giwcodec.TargetForBackend(backendName, linalg.PreferredInt4Layout(4, 32, 32) == linalg.LayoutRow4)
}

// ParseGIWTarget parses a -target flag value.
func ParseGIWTarget(s string) (GIWTarget, error) {
	return giwcodec.ParseTarget(s, linalg.PreferredInt4Layout(4, 32, 32) == linalg.LayoutRow4)
}

var (
	w4a8PreOff   = false
	w4a8PreCalls atomic.Int64
)

// Int4F32 returns w's canonical int4 nibbles and its binary16 group scales widened, exactly, to f32 in a NEW
// slice — for load-time consumers written against f32 scales (GPU uploads, the .giw writer). ok and group are
// linalg.WeightMat.Int4F16's. It allocates, so it is not for a per-token path; decode reads Int4F16.
func Int4F32(w *linalg.WeightMat) (q4 []byte, scales []float32, group int, ok bool) {
	q4, s16, group, ok := w.Int4F16()
	if s16 != nil {
		scales = make([]float32, len(s16))
		linalg.F16ToF32Slice(scales, s16)
	}
	return q4, scales, group, ok
}
