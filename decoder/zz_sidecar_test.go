package decoder

import (
	"context"
	"os"

	"github.com/townsendmerino/goinfer/internal/runtime/sidecar"
)

type sidecarTestHooks = sidecar.TestHooks

const giwReplacedInfix = sidecar.ReplacedInfix

func publishGIW(tmp, out string) error { return sidecar.PublishGIW(sidecarOps(), tmp, out) }
func gatherTableTag(src, quant string, embedInt4 bool) string {
	return sidecar.GatherTableTag(sidecarOps(), src, quant, embedInt4)
}
func streamCachePath(ggufPath, quant string, embedInt4 bool, target GIWTarget) string {
	return sidecar.StreamCachePath(sidecarOps(), ggufPath, quant, embedInt4, target)
}
func legacyCachePath(ggufPath, quant string, embedInt4 bool, target GIWTarget) string {
	return sidecar.LegacyCachePath(sidecarOps(), ggufPath, quant, embedInt4, target)
}
func projectedSidecarBytes(ggufPath, quant string) (int64, bool) {
	return sidecar.ProjectedSidecarBytes(sidecarOps(), ggufPath, quant)
}
func projectedDirSidecarBytes(dir, quant string) (int64, bool) {
	return sidecar.ProjectedDirSidecarBytes(dir, quant)
}
func withSidecarTestHooks(ctx context.Context, h sidecarTestHooks) context.Context {
	return sidecar.WithTestHooks(ctx, h)
}

func cacheNewer(cache, src string) bool   { return sidecar.CacheNewer(sidecarOps(), cache, src) }
func cacheFresh(cache, src string) bool   { return sidecar.CacheFresh(sidecarOps(), cache, src) }
func carryVerifiedMarker(tmp, out string) { sidecar.CarryVerifiedMarker(sidecarOps(), tmp, out) }
func removeTempGIW(tmp string)            { sidecar.RemoveTempGIW(sidecarOps(), tmp) }
func transcodeDir(ctx context.Context, dir, lora, out, quant string, embedInt4 bool, target GIWTarget) (TranscodePhases, error) {
	return sidecar.TranscodeDir(sidecarOps(), ctx, dir, lora, out, quant, embedInt4, target)
}

var errReadBack = sidecar.ErrReadBack

func sourceBytes(source string) int64 { return sidecar.SourceBytes(source) }
func startReadBack(f *os.File, bodyStart int64) *sidecar.ReadBack {
	return sidecar.StartReadBack(f, bodyStart)
}
