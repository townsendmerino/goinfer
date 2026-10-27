//go:build cuda

package cuda

import "github.com/townsendmerino/goinfer/decoder"

// The CUDA build's account of a load, before the build, for the fit guard (decoder.BuildDemand). Which architectures run
// resident is decoder's feature table (BuildQuery.FeaturesCovered); the bytes are this package's.
func init() { decoder.RegisterBuildDemand("cuda", buildDemand) }

// packStageHostBytes is the host staging BuildResident streams packed weights through: up to four blocks of stagePackWords
// words (8 MiB) with their scale scratch, 20 MiB each.
const packStageHostBytes = 4 * (20 << 20)

// pinnedExpertNum and pinnedExpertDen are a pinned expert copy's bytes per weight: int4 with f16 scales per group of 32,
// 9/16 of a byte.
const pinnedExpertNum, pinnedExpertDen = 9, 16

// buildDemand: resident when the load is CUDA's, does not stream, and the architecture is covered; its KV is then device
// memory, and packing adds one packed layer beside the weights and the staging. A C' expert cache built from a .gguf holds the
// weights, a packed copy and a pinned copy of the experts at once; a .giw load's cache pins every routed expert.
func buildDemand(q decoder.BuildQuery) decoder.BuildDemand {
	o := q.Options
	var d decoder.BuildDemand
	d.Resident = o.Backend == "cuda" && !o.StreamWeights && q.FeaturesCovered
	d.KVOnDevice = d.Resident
	if d.Resident {
		d.HostPackBytes = q.WeightBytes/int64(max(q.Layers, 1)) + packStageHostBytes
	}
	if o.Backend == "cuda" && q.GGUF && !o.StreamWeights && q.ExpertBytes > 0 && q.ExpertCache {
		d.HostBuildPeak = 2*q.WeightBytes + q.ExpertBytes
	}
	if q.ExpertCache && d.Resident && q.NumExperts > 0 && q.ExpertIntermediate > 0 {
		d.PinnedExpertBytes = int64(q.MoELayers) * int64(q.NumExperts) * 3 * int64(q.ExpertIntermediate) * int64(q.Hidden) *
			pinnedExpertNum / pinnedExpertDen
	}
	return d
}
