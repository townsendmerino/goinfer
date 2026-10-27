//go:build darwin

package metal

import "github.com/townsendmerino/goinfer/decoder"

// The Metal build's account of a load, before the build, for the fit guard (decoder.BuildDemand): resident when the load is
// Metal's and the architecture is covered (decoder's feature table). Its KV is unified memory, which the fit guard prices as
// Metal allocates it, and the build adds no host term the guard prices beside the weights.
func init() { decoder.RegisterBuildDemand("metal", buildDemand) }

func buildDemand(q decoder.BuildQuery) decoder.BuildDemand {
	return decoder.BuildDemand{Resident: q.Options.Backend == "metal" && q.FeaturesCovered}
}
