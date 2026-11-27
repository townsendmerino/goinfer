package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/paging"
)

type (
	expertPager      = paging.ExpertPager
	expertBufferPool = paging.ExpertBufferPool
	layerPager       = paging.LayerPager
	poolMember       = paging.PoolMember
	expertField      = paging.ExpertField
)

// MoEPagerDefault is the registered default pager mode for goos ("pool" on darwin, "mmap" elsewhere).
func MoEPagerDefault(goos string) string { return paging.DefaultMoEPagerMode(goos) }

func resolveMoEPagerPool(opt, preadCPU string) bool { return paging.UseMoEPagerPool(opt, preadCPU) }

func newExpertPager(w *darch.Weights, mapping []byte, budget int64, giwPath string, pool bool) *expertPager {
	return paging.NewExpertPager(w, mapping, budget, giwPath, pool, processFaultCounts)
}

func newLayerPager(w *darch.Weights, mapping []byte, budget int64) *layerPager {
	return paging.OpenLayerPager(w, mapping, budget)
}

func pagerSummary(p *expertPager) string     { return paging.ExpertPagerSummary(p) }
func layerPagerSummary(p *layerPager) string { return paging.SummarizeLayerPager(p) }
