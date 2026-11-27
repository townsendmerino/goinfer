package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/banner"

type bannerFacts = banner.Facts

const (
	harnessCtx = banner.HarnessCtx
	kvI8Cost   = banner.KVI8Cost
)

var (
	samplingLine = banner.SamplingLine
	harnessLine  = banner.HarnessLine
)

func (cfg config) bannerConfig() banner.Config {
	return banner.Config{
		Ctx:            cfg.load.Ctx,
		KVSessions:     cfg.kvSessions,
		KVSessionsSet:  cfg.kvSessionsSet,
		KVHostBytes:    cfg.kvHostBytes,
		KVHostBytesHow: cfg.kvHostBytesHow,
		KVDemote:       cfg.kvDemote,
		MaxConcurrent:  cfg.maxConcurrent,
		PrefillChunk:   cfg.prefillChunk,
		CPUBatch:       cfg.cpuBatch,
	}
}

func factsOf(lm *loadedModel) bannerFacts {
	_, prefillWhy := lm.model.PrefillPath()
	f := bannerFacts{
		DecodePath:   lm.model.DecodePath(),
		HeadTable:    lm.model.HeadTable(),
		PrefillPath:  prefillWhy,
		Resident:     lm.model.ResidentActive(),
		Snapshots:    lm.model.KVSnapshotsSupported(),
		HasTemplate:  lm.tmpl != nil,
		Spec:         lm.spec,
		BlockDrafter: lm.blockSpec != nil,
	}
	if lm.tmpl != nil {
		_, _, _, _, f.ToolCallForm = lm.tmpl.ToolCallWrapper()
	}
	f.FitCtx, f.FitKVBytes, f.FitWeightBytes, f.FitBudgetBytes, f.FitKnown = lm.model.FitBudgetSummary()
	f.CtxWindow = lm.contextWindow(lm.adapter == "")
	f.MaxPositions = lm.model.Config().MaxPositions
	f.KV = lm.model.KVPrecisionReports()
	f.CtxNote = lm.model.ResidentContextNote()
	f.HarnessKVBytes = lm.model.ResidentContextKVBytes(harnessCtx)
	f.KVSlots = lm.model.ResidentKVSlots()
	f.KVPoolDetail = lm.model.ResidentKVPool()
	f.KVShareState, f.KVShareWhy = lm.model.KVShareState()
	f.KVPool, f.KVShare = f.KVPoolDetail.Positions, f.KVPoolDetail.Guaranteed
	if lm.model.EffectiveBackend() != "cpu" {
		f.ResidentDecline = lm.model.ResidentDecline()
	}
	f.Concurrent = lm.concurrent
	f.Sampling = lm.sampling
	if v, _ := lm.model.Knob("GOINFER_NO_RESIDENT_REUSE"); v != "" {
		f.ResidentReuseOff = true
	}
	return f
}

func modelBanner(lm *loadedModel, cfg config) []string {
	f := factsOf(lm)
	f.TowerReserve = towerReserve(cfg, lm.source)
	return modelBannerFrom(f, cfg)
}

func modelBannerFrom(f bannerFacts, cfg config) []string {
	return banner.ModelLines(f, cfg.bannerConfig())
}

func serverBanner(s *server, cfg config) []string {
	loadSum := ""
	for _, lm := range s.models {
		if sum := lm.model.LoadProfile().Summary(); sum != "" {
			loadSum = sum
			break
		}
	}
	return banner.ServerLines(banner.ServerConfig{
		HasModels:   len(s.models) > 0,
		HasSpeech:   len(s.speech) > 0,
		Web:         cfg.web,
		AllowAdmin:  cfg.allowAdmin,
		LoadSummary: loadSum,
	})
}

func concurrencyLine(f bannerFacts, cfg config) string {
	return banner.ConcurrencyLine(f, cfg.bannerConfig())
}
