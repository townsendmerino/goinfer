package serveapp

import (
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/engine"
	"github.com/townsendmerino/goinfer/engine/enginetest"
)

// genReport is a generation's account as the shell reads it.
type genReport = engine.Report

// reportNow is report's account now, or nil when there is no report function.
func reportNow(report func() *genReport) *genReport {
	if report == nil {
		return nil
	}
	return report()
}

// genEngine is the model's engine: a test's fake engine or the scripted seam when one is set, else the decoder.
func (lm *loadedModel) genEngine() engine.Engine {
	if lm.fakeEngine != nil {
		return lm.fakeEngine
	}
	if lm.scripted != nil {
		return enginetest.Scripted{IDs: lm.scripted, OnStart: func(req engine.Request) {
			if lm.onScripted != nil {
				lm.onScripted(req.PromptIDs)
			}
			if lm.onScriptedScope != nil {
				lm.onScriptedScope(req.Scope)
			}
		}}
	}
	return engine.Local{Model: asDecoder(lm.model), Pool: sessionPool{lm}, Drafter: asBlockSpec(lm.blockSpec), Speculative: lm.spec, AdapterBound: lm.adapter != "",
		Encode: func(s string) []int { ids, _ := lm.tk.Encode(s, false); return ids }}
}

// sessionPool is lm's session LRU as an engine.SessionPool, each operation under sessMu (which covers the LRU
// operation only, never the generation: a session in use is checked out).
type sessionPool struct{ lm *loadedModel }

func (p sessionPool) Acquire(prompt []int) *decoder.Session {
	p.lm.sessMu.Lock()
	defer p.lm.sessMu.Unlock()
	return p.lm.sessions.acquire(prompt)
}

func (p sessionPool) Checkin(s *decoder.Session) {
	p.lm.sessMu.Lock()
	defer p.lm.sessMu.Unlock()
	p.lm.sessions.checkin(s)
}
