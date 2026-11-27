package serveapp

import (
	"net/http"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/netguard"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

const (
	maxBodyBytes        = netguard.MaxBodyBytes
	maxVisionBodyBytes  = netguard.MaxVisionBodyBytes
	maxEmbedBodyBytes   = netguard.MaxEmbedBodyBytes
	maxBatchFileBytes   = netguard.MaxBatchFileBytes
	maxFileUploadMemory = netguard.MaxFileUploadMemory
	statusCancelled     = netguard.StatusCancelled
)

type (
	sseWriter     = streamframe.SSEWriter
	delta         = wirecore.Delta
	streamOptions = wirecore.StreamOptions
)

var (
	maxBytes          = netguard.MaxBytes
	limitInflight     = netguard.LimitInflight
	requireAuth       = netguard.RequireAuth
	writeBodyErr      = netguard.WriteBodyErr
	parseUpload       = netguard.ParseUpload
	jsonDecodeMessage = netguard.JSONDecodeMessage
	decodeJSON        = netguard.DecodeJSON
	serveHandler      = netguard.ServeHandler
	sameOriginGuard   = netguard.SameOriginGuard
	splitHosts        = netguard.SplitHosts

	newSSEWriter  = streamframe.NewSSEWriter
	sseJSONSender = streamframe.JSONSender
	sseSend       = streamframe.SendData
	sseDone       = streamframe.SendDone
	sseErr        = streamframe.SendOpenAIError
	sseHeartbeat  = streamframe.Heartbeat

	writeJSON      = netguard.WriteJSON
	writeErr       = netguard.WriteErr
	writeServerErr = netguard.WriteServerErr

	chatChunk          = wirecore.ChatChunk
	usageChunk         = wirecore.UsageChunk
	completionChunk    = wirecore.CompletionChunk
	parseStop          = wirecore.ParseStop
	singlePromptString = wirecore.SinglePromptString
	firstStop          = wirecore.FirstStop
	stopTailHold       = wirecore.StopTailHold
	completeUTF8       = wirecore.CompleteUTF8
	humanBytes         = wirecore.HumanBytes
	reqID              = stores.MintID
)

func sseStart(w http.ResponseWriter) (*sseWriter, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return newSSEWriter(w, f), true
}

func sendUsage(ss *sseWriter, clock *jobs.TurnClock, so *streamOptions, id string, created int64, model string, u usage) {
	if so == nil || !so.IncludeUsage {
		return
	}
	sseSend(ss, clock.AddTurn(usageChunk(id, created, model, u)))
}

func deref[T any](p *T, def T) T {
	if p != nil {
		return *p
	}
	return def
}
