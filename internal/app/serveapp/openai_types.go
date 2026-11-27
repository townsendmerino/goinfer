package serveapp

import (
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
	samplingp "github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

type (
	sampling      = wirecore.Sampling
	respFormat    = wirecore.RespFormat
	chatReq       = wirecore.ChatReq
	chatMessage   = wirecore.ChatMessage
	toolSpec      = wirecore.ToolSpec
	apiToolCall   = wirecore.APIToolCall
	completionReq = wirecore.CompletionReq
	usage         = jobs.Usage
	prefillReuse  = wirecore.PrefillReuse
	confResult    = wirecore.ConfResult
	kvScope       = decoder.KVScope
)

var (
	rawStr                 = wirecore.RawStr
	functionTools          = wirecore.FunctionTools
	dropUnrunnableTools    = wirecore.DropUnrunnableTools
	reuseOf                = wirecore.ReuseOf
	newUsage               = wirecore.NewUsage
	validateN              = wirecore.ValidateN
	grammarFor             = wirecore.GrammarFor
	messagesToTurns        = wirecore.MessagesToTurns
	messagesToTurnsCarried = wirecore.MessagesToTurnsCarried
	rawPrompt              = wirecore.RawPrompt
	parseLogitBias         = wirecore.ParseLogitBias
)

// turnResult reports a finished generation's finish reason, token count, logprobs, stop hit, KV reuse, and cancel reason.
type turnResult struct {
	finish       string
	nComp        int
	logprobs     []decoder.SampleInfo
	stopHit      string
	reused       prefillReuse
	cancelReason string
}

// genRequest is a prepared generation ready for drive or driveVL.
type genRequest struct {
	promptIDs     []int
	sp            samplingp.SamplingParams
	maxTokens     int
	stopStrings   []string
	id            string
	masker        *constrain.Masker
	lazy          *constrain.LazyMasker
	conf          *confResult
	think         *thinkOut
	outcome       *turnOutcome
	boundary      int
	keep          int
	msgBoundary   int
	fromModel     []string
	callPostCheck bool
	clock         *jobs.TurnClock
}

func (gr *genRequest) kvScope() kvScope {
	return kvScope{Boundary: gr.boundary, Keep: gr.keep, MsgBoundary: gr.msgBoundary}
}
