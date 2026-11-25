package serveapp

import (
	"net/http"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/carriedids"
)

// Carried-ids protocol types and helpers live in internal/app/serveapp/carriedids; the aliases below keep existing
// serveapp callers and tests concise while this file owns only the server and loadedModel wiring.
const (
	reasoningBlobV2Prefix = carriedids.V2Prefix
	idsV1                 = carriedids.ReasonV1
	idsParse              = carriedids.ReasonParse
	idsMAC                = carriedids.ReasonMAC
	idsKeyR               = carriedids.ReasonKey
	idsModel              = carriedids.ReasonModel
	idsNS                 = carriedids.ReasonNS
	idsSpecial            = carriedids.ReasonSpecial
	idsDecode             = carriedids.ReasonDecode
	idsHistory            = carriedids.ReasonHistory
	idsTemplate           = carriedids.ReasonTemplate
)

type (
	idsKey      = carriedids.Key
	idsBlob     = carriedids.Blob
	idsCarrier  = carriedids.Carrier
	idsOutcome  = carriedids.Outcome
	idsCounters = carriedids.Counters
	idsCheck    = carriedids.Check
)

var (
	newIDsKey         = carriedids.NewKey
	loadIDsKey        = carriedids.LoadKey
	idsKeyFileWarning = carriedids.KeyFileWarning
	parseIDsBlob      = carriedids.ParseBlob
	idsBytes          = carriedids.Bytes
	idsSHA256         = carriedids.SHA256
)

// idsCheckFor is the check for a request to lm, nil when the server has no key (the feature is off) or no turn carries a blob.
func (s *server) idsCheckFor(lm *loadedModel, r *http.Request, carriers map[int]*idsCarrier) *idsCheck {
	if s.idsKey == nil || len(carriers) == 0 {
		return nil
	}
	return &idsCheck{Key: s.idsKey, Model: lm.idsIdentity(), NS: requestNamespace(r), Carriers: carriers}
}

// requestNamespace is the request's KV sharing namespace (decoder.KVScope). serve has one API key, so every request is in
// the single local namespace. It is inside the MAC so that a blob minted in one namespace never decides another's
// tokenization once there are several.
func requestNamespace(*http.Request) string { return "" }

// idsIdentity names the model and tokenizer a blob is minted for, cached once per loadedModel.
func (lm *loadedModel) idsIdentity() string {
	lm.caches.idsIdentOnce.Do(func() {
		lm.caches.idsIdent = carriedids.Identity(lm.tk, lm.fp)
	})
	return lm.caches.idsIdent
}

// encodePrompt renders and tokenizes a prompt as the generation routes do, substituting verified carried token ids where valid.
func (lm *loadedModel) encodePrompt(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ck *idsCheck) ([]int, idsOutcome, error) {
	return carriedids.EncodePrompt(lm.tk, tm, system, turns, tools, ck, func() ([]int, error) {
		return lm.promptForT(nil, system, turns)
	})
}
