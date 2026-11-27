package serveapp

import (
	"net/http"
	"slices"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/responsesapi"
)

// carriedMint mints the blobs one reply carries (carried_ids.go). A nil *carriedMint, or a reply with no ids, mints nothing
// and the reasoning item keeps the v1 text blob.
type carriedMint struct {
	key       *idsKey
	lm        *loadedModel
	model, ns string
	out       *turnOutcome
}

// mintFor is the minter for gr's reply, nil when the server has no key.
func (s *server) mintFor(lm *loadedModel, r *http.Request, gr genRequest) *carriedMint {
	if s.idsKey == nil || gr.outcome == nil || lm.tk == nil {
		return nil
	}
	return &carriedMint{key: s.idsKey, lm: lm, model: lm.idsIdentity(), ns: requestNamespace(r), out: gr.outcome}
}

// blob is the encrypted_content for the reply's reasoning item holding text: the whole turn's ids once the generation has
// ended; before that (a streamed reasoning item closes when the answer opens) the ids up to the last added token so far,
// which ends the reasoning block. "" (the v1 text blob) when there are no ids to carry, or a stop string ended the turn
// (its ids then run past the text the client received).
func (m *carriedMint) blob(text string) string {
	if m == nil || m.out == nil || m.out.StopHit {
		return ""
	}
	ids, whole := m.out.IDs, m.out.Done
	if !whole {
		cut := -1
		for i, id := range slices.Backward(ids) {
			if m.lm.tk.IsAdded(id) {
				cut = i + 1
				break
			}
		}
		if cut <= 0 {
			return ""
		}
		ids = ids[:cut]
	}
	if len(ids) == 0 {
		return ""
	}
	return m.key.Mint(m.model, m.ns, text, ids, whole)
}

var wantsCarrier = responsesapi.WantsCarrier

// withCarrier puts a carrier-only reasoning item first in a reply that has no reasoning item, when the request asked for
// one and the reply has ids to carry.
func withCarrier(rs *respStream, mint *carriedMint, include []string, id string, items []any) []any {
	var fn func(string) string
	if mint != nil {
		fn = mint.blob
	}
	return responsesapi.WithCarrier(rs, fn, include, id, items)
}

// noteIDs records a request's carried-id outcome: the server's totals, and the request's -log-requests line.
func (s *server) noteIDs(r *http.Request, o idsOutcome) {
	s.idsStats.Add(o)
	if tr := traceFrom(r.Context()); tr != nil {
		tr.carried(o)
	}
}
