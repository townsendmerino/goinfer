package serveapp

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"slices"
	"sort"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// idsCheck is what verifying one request's carriers needs: the server's key, the identity a blob must name, the request's
// namespace, and the carrier of each assistant turn that has one (by turn index).
type idsCheck struct {
	key      *idsKey
	model    string
	ns       string
	carriers map[int]*idsCarrier
}

// idsCheckFor is the check for a request to lm, nil when the server has no key (the feature is off) or no turn carries a blob.
func (s *server) idsCheckFor(lm *loadedModel, r *http.Request, carriers map[int]*idsCarrier) *idsCheck {
	if s.idsKey == nil || len(carriers) == 0 {
		return nil
	}
	return &idsCheck{key: s.idsKey, model: lm.idsIdentity(), ns: requestNamespace(r), carriers: carriers}
}

// requestNamespace is the request's KV sharing namespace (decoder.KVScope). serve has one API key, so every request is in
// the single local namespace. It is inside the MAC so that a blob minted in one namespace never decides another's
// tokenization once there are several.
func requestNamespace(*http.Request) string { return "" }

// idsIdentity names the model and tokenizer a blob is minted for: a hash of the tokenizer's vocabulary (every id's surface
// and whether it is an added token) and of the weights' identity (the fingerprint less its mtime, so a copy of the same file
// on another server matches, and the adapter).
func (lm *loadedModel) idsIdentity() string {
	lm.idsIdentOnce.Do(func() {
		h := sha256.New()
		if lm.tk != nil {
			for id := range lm.tk.Size() {
				if lm.tk.IsAdded(id) {
					h.Write([]byte{1})
				} else {
					h.Write([]byte{0})
				}
				h.Write(lm.tk.TokenText(id))
				h.Write([]byte{0xff})
			}
		}
		fp, adapter, _ := strings.Cut(lm.fp, "+adapter:")
		if f := strings.Split(fp, "|"); len(f) == 4 { // name|size|mtime|quant
			fp = f[0] + "|" + f[1] + "|" + f[3]
		}
		h.Write([]byte("\x00" + fp + "\x00" + adapter))
		lm.idsIdent = hex.EncodeToString(h.Sum(nil)[:16])
	})
	return lm.idsIdent
}

// encodePrompt renders and tokenizes a prompt as the generation routes do: RenderToolsSegments when tools are given, else
// RenderSegments, then EncodeSegments (the raw fallback when tm is nil). With a check, each assistant turn that carries a
// verified blob whose ids decode exactly to the turn's text as rendered is tokenized as those ids; every other turn, and
// every byte outside the carried regions, is tokenized exactly as without the check.
func (lm *loadedModel) encodePrompt(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ck *idsCheck) ([]int, idsOutcome, error) {
	var out idsOutcome
	if tm == nil {
		if ck != nil {
			for _, k := range sortedKeys(ck.carriers) {
				out.note(ck.reasonBeforeRender(ck.carriers[k], idsSpecial))
			}
		}
		ids, err := lm.promptForT(nil, system, turns)
		return ids, out, err
	}
	render := func(turns []chat.Turn) []chat.Segment {
		if len(tools) > 0 {
			return tm.RenderToolsSegments(system, turns, tools)
		}
		return tm.RenderSegments(system, turns)
	}
	segs := render(turns)
	if ck == nil {
		ids, err := lm.tk.EncodeSegments(segs, false)
		return ids, out, err
	}
	pl := newSegPlan(segs)
	var regs []idsRegion
	for _, k := range sortedKeys(ck.carriers) {
		reason, reg := lm.idsTurn(ck, ck.carriers[k], k, turns, render, pl, newIDsFormat(tm))
		if reason == "" && len(regs) > 0 && reg.a < regs[len(regs)-1].b {
			reason = idsDecode // two carriers claim the same bytes; cannot happen for turns rendered in order
		}
		out.note(reason)
		if reason == "" {
			regs = append(regs, reg)
			if reg.formatted {
				out.formatted++
			}
		}
	}
	ids, err := lm.encodeRegions(pl, regs)
	return ids, out, err
}

func sortedKeys(m map[int]*idsCarrier) []int {
	ks := make([]int, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Ints(ks)
	return ks
}

// reasonBeforeRender is the reason a carrier fails before its ids are compared with the rendered turn ("" when it passes
// these checks); otherwise is returned when it passes them, for a caller that cannot compare.
func (ck *idsCheck) reasonBeforeRender(c *idsCarrier, otherwise string) string {
	if c.reason != "" {
		return c.reason
	}
	b := c.blob
	switch {
	case b.keyID != ck.key.id:
		return idsKeyR
	case !ck.key.verifyMAC(b):
		return idsMAC
	case b.model != ck.model:
		return idsModel
	case b.ns != ck.ns:
		return idsNS
	}
	return otherwise
}

// segPlan is a rendered prompt's text and where each segment starts in it (starts[len(segs)] is the text's length).
type segPlan struct {
	segs   []chat.Segment
	text   string
	starts []int
}

func newSegPlan(segs []chat.Segment) *segPlan {
	var b strings.Builder
	starts := make([]int, 0, len(segs)+1)
	for _, s := range segs {
		starts = append(starts, b.Len())
		b.WriteString(s.Text)
	}
	starts = append(starts, b.Len())
	return &segPlan{segs: segs, text: b.String(), starts: starts}
}

// segAt is the segment holding byte x (x < len(text)).
func (p *segPlan) segAt(x int) int {
	return sort.Search(len(p.segs), func(i int) bool { return p.starts[i+1] > x })
}

// idsPiece is one special id or one run of ordinary ids, as text.
type idsPiece struct {
	special bool
	text    string
}

// idsRegion is a byte range of the rendered text whose tokens are carried ids. formatted marks a range whose rendered bytes
// differ from the ids' text in template-owned formatting only (idsFormat).
type idsRegion struct {
	a, b      int
	ids       []int
	formatted bool
}

// pieces splits ids at their added (special) ids and decodes each run between them; ok is false when an id is out of the
// vocabulary or a run does not decode.
func (lm *loadedModel) pieces(ids []int) (ps []idsPiece, ok bool) {
	n := lm.tk.Size()
	for i := 0; i < len(ids); {
		if ids[i] < 0 || ids[i] >= n {
			return nil, false
		}
		if lm.tk.IsAdded(ids[i]) {
			ps = append(ps, idsPiece{special: true, text: string(lm.tk.TokenText(ids[i]))})
			i++
			continue
		}
		j := i
		for j < len(ids) && ids[j] >= 0 && ids[j] < n && !lm.tk.IsAdded(ids[j]) {
			j++
		}
		s, err := lm.tk.DecodeContinuation(ids[i:j])
		if err != nil {
			return nil, false
		}
		ps = append(ps, idsPiece{text: s})
		i = j
	}
	return ps, true
}

// aligned reports whether pieces laid down from byte a of the rendered text put each special id exactly on a special segment
// and every run inside text that is not special.
func (p *segPlan) aligned(a int, ps []idsPiece) bool {
	x := a
	for _, pc := range ps {
		end := x + len(pc.text)
		if pc.special {
			i := p.segAt(x)
			if p.starts[i] != x || p.starts[i+1] != end || !p.segs[i].Special {
				return false
			}
		} else if end > x {
			for i := p.segAt(x); i < len(p.segs) && p.starts[i] < end; i++ {
				if p.segs[i].Special {
					return false
				}
			}
		}
		x = end
	}
	return true
}

// idsTurn decides whether turn k uses its carried ids, and where they go. The turn's rendered bytes are located by rendering
// the conversation once more with the turn's content, reasoning and calls emptied: the range where the two texts differ lies
// inside the turn's generated part. The ids must decode to a run of the rendered text that covers that range (all of it for a
// whole-turn blob, its start for a reasoning-only one) and line up with the template's special tokens, or differ from such a
// run only by fm's template-owned formatting, in which case the ids replace the run.
func (lm *loadedModel) idsTurn(ck *idsCheck, c *idsCarrier, k int, turns []chat.Turn, render func([]chat.Turn) []chat.Segment, pl *segPlan, fm idsFormat) (string, idsRegion) {
	if r := ck.reasonBeforeRender(c, ""); r != "" {
		return r, idsRegion{}
	}
	b := c.blob
	ps, ok := lm.pieces(b.ids)
	if !ok || len(ps) == 0 {
		return idsDecode, idsRegion{}
	}
	var d strings.Builder
	for _, pc := range ps {
		d.WriteString(pc.text)
	}
	D := d.String()
	T := pl.text
	empty := slices.Clone(turns)
	empty[k].Content, empty[k].Reasoning, empty[k].ToolCalls = "", "", nil
	E := newSegPlan(render(empty)).text
	// p: the bytes before it are the same in both renders; sfx: the bytes after len(T)-sfx are. The two can claim the same
	// byte (an insertion next to an equal byte), so each is measured on its own and a range of T holds every difference when
	// it starts at or before p and ends at or after len(T)-sfx, with what it leaves outside fitting in E.
	p := 0
	for p < len(T) && p < len(E) && T[p] == E[p] {
		p++
	}
	sfx := 0
	for sfx < len(T) && sfx < len(E) && T[len(T)-1-sfx] == E[len(E)-1-sfx] {
		sfx++
	}
	textHit, found, n := false, idsRegion{}, 0
	for from := 0; from <= len(T)-len(D); {
		i := strings.Index(T[from:], D)
		if i < 0 {
			break
		}
		a := from + i
		end := a + len(D)
		covers := T != E && a <= p && end > p
		if b.whole {
			covers = covers && len(T)-end <= sfx && a+len(T)-end <= len(E)
		}
		if covers {
			textHit = true
			if pl.aligned(a, ps) {
				found, n = idsRegion{a: a, b: end, ids: b.ids}, n+1
			}
		}
		from = a + 1
	}
	switch {
	case n == 1:
		return "", found
	case n > 1:
		return idsDecode, idsRegion{} // the turn's text occurs twice over its own range: no single place to put the ids
	case textHit:
		return idsSpecial, idsRegion{}
	}
	if turns[k].Reasoning != "" {
		noR := slices.Clone(turns)
		noR[k].Reasoning = ""
		if newSegPlan(render(noR)).text == T {
			return idsHistory, idsRegion{}
		}
	}
	formatOnly := false
	n = 0
	for a := min(p, len(T)); a >= max(0, p-len(D)); a-- {
		end, at, differs, ok := fm.align(D, T[a:])
		if !ok || !differs {
			continue
		}
		formatOnly = true
		end += a
		for i := range at {
			at[i] += a
		}
		covers := T != E && a <= p && end > p
		if b.whole {
			covers = covers && len(T)-end <= sfx && a+len(T)-end <= len(E)
		}
		if covers && pl.alignedAt(at, ps) {
			found, n = idsRegion{a: a, b: end, ids: b.ids, formatted: true}, n+1
		}
	}
	switch {
	case n == 1:
		return "", found
	case formatOnly:
		return idsTemplate, idsRegion{} // the difference is template-owned, but no single run of the render takes the ids
	}
	return idsDecode, idsRegion{}
}

// encodeRegions tokenizes the rendered prompt with each region's ids in place of its bytes. Text between regions is
// tokenized as EncodeSegments tokenizes it, a segment cut by a region's edge as text of its own; after a region that ends on
// a special token, that token leads the next stretch, so a token that swallows the whitespace after it still does.
func (lm *loadedModel) encodeRegions(pl *segPlan, regs []idsRegion) ([]int, error) {
	var out []int
	pos := 0
	var lead *chat.Segment
	stretch := func(x, y int) error {
		var part []chat.Segment
		if lead != nil {
			part = append(part, *lead)
		}
		if x < y {
			for i := pl.segAt(x); i < len(pl.segs) && pl.starts[i] < y; i++ {
				s := pl.segs[i]
				lo, hi := max(x, pl.starts[i]), min(y, pl.starts[i+1])
				if lo < hi {
					part = append(part, chat.Segment{Text: pl.text[lo:hi], Special: s.Special})
				}
			}
		}
		ids, err := lm.tk.EncodeSegments(part, false)
		if err != nil {
			return err
		}
		if lead != nil {
			one, err := lm.tk.EncodeSegments([]chat.Segment{*lead}, false)
			if err != nil {
				return err
			}
			ids = ids[len(one):]
		}
		out = append(out, ids...)
		return nil
	}
	for _, r := range regs {
		if err := stretch(pos, r.a); err != nil {
			return nil, err
		}
		out = append(out, r.ids...)
		pos, lead = r.b, nil
		if i := pl.segAt(r.b - 1); pl.starts[i+1] == r.b && pl.segs[i].Special {
			lead = &pl.segs[i]
		}
	}
	if err := stretch(pos, len(pl.text)); err != nil {
		return nil, err
	}
	return out, nil
}
