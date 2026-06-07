package constrain

// schemaGrammar is the incremental byte-level acceptor for a compiled schema. It
// is a pushdown automaton: a stack of frames, each one a position inside the
// value of some schema node. The top frame consumes bytes; when a value
// completes it pops and the parent advances. Frames are plain value structs
// (state in ints + ≤64-bit masks, the node pointer is immutable), so snapshot is
// a slice copy — TryBytes snapshots, steps the candidate bytes, and restores.

// frame sub-states.
const (
	fsValue           = iota // pending dispatch: the first byte picks the concrete shape
	fsObjKeyOrClose          // after '{': a key or '}'
	fsObjKey                 // after ',': a key
	fsObjKeyStr              // inside a key string, matching against unseen property names
	fsObjColon               // after a key: ':'
	fsObjComma               // after a value: ',' or '}'
	fsArrValueOrClose        // after '[': a value or ']'
	fsArrValue               // after ',': a value
	fsArrComma               // after a value: ',' or ']'
	fsStr                    // inside a string
	fsStrEsc                 // inside a string, after '\'
	fsStrU                   // inside a string, reading \uXXXX (num = hex left)
	fsNum                    // inside a number (num = sub-state)
	fsEnum                   // matching a fixed literal set (cand mask, pos)
)

// number sub-states (held in frame.num while state == fsNum).
const (
	snNeg = iota
	snZero
	snInt
	snDot
	snFrac
	snExpSign
	snExpDigit
	snExp
)

func snTerminal(n int) bool { return n == snZero || n == snInt || n == snFrac || n == snExp }

type frame struct {
	n     *node
	state int
	seen  uint64 // object: properties already parsed (bit per props index)
	cand  uint64 // enum: candidate literals; object-key: candidate property names
	pos   int    // enum/key: bytes matched so far
	count int    // array: items parsed
	sel   int    // object: the property index selected by the current key
	num   int    // number sub-state; string: \u hex digits remaining
}

type schemaGrammar struct {
	root  *node
	stack []frame
	done  bool // the root value is complete (closing delimiter consumed)

	// snapshot buffers reused by TryBytes (no per-trial allocation).
	sStack []frame
	sDone  bool
}

func (g *schemaGrammar) Reset() {
	g.stack = append(g.stack[:0], frame{n: g.root, state: fsValue})
	g.done = false
}

// CanEnd reports whether the committed output is a complete document: the root
// value finished, or we're sitting on a complete top-level scalar (a number or
// enum literal has no closing delimiter, so it's done as soon as it can't extend).
func (g *schemaGrammar) CanEnd() bool {
	if g.done {
		return true
	}
	if len(g.stack) != 1 {
		return false
	}
	f := &g.stack[0]
	switch {
	case f.state == fsNum && snTerminal(f.num):
		return true
	case f.state == fsEnum && g.enumTerminal(f):
		return true
	}
	return false
}

func (g *schemaGrammar) TryBytes(bs []byte) bool {
	g.snapshot()
	ok := true
	for _, b := range bs {
		if !g.step(b) {
			ok = false
			break
		}
	}
	g.restore()
	return ok
}

func (g *schemaGrammar) Commit(bs []byte) {
	for _, b := range bs {
		g.step(b)
	}
}

func (g *schemaGrammar) snapshot() {
	g.sStack = append(g.sStack[:0], g.stack...)
	g.sDone = g.done
}

func (g *schemaGrammar) restore() {
	g.stack = append(g.stack[:0], g.sStack...)
	g.done = g.sDone
}

// step advances over one byte, returning false if it isn't a legal next byte.
// A loop lets a lazily-completing scalar (number/enum) finish and re-feed the
// byte that ended it to the parent frame.
func (g *schemaGrammar) step(b byte) bool {
	for {
		if len(g.stack) == 0 {
			return isWS(b) // document complete; only trailing whitespace
		}
		f := &g.stack[len(g.stack)-1]
		switch f.state {
		case fsValue:
			if isWS(b) {
				return true
			}
			return g.dispatch(f, b)

		case fsObjKeyOrClose:
			if isWS(b) {
				return true
			}
			if b == '}' {
				if g.objCanClose(f) {
					g.complete()
					return true
				}
				return false
			}
			if b == '"' && g.objHasUnseen(f) {
				g.enterKey(f)
				return true
			}
			return false

		case fsObjKey:
			if isWS(b) {
				return true
			}
			if b == '"' && g.objHasUnseen(f) {
				g.enterKey(f)
				return true
			}
			return false

		case fsObjKeyStr:
			return g.keyStep(f, b)

		case fsObjColon:
			if isWS(b) {
				return true
			}
			if b == ':' {
				child := f.n.props[f.sel].schema
				f.state = fsObjComma
				g.push(child)
				return true
			}
			return false

		case fsObjComma:
			if isWS(b) {
				return true
			}
			if b == ',' && g.objHasUnseen(f) {
				f.state = fsObjKey
				return true
			}
			if b == '}' {
				if g.objCanClose(f) {
					g.complete()
					return true
				}
				return false
			}
			return false

		case fsArrValueOrClose:
			if isWS(b) {
				return true
			}
			if b == ']' {
				if f.count >= f.n.minItems {
					g.complete()
					return true
				}
				return false
			}
			if f.n.maxItems >= 0 && f.count >= f.n.maxItems {
				return false
			}
			f.state = fsArrComma
			f.count++
			g.push(f.n.items)
			continue // re-feed b to the new child value frame

		case fsArrValue:
			if isWS(b) {
				return true
			}
			if f.n.maxItems >= 0 && f.count >= f.n.maxItems {
				return false
			}
			f.state = fsArrComma
			f.count++
			g.push(f.n.items)
			continue

		case fsArrComma:
			if isWS(b) {
				return true
			}
			if b == ',' && (f.n.maxItems < 0 || f.count < f.n.maxItems) {
				f.state = fsArrValue
				return true
			}
			if b == ']' {
				if f.count >= f.n.minItems {
					g.complete()
					return true
				}
				return false
			}
			return false

		case fsStr:
			switch {
			case b == '"':
				g.complete()
				return true
			case b == '\\':
				f.state = fsStrEsc
				return true
			case b < 0x20:
				return false
			default:
				return true
			}

		case fsStrEsc:
			switch b {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				f.state = fsStr
				return true
			case 'u':
				f.state, f.num = fsStrU, 4
				return true
			}
			return false

		case fsStrU:
			if !isHex(b) {
				return false
			}
			f.num--
			if f.num == 0 {
				f.state = fsStr
			}
			return true

		case fsNum:
			if numStepSchema(f, b) {
				return true
			}
			if !snTerminal(f.num) {
				return false
			}
			g.complete()
			continue // number ended; re-feed b to the parent

		case fsEnum:
			if g.enumStep(f, b) {
				return true
			}
			if !g.enumTerminal(f) {
				return false
			}
			g.complete()
			continue
		}
		return false
	}
}

// dispatch picks the concrete shape on the first byte of a pending value.
func (g *schemaGrammar) dispatch(f *frame, b byte) bool {
	switch f.n.kind {
	case kObject:
		if b == '{' {
			f.state = fsObjKeyOrClose
			return true
		}
	case kArray:
		if b == '[' {
			f.state = fsArrValueOrClose
			return true
		}
	case kString:
		if b == '"' {
			f.state = fsStr
			return true
		}
	case kNumber:
		if numStartSchema(f, b) {
			f.state = fsNum
			return true
		}
	case kEnum:
		var cand uint64
		for i, e := range f.n.enum {
			if len(e) > 0 && e[0] == b {
				cand |= 1 << uint(i)
			}
		}
		if cand != 0 {
			f.cand, f.pos, f.state = cand, 1, fsEnum
			return true
		}
	}
	return false
}

// complete pops the finished top frame; emptying the stack means the whole
// document is done. Parent frames were advanced to their post-value state before
// their child was pushed, so the parent resumes correctly.
func (g *schemaGrammar) complete() {
	g.stack = g.stack[:len(g.stack)-1]
	if len(g.stack) == 0 {
		g.done = true
	}
}

func (g *schemaGrammar) push(n *node) {
	g.stack = append(g.stack, frame{n: n, state: fsValue})
}

func (g *schemaGrammar) objCanClose(f *frame) bool {
	return f.seen&f.n.requiredMask == f.n.requiredMask
}

// objHasUnseen reports whether any declared property hasn't been parsed yet — so
// a key (or the comma before one) is only allowed while there's room for it.
func (g *schemaGrammar) objHasUnseen(f *frame) bool {
	n := len(f.n.props)
	var all uint64
	if n >= 64 {
		all = ^uint64(0)
	} else {
		all = (uint64(1) << uint(n)) - 1
	}
	return all&^f.seen != 0
}

// enterKey begins matching an object key against the property names not yet seen.
func (g *schemaGrammar) enterKey(f *frame) {
	all := uint64(0)
	if len(f.n.props) < 64 {
		all = (uint64(1) << uint(len(f.n.props))) - 1
	} else {
		all = ^uint64(0)
	}
	f.cand = all &^ f.seen
	f.pos = 0
	f.state = fsObjKeyStr
}

// keyStep matches one byte of an object key against the candidate property
// names; on the closing quote it selects a fully-matched, unseen property.
func (g *schemaGrammar) keyStep(f *frame, b byte) bool {
	switch {
	case b == '"':
		for i := range f.n.props {
			if f.cand&(1<<uint(i)) != 0 && len(f.n.props[i].name) == f.pos {
				f.sel = i
				f.seen |= 1 << uint(i)
				f.state = fsObjColon
				return true
			}
		}
		return false
	case b == '\\' || b < 0x20:
		return false // simple key names only (no escapes) — the supported subset
	default:
		var next uint64
		for i := range f.n.props {
			if f.cand&(1<<uint(i)) == 0 {
				continue
			}
			name := f.n.props[i].name
			if len(name) > f.pos && name[f.pos] == b {
				next |= 1 << uint(i)
			}
		}
		if next == 0 {
			return false
		}
		f.cand, f.pos = next, f.pos+1
		return true
	}
}

// enumStep advances the candidate set by one byte; false if none extend.
func (g *schemaGrammar) enumStep(f *frame, b byte) bool {
	var next uint64
	for i, e := range f.n.enum {
		if f.cand&(1<<uint(i)) == 0 {
			continue
		}
		if len(e) > f.pos && e[f.pos] == b {
			next |= 1 << uint(i)
		}
	}
	if next == 0 {
		return false
	}
	f.cand, f.pos = next, f.pos+1
	return true
}

// enumTerminal reports whether a candidate literal is exactly matched (so the
// value is complete).
func (g *schemaGrammar) enumTerminal(f *frame) bool {
	for i, e := range f.n.enum {
		if f.cand&(1<<uint(i)) != 0 && len(e) == f.pos {
			return true
		}
	}
	return false
}

// numStartSchema sets the initial number sub-state from the first byte.
func numStartSchema(f *frame, b byte) bool {
	switch {
	case b == '-':
		f.num = snNeg
	case b == '0':
		f.num = snZero
	case b >= '1' && b <= '9':
		f.num = snInt
	default:
		return false
	}
	return true
}

// numStepSchema advances the number sub-state if b continues it. intOnly nodes
// reject the fraction/exponent transitions, so only integers are accepted.
func numStepSchema(f *frame, b byte) bool {
	d := b >= '0' && b <= '9'
	frac := !f.n.intOnly
	switch f.num {
	case snNeg:
		if b == '0' {
			f.num = snZero
			return true
		}
		if b >= '1' && b <= '9' {
			f.num = snInt
			return true
		}
	case snZero:
		if frac && b == '.' {
			f.num = snDot
			return true
		}
		if frac && (b == 'e' || b == 'E') {
			f.num = snExpSign
			return true
		}
	case snInt:
		if d {
			return true
		}
		if frac && b == '.' {
			f.num = snDot
			return true
		}
		if frac && (b == 'e' || b == 'E') {
			f.num = snExpSign
			return true
		}
	case snDot:
		if d {
			f.num = snFrac
			return true
		}
	case snFrac:
		if d {
			return true
		}
		if b == 'e' || b == 'E' {
			f.num = snExpSign
			return true
		}
	case snExpSign:
		if b == '+' || b == '-' {
			f.num = snExpDigit
			return true
		}
		if d {
			f.num = snExp
			return true
		}
	case snExpDigit:
		if d {
			f.num = snExp
			return true
		}
	case snExp:
		if d {
			return true
		}
	}
	return false
}
