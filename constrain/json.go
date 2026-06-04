package constrain

// Streaming byte-level JSON validator — a pushdown automaton that decides, byte
// by byte, whether the input so far is a valid PREFIX of some JSON document
// (could still be completed) and whether it is already a complete value. It is
// the grammar behind JSON-constrained decoding: at each step a candidate token's
// bytes are tried (TryBytes) without committing, and only tokens that keep the
// document a valid prefix survive the logit mask.
//
// It works on bytes, not runes, so multi-byte UTF-8 inside strings passes
// through unvalidated (any byte ≥ 0x20 that isn't " or \ is legal string
// content) and the grammar is independent of the tokenizer's encoding.

// parser states.
const (
	jsValue           = iota // expecting the start of a value (top, after ':' or ',' in array)
	jsArrValueOrClose        // right after '[': a value or ']'
	jsObjKeyOrClose          // right after '{': a "key" or '}'
	jsObjKey                 // after ',' in an object: a "key"
	jsColon                  // after a key: ':'
	jsObjNext                // after a value in an object: ',' or '}'
	jsArrNext                // after a value in an array: ',' or ']'
	jsEnd                    // complete top-level value; only whitespace
	jsString                 // inside a string
	jsStringEsc              // inside a string, after '\'
	jsStringU                // inside a string, reading \uXXXX (num = hex digits left)
	jsNumber                 // inside a number (num = sub-state below)
	jsLiteral                // matching the rest of true/false/null (lit = remaining)
)

// number sub-states (held in num while state == jsNumber).
const (
	numNeg      = iota // saw '-', need a digit
	numZero            // saw a single 0 (terminal)
	numInt             // integer digits, leading 1-9 (terminal)
	numDot             // saw '.', need a fraction digit
	numFrac            // fraction digits (terminal)
	numExpSign         // saw e/E, optional sign or digit
	numExpDigit        // saw e/E[sign], need a digit
	numExp             // exponent digits (terminal)
)

func numTerminal(n int) bool {
	return n == numZero || n == numInt || n == numFrac || n == numExp
}

// jsonGrammar is the JSON acceptor state. The zero value is the initial state
// (jsValue, empty stack), so new(jsonGrammar) is ready to use.
type jsonGrammar struct {
	stack []byte // container nesting: '{' or '['
	state int
	isKey bool   // the string being read is an object key
	lit   string // jsLiteral: bytes still to match
	num   int    // jsNumber sub-state; jsStringU hex-digits-remaining

	// snapshot buffers reused by TryBytes (no per-trial allocation).
	sStack       []byte
	sState, sNum int
	sIsKey       bool
	sLit         string
}

// JSON returns a fresh JSON grammar (a complete single JSON value, RFC 8259).
func JSON() Grammar { return &jsonGrammar{} }

func (g *jsonGrammar) Reset() {
	g.stack = g.stack[:0]
	g.state, g.isKey, g.lit, g.num = jsValue, false, "", 0
}

// CanEnd reports whether the input so far is a complete, valid JSON value (so
// EOS may be emitted): the container stack is empty and we are either past a
// finished value, or sitting on a complete top-level number (which has no
// closing delimiter — "123" is done as soon as the digits stop).
func (g *jsonGrammar) CanEnd() bool {
	if len(g.stack) != 0 {
		return false
	}
	if g.state == jsEnd {
		return true
	}
	return g.state == jsNumber && numTerminal(g.num)
}

// TryBytes reports whether appending bs keeps the document a valid prefix,
// leaving the grammar state unchanged (snapshot/restore around the trial).
func (g *jsonGrammar) TryBytes(bs []byte) bool {
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

// Commit advances the state over bs (which must have passed TryBytes).
func (g *jsonGrammar) Commit(bs []byte) {
	for _, b := range bs {
		g.step(b)
	}
}

func (g *jsonGrammar) snapshot() {
	g.sStack = append(g.sStack[:0], g.stack...)
	g.sState, g.sIsKey, g.sLit, g.sNum = g.state, g.isKey, g.lit, g.num
}

func (g *jsonGrammar) restore() {
	g.stack = append(g.stack[:0], g.sStack...)
	g.state, g.isKey, g.lit, g.num = g.sState, g.sIsKey, g.sLit, g.sNum
}

// step advances over one byte, returning false if it is not a legal next byte.
func (g *jsonGrammar) step(b byte) bool {
	for {
		switch g.state {
		case jsValue:
			if isWS(b) {
				return true
			}
			return g.beginValue(b)
		case jsArrValueOrClose:
			if isWS(b) {
				return true
			}
			if b == ']' {
				g.pop()
				g.valueComplete()
				return true
			}
			return g.beginValue(b)
		case jsObjKeyOrClose:
			if isWS(b) {
				return true
			}
			if b == '}' {
				g.pop()
				g.valueComplete()
				return true
			}
			if b == '"' {
				g.state, g.isKey = jsString, true
				return true
			}
			return false
		case jsObjKey:
			if isWS(b) {
				return true
			}
			if b == '"' {
				g.state, g.isKey = jsString, true
				return true
			}
			return false
		case jsColon:
			if isWS(b) {
				return true
			}
			if b == ':' {
				g.state = jsValue
				return true
			}
			return false
		case jsObjNext:
			if isWS(b) {
				return true
			}
			if b == ',' {
				g.state = jsObjKey
				return true
			}
			if b == '}' {
				g.pop()
				g.valueComplete()
				return true
			}
			return false
		case jsArrNext:
			if isWS(b) {
				return true
			}
			if b == ',' {
				g.state = jsValue
				return true
			}
			if b == ']' {
				g.pop()
				g.valueComplete()
				return true
			}
			return false
		case jsEnd:
			return isWS(b)
		case jsString:
			switch {
			case b == '"':
				if g.isKey {
					g.state = jsColon
				} else {
					g.valueComplete()
				}
				return true
			case b == '\\':
				g.state = jsStringEsc
				return true
			case b < 0x20:
				return false // raw control chars are illegal in JSON strings
			default:
				return true
			}
		case jsStringEsc:
			switch b {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				g.state = jsString
				return true
			case 'u':
				g.state, g.num = jsStringU, 4
				return true
			}
			return false
		case jsStringU:
			if !isHex(b) {
				return false
			}
			g.num--
			if g.num == 0 {
				g.state = jsString
			}
			return true
		case jsLiteral:
			if len(g.lit) == 0 || g.lit[0] != b {
				return false
			}
			g.lit = g.lit[1:]
			if len(g.lit) == 0 {
				g.valueComplete()
			}
			return true
		case jsNumber:
			if g.numStep(b) {
				return true
			}
			// b does not continue the number: the number ends here if it is in a
			// terminal sub-state, then b is reprocessed in the new state.
			if !numTerminal(g.num) {
				return false
			}
			g.valueComplete()
			continue
		}
		return false
	}
}

// beginValue dispatches the first byte of a value.
func (g *jsonGrammar) beginValue(b byte) bool {
	switch {
	case b == '{':
		g.push('{')
		g.state = jsObjKeyOrClose
	case b == '[':
		g.push('[')
		g.state = jsArrValueOrClose
	case b == '"':
		g.state, g.isKey = jsString, false
	case b == '-':
		g.state, g.num = jsNumber, numNeg
	case b == '0':
		g.state, g.num = jsNumber, numZero
	case b >= '1' && b <= '9':
		g.state, g.num = jsNumber, numInt
	case b == 't':
		g.state, g.lit = jsLiteral, "rue"
	case b == 'f':
		g.state, g.lit = jsLiteral, "alse"
	case b == 'n':
		g.state, g.lit = jsLiteral, "ull"
	default:
		return false
	}
	return true
}

// numStep advances the number sub-state if b continues the number.
func (g *jsonGrammar) numStep(b byte) bool {
	d := b >= '0' && b <= '9'
	switch g.num {
	case numNeg:
		if b == '0' {
			g.num = numZero
			return true
		}
		if b >= '1' && b <= '9' {
			g.num = numInt
			return true
		}
	case numZero:
		if b == '.' {
			g.num = numDot
			return true
		}
		if b == 'e' || b == 'E' {
			g.num = numExpSign
			return true
		}
	case numInt:
		if d {
			return true
		}
		if b == '.' {
			g.num = numDot
			return true
		}
		if b == 'e' || b == 'E' {
			g.num = numExpSign
			return true
		}
	case numDot:
		if d {
			g.num = numFrac
			return true
		}
	case numFrac:
		if d {
			return true
		}
		if b == 'e' || b == 'E' {
			g.num = numExpSign
			return true
		}
	case numExpSign:
		if b == '+' || b == '-' {
			g.num = numExpDigit
			return true
		}
		if d {
			g.num = numExp
			return true
		}
	case numExpDigit:
		if d {
			g.num = numExp
			return true
		}
	case numExp:
		if d {
			return true
		}
	}
	return false
}

// valueComplete transitions after a value finishes, per the enclosing container.
func (g *jsonGrammar) valueComplete() {
	if len(g.stack) == 0 {
		g.state = jsEnd
		return
	}
	switch g.stack[len(g.stack)-1] {
	case '{':
		g.state = jsObjNext
	case '[':
		g.state = jsArrNext
	}
}

func (g *jsonGrammar) push(c byte) { g.stack = append(g.stack, c) }

func (g *jsonGrammar) pop() {
	if len(g.stack) > 0 {
		g.stack = g.stack[:len(g.stack)-1]
	}
}

func isWS(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }
func isHex(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F')
}
