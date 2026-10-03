package clef

// A JSON value tree and a renderer that reproduces Python's
//
//	json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
//
// byte for byte, which is what Cloudflare's reference encoder (joint_schema_model.py, render) puts in the prompt. Go's encoding/json
// differs in four ways that all change the token ids: it keeps map order unless sorted, it escapes <, > and & and U+2028/2029, it
// spells floats differently (1e+21 vs Python's 1e+21, 100000 vs 100000.0, 1e-7 vs 1e-07), and it decodes into float64, losing
// integers past 2^53. So the tree keeps every number as its source literal and renders from that.
//
// What it refuses, all of which Python's json.loads would accept: NaN, Infinity and -Infinity literals, a lone surrogate escape,
// and invalid UTF-8. A request carrying one is an error here rather than a prompt the reference would have built differently.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

type kind uint8

const (
	kNull kind = iota
	kBool
	kNumber
	kString
	kArray
	kObject
)

// value is one parsed JSON value. Numbers keep their source literal (num); an object keeps its keys in document order, with a
// repeated key updating the first occurrence's value in place, which is what a Python dict does on json.loads.
type value struct {
	k    kind
	b    bool
	num  string
	s    string
	arr  []*value
	keys []string
	vals []*value
}

func (v *value) isNull() bool { return v == nil || v.k == kNull }

// get returns the member of an object, or nil if absent (or v is not an object).
func (v *value) get(key string) *value {
	if v == nil || v.k != kObject {
		return nil
	}
	for i, k := range v.keys {
		if k == key {
			return v.vals[i]
		}
	}
	return nil
}

// falsy is Python truthiness for a parsed JSON value: None, False, 0, 0.0, "", [] and {}.
func (v *value) falsy() bool {
	if v == nil {
		return true
	}
	switch v.k {
	case kNull:
		return true
	case kBool:
		return !v.b
	case kNumber:
		f, err := strconv.ParseFloat(v.num, 64)
		return err == nil && f == 0
	case kString:
		return v.s == ""
	case kArray:
		return len(v.arr) == 0
	default:
		return len(v.keys) == 0
	}
}

// parse decodes one JSON document. It validates with encoding/json first, so the hand parser below only ever sees well-formed input.
func parse(raw []byte) (*value, error) {
	if !utf8.Valid(raw) {
		return nil, errors.New("clef: request is not valid UTF-8")
	}
	if !json.Valid(raw) {
		return nil, errors.New("clef: request is not valid JSON")
	}
	p := &parser{b: raw}
	v, err := p.value()
	if err != nil {
		return nil, err
	}
	return v, nil
}

type parser struct {
	b []byte
	i int
}

func (p *parser) ws() {
	for p.i < len(p.b) && (p.b[p.i] == ' ' || p.b[p.i] == '\t' || p.b[p.i] == '\n' || p.b[p.i] == '\r') {
		p.i++
	}
}

func (p *parser) value() (*value, error) {
	p.ws()
	switch c := p.b[p.i]; {
	case c == '{':
		p.i++
		v := &value{k: kObject}
		idx := map[string]int{}
		p.ws()
		if p.b[p.i] == '}' {
			p.i++
			return v, nil
		}
		for {
			p.ws()
			key, err := p.str()
			if err != nil {
				return nil, err
			}
			p.ws()
			p.i++ // ':'
			val, err := p.value()
			if err != nil {
				return nil, err
			}
			if at, dup := idx[key]; dup {
				v.vals[at] = val
			} else {
				idx[key] = len(v.keys)
				v.keys, v.vals = append(v.keys, key), append(v.vals, val)
			}
			p.ws()
			c := p.b[p.i]
			p.i++
			if c == '}' {
				return v, nil
			}
		}
	case c == '[':
		p.i++
		v := &value{k: kArray}
		p.ws()
		if p.b[p.i] == ']' {
			p.i++
			return v, nil
		}
		for {
			el, err := p.value()
			if err != nil {
				return nil, err
			}
			v.arr = append(v.arr, el)
			p.ws()
			c := p.b[p.i]
			p.i++
			if c == ']' {
				return v, nil
			}
		}
	case c == '"':
		s, err := p.str()
		if err != nil {
			return nil, err
		}
		return &value{k: kString, s: s}, nil
	case c == 't':
		p.i += 4
		return &value{k: kBool, b: true}, nil
	case c == 'f':
		p.i += 5
		return &value{k: kBool}, nil
	case c == 'n':
		p.i += 4
		return &value{k: kNull}, nil
	default:
		start := p.i
		for p.i < len(p.b) && strings.IndexByte("+-0123456789.eE", p.b[p.i]) >= 0 {
			p.i++
		}
		return &value{k: kNumber, num: string(p.b[start:p.i])}, nil
	}
}

// str decodes a JSON string starting at the opening quote.
func (p *parser) str() (string, error) {
	p.i++ // opening quote
	var out strings.Builder
	for {
		c := p.b[p.i]
		switch {
		case c == '"':
			p.i++
			return out.String(), nil
		case c != '\\':
			out.WriteByte(c)
			p.i++
		default:
			e := p.b[p.i+1]
			p.i += 2
			switch e {
			case '"', '\\', '/':
				out.WriteByte(e)
			case 'b':
				out.WriteByte('\b')
			case 'f':
				out.WriteByte('\f')
			case 'n':
				out.WriteByte('\n')
			case 'r':
				out.WriteByte('\r')
			case 't':
				out.WriteByte('\t')
			case 'u':
				r, err := p.hex4()
				if err != nil {
					return "", err
				}
				if utf16.IsSurrogate(r) {
					if r < 0xDC00 && p.i+1 < len(p.b) && p.b[p.i] == '\\' && p.b[p.i+1] == 'u' {
						save := p.i
						p.i += 2
						lo, err := p.hex4()
						if err == nil && lo >= 0xDC00 && lo < 0xE000 {
							out.WriteRune(utf16.DecodeRune(r, lo))
							continue
						}
						p.i = save
					}
					return "", errors.New("clef: a lone surrogate escape in the request; the reference would tokenize a surrogate this port cannot represent")
				}
				out.WriteRune(r)
			}
		}
	}
}

func (p *parser) hex4() (rune, error) {
	if p.i+4 > len(p.b) {
		return 0, errors.New("clef: truncated \\u escape")
	}
	n, err := strconv.ParseUint(string(p.b[p.i:p.i+4]), 16, 32)
	if err != nil {
		return 0, err
	}
	p.i += 4
	return rune(n), nil
}

// dumps renders v like Python's json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True).
func dumps(v *value) (string, error) {
	var b bytes.Buffer
	if err := writeValue(&b, v); err != nil {
		return "", err
	}
	return b.String(), nil
}

func writeValue(b *bytes.Buffer, v *value) error {
	if v == nil {
		b.WriteString("null")
		return nil
	}
	switch v.k {
	case kNull:
		b.WriteString("null")
	case kBool:
		if v.b {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case kNumber:
		s, err := pyNumber(v.num)
		if err != nil {
			return err
		}
		b.WriteString(s)
	case kString:
		writeString(b, v.s)
	case kArray:
		b.WriteByte('[')
		for i, el := range v.arr {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeValue(b, el); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case kObject:
		order := make([]int, len(v.keys))
		for i := range order {
			order[i] = i
		}
		// UTF-8 byte order is code point order, which is Python's str ordering.
		sort.Slice(order, func(a, c int) bool { return v.keys[order[a]] < v.keys[order[c]] })
		b.WriteByte('{')
		for n, i := range order {
			if n > 0 {
				b.WriteByte(',')
			}
			writeString(b, v.keys[i])
			b.WriteByte(':')
			if err := writeValue(b, v.vals[i]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	}
	return nil
}

// writeString is json.dumps's string encoding with ensure_ascii=False: it escapes the quote, the backslash and the C0 controls, and
// nothing else (0x7f, U+2028 and everything above ASCII go through raw).
func writeString(b *bytes.Buffer, s string) {
	b.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if c < 0x20 {
				fmt.Fprintf(b, `\u%04x`, c)
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteByte('"')
}

// pyNumber renders a JSON number literal as Python would after json.loads: an integer literal as the integer (so "-0" is "0" and
// "1E5", being a float literal, is "100000.0"), and a float as repr(float).
func pyNumber(lit string) (string, error) {
	if !strings.ContainsAny(lit, ".eE") {
		neg := strings.HasPrefix(lit, "-")
		digits := strings.TrimPrefix(lit, "-")
		if strings.Trim(digits, "0") == "" {
			return "0", nil
		}
		if neg {
			return "-" + digits, nil
		}
		return digits, nil
	}
	f, err := strconv.ParseFloat(lit, 64)
	if err != nil && !math.IsInf(f, 0) {
		return "", err
	}
	if math.IsInf(f, 0) {
		return "", fmt.Errorf("clef: number %s overflows a float64; Python renders it as Infinity, which is not JSON", lit)
	}
	return pyFloatRepr(f), nil
}

// pyFloatRepr is repr(float): the shortest digits that round-trip, in fixed notation when the decimal point sits in (-4, 16] relative
// to the digit string and in exponent notation (two-digit minimum, explicit sign) otherwise, with ".0" on an integral fixed value.
func pyFloatRepr(f float64) string {
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	e := strconv.FormatFloat(math.Abs(f), 'e', -1, 64) // d[.ddd]e±XX
	mant, expS, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expS)
	digits := strings.Replace(mant, ".", "", 1)
	decpt := exp + 1 // value = 0.DIGITS x 10^decpt
	var out string
	if decpt > -4 && decpt <= 16 {
		switch {
		case decpt <= 0:
			out = "0." + strings.Repeat("0", -decpt) + digits
		case decpt >= len(digits):
			out = digits + strings.Repeat("0", decpt-len(digits)) + ".0"
		default:
			out = digits[:decpt] + "." + digits[decpt:]
		}
	} else {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		sign := "+"
		x := decpt - 1
		if x < 0 {
			sign, x = "-", -x
		}
		out = fmt.Sprintf("%se%s%02d", m, sign, x)
	}
	if f < 0 {
		return "-" + out
	}
	return out
}

// render is the reference's render(value): a string is returned as is, anything else is dumps.
func render(v *value) (string, error) {
	if v != nil && v.k == kString {
		return v.s, nil
	}
	return dumps(v)
}
