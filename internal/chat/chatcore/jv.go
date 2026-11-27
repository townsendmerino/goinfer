package chatcore

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// JKind is the JSON value kind held by JV.
type JKind uint8

const (
	JNull JKind = iota
	JBool
	JNum
	JStr
	JArr
	JObj
)

// JV is an order-preserving JSON value that retains object key order and number source text.
type JV struct {
	Kind JKind
	B    bool
	S    string
	Arr  []JV
	Keys []string
	Vals map[string]JV
}

// ParseJV decodes a single JSON value while preserving object key order and number literals.
func ParseJV(raw []byte) (JV, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := readJV(dec)
	if err != nil {
		return JV{}, err
	}
	if dec.More() {
		return JV{}, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func readJV(dec *json.Decoder) (JV, error) {
	tok, err := dec.Token()
	if err != nil {
		return JV{}, err
	}
	switch t := tok.(type) {
	case json.Delim:
		if t == '{' {
			v := JV{Kind: JObj, Vals: map[string]JV{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return JV{}, err
				}
				key, _ := kt.(string)
				val, err := readJV(dec)
				if err != nil {
					return JV{}, err
				}
				if _, dup := v.Vals[key]; !dup {
					v.Keys = append(v.Keys, key)
				}
				v.Vals[key] = val
			}
			if _, err := dec.Token(); err != nil {
				return JV{}, err
			}
			return v, nil
		}
		v := JV{Kind: JArr}
		for dec.More() {
			el, err := readJV(dec)
			if err != nil {
				return JV{}, err
			}
			v.Arr = append(v.Arr, el)
		}
		if _, err := dec.Token(); err != nil {
			return JV{}, err
		}
		return v, nil
	case string:
		return JV{Kind: JStr, S: t}, nil
	case json.Number:
		return JV{Kind: JNum, S: t.String()}, nil
	case bool:
		return JV{Kind: JBool, B: t}, nil
	}
	return JV{Kind: JNull}, nil
}

// Get looks up key k on an object JV.
func (v JV) Get(k string) (JV, bool) {
	if v.Kind != JObj {
		return JV{}, false
	}
	x, ok := v.Vals[k]
	return x, ok
}

// Truthy reports Python/Jinja truthiness of v.
func (v JV) Truthy() bool {
	switch v.Kind {
	case JBool:
		return v.B
	case JNum:
		return v.S != "0" && v.S != "0.0"
	case JStr:
		return v.S != ""
	case JArr:
		return len(v.Arr) > 0
	case JObj:
		return len(v.Keys) > 0
	}
	return false
}

// Text formats v the way Jinja string concatenation renders it.
func (v JV) Text() string {
	switch v.Kind {
	case JStr, JNum:
		return v.S
	case JBool:
		if v.B {
			return "True"
		}
		return "False"
	case JNull:
		return "None"
	}
	return PyDumps(v)
}

// PyStr formats v the way Python's str() renders decoded JSON arguments in XML tool calls.
func PyStr(v JV) string {
	switch v.Kind {
	case JStr, JNum:
		return v.S
	case JBool, JNull:
		return v.Text()
	}
	return PyDumps(v)
}

// PyDumps formats v like Python's json.dumps(..., ensure_ascii=False) with ", " and ": " separators.
func PyDumps(v JV) string {
	var b strings.Builder
	writePy(&b, v)
	return b.String()
}

func writePy(b *strings.Builder, v JV) {
	switch v.Kind {
	case JNull:
		b.WriteString("null")
	case JBool:
		if v.B {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case JNum:
		b.WriteString(v.S)
	case JStr:
		b.WriteString(PyQuote(v.S))
	case JArr:
		b.WriteByte('[')
		for i, el := range v.Arr {
			if i > 0 {
				b.WriteString(", ")
			}
			writePy(b, el)
		}
		b.WriteByte(']')
	case JObj:
		b.WriteByte('{')
		for i, k := range v.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(PyQuote(k))
			b.WriteString(": ")
			writePy(b, v.Vals[k])
		}
		b.WriteByte('}')
	}
}

// PyQuote formats a string like Python's json.dumps(s, ensure_ascii=False).
func PyQuote(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// PyArgs formats raw JSON call arguments with PyDumps, preserving malformed raw bytes verbatim.
func PyArgs(raw json.RawMessage) string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "{}"
	}
	v, err := ParseJV(raw)
	if err != nil {
		return string(raw)
	}
	return PyDumps(v)
}
