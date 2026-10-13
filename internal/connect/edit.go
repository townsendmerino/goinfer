package connect

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// member is one key of a JSON object with its value's raw text.
type member struct {
	key string
	val json.RawMessage
}

// object is a JSON object that remembers the order of its keys, so a file a person wrote is given back with their
// keys where they put them.
type object []member

// parseObject reads raw as one JSON object. It refuses anything else, and anything encoding/json refuses: a file with
// comments or trailing commas (JSONC) cannot be patched without losing them.
func parseObject(raw []byte) (object, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, errors.New("not a JSON object")
	}
	var o object
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := kt.(string)
		if !ok {
			return nil, errors.New("object key is not a string")
		}
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, err
		}
		o = append(o, member{key, v})
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, errors.New("text after the object")
	}
	return o, nil
}

// get returns the value of key as an object, empty when the key is absent, and an error when it holds something else.
func (o object) get(key string) (object, error) {
	for _, m := range o {
		if m.key == key {
			sub, err := parseObject(m.val)
			if err != nil {
				return nil, fmt.Errorf("%q is not an object: %v", key, err)
			}
			return sub, nil
		}
	}
	return nil, nil
}

// set replaces key's value, or appends the key when it is new.
func (o *object) set(key string, v any) {
	raw, ok := v.(json.RawMessage)
	if !ok {
		if sub, isObj := v.(object); isObj {
			raw = sub.compact()
		} else {
			raw, _ = json.Marshal(v)
		}
	}
	for i := range *o {
		if (*o)[i].key == key {
			(*o)[i].val = raw
			return
		}
	}
	*o = append(*o, member{key, raw})
}

func (o object) compact() json.RawMessage {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, m := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(m.key)
		b.Write(k)
		b.WriteByte(':')
		b.Write(m.val)
	}
	b.WriteByte('}')
	return b.Bytes()
}

// text is the object as an indented file.
func (o object) text() []byte {
	var b bytes.Buffer
	_ = json.Indent(&b, o.compact(), "", "  ")
	b.WriteByte('\n')
	return b.Bytes()
}

// tomlHeader reports whether a line opens a TOML table, and its name.
func tomlHeader(line string) (string, bool) {
	t := strings.TrimSpace(line)
	if i := strings.Index(t, "#"); i >= 0 && strings.HasSuffix(strings.TrimSpace(t[:i]), "]") {
		t = strings.TrimSpace(t[:i])
	}
	if strings.HasPrefix(t, "[") && strings.HasSuffix(t, "]") {
		return strings.TrimSpace(strings.Trim(t, "[]")), true
	}
	return "", false
}

// tomlKey is the bare key a `key = value` line sets, or "".
func tomlKey(line string) string {
	t := strings.TrimSpace(line)
	i := strings.Index(t, "=")
	if i <= 0 || strings.HasPrefix(t, "#") {
		return ""
	}
	return strings.TrimSpace(t[:i])
}

// tomlEdit patches a TOML file as text, since the standard library reads none: it sets top-level keys (the lines
// before the first table) and replaces or appends one table, leaving every other line as it was. It refuses a file
// that already holds that table in a form it cannot see as one block: an inline table, or dotted keys.
func tomlEdit(src string, top []tomlRow, table string, rows []tomlRow) (string, error) {
	parent := table[:strings.LastIndex(table, ".")]
	leaf := table[strings.LastIndex(table, ".")+1:]
	lines := strings.Split(strings.TrimRight(src, "\n"), "\n")
	if strings.TrimSpace(src) == "" {
		lines = nil
	}
	first := len(lines) // the first table header: top-level keys live above it
	for i, ln := range lines {
		if _, ok := tomlHeader(ln); ok {
			first = i
			break
		}
	}
	inTable := ""
	for i, ln := range lines {
		if h, ok := tomlHeader(ln); ok {
			inTable = h
			continue
		}
		k := tomlKey(ln)
		switch {
		case i < first && (k == parent || strings.HasPrefix(k, table+".") || strings.HasPrefix(k, parent+".")):
			return "", fmt.Errorf("line %d sets %s without a [%s] table; edit it by hand", i+1, k, table)
		case inTable == parent && (k == leaf || strings.HasPrefix(k, leaf+".")):
			return "", fmt.Errorf("line %d sets %s inside [%s] without a [%s] table; edit it by hand", i+1, k, parent, table)
		}
	}
	head := append([]string(nil), lines[:first]...)
	rest := append([]string(nil), lines[first:]...)
	for _, kv := range top {
		line, done := kv.key+" = "+kv.val, false
		for i, ln := range head {
			if tomlKey(ln) == kv.key {
				head[i], done = line, true
				break
			}
		}
		if !done {
			end := len(head)
			for end > 0 && strings.TrimSpace(head[end-1]) == "" {
				end--
			}
			head = append(head[:end], append([]string{line}, head[end:]...)...)
		}
	}
	block := []string{"[" + table + "]"}
	for _, kv := range rows {
		block = append(block, kv.key+" = "+kv.val)
	}
	start, end := -1, len(rest)
	for i, ln := range rest {
		h, ok := tomlHeader(ln)
		if !ok {
			continue
		}
		if start >= 0 {
			end = i
			break
		}
		if h == table {
			start = i
		}
	}
	if start >= 0 {
		for end > start+1 && strings.TrimSpace(rest[end-1]) == "" {
			end-- // keep the blank lines that separated the table from the next
		}
		rest = append(rest[:start], append(block, rest[end:]...)...)
	} else {
		if len(rest) > 0 && strings.TrimSpace(rest[len(rest)-1]) != "" {
			rest = append(rest, "")
		}
		rest = append(rest, block...)
	}
	if len(head) > 0 && strings.TrimSpace(head[len(head)-1]) != "" {
		head = append(head, "")
	}
	return strings.Join(append(head, rest...), "\n") + "\n", nil
}

// tomlRow is one TOML `key = value` line; val is the value's TOML text.
type tomlRow struct{ key, val string }
