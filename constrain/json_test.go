package constrain

import "testing"

// validPrefix reports whether s is a valid prefix of some JSON document.
func validPrefix(s string) bool { return JSON().TryBytes([]byte(s)) }

// validComplete reports whether s is a complete, valid JSON value.
func validComplete(s string) bool {
	g := JSON()
	if !g.TryBytes([]byte(s)) {
		return false
	}
	g.Commit([]byte(s))
	return g.CanEnd()
}

func TestJSON_completeValues(t *testing.T) {
	ok := []string{
		`0`, `-0`, `123`, `-1.5`, `-1.5e+10`, `1E10`, `0.5`, `3.14159`,
		`"hi"`, `""`, `"a\"b\\c\/\b\f\n\r\t"`, `"é\uD83D"`,
		`true`, `false`, `null`,
		`{}`, `[]`, `{"a":1}`, `[1,2,3]`, `["x"]`,
		`{"a":[1,{"b":"c"}],"d":null,"e":true}`,
		`  42  `, "\n{\n\t\"k\" : [ 1 , 2 ]\n}\n", `[[[]]]`,
	}
	for _, s := range ok {
		if !validComplete(s) {
			t.Errorf("validComplete(%q) = false, want true", s)
		}
	}
}

func TestJSON_validPrefixesNotComplete(t *testing.T) {
	// Valid so far, but not yet a complete document (EOS must be masked).
	partial := []string{
		``, `{`, `[`, `[1`, `[1,`, `{"a"`, `{"a":`, `{"a":1`, `{"a":1,`,
		`tru`, `fals`, `nul`, `-`, `1.`, `1e`, `1e+`, `"unterminated`,
		`"esc\`, `"\u00`, `  `, `[1,2`,
	}
	for _, s := range partial {
		if !validPrefix(s) {
			t.Errorf("validPrefix(%q) = false, want true (valid prefix)", s)
		}
		if validComplete(s) {
			t.Errorf("validComplete(%q) = true, want false (incomplete)", s)
		}
	}
}

func TestJSON_invalidPrefixes(t *testing.T) {
	// Not a valid prefix of any JSON document — TryBytes must reject.
	bad := []string{
		`}`, `]`, `,`, `:`, `01`, `00`, `1.2.3`, `--1`, `+1`, `.5`,
		`[1,]`, `[,]`, `{,}`, `{"a" 1}`, `{"a":1 "b":2}`, `[1 2]`, `{1:2}`,
		`truex`, `nulll`, `tru e`, `"\x"`, `"a` + "\n" + `b"`, `{}x`, `[]]`,
		`{"a":1}}`, `nan`, `Infinity`, `'single'`,
	}
	for _, s := range bad {
		if validPrefix(s) {
			t.Errorf("validPrefix(%q) = true, want false (invalid)", s)
		}
	}
}

// TestJSON_byteByByteMatchesWhole: feeding bytes one Commit at a time must reach
// the same accept state as one Commit of the whole string (the grammar is a
// proper streaming automaton — no lookahead beyond one byte).
func TestJSON_byteByByte(t *testing.T) {
	for _, s := range []string{`{"a":[1,2.5e3,null],"b":true}`, `-0.5`, `"xAy"`} {
		g := JSON()
		for i := 0; i < len(s); i++ {
			if !g.TryBytes([]byte{s[i]}) {
				t.Fatalf("%q: byte %d (%q) rejected", s, i, s[i])
			}
			g.Commit([]byte{s[i]})
		}
		if !g.CanEnd() {
			t.Errorf("%q: not complete after byte-by-byte commit", s)
		}
	}
}
