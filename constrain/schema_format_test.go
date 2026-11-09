package constrain

import "testing"

// TestJSONSchema_formatIsEnforced: a string schema's common `format` (date, time, date-time, uuid, ipv4, email) is held by
// the grammar, so a value of the wrong shape cannot be generated; any other format stays annotation-only.
func TestJSONSchema_formatIsEnforced(t *testing.T) {
	cases := []struct {
		format    string
		ok, notOK []string
	}{
		{"date", []string{`"2026-11-09"`, `"1999-12-31"`, `"2024-02-29"`},
			[]string{`"2026-13-01"`, `"2026-00-10"`, `"2026-11-32"`, `"2026-11-00"`, `"26-11-09"`, `"2026-11-9"`, `"2026/11/09"`, `"today"`, `""`}},
		{"time", []string{`"13:05:00Z"`, `"23:59:60z"`, `"00:00:00.5+01:00"`, `"07:30:15.123-08:00"`},
			[]string{`"24:00:00Z"`, `"12:60:00Z"`, `"12:00:00"`, `"12:00Z"`, `"12:00:00.Z"`, `"12:00:00+1:00"`, `"noon"`}},
		{"date-time", []string{`"2026-11-09T13:05:00Z"`, `"2026-11-09t13:05:00.25+05:30"`},
			[]string{`"2026-11-09 13:05:00Z"`, `"2026-11-09T13:05:00"`, `"2026-11-09"`, `"13:05:00Z"`}},
		{"uuid", []string{`"123e4567-e89b-12d3-a456-426614174000"`, `"ABCDEF01-2345-6789-ABCD-EF0123456789"`},
			[]string{`"123e4567e89b12d3a456426614174000"`, `"123e4567-e89b-12d3-a456-42661417400"`, `"123e4567-e89b-12d3-a456-4266141740000"`, `"g23e4567-e89b-12d3-a456-426614174000"`}},
		{"ipv4", []string{`"192.168.0.1"`, `"0.0.0.0"`, `"255.255.255.255"`},
			[]string{`"256.1.1.1"`, `"1.2.3"`, `"1.2.3.4.5"`, `"01.2.3.4"`, `"1..2.3"`, `"a.b.c.d"`, `"1.2.3.4."`}},
		{"email", []string{`"a@b.co"`, `"first.last+tag@mail.example.org"`},
			[]string{`"a@b"`, `"@b.co"`, `"a@.co"`, `"a b@c.co"`, `".a@b.co"`, `"a..b@c.co"`, `"a@b..co"`, `"a@b.co."`, `"ab.co"`}},
	}
	for _, c := range cases {
		g, err := JSONSchema([]byte(`{"type":"string","format":"` + c.format + `"}`))
		if err != nil {
			t.Fatalf("%s: %v", c.format, err)
		}
		for _, s := range c.ok {
			if !accepts(g, s) {
				t.Errorf("format %s rejects %s", c.format, s)
			}
		}
		for _, s := range c.notOK {
			if accepts(g, s) {
				t.Errorf("format %s accepts %s", c.format, s)
			}
		}
	}
	// A format this package does not know stays an annotation: any string.
	g, err := JSONSchema([]byte(`{"type":"string","format":"hostname"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !accepts(g, `"any thing at all"`) {
		t.Error("an unknown format constrained the string")
	}
	// Inside an object, the formatted property is held and the rest are free.
	g, err = JSONSchema([]byte(`{"type":"object","properties":{"when":{"type":"string","format":"date"},"note":{"type":"string"}},"required":["when","note"],"additionalProperties":false}`))
	if err != nil {
		t.Fatal(err)
	}
	if !accepts(g, `{"when":"2026-11-09","note":"x"}`) || accepts(g, `{"when":"soon","note":"x"}`) {
		t.Error("a date property inside an object is not held to its format")
	}
}
