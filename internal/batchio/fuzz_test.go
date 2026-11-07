package batchio

import "testing"

// A batch input file is attacker-supplied. A line ParseInput accepts must carry the id every later step keys on.
func FuzzParseInput(f *testing.F) {
	for _, s := range []string{
		`{"custom_id":"a1","method":"POST","url":"/v1/chat/completions","body":{"messages":[]}}`,
		"{\"custom_id\":\"a\"}\n\n  \n{\"custom_id\":\"a\"}\n",
		`{"custom_id":""}`, `{"custom_id":1}`, `[]`, "", "\n\n", "not json", "{",
	} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		lines, err := ParseInput(data)
		if err != nil {
			return
		}
		if len(lines) == 0 {
			t.Fatal("no lines and no error")
		}
		for i, l := range lines {
			if l.CustomID == "" {
				t.Fatalf("line %d was accepted with no custom_id", i)
			}
		}
		_ = CheckUnique(lines)
	})
}
