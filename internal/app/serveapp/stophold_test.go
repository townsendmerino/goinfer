package serveapp

import "testing"

// TestStopTailHold gates M2: the streamer must hold back a trailing partial stop
// match so a boundary-spanning stop ("END" as "E"+"ND") never leaks its prefix.
func TestStopTailHold(t *testing.T) {
	for _, c := range []struct {
		name  string
		text  string
		stops []string
		want  int
	}{
		{"no stops", "hello E", nil, 0},
		{"no partial match", "hello world", []string{"END"}, 0},
		{"one-char partial", "hello E", []string{"END"}, 1},
		{"two-char partial", "hello EN", []string{"END"}, 2},
		{"full match is not held (firstStop handles it)", "hello END", []string{"END"}, 0},
		{"past-full trailing text not held", "hello ENDx", []string{"END"}, 0},
		{"longest across stops wins", "chat <|", []string{"<|im_end|>", "<|"}, 2},
		{"partial shorter than a full stop earlier in text", "a<|b<", []string{"<|end|>"}, 1},
		{"empty stop ignored", "abc", []string{""}, 0},
		{"whole text is a proper prefix", "EN", []string{"END"}, 2},
		{"text shorter, matches short prefix", "E", []string{"END"}, 1},
		{"newline-user multi-token stop", "reply\n\nUser", []string{"\n\nUser:"}, 6},
	} {
		if got := stopTailHold(c.text, c.stops); got != c.want {
			t.Errorf("%s: stopTailHold(%q, %v) = %d, want %d", c.name, c.text, c.stops, got, c.want)
		}
	}
}
