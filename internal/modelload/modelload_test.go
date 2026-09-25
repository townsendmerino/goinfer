package modelload

import (
	"context"
	"strings"
	"testing"
)

// N-25, now in the one place all three apps read a .giw's tok half: when the bytes are neither a GGUF
// nor a tokenizer.json, the error names BOTH attempts. serve's copy used to return only the JSON one,
// so a corrupt GGUF-sourced bundle reported "invalid JSON" and pointed at the wrong half of the file.
func TestTokenizerFromTok_reportsBothErrors(t *testing.T) {
	_, err := TokenizerFromTok([]byte("neither a GGUF nor JSON"))
	if err == nil {
		t.Fatal("garbage tok half parsed")
	}
	msg := err.Error()
	if !strings.Contains(msg, "not a GGUF") || !strings.Contains(msg, "not tokenizer.json") {
		t.Errorf("error must name both the GGUF and the JSON attempt, got: %v", err)
	}
}

// A plain path is returned untouched — the property that lets fit, serve and chat all resolve every
// --model value without changing what an existing path means.
func TestResolve_plainPathUntouched(t *testing.T) {
	for _, p := range []string{"/models/x.gguf", "rel/dir", "x.giw", "hfdir/hf:not-a-prefix"} {
		got, err := Resolve(context.Background(), p)
		if err != nil || got != p {
			t.Errorf("Resolve(%q) = %q, %v; want the path unchanged", p, got, err)
		}
	}
}
