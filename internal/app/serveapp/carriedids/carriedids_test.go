package carriedids

import (
	"bytes"
	"crypto/sha256"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// The v2 blob's format: every field survives a round trip, and a blob that is cut short or garbled does not parse.
func TestBlob_roundTrip(t *testing.T) {
	k := NewKey([]byte("round-trip test key of 32 bytes!"))
	enc := k.Mint("model-x", "ns-y", "reasoning text\nwith a newline", []int{0, 1, 127, 128, 300000, 5}, false)
	if !strings.HasPrefix(enc, V2Prefix) {
		t.Fatalf("minted %q", enc)
	}
	b, err := ParseBlob(enc)
	if err != nil {
		t.Fatal(err)
	}
	if b.Whole || b.KeyID != k.ID || b.Model != "model-x" || b.NS != "ns-y" || b.Text != "reasoning text\nwith a newline" ||
		!slices.Equal(b.IDs, []int{0, 1, 127, 128, 300000, 5}) || len(b.MAC) != 32 {
		t.Errorf("parsed %+v", b)
	}
	if !k.VerifyMAC(b) {
		t.Error("the key does not verify its own MAC")
	}
	b.IDs[2]++
	if k.VerifyMAC(b) {
		t.Error("the MAC verifies after an id changed")
	}
	for _, bad := range []string{enc[:len(enc)-6], enc[:len(V2Prefix)+3], V2Prefix + "%%%", V2Prefix} {
		if _, err := ParseBlob(bad); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// --ids-key-file: 32 bytes or more, mode 0600 or stricter, else refused; the key id is the first 8 bytes of SHA-256 of the key.
func TestKeyFile(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte, mode os.FileMode) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, mode); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(p, mode); err != nil {
			t.Fatal(err)
		}
		return p
	}
	key := bytes.Repeat([]byte{7}, 40)
	k, err := ReadKeyFile(write("ok", key, 0o600))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(key)
	if !bytes.Equal(k.ID[:], sum[:8]) || !bytes.Equal(k.key, key) {
		t.Errorf("key id %x, want %x", k.ID, sum[:8])
	}
	if _, err := ReadKeyFile(write("short", key[:31], 0o600)); err == nil {
		t.Error("a 31-byte key was accepted")
	}
	if _, err := ReadKeyFile(write("open", key, 0o644)); err == nil && runtime.GOOS != "windows" { // Windows carries no mode bits
		t.Error("a key file readable by others was accepted")
	}
	if _, err := ReadKeyFile(filepath.Join(dir, "missing")); err == nil {
		t.Error("a missing key file was accepted")
	}
	if w := KeyFileWarning("windows", "k"); !strings.Contains(w, "permissions not checked on Windows") {
		t.Errorf("windows: warning %q", w)
	}
	if KeyFileWarning("linux", "k") != "" || KeyFileWarning("windows", "") != "" {
		t.Error("a warning off Windows, or with no key file")
	}
	r1, err1 := RandomKey()
	r2, err2 := RandomKey()
	if err1 != nil || err2 != nil || len(r1.key) != 32 || bytes.Equal(r1.key, r2.key) {
		t.Error("the default key is not 32 random bytes per start")
	}
}

// The closed list (Format) on its own, with no tokenizer: what it lets through and what stays content. The Mellum2.1
// cases themselves are TestIDsCarried_templateFormat in serveapp, which needs the per-machine tokenizer.
func TestFormat_align(t *testing.T) {
	f := Format{ROpen: "<think>", RClose: "</think>", COpen: "<tool_call>", CClose: "</tool_call>"}
	call := func(obj string) string { return "<tool_call>\n" + obj + "\n</tool_call>" }
	for _, tc := range []struct {
		name, d, t string
		ok         bool
	}{
		{"a newline the template adds before the closing delimiter", "<think>\nNow</think>\n\nok", "<think>\nNow\n</think>\n\nok", true},
		{"a newline the template drops after the opening delimiter", "<think>\n\nNow\n</think>", "<think>\nNow\n</think>", true},
		{"the template's space after a call's key", call(`{"name": "f", "arguments":{"cmd":"a b"}}`), call(`{"name": "f", "arguments": {"cmd":"a b"}}`), true},
		{"separators re-spaced throughout a call", call(`{"name":"f","arguments":{"a":[1,2]}}`), call(`{"name": "f", "arguments": {"a": [1, 2]}}`), true},
		{"identical", "<think>\nx\n</think>", "<think>\nx\n</think>", true},

		{"a space inside an argument string", call(`{"arguments":{"cmd":"a b"}}`), call(`{"arguments":{"cmd":"a  b"}}`), false},
		{"a newline inside an argument string", call(`{"arguments":{"cmd":"ab"}}`), call(`{"arguments":{"cmd":"a\nb"}}`), false},
		{"a number split by a space", call(`{"arguments":{"n":12}}`), call(`{"arguments":{"n":1 2}}`), false},
		{"a space, not a newline, before the closing delimiter", "<think>\nNow</think>", "<think>\nNow </think>", false},
		{"a newline inside the reasoning, away from the delimiters", "<think>\na b\n</think>", "<think>\na\nb\n</think>", false},
		{"JSON spacing outside a call", `{"a":1}`, `{"a": 1}`, false},
		{"JSON spacing after a call has closed", call(`{"a":1}`) + `{"b":2}`, call(`{"a":1}`) + `{"b": 2}`, false},
		{"a one-character content edit", "<think>\nNow</think>", "<think>\nNox\n</think>", false},
		{"a difference before the first byte", "<think>\nNow</think>", "\n<think>\nNow</think>", false},
	} {
		end, at, differs, ok := f.Align(tc.d, tc.t)
		if ok != tc.ok {
			t.Errorf("%s: ok=%v, want %v", tc.name, ok, tc.ok)
			continue
		}
		if !ok {
			continue
		}
		if end != len(tc.t) || differs != (tc.d != tc.t) || len(at) != len(tc.d)+1 || at[len(tc.d)] != end {
			t.Errorf("%s: end %d of %d, differs %v, at %d entries", tc.name, end, len(tc.t), differs, len(at))
		}
		for i := range len(tc.d) { // every byte stands for an equal byte of t, or for a skipped whitespace run
			if c := tc.d[i]; tc.t[at[i]] != c && c != ' ' && c != '\n' {
				t.Errorf("%s: d[%d]=%q stands for t[%d]=%q", tc.name, i, c, at[i], tc.t[at[i]])
			}
		}
	}
}
