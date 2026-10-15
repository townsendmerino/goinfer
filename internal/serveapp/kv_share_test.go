package serveapp

import "testing"

// TestKVShareOffFor is G4(c)'s serve half (docs/measurements/kv2-2026-10-14.md): sharing is off under -kv-share off,
// and whenever the server answers the network without an API key; otherwise every request is one scope.
func TestKVShareOffFor(t *testing.T) {
	for _, tc := range []struct {
		name                        string
		kvShare, key, addr, allowed string
		want                        bool
	}{
		{"loopback, no key: one local user, shared", "on", "", "127.0.0.1:8080", "", false},
		{"loopback, a key: shared", "on", "k", "127.0.0.1:8080", "", false},
		{"beyond loopback with a key: shared", "on", "k", "0.0.0.0:8080", "", false},
		{"beyond loopback, no key (serve refuses to start; the rule still says off)", "on", "", "0.0.0.0:8080", "", true},
		{"loopback behind a proxy (-allowed-hosts), no key: off", "on", "", "127.0.0.1:8080", "llm.example.com", true},
		{"behind a proxy with a key: shared", "on", "k", "127.0.0.1:8080", "llm.example.com", false},
		{"-kv-share off", "off", "k", "127.0.0.1:8080", "", true},
	} {
		if got := kvShareOffFor(tc.kvShare, tc.key, tc.addr, tc.allowed); got != tc.want {
			t.Errorf("%s: kvShareOffFor = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestKVShare_reachesTheModel: the flag's answer is the decoder option every model is loaded with.
func TestKVShare_reachesTheModel(t *testing.T) {
	for _, off := range []bool{false, true} {
		cfg := config{kvShareOff: off}
		if got := (modelSpec{}).options(cfg).KVShareOff; got != off {
			t.Errorf("kvShareOff %v: decoder.Options.KVShareOff = %v", off, got)
		}
	}
}

func TestKVShare_validate(t *testing.T) {
	if err := (config{kvShare: "maybe"}).validate(); err == nil {
		t.Error("-kv-share maybe was accepted")
	}
}
