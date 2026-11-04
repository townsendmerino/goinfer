package serveapp

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A DNS-rebinding request reaches the loopback listener with the attacker's name in Host (and the same name in Origin, which is what
// defeated sameOrigin). hostGuard must refuse it before any handler runs, and answer every name that legitimately reaches loopback.
func TestHostGuard_refusesRebindingNamesOnALoopbackListener(t *testing.T) {
	reached := 0
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached++; w.WriteHeader(http.StatusOK) })
	h := hostGuard(inner, "127.0.0.1:8080", []string{"goinfer.lan", " Proxy.Example.com. "})

	for _, c := range []struct {
		host string
		want int
	}{
		{"127.0.0.1:8080", 200}, {"127.0.0.1", 200}, {"localhost:8080", 200}, {"LocalHost", 200}, {"[::1]:8080", 200}, {"[::1]", 200},
		{"goinfer.lan:8080", 200}, {"proxy.example.com", 200}, // the operator's names, normalized
		{"", 200}, // no Host (HTTP/1.0, a raw client): a browser always sends one
		{"evil.example:8080", 421}, {"evil.example", 421}, {"127.0.0.1.evil.example:8080", 421}, {"localhost.evil.example", 421},
		{"192.168.1.5:8080", 421}, {"0.0.0.0:8080", 421}, {"[::2]:8080", 421}, {"localhost:8080@evil.example", 421},
	} {
		r := httptest.NewRequest("GET", "/v1/models", nil)
		r.Host = c.host
		r.Header.Set("Origin", "http://"+c.host)
		w := httptest.NewRecorder()
		before := reached
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("Host %q: status %d, want %d", c.host, w.Code, c.want)
		}
		if c.want == 421 {
			if reached != before {
				t.Errorf("Host %q: the handler ran for a refused request", c.host)
			}
			if !strings.Contains(w.Body.String(), "allowed-hosts") {
				t.Errorf("Host %q: the refusal does not name the flag: %s", c.host, w.Body.String())
			}
		}
	}
}

// End to end on a real socket: connect to the loopback address but send a foreign Host, as a rebound browser does.
func TestHostGuard_realSocketRebinding(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skip(err)
	}
	srv := &http.Server{Handler: hostGuard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("secret")) }), ln.Addr().String(), nil)}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()
	for host, want := range map[string]int{ln.Addr().String(): 200, "evil.example:" + portOf(ln): 421} {
		req, _ := http.NewRequest("GET", "http://"+ln.Addr().String()+"/", nil)
		req.Host = host
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Errorf("Host %q: %d, want %d", host, resp.StatusCode, want)
		}
	}
}

func portOf(ln net.Listener) string { _, p, _ := net.SplitHostPort(ln.Addr().String()); return p }

// The guard is wired only for a loopback -addr: addrIsLoopback is its predicate, and a listener bound beyond loopback must not refuse names.
func TestHostGuard_predicate(t *testing.T) {
	for addr, loop := range map[string]bool{"127.0.0.1:8080": true, "localhost:8080": true, "[::1]:8080": true, "0.0.0.0:8080": false, ":8080": false, "10.0.0.5:8080": false} {
		if got := addrIsLoopback(addr); got != loop {
			t.Errorf("addrIsLoopback(%q) = %v, want %v", addr, got, loop)
		}
	}
	if got := splitHosts(" a.example , ,b.example,"); len(got) != 2 || got[0] != "a.example" || got[1] != "b.example" {
		t.Errorf("splitHosts: %v", got)
	}
}
