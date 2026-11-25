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

// A key-free loopback server refuses a mutating /v1/ request a browser sends cross-origin (a page open in the user's browser can
// send one; the browser blocks only the reading of the reply), and passes same-origin browser requests, requests with neither
// Origin nor Sec-Fetch-Site (curl, the SDKs), reads, and everything on a keyed or non-loopback server.
func TestBrowserGuard_refusesCrossOriginWritesOnAKeyFreeLoopbackServer(t *testing.T) {
	const addr = "127.0.0.1:8080"
	reached := 0
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached++; w.WriteHeader(http.StatusOK) })
	for _, c := range []struct {
		name, addr, key, method, path, host, origin, fetchSite string
		want                                                   int
		allowed                                                string
	}{
		{"cross-origin Origin", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "http://evil.example", "", 403, ""},
		{"another local port", addr, "", "POST", "/v1/chat/completions", "localhost:8080", "http://localhost:3000", "", 403, ""},
		{"Origin null", addr, "", "POST", "/v1/messages", "127.0.0.1:8080", "null", "", 403, ""},
		{"unparsable Origin", addr, "", "POST", "/v1/messages", "127.0.0.1:8080", "http://[::1", "", 403, ""},
		{"Sec-Fetch-Site cross-site", addr, "", "POST", "/v1/files", "127.0.0.1:8080", "", "cross-site", 403, ""},
		{"DELETE cross-origin", addr, "", "DELETE", "/v1/jobs/x", "127.0.0.1:8080", "http://evil.example", "", 403, ""},
		{"PUT cross-origin", addr, "", "PUT", "/v1/x", "127.0.0.1:8080", "http://evil.example", "", 403, ""},
		{"PATCH cross-origin", addr, "", "PATCH", "/v1/x", "127.0.0.1:8080", "http://evil.example", "", 403, ""},
		{"same-origin Origin", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "http://127.0.0.1:8080", "same-origin", 200, ""},
		{"same-origin, case and default port", "localhost:80", "", "POST", "/v1/chat/completions", "LOCALHOST", "http://localhost:80", "", 200, ""},
		{"same-origin IPv6", "[::1]:8080", "", "POST", "/v1/chat/completions", "[::1]:8080", "http://[::1]:8080", "", 200, ""},
		{"absent Origin (SDK, curl)", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "", "", 200, ""},
		{"GET cross-origin", addr, "", "GET", "/v1/models", "127.0.0.1:8080", "http://evil.example", "cross-site", 200, ""},
		{"not under /v1/", addr, "", "POST", "/health", "127.0.0.1:8080", "http://evil.example", "", 200, ""},
		{"keyed server", addr, "k", "POST", "/v1/chat/completions", "127.0.0.1:8080", "http://evil.example", "cross-site", 200, ""},
		{"non-loopback server", "0.0.0.0:8080", "", "POST", "/v1/chat/completions", "10.0.0.5:8080", "http://evil.example", "cross-site", 200, ""},
		// A reverse proxy that rewrites Host to the loopback (nginx's default proxy_set_header Host $proxy_host).
		{"proxied page under an -allowed-hosts name", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "https://llm.example.com", "same-origin", 200, "llm.example.com"},
		{"proxied page, name not listed", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "https://llm.example.com", "same-origin", 403, ""},
		{"foreign Origin with -allowed-hosts set", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "https://evil.example", "", 403, "llm.example.com"},
		{"listed name but Sec-Fetch-Site cross-site", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "https://llm.example.com", "cross-site", 403, "llm.example.com"},
	} {
		h := serveHandler(inner, c.addr, c.key, c.allowed)
		r := httptest.NewRequest(c.method, c.path, nil)
		r.Host = c.host
		if c.origin != "" {
			r.Header.Set("Origin", c.origin)
		}
		if c.fetchSite != "" {
			r.Header.Set("Sec-Fetch-Site", c.fetchSite)
		}
		w := httptest.NewRecorder()
		before := reached
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("%s: status %d, want %d (%s)", c.name, w.Code, c.want, w.Body.String())
		}
		if c.want == 403 && reached != before {
			t.Errorf("%s: the handler ran for a refused request", c.name)
		}
	}
}
